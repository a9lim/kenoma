"""Terminal front end: an inline prompt that passes for the user's shell,
with ghost text and loom navigation, plus a full-screen tree explorer."""

from __future__ import annotations

import asyncio
import re
import shutil
import sys
from dataclasses import dataclass
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application, get_app
from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
from prompt_toolkit.data_structures import Point
from prompt_toolkit.formatted_text import ANSI, StyleAndTextTuples
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.utils import get_cwidth

from kenoma.session import Node, Session

# Model output goes to the real terminal, so control bytes (escape
# sequences, C1 controls, carriage returns) are dropped; tab and newline stay.
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
_SGR = re.compile(r"\x1b\[[0-9;:]*m")

HINT = ("[kenoma: Alt-r re-roll · Alt-←/→ alternatives · Alt-↑/↓ back/forward · "
        "Alt-t tree · Ctrl-D exit]")

NAV_KEYS = [
    (("escape", "r"), "reroll"),
    (("escape", "left"), "prev"),
    (("escape", "right"), "next"),
    (("escape", "up"), "parent"),
    (("escape", "down"), "child"),
    (("escape", "t"), "tree"),
]


def sanitize(s: str) -> str:
    return _CTRL.sub("", s)


@dataclass
class Nav:
    kind: str
    text: str  # what was typed in the input line, restored afterwards


class _Ghost(AutoSuggest):
    """Shows the model's guess at the next command while the typed text is a
    prefix of it."""

    def __init__(self) -> None:
        self.text = ""

    def get_suggestion(self, buffer: Any, document: Any) -> Suggestion | None:
        typed = document.text
        if len(self.text) > len(typed) and self.text.startswith(typed):
            return Suggestion(self.text[len(typed):])
        return None


class UI:
    def __init__(self, session: Session, history: list[str]):
        self.s = session
        self.ghost = _Ghost()
        hist = InMemoryHistory()
        for cmd in history:
            hist.append_string(cmd)
        self.ps: PromptSession[Any] = PromptSession(history=hist, auto_suggest=self.ghost,
                                                    key_bindings=self._keys())

    def _keys(self) -> KeyBindings:
        kb = KeyBindings()

        def nav(kind: str) -> Any:
            def handler(event: Any) -> None:
                event.app.exit(result=Nav(kind, event.app.current_buffer.text))
            return handler

        for keys, kind in NAV_KEYS:
            kb.add(*keys)(nav(kind))
        return kb

    def run(self) -> None:
        print(HINT, file=sys.stderr)
        default = ""
        while True:
            node = self.s.current
            try:
                res = self.ps.prompt(ANSI(self.s.display_prompt(node)), default=default,
                                     pre_run=self._start_ghost)
            except KeyboardInterrupt:
                default = ""
                continue
            except EOFError:
                return
            default = ""
            if isinstance(res, Nav):
                default = res.text
                self._navigate(res.kind)
            else:
                self._turn(node, res)

    # -- ghost text ---------------------------------------------------------

    def _show_ghost(self, app: Any) -> None:
        buf = app.current_buffer
        buf.suggestion = self.ghost.get_suggestion(buf, buf.document)
        app.invalidate()

    def _start_ghost(self) -> None:
        node = self.s.current
        app = get_app()
        self.ghost.text = sanitize(node.ghost or "").replace("\t", " ")
        self._show_ghost(app)
        if node.ghost is None:
            app.create_background_task(self._ghost_task(app, node))

    async def _ghost_task(self, app: Any, node: Node) -> None:
        # Decoding runs on the event loop one token at a time, yielding in
        # between, so keystrokes are never more than a token behind and the
        # engine is never used from two places at once.
        it = self.s.ghost(node)
        try:
            for guess in it:
                self.ghost.text = sanitize(guess).replace("\t", " ")
                self._show_ghost(app)
                await asyncio.sleep(0)
        except Exception:
            pass  # the guess is optional; a failure just leaves none
        finally:
            it.close()  # type: ignore[attr-defined]

    # -- turns and navigation -----------------------------------------------

    def _turn(self, parent: Node, command: str) -> None:
        gen = self.s.turn(parent, command)
        shown = ""
        while True:
            try:
                piece = next(gen)
                shown += piece
                sys.stdout.write(sanitize(piece))
                sys.stdout.flush()
            except StopIteration as stop:
                node: Node = stop.value
                break
            except KeyboardInterrupt:
                # ^C arrived between pieces: hand it to the turn. If the turn
                # already died of it (it landed while the turn was being
                # recorded), there's no node to show.
                try:
                    gen.throw(KeyboardInterrupt())
                except StopIteration as stop:
                    node = stop.value
                    break
                except KeyboardInterrupt:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    return
            except Exception as e:
                gen.close()
                print(f"\n[kenoma: generation failed: {e}]", file=sys.stderr)
                return
        if node.interrupted:
            sys.stdout.write("\n")  # the terminal already echoed ^C
        elif node.body.startswith(shown):
            sys.stdout.write(sanitize(node.body[len(shown):]))
        sys.stdout.flush()
        self.s.visit(node)

    def _navigate(self, kind: str) -> None:
        cur = self.s.current
        target: Node | None
        if kind == "reroll":
            if cur.parent is None:
                return self._bell()
            self._redraw(cur.parent, cur.command)
            return self._turn(cur.parent, cur.command)
        if kind == "tree":
            target = tree_view(self.s)
            if target is None or target is cur:
                return  # leaving the alternate screen restored the old view
        elif kind == "parent":
            target = cur.parent
        elif kind == "child":
            target = cur.last_child
        else:
            sibs = self.s.siblings(cur)
            j = sibs.index(cur) + (1 if kind == "next" else -1)
            target = sibs[j] if 0 <= j < len(sibs) else None
        if target is None:
            return self._bell()
        self.s.visit(target)
        self._redraw(target)

    def _bell(self) -> None:
        sys.stdout.write("\a")
        sys.stdout.flush()

    def _redraw(self, node: Node, command: str | None = None) -> None:
        """Clear the screen and show the end of the transcript up to `node`
        (plus `command`, about to be re-run), leaving room for the prompt."""
        show = self.s.display_prompt
        text = "".join(show(n.parent) + n.command + "\n" + sanitize(n.body)
                       for n in node.path()[1:] if n.parent is not None)
        if command is not None:
            text += show(node) + command + "\n"
        cols, rows = shutil.get_terminal_size()
        reserve = 0 if command is not None else show(node).count("\n") + 1
        sys.stdout.write("\x1b[H\x1b[2J" + _tail(text, rows - reserve, cols))
        sys.stdout.flush()


def _tail(text: str, rows: int, cols: int) -> str:
    """The last lines of `text` that fit in `rows` terminal rows."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    out: list[str] = []
    used = 0
    for line in reversed(lines):
        width = get_cwidth(_SGR.sub("", line).expandtabs())
        height = max(1, -(-width // max(cols, 1)))
        if used + height > rows:
            break
        out.append(line)
        used += height
    return "".join(line + "\n" for line in reversed(out))


# -- tree explorer -----------------------------------------------------------

def _rows(root: Node) -> list[tuple[Node, str]]:
    """Flatten the tree for display. Chains of single children stay at one
    indentation; only branch points indent, with box-drawing guides."""
    rows: list[tuple[Node, str]] = []
    stack: list[tuple[Node, str, str]] = [(root, "", "")]
    while stack:
        n, first, cont = stack.pop()
        rows.append((n, first))
        while len(n.children) == 1:
            n = n.children[0]
            rows.append((n, cont))
        for i in reversed(range(len(n.children))):
            last = i == len(n.children) - 1
            stack.append((n.children[i], cont + ("└─ " if last else "├─ "),
                          cont + ("   " if last else "│  ")))
    return rows


def _first_line(s: str) -> str:
    for line in s.splitlines():
        if line.strip():
            return line.strip()
    return ""


def tree_view(session: Session) -> Node | None:
    """Full-screen explorer of the session tree. Returns the chosen node, or
    None if cancelled."""
    rows = _rows(session.root)
    index = {id(n): i for i, (n, _) in enumerate(rows)}
    on_path = {id(n) for n in session.current.path()}
    sel = [index[id(session.current)]]
    def tree_text() -> StyleAndTextTuples:
        out: StyleAndTextTuples = []
        for i, (n, guide) in enumerate(rows):
            label = "(session start)" if n.parent is None else (n.command or "⏎")
            base = "reverse " if i == sel[0] else ""
            if id(n) in on_path:
                base += "bold "
            out.append((base + "fg:ansibrightblack", guide))
            out.append((base, sanitize(label)))
            peek = _first_line(sanitize(n.body))
            if peek:
                out.append((base + "fg:ansibrightblack", "  " + peek))
            if n is session.current:
                out.append((base + "fg:ansicyan", "  ◀"))
            out.append(("", "\n"))
        return out

    def preview_text() -> Any:
        n = rows[sel[0]][0]
        if n.parent is None:
            return ANSI(sanitize(n.text[-4000:]) + session.display_prompt(n))
        return ANSI(session.display_prompt(n.parent) + sanitize(n.command) + "\n" + sanitize(n.body))

    def status_text() -> StyleAndTextTuples:
        n = rows[sel[0]][0]
        sibs = session.siblings(n)
        alt = f"  alternative {sibs.index(n) + 1}/{len(sibs)}" if len(sibs) > 1 else ""
        return [("reverse", f" ↑↓ move  ←→ parent/child  ⏎ jump  q cancel{alt} ")]

    kb = KeyBindings()

    def move(i: int) -> None:
        sel[0] = max(0, min(len(rows) - 1, i))

    @kb.add("up")
    @kb.add("k")
    def _(event: Any) -> None:
        move(sel[0] - 1)

    @kb.add("down")
    @kb.add("j")
    def _(event: Any) -> None:
        move(sel[0] + 1)

    @kb.add("left")
    @kb.add("h")
    def _(event: Any) -> None:
        n = rows[sel[0]][0]
        if n.parent is not None:
            move(index[id(n.parent)])

    @kb.add("right")
    @kb.add("l")
    def _(event: Any) -> None:
        n = rows[sel[0]][0]
        child = n.last_child or (n.children[0] if n.children else None)
        if child is not None:
            move(index[id(child)])

    @kb.add("enter")
    def _(event: Any) -> None:
        event.app.exit(result=rows[sel[0]][0])

    @kb.add("q")
    @kb.add("escape")
    @kb.add("c-c")
    def _(event: Any) -> None:
        event.app.exit(result=None)

    tree = Window(FormattedTextControl(tree_text, focusable=True, show_cursor=False,
                                       get_cursor_position=lambda: Point(0, sel[0])),
                  width=Dimension(weight=2))
    preview = Window(FormattedTextControl(preview_text), wrap_lines=True, width=Dimension(weight=3))
    layout = Layout(HSplit([
        VSplit([tree, Window(width=1, char="│", style="fg:ansibrightblack"), preview]),
        Window(FormattedTextControl(status_text), height=1),
    ]), focused_element=tree)
    app: Application[Node | None] = Application(layout=layout, key_bindings=kb, full_screen=True)
    return app.run()
