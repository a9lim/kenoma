"""The fake session as a tree of turns.

Each `Node` is one turn: the command the user typed, the output the model
wrote, and the prompt that followed. The transcript the model sees is the
path from the root (the seed) to the current node, so a re-rolled output or
a different command is just another child of the same parent.

A turn streams model output through a `Recognizer` until the model writes
the next prompt. The model's own tokens up to that prompt stay in the
transcript as they were sampled; whatever they don't cover (part of a token,
or a prompt kenoma had to supply) is carried as `leftover` text at the start
of the next segment, so it's tokenized together with the next command.
"""

from __future__ import annotations

import bisect
from collections.abc import Generator, Iterator
from dataclasses import dataclass, field

from kenoma.engine import Engine, Piece, Segment
from kenoma.template import Recognizer, Template

GHOST_MAX_TOKENS = 48
GHOST_MAX_CHARS = 160


@dataclass(eq=False)
class Node:
    parent: Node | None
    command: str  # "" for the root
    text: str  # model-facing text this turn adds to the transcript
    tokens: list[int] | None  # its token ids (local engine), or None
    leftover: str  # model-facing text owed to the start of the next segment
    body: str  # the command's output as written by the model
    values: dict[str, str]  # prompt hole values after this turn
    forced: bool  # kenoma supplied the prompt (the model didn't write one)
    interrupted: bool = False  # the user hit ^C while it ran
    prompt: str | None = None  # the model's own prompt, when it didn't fit the template
    ghost: str | None = None  # predicted next command, once computed
    children: list[Node] = field(default_factory=list)
    last_child: Node | None = None

    def path(self) -> list[Node]:
        out: list[Node] = []
        n: Node | None = self
        while n is not None:
            out.append(n)
            n = n.parent
        return out[::-1]


class Session:
    def __init__(self, engine: Engine, template: Template, *, max_new_tokens: int,
                 context_tokens: int, ghost: bool = True):
        self.engine = engine
        self.tmpl = template
        self.max_new_tokens = max_new_tokens
        self.budget = context_tokens
        self.ghost_enabled = ghost
        self.root = self.current = self._root("")
        # node -> (live stream positioned right after its prompt, ghost text so far)
        self._pending: dict[Node, tuple[Iterator[Piece], str]] = {}

    def _root(self, seed: str) -> Node:
        return Node(None, "", seed, self.engine.encode(seed), self.tmpl.text(self.tmpl.values),
                    "", dict(self.tmpl.values), forced=True)

    def start(self, seed: str) -> None:
        """Install the seed transcript (its newest lines, if it's over half
        the context limit) and get it into the engine's cache."""
        self.root = self.current = self._root(self._fit(seed, self._limit() // 2))
        self.engine.warm(self._segments(self.root))

    def _fit(self, seed: str, limit: int) -> str:
        while seed:
            cost = self._text_cost(seed)
            if cost <= limit:
                break
            cut = seed.find("\n", int(len(seed) * (1 - limit / cost)))
            seed = seed[cut + 1:] if cut >= 0 else ""
        return seed

    # -- context ------------------------------------------------------------

    def _text_cost(self, text: str) -> int:
        toks = self.engine.encode(text)
        return len(toks) if toks is not None else len(text) // 3 + 1

    def _cost(self, n: Node) -> int:
        return len(n.tokens) if n.tokens is not None else len(n.text) // 3 + 1

    def _limit(self) -> int:
        """The context budget left after reserving room for one turn's output."""
        return max(self.budget - self.max_new_tokens, self.budget // 2)

    def window(self, node: Node, extra: int = 0) -> list[Node]:
        """The nodes whose text the model sees after `node`, with `extra`
        tokens about to be appended. A pure function of the path, so a
        re-roll always sees the same history as the original.

        Positions count cost from the root. Past the limit, the window starts
        at the first node beyond a cut that advances in steps of half the
        limit, so consecutive turns keep the same start (and the engine's
        cached prefix) until the next step. The last node is always kept,
        even if it alone is over the limit."""
        path = node.path()
        costs = [self._cost(n) for n in path]
        limit = self._limit()
        total = sum(costs) + extra
        if total <= limit:
            return path
        half = max(limit // 2, 1)
        cut = ((total - limit) // half + 1) * half
        pos = 0
        for i, c in enumerate(costs):
            if pos >= cut:
                return path[i:]
            pos += c
        return path[-1:]

    def _segments(self, node: Node, extra: int = 0) -> list[Segment]:
        return [(n.text, n.tokens) for n in self.window(node, extra)]

    def _prompt_text(self, node: Node) -> str:
        """The model-facing prompt after `node`: the model's own when it
        didn't fit the template, else the template filled with its values."""
        return node.prompt if node.prompt is not None else self.tmpl.text(node.values)

    def _drop_pending(self) -> None:
        for gen, _ in self._pending.values():
            gen.close()  # type: ignore[attr-defined]
        self._pending.clear()

    # -- turns --------------------------------------------------------------

    def turn(self, parent: Node, command: str) -> Generator[str, None, Node]:
        """Run `command` after `parent`. Yields output text as it becomes safe
        to show and returns the new child node. A KeyboardInterrupt (raised
        inside, or thrown in at a yield) ends the command like ^C in a real
        terminal: what was shown stays, followed by `^C`."""
        self._drop_pending()
        suffix = parent.leftover + command + "\n"
        stoks = self.engine.encode(suffix)
        extra = len(stoks) if stoks is not None else len(suffix) // 3 + 1
        gen = self.engine.stream(self._segments(parent, extra) + [(suffix, stoks)], self.max_new_tokens)
        rec = Recognizer(self.tmpl)
        text = ""
        toks: list[int] = []
        ends: list[int] = []  # streamed text length after each token
        match = None
        interrupted = False
        try:
            for piece, tok in gen:
                text += piece
                if tok is not None:
                    toks.append(tok)
                    ends.append(len(text))
                out, match = rec.feed(piece)
                if out:
                    yield out
                if match is not None:
                    break
            else:
                out = rec.rest()
                if out:
                    yield out
        except KeyboardInterrupt:
            interrupted = True
        except BaseException:
            gen.close()  # type: ignore[attr-defined]
            raise
        if match is not None:
            interrupted = False  # the command had already finished

        if match is None:
            gen.close()  # type: ignore[attr-defined]
            if interrupted:
                text = text[:rec.shown]
                body = text + "^C\n"
            else:
                body = text if not text or text.endswith("\n") else text + "\n"
            covered, k = self._covered(text, toks, ends, len(text))
            leftover = body[len(covered):] + self._prompt_text(parent)
            values = parent.values
        else:
            body = text[:match.start]
            covered, k = self._covered(text, toks, ends, match.end)
            leftover = text[len(covered):match.end] + ("" if match.trailed else self.tmpl.trail)
            values = {**parent.values, **match.values}
        node = Node(parent, command, suffix + covered,
                    stoks + toks[:k] if stoks is not None else None,
                    leftover, body, values, forced=match is None, interrupted=interrupted)
        if match is None:
            node.prompt = parent.prompt  # a supplied prompt repeats the last one shown
        elif not match.exact:
            node.prompt = text[match.start:match.end] + ("" if match.trailed else self.tmpl.trail)
        if match is not None:
            # A local stream can keep going for the guess; a remote one is
            # replaced by a fresh, quick-to-give-up request.
            if match.trailed and self.ghost_enabled and self.engine.tokenized:
                self._pending[node] = (gen, text[match.end:])
            else:
                gen.close()  # type: ignore[attr-defined]
                node.ghost = "" if not match.trailed else None
        parent.children.append(node)
        parent.last_child = node
        return node

    def _covered(self, text: str, toks: list[int], ends: list[int], end: int) -> tuple[str, int]:
        """The longest prefix of the model's tokens whose text lies within
        text[:end]: returns (its text, token count). Without token ids
        (remote engine) the text itself is the transcript."""
        if not self.engine.tokenized:
            # Keep the prompt's trailing space out of the transcript, as the
            # token path does, so the next segment starts with it.
            covered = text[:end]
            trail = self.tmpl.trail
            return (covered[:-len(trail)] if trail and covered.endswith(trail) else covered), 0
        k = bisect.bisect_right(ends, end)
        while k > 0:
            d = self.engine.decode(toks[:k])
            if len(d) <= end and text.startswith(d):
                return d, k
            k -= 1
        return "", 0

    # -- ghost text ---------------------------------------------------------

    def ghost(self, node: Node) -> Iterator[str]:
        """Yield the model's growing guess at the next command after `node`.

        Right after a turn this just keeps reading the same stream past the
        prompt. Otherwise (a prompt kenoma supplied, or the stream is gone)
        it starts a new one from the end of the prompt."""
        if node.ghost is not None or not self.ghost_enabled:
            if node.ghost:
                yield node.ghost
            return
        pending = self._pending.pop(node, None)
        if pending is not None:
            gen, raw = pending
            trail = ""
        else:
            self._drop_pending()
            trail = self.tmpl.trail
            if trail and node.leftover.endswith(trail):
                core = node.leftover[:-len(trail)]
            else:
                core, trail = node.leftover, ""  # the space is already in the transcript
            gen = self.engine.stream(self._segments(node) + [(core, self.engine.encode(core))],
                                     GHOST_MAX_TOKENS, interactive=True)
            raw = ""
        guess = ""
        try:
            for _ in range(GHOST_MAX_TOKENS):
                if "\n" in raw or len(raw) > GHOST_MAX_CHARS:
                    break
                if len(raw) >= len(trail):
                    if not raw.startswith(trail):  # the model didn't start a command
                        break
                    guess = raw[len(trail):]
                # Yield after every piece, changed or not: each yield is where
                # the UI lets keystrokes through.
                yield guess
                try:
                    piece, _ = next(gen)
                except StopIteration:
                    break
                raw += piece
            guess = raw[len(trail):].split("\n", 1)[0][:GHOST_MAX_CHARS] if raw.startswith(trail) else ""
            node.ghost = guess
            yield guess
        finally:
            if node.ghost is None:
                node.ghost = guess
            gen.close()  # type: ignore[attr-defined]

    # -- navigation ---------------------------------------------------------

    def display_prompt(self, node: Node) -> str:
        """The prompt shown after `node`, with the user's colors when the
        model's prompt fit the template."""
        return node.prompt if node.prompt is not None else self.tmpl.display(node.values)

    def siblings(self, node: Node) -> list[Node]:
        return node.parent.children if node.parent is not None else [node]

    def visit(self, node: Node) -> None:
        self.current = node
        n = node
        while n.parent is not None:
            n.parent.last_child = n
            n = n.parent
