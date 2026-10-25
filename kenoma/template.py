"""The user's prompt as a template: literal text with typed holes.

A rendered PS1 is split into literals and holes. Holes are the parts that
legitimately drift during a fake session (the cwd, a git branch, a clock);
everything else must be reproduced verbatim. The model-facing form is plain
text; the display form keeps the prompt's SGR colors.

`Recognizer` watches the model's streamed output for the start of a prompt
and holds back only text that could still turn into one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Escapes inside a rendered prompt. SGR (colors) survives into the display
# form; OSC titles, cursor movement and readline's \x01/\x02 markers are
# dropped from both forms.
_ESC = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?|\x1b[@-Z\\-_]|[\x01\x02]")
_SGR = re.compile(r"\x1b\[[0-9;:]*m")

# A hole value must not run into neighboring path-ish characters, so that a
# cwd basename like "kenoma" isn't found inside "kenoma-dev".
_BOUND = re.compile(r"[\w.~/-]")
_TIME = re.compile(r"(?<![\d:])\d{1,2}:\d{2}(?::\d{2})?(?![\d:])")

_HOLE_MAX = {"cwd": 256, "branch": 128, "time": 8, "any": 256}

FAIL, PARTIAL, MATCH = 0, 1, 2


@dataclass(frozen=True)
class Hole:
    kind: str  # "cwd" | "branch" | "time" | "any"


def _hole_char(kind: str, c: str, spaced: bool) -> bool:
    if kind == "time":
        return c.isdigit() or c == ":"
    return c.isprintable() and (spaced or not c.isspace())


@dataclass(frozen=True)
class Match:
    start: int  # offset of the prompt's first character in the recognizer's text
    end: int  # offset just past the prompt, trailing whitespace included if present
    values: dict[str, str]
    trailed: bool  # False when the line ended where the trailing whitespace belongs
    exact: bool = True  # False for a loose match: the model's own variation on the prompt


class Template:
    def __init__(self, parts: list[str | Hole], dparts: list[str | Hole], trail: str,
                 values: dict[str, str]):
        self.parts = parts  # model-facing core, trailing whitespace removed
        self.dparts = dparts  # display form; same holes in the same order
        self.trail = trail  # trailing whitespace (where the user's command starts)
        self.values = values  # hole values in the captured render
        # A hole with literal text before it on its line may contain spaces
        # (a model writing `user@host kenoma src %` after `cd src`); the
        # literal after it still decides where it ends. A hole that starts a
        # line or ends the prompt can't, or it would swallow whole lines.
        self._spaced = {
            k for k, p in enumerate(parts)
            if isinstance(p, Hole) and 0 < k < len(parts) - 1
            and isinstance(parts[k - 1], str) and str(parts[k - 1]).rsplit("\n", 1)[-1] != ""
        }
        self.loose = self._loose()

    def _loose(self) -> Template | None:
        """A fallback that only asks for the prompt's opening text and its
        closing symbol, for prompts whose optional parts come and go (a git
        segment vanishing after `cd /tmp`)."""
        if not self.parts:
            return None
        first, last = self.parts[0], self.parts[-1]
        if not (isinstance(first, str) and isinstance(last, str)) or len(first.strip()) < 3:
            return None
        suffix = last[last.rfind(" "):] if " " in last else last
        if len(self.parts) <= 3 and suffix == last:
            return None  # the strict template is already this loose
        return Template([first, Hole("any"), suffix], [], self.trail, {})

    @property
    def has_cwd(self) -> bool:
        return any(isinstance(p, Hole) and p.kind == "cwd" for p in self.parts)

    @property
    def anchored(self) -> bool:
        """Has literal text to recognize. A prompt that is nothing but a hole
        (PS1='%~') would match any output line."""
        return any(isinstance(p, str) and p.strip() for p in self.parts)

    def core(self, values: dict[str, str]) -> str:
        return "".join(values.get(p.kind, "") if isinstance(p, Hole) else p for p in self.parts)

    def text(self, values: dict[str, str]) -> str:
        return self.core(values) + self.trail

    def display(self, values: dict[str, str]) -> str:
        return "".join(values.get(p.kind, "") if isinstance(p, Hole) else p for p in self.dparts)

    # -- construction -------------------------------------------------------

    @classmethod
    def from_render(cls, raw: str, cwd: str, home: str, branch: str | None = None) -> Template:
        # Split into visible characters, remembering the SGR escapes that
        # precede each one. Dropped characters (leading newlines, stray
        # control bytes) hand their escapes on to the next visible one.
        vis: list[str] = []
        pre: list[str] = []
        pend = ""
        pos = 0

        def emit(chunk: str) -> None:
            nonlocal pend
            for ch in chunk:
                if (ch == "\n" and not vis) or not (ch in "\n\t" or ch.isprintable()):
                    continue
                vis.append(ch)
                pre.append(pend)
                pend = ""

        for m in _ESC.finditer(raw):
            emit(raw[pos:m.start()])
            if _SGR.fullmatch(m.group()):
                pend += m.group()
            pos = m.end()
        emit(raw[pos:])
        plain = "".join(vis)

        parts: list[str | Hole] = []
        dparts: list[str | Hole] = []
        values: dict[str, str] = {}
        i = 0
        for a, b, kind in _find_holes(plain, cwd, home, branch):
            parts.append(plain[i:a])
            dparts.append("".join(pre[j] + vis[j] for j in range(i, a)) + pre[a])
            parts.append(Hole(kind))
            dparts.append(Hole(kind))
            # Escapes inside the hole's span keep the color state right after it.
            dparts.append("".join(pre[j] for j in range(a + 1, b)))
            values[kind] = plain[a:b]
            i = b
        parts.append(plain[i:])
        dparts.append("".join(pre[j] + vis[j] for j in range(i, len(vis))) + pend)

        trail = ""
        if isinstance(parts[-1], str):
            core = parts[-1].rstrip(" \t")
            trail = parts[-1][len(core):]
            parts[-1] = core
        return cls(_merge(parts), _merge(dparts), trail, values)

    # -- matching -----------------------------------------------------------

    def scan(self, s: str) -> tuple[int, Match | None]:
        """Match `s` (which starts where a prompt would) against the template.

        Returns (MATCH, match) for the match that ends earliest, (PARTIAL,
        None) when `s` could still grow into a prompt, or (FAIL, None). Taking
        the earliest end makes the result independent of how the text arrived
        in chunks: once any match is complete, no path that still needs more
        text can finish before it."""
        found: list[tuple[int, dict[str, str], bool]] = []
        partial = self._walk(0, s, 0, {}, found)
        if found:
            end, values, trailed = min(found, key=lambda f: f[0])
            return MATCH, Match(0, end, values, trailed)
        return (PARTIAL if partial else FAIL), None

    def _walk(self, k: int, s: str, i: int, vals: dict[str, str],
              found: list[tuple[int, dict[str, str], bool]]) -> bool:
        """Try every way parts[k:] can match s[i:], appending complete
        matches to `found`. Returns True if some path needs more text."""
        if k == len(self.parts):
            return self._trail(s, i, vals, found)
        p = self.parts[k]
        if isinstance(p, str):
            n = min(len(p), len(s) - i)
            if s[i:i + n] != p[:n]:
                return False
            if n < len(p):
                return True
            return self._walk(k + 1, s, i + n, vals, found)
        limit = _HOLE_MAX[p.kind]
        j = i
        if k + 1 == len(self.parts):
            # A hole that ends the prompt is greedy: it runs until a character
            # outside its class, otherwise "R" would already match "README".
            while j < len(s) and j - i < limit and _hole_char(p.kind, s[j], False):
                j += 1
            if j == len(s):
                return True
            if j == i:
                return False
            return self._trail(s, j, {**vals, p.kind: s[i:j]}, found)
        spaced = k in self._spaced
        partial = False
        while j < len(s) and j - i < limit and _hole_char(p.kind, s[j], spaced):
            j += 1
            partial |= self._walk(k + 1, s, j, {**vals, p.kind: s[i:j]}, found)
        return partial or (j == len(s) and j - i < limit)

    def _trail(self, s: str, i: int, vals: dict[str, str],
               found: list[tuple[int, dict[str, str], bool]]) -> bool:
        rest = s[i:]
        if rest.startswith(self.trail):
            found.append((i + len(self.trail), vals, True))
            return False
        if rest.startswith("\n"):
            found.append((i, vals, False))
            return False
        return self.trail.startswith(rest)


class Recognizer:
    """Find the first prompt in a stream of model output.

    Every line start, including the start of the stream (a silent command's
    output is just the next prompt), is a candidate. Text is held back only
    while some candidate could still become a prompt."""

    def __init__(self, template: Template):
        self.t = template
        self.text = ""
        self.shown = 0
        self.cands = [0]

    def feed(self, piece: str) -> tuple[str, Match | None]:
        """Add a streamed piece. Returns (text safe to show, match or None)."""
        base = len(self.text)
        self.text += piece
        i = piece.find("\n")
        while i >= 0:
            self.cands.append(base + i + 1)
            i = piece.find("\n", i + 1)
        best: Match | None = None
        live = []
        for c in self.cands:
            status, m = self.t.scan(self.text[c:])
            exact = True
            if self.t.loose is not None:
                loose_status, lm = self.t.loose.scan(self.text[c:])
                if lm is not None and (m is None or lm.end < m.end):
                    m, exact = lm, False
                status = max(status, loose_status)
            if m is not None:
                hit = Match(c, c + m.end, m.values if exact else {}, m.trailed, exact)
                # Earliest end wins (then earliest start): the same prompt
                # whatever the chunking.
                if best is None or (hit.end, hit.start) < (best.end, best.start):
                    best = hit
            elif status == PARTIAL:
                live.append(c)
        if best is not None:
            out = self.text[self.shown:best.start]
            self.shown = best.start
            return out, best
        self.cands = live
        hold = live[0] if live else len(self.text)
        out = self.text[self.shown:hold]
        self.shown = max(self.shown, hold)
        return out, None

    def rest(self) -> str:
        """Release everything still held (the stream ended without a prompt)."""
        out = self.text[self.shown:]
        self.shown = len(self.text)
        return out


def _merge(parts: list[str | Hole]) -> list[str | Hole]:
    out: list[str | Hole] = []
    for p in parts:
        if isinstance(p, str):
            if not p:
                continue
            if out and isinstance(out[-1], str):
                out[-1] += p
                continue
        out.append(p)
    return out


def _cwd_forms(cwd: str, home: str) -> list[str]:
    """Ways a prompt might show the cwd, most specific first: absolute,
    ~-relative, then trailing component suffixes (`Work/kenoma`, `kenoma`)."""
    forms = [cwd]
    if cwd == home:
        forms.append("~")
    elif cwd.startswith(home.rstrip("/") + "/"):
        forms.append("~" + cwd[len(home.rstrip("/")):])
    comps = [c for c in cwd.split("/") if c]
    for k in range(len(comps) - 1, 0, -1):
        forms.append("/".join(comps[-k:]))
    return list(dict.fromkeys(forms))


def _find_holes(plain: str, cwd: str, home: str, branch: str | None) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []

    def free(a: int, b: int) -> bool:
        return not any(a < y and x < b for x, y, _ in spans)

    def take(value: str, kind: str) -> bool:
        end = len(plain)
        while value:
            a = plain.rfind(value, 0, end)
            if a < 0:
                return False
            b = a + len(value)
            if ((a == 0 or not _BOUND.match(plain[a - 1]))
                    and (b == len(plain) or not _BOUND.match(plain[b])) and free(a, b)):
                spans.append((a, b, kind))
                return True
            end = b - 1
        return False

    for form in _cwd_forms(cwd, home):
        if take(form, "cwd"):
            break
    if branch:
        take(branch, "branch")
    for m in _TIME.finditer(plain):
        if free(m.start(), m.end()):
            spans.append((m.start(), m.end(), "time"))
            break
    return sorted(spans)
