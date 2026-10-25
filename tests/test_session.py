from collections.abc import Iterator, Sequence

import pytest

from kenoma.engine import Piece, Segment
from kenoma.session import Node, Session
from kenoma.template import Template


class FakeEngine:
    """Plays one scripted list of pieces per stream() call; each piece is one
    token. Token ids come from a vocabulary built on the fly."""

    tokenized = True

    def __init__(self, *scripts: list[str]):
        self.scripts = list(scripts)
        self.vocab: dict[int, str] = {}
        self.calls: list[tuple[list[Segment], int]] = []

    def _id(self, text: str) -> int:
        for k, v in self.vocab.items():
            if v == text:
                return k
        self.vocab[len(self.vocab) + 1] = text
        return len(self.vocab)

    def encode(self, text: str) -> list[int] | None:
        return [self._id(c) for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(self.vocab[t] for t in tokens)

    def warm(self, segments: Sequence[Segment]) -> None:
        pass

    def stream(self, segments: Sequence[Segment], max_tokens: int,
               interactive: bool = False) -> Iterator[Piece]:
        self.calls.append((list(segments), max_tokens))
        for piece in self.scripts.pop(0)[:max_tokens]:
            if piece == "^C":
                raise KeyboardInterrupt
            yield piece, self._id(piece)


class FakeRemote(FakeEngine):
    tokenized = False

    def encode(self, text: str) -> list[int] | None:
        return None

    def stream(self, segments: Sequence[Segment], max_tokens: int,
               interactive: bool = False) -> Iterator[Piece]:
        for piece, _ in super().stream(segments, max_tokens):
            yield piece, None


def make(engine: FakeEngine, seed: str = "", budget: int = 10_000) -> Session:
    t = Template.from_render("a9lim@Mac kenoma % ", "/Users/a9lim/Work/kenoma", "/Users/a9lim")
    s = Session(engine, t, max_new_tokens=100, context_tokens=budget)
    s.start(seed)
    return s


def run(s: Session, parent: Node, command: str) -> tuple[str, Node]:
    gen = s.turn(parent, command)
    shown = ""
    try:
        while True:
            shown += next(gen)
    except StopIteration as stop:
        return shown, stop.value


def ghost(s: Session, node: Node) -> str:
    out = ""
    for out in s.ghost(node):
        pass
    return out


def test_turn_keeps_model_tokens_up_to_the_prompt():
    e = FakeEngine(["total 8\n", "README.md\n", "a9lim@Mac", " kenoma", " %", " ls", "\n"])
    s = make(e)
    shown, node = run(s, s.root, "ls -la")
    assert shown == node.body == "total 8\nREADME.md\n"
    assert node.values == {"cwd": "kenoma"} and not node.forced
    # " ls" straddles the prompt's trailing space: it stays out of the
    # transcript, and the space moves to the next segment as leftover.
    assert node.text == "a9lim@Mac kenoma % ls -la\ntotal 8\nREADME.md\na9lim@Mac kenoma %"
    assert node.leftover == " "
    assert e.decode(node.tokens or []) == node.text
    assert ghost(s, node) == "ls"
    assert node.ghost == "ls"


def test_silent_command_ends_at_once():
    s = make(FakeEngine(["a9lim@Mac", " tmp", " %", " pwd", "\n"]))
    shown, node = run(s, s.root, "cd /tmp")
    assert shown == "" and node.body == ""
    assert node.values == {"cwd": "tmp"}
    assert s.tmpl.display(node.values) == "a9lim@Mac tmp % "


def test_prompt_inside_one_token():
    e = FakeEngine(["a9lim@Mac", " kenoma % ls", "\n"])
    s = make(e)
    _, node = run(s, s.root, "true")
    assert node.text.endswith("\na9lim@Mac") and node.leftover == " kenoma % "
    assert ghost(s, node) == "ls"


def test_empty_predicted_command():
    s = make(FakeEngine(["a9lim@Mac kenoma %\n", "x"]))
    _, node = run(s, s.root, "clear")
    assert node.leftover == "a9lim@Mac kenoma % " and node.ghost == ""


def test_no_prompt_means_kenoma_supplies_it():
    e = FakeEngine(["foo\n", "bar"], [" make", "\n"])
    s = make(e)
    shown, node = run(s, s.root, "cat x")
    assert shown == "foo\nbar" and node.body == "foo\nbar\n"
    assert node.forced and node.values == s.root.values
    assert node.leftover == "\na9lim@Mac kenoma % "
    assert ghost(s, node) == "make"
    # The guess stream starts at the end of the prompt, before its space.
    assert e.calls[-1][0][-1][0] == "\na9lim@Mac kenoma %"


def test_next_turn_continues_the_transcript():
    e = FakeEngine(["a9lim@Mac", " src", " %", " ls", "\n"], ["main.rs\n", "a9lim@Mac src % "])
    s = make(e, seed="old output\n")
    _, a = run(s, s.root, "cd src")
    s._drop_pending()
    _, b = run(s, a, "ls")
    segments = e.calls[-1][0]
    assert [t for t, _ in segments] == [s.root.text, a.text, " ls\n"]
    assert b.body == "main.rs\n"


def test_reroll_is_a_sibling():
    e = FakeEngine(["one\n", "a9lim@Mac kenoma % "], ["two\n", "a9lim@Mac kenoma % "])
    s = make(e)
    _, a = run(s, s.root, "fortune")
    _, b = run(s, s.root, "fortune")
    assert s.root.children == [a, b] and s.root.last_child is b
    assert s.siblings(a) == [a, b]
    assert (a.body, b.body) == ("one\n", "two\n")


def test_interrupt_keeps_what_was_shown():
    s = make(FakeEngine(["y\n", "y\n", "a9lim@M", "^C"]))
    gen = s.turn(s.root, "yes")
    assert next(gen) == "y\n" and next(gen) == "y\n"
    with pytest.raises(StopIteration) as stop:
        next(gen)  # the held-back "a9lim@M" is dropped, not shown
    node = stop.value.value
    assert node.interrupted and node.forced and s.root.children == [node]
    assert node.body == "y\ny\n^C\n"
    assert node.text.endswith("yes\ny\ny\n") and node.leftover == "^C\na9lim@Mac kenoma % "


def test_interrupt_thrown_in_at_a_yield():
    s = make(FakeEngine(["partial", " output", " never seen"]))
    gen = s.turn(s.root, "cat")
    assert next(gen) == "partial"
    with pytest.raises(StopIteration) as stop:
        gen.throw(KeyboardInterrupt())
    assert stop.value.value.body == "partial^C\n"


def test_window_drops_old_turns_and_stays_under_the_limit():
    scripts = [[f"out{i}\n", "a9lim@Mac kenoma % "] for i in range(6)]
    s = make(FakeEngine(*scripts), seed="s" * 40, budget=60)
    node = s.root
    for i in range(6):
        _, node = run(s, node, f"c{i}")
    w = s.window(node)
    assert w[-1] is node and s.root not in w
    assert sum(len(n.tokens or []) for n in w) <= s._limit()


def test_window_depends_only_on_the_path():
    scripts = [[f"out{i}\n", "a9lim@Mac kenoma % "] for i in range(6)]
    s = make(FakeEngine(*scripts), budget=90)
    nodes = [s.root]
    for i in range(6):
        nodes.append(run(s, nodes[-1], f"c{i}")[1])
    before = s.window(nodes[3])
    s.window(nodes[6])  # visiting further down must not change it
    assert s.window(nodes[3]) == before


def test_seed_is_trimmed_to_its_newest_lines():
    s = make(FakeEngine(), seed="".join(f"line {i}\n" for i in range(100)), budget=200)
    assert len(s.root.tokens or []) <= s._limit() // 2
    assert s.root.text.endswith("line 99\n") and s.root.text.startswith("line ")


def test_loose_prompt_is_shown_as_the_model_wrote_it():
    e = FakeEngine(["a9lim@Mac tmp % ", "ls", "\n"])
    t = Template.from_render("a9lim@Mac kenoma (main) % ", "/Users/a9lim/Work/kenoma", "/Users/a9lim", "main")
    s = Session(e, t, max_new_tokens=100, context_tokens=10_000)
    s.start("")
    _, node = run(s, s.root, "cd /tmp")
    assert node.prompt == "a9lim@Mac tmp % " and s.display_prompt(node) == "a9lim@Mac tmp % "
    assert node.values == s.root.values and not node.forced
    assert ghost(s, node) == "ls"


def test_forced_prompt_repeats_a_loose_one():
    e = FakeEngine(["a9lim@Mac tmp % ", "ls", "\n"], ["hi\n"])
    t = Template.from_render("a9lim@Mac kenoma (main) % ", "/Users/a9lim/Work/kenoma", "/Users/a9lim", "main")
    s = Session(e, t, max_new_tokens=100, context_tokens=10_000)
    s.start("")
    _, a = run(s, s.root, "cd /tmp")
    _, b = run(s, a, "echo hi")
    assert b.forced and b.prompt == "a9lim@Mac tmp % "
    assert b.leftover.endswith("a9lim@Mac tmp % ")


def test_guess_restarts_when_the_prompt_is_fully_covered():
    e = FakeEngine(["a9lim@Mac kenoma %", " ", "ls", "\n"], ["ls", "\n"])
    s = make(e)
    _, node = run(s, s.root, "true")
    assert node.leftover == "" and node.text.endswith("kenoma % ")
    s._drop_pending()  # the original stream is gone
    assert ghost(s, node) == "ls"  # no trailing space demanded


def test_interrupt_after_the_prompt_was_found_is_a_normal_finish():
    s = make(FakeEngine(["done\na9lim@Mac kenoma % ls", "\n"]))
    gen = s.turn(s.root, "make")
    assert next(gen) == "done\n"
    with pytest.raises(StopIteration) as stop:
        gen.throw(KeyboardInterrupt())
    node = stop.value.value
    assert not node.interrupted and not node.forced and node.body == "done\n"


def test_remote_engine_keeps_text_only():
    e = FakeRemote(["hi\n", "a9lim@Mac kenoma % ", "ls"], [" pwd", "\n"])
    s = make(e)
    _, node = run(s, s.root, "echo hi")
    assert node.tokens is None
    assert node.text == "a9lim@Mac kenoma % echo hi\nhi\na9lim@Mac kenoma %"
    assert node.leftover == " "
    # The guess is a fresh request ending before the prompt's space.
    assert ghost(s, node) == "pwd"
    assert "".join(t for t, _ in e.calls[-1][0]).endswith("hi\na9lim@Mac kenoma %")
