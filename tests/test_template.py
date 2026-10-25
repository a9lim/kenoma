from kenoma.template import FAIL, MATCH, PARTIAL, Hole, Recognizer, Template

HOME = "/Users/a9lim"
CWD = "/Users/a9lim/Work/kenoma"


def macos_default(cwd: str = CWD) -> Template:
    # zsh's stock macOS prompt, %n@%m %1~ %#: only the cwd's last component.
    return Template.from_render("a9lim@Mac kenoma % ", cwd, HOME)


def test_basename_prompt_gets_a_cwd_hole():
    t = macos_default()
    assert t.parts == ["a9lim@Mac ", Hole("cwd"), " %"]
    assert t.trail == " "
    assert t.values == {"cwd": "kenoma"}
    assert t.text({"cwd": "tmp"}) == "a9lim@Mac tmp % "


def test_home_is_tilde_not_the_username():
    t = Template.from_render("a9lim@Mac ~ % ", HOME, HOME)
    assert t.parts == ["a9lim@Mac ", Hole("cwd"), " %"]
    assert t.values == {"cwd": "~"}


def test_colors_stay_in_display_only():
    raw = "\x1b[32ma9lim@Mac\x1b[0m \x1b[34m~/Work/kenoma\x1b[0m $ "
    t = Template.from_render(raw, CWD, HOME)
    assert t.core(t.values) == "a9lim@Mac ~/Work/kenoma $"
    assert t.display({"cwd": "~/tmp"}) == "\x1b[32ma9lim@Mac\x1b[0m \x1b[34m~/tmp\x1b[0m $ "


def test_bash_markers_and_title_escape_are_dropped():
    raw = "\x01\x1b]0;a9lim@host: ~\x07\x02a9lim@host:~/Work/kenoma$ "
    t = Template.from_render(raw, CWD, HOME)
    assert t.parts == ["a9lim@host:", Hole("cwd"), "$"]
    assert "\x1b]" not in t.display(t.values)


def test_multiline_prompt_with_branch_and_leading_newline():
    raw = "\n\x1b[1;36mkenoma\x1b[0m on \x1b[1;35m main\x1b[0m\n\x1b[1;32m❯\x1b[0m "
    t = Template.from_render(raw, CWD, HOME, branch="main")
    assert t.parts == [Hole("cwd"), " on  ", Hole("branch"), "\n❯"]
    assert t.trail == " "
    assert t.display({"cwd": "src", "branch": "dev"}).startswith("\x1b[1;36msrc")


def test_clock_hole():
    t = Template.from_render("[09:41:07] a9lim % ", "/elsewhere", HOME)
    assert t.parts == ["[", Hole("time"), "] a9lim %"]
    assert t.anchored


def test_bare_hole_prompt_is_not_anchored():
    assert not Template.from_render("~/Work/kenoma", CWD, HOME).anchored


def test_scan_statuses():
    t = macos_default()
    assert t.scan("")[0] == PARTIAL
    assert t.scan("a9lim@M")[0] == PARTIAL
    assert t.scan("a9lim@Mac src/lib")[0] == PARTIAL
    assert t.scan("a9lim@Mac kenoma %")[0] == PARTIAL  # trailing space not seen yet
    assert t.scan("total 48")[0] == FAIL
    assert t.scan("a9lim@Mac kenoma %x")[0] == PARTIAL  # "... %x src % " is still possible
    assert t.scan("a9lim@Mac kenoma %x\n")[0] == FAIL
    status, m = t.scan("a9lim@Mac /tmp % ls")
    assert status == MATCH and m is not None
    assert (m.end, m.values, m.trailed) == (len("a9lim@Mac /tmp % "), {"cwd": "/tmp"}, True)


def test_hole_between_literals_may_contain_spaces():
    status, m = macos_default().scan("a9lim@Mac kenoma src % cat x")
    assert status == MATCH and m is not None and m.values == {"cwd": "kenoma src"}


def test_hole_starting_a_line_may_not():
    t = Template.from_render("kenoma on  main\n❯ ", CWD, HOME, branch="main")
    assert t.scan("Depends")[0] == PARTIAL
    assert t.scan("Depends on")[0] == PARTIAL  # could still be "Depends on  main"
    assert t.scan("Depends on it")[0] == FAIL


def test_loose_fallback_when_an_optional_segment_vanishes():
    t = Template.from_render("a9lim@Mac kenoma (main) % ", CWD, HOME, branch="main")
    assert t.parts == ["a9lim@Mac ", Hole("cwd"), " (", Hole("branch"), ") %"]
    assert t.loose is not None and t.loose.parts == ["a9lim@Mac ", Hole("any"), " %"]
    assert macos_default().loose is None
    _, exact = feed_all(Recognizer(t), list("ok\na9lim@Mac kenoma (dev) % ls"))
    assert exact is not None and exact.exact and exact.values == {"cwd": "kenoma", "branch": "dev"}
    shown, loose = feed_all(Recognizer(t), list("ok\na9lim@Mac tmp % ls"))
    assert shown == "ok\n"
    assert loose is not None and not loose.exact and loose.values == {}
    assert (loose.start, loose.end) == (3, len("ok\na9lim@Mac tmp % "))


def test_scan_newline_where_the_trailing_space_belongs():
    status, m = macos_default().scan("a9lim@Mac kenoma %\nnext")
    assert status == MATCH and m is not None
    assert (m.end, m.trailed) == (len("a9lim@Mac kenoma %"), False)


def test_hole_at_the_end_is_greedy():
    t = Template.from_render("a9lim:~/Work/kenoma", CWD, HOME)
    assert t.parts == ["a9lim:", Hole("cwd")] and t.trail == ""
    assert t.scan("a9lim:~/src")[0] == PARTIAL
    status, m = t.scan("a9lim:~/src ls")
    assert status == MATCH and m is not None and m.values == {"cwd": "~/src"}


def feed_all(rec: Recognizer, pieces: list[str]):
    shown = ""
    for p in pieces:
        out, m = rec.feed(p)
        shown += out
        if m is not None:
            return shown, m
    return shown + rec.rest(), None


def test_silent_command_prompt_at_stream_start():
    shown, m = feed_all(Recognizer(macos_default()), ["a9lim@Mac", " tmp", " %", " ls"])
    assert shown == ""
    assert m is not None and (m.start, m.values) == (0, {"cwd": "tmp"})


def test_output_streams_without_line_holdback():
    rec = Recognizer(macos_default())
    assert rec.feed("total")[0] == "total"  # can't start a prompt: shown at once
    assert rec.feed(" 48\nREADME")[0] == " 48\nREADME"
    assert rec.feed(".md\na9lim@")[0] == ".md\n"  # could be a prompt: held
    out, m = rec.feed("Mac kenoma % git")
    assert out == "" and m is not None and m.start == len("total 48\nREADME.md\n")


def test_chunking_does_not_matter():
    text = "ok\na9lim@Macintosh\nx a9lim@Mac kenoma % y\na9lim@Mac src % next\n"
    whole = feed_all(Recognizer(macos_default()), [text])
    chars = feed_all(Recognizer(macos_default()), list(text))
    assert whole == chars
    shown, m = whole
    assert shown == "ok\na9lim@Macintosh\nx a9lim@Mac kenoma % y\n"
    assert m is not None and m.values == {"cwd": "src"}


def test_strict_and_loose_agree_whatever_the_chunking():
    # Strict could also match later on this line (cwd "tmp % hi", branch
    # "dev"); the loose match completes first, and must win either way.
    t = Template.from_render("a9lim@Mac kenoma (main) % ", CWD, HOME, branch="main")
    text = "a9lim@Mac tmp % hi (dev) % ls\n"
    whole = feed_all(Recognizer(t), [text])
    chars = feed_all(Recognizer(t), list(text))
    assert whole == chars
    m = whole[1]
    assert m is not None and not m.exact and m.end == len("a9lim@Mac tmp % ")


def test_earliest_end_across_candidates():
    # A two-line prompt starting at line 1 vs a one-line match on line 2.
    t = Template.from_render("kenoma\n❯ ", CWD, HOME)
    text = "x\n❯ y\nsrc\n❯ z"
    assert feed_all(Recognizer(t), [text]) == feed_all(Recognizer(t), list(text))


def test_empty_or_escape_only_prompts_are_unanchored():
    for raw in ("", "   ", "\x1b[32m\x1b[0m"):
        t = Template.from_render(raw, CWD, HOME)
        assert not t.anchored and t.loose is None


def test_multiline_prompt_recognized_across_lines():
    t = Template.from_render("kenoma on  main\n❯ ", CWD, HOME, branch="main")
    shown, m = feed_all(Recognizer(t), list("README.md\nsrc on  dev\n❯ make\n"))
    assert shown == "README.md\n"
    assert m is not None and m.values == {"cwd": "src", "branch": "dev"}


def test_stream_without_prompt_releases_everything():
    shown, m = feed_all(Recognizer(macos_default()), ["done\na9lim@Ma"])
    assert (shown, m) == ("done\na9lim@Ma", None)
