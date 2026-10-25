import shutil
import sys
from pathlib import Path

import pytest

from kenoma import shell


def test_zsh_extended_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    hist = tmp_path / ".zsh_history"
    hist.write_text(": 1700000000:0;git status\n: 1700000001:0;echo one \\\ntwo\n\nls\n")
    monkeypatch.setenv("SHELL", "/bin/zsh")
    monkeypatch.setenv("HISTFILE", str(hist))
    assert shell.read_history() == ["git status", "echo one \ntwo", "ls"]


def test_history_seed_shows_history_as_output(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, "platform", "linux")
    seed = shell.history_seed("me@box ~ % ", ["a", "b", "c"], 2, "/home/me")
    assert seed == ("me@box ~ % history | tail -2\n"
                    "    2  b\n"
                    "    3  c\n"
                    "me@box ~ % pwd\n"
                    "/home/me\n")


@pytest.mark.skipif(shutil.which("zsh") is None, reason="needs zsh")
def test_capture_runs_precmd_and_keeps_colors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / ".zshrc").write_text(
        "precmd() { mood=calm }\n"
        "setopt PROMPT_SUBST\n"
        "PROMPT='%F{green}box%f ${mood} %# '\n"
        "echo 'startup chatter'\n")
    monkeypatch.setenv("ZDOTDIR", str(tmp_path))
    monkeypatch.setenv("SHELL", shutil.which("zsh") or "zsh")
    raw = shell.capture_prompt()
    assert raw is not None
    assert raw.endswith("box\x1b[39m calm % ") and raw.startswith("\x1b[32m")
