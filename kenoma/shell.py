"""What kenoma reads from the real environment at startup: the rendered
prompt, the tmux scrollback, shell history and the git branch."""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import time

_BEGIN, _END = "\x1e<<PS1_BEGIN>>\x1e", "\x1e<<PS1_END>>\x1e"
_ZSH_EXT_PREFIX = re.compile(r"^: \d+:\d+;")


def capture_tmux_pane(lines: int) -> str:
    """The last `lines` lines of the current tmux pane (real commands and
    their outputs), minus the row that launched kenoma. Must run before
    kenoma writes anything to the terminal."""
    if lines <= 0 or not os.environ.get("TMUX"):
        return ""
    try:
        out = subprocess.run(
            ["tmux", "capture-pane", "-p", "-J", "-S", f"-{lines}"],
            capture_output=True, text=True, timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    rows = out.stdout.rstrip("\n").split("\n")
    return "\n".join(rows[:-1]) if len(rows) > 1 else ""


def read_history() -> list[str]:
    """Commands from $HISTFILE, ~/.zsh_history or ~/.bash_history, oldest
    first. Handles zsh extended history (`: <ts>:<dur>;<cmd>`) and its
    trailing-`\\` continuations; bash newline-separates and has no metadata
    lines."""
    is_zsh = "zsh" in os.path.basename(os.environ.get("SHELL", ""))
    histfile = os.environ.get("HISTFILE") or os.path.expanduser(
        "~/.zsh_history" if is_zsh else "~/.bash_history")
    try:
        with open(histfile, errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return []
    cmds: list[str] = []
    pending = ""
    for line in lines:
        if not line:
            continue
        if is_zsh:
            m = _ZSH_EXT_PREFIX.match(line)
            if m:
                line = line[m.end():]
            if line.endswith("\\"):
                pending += line[:-1] + "\n"
                continue
        cmd = pending + line
        pending = ""
        if cmd.strip():
            cmds.append(cmd)
    return cmds


def history_seed(prompt: str, history: list[str], n: int, cwd: str) -> str:
    """A transcript for when there's no tmux scrollback: the last n commands
    shown as the output of `history`, then a `pwd`. Bare commands back to
    back would teach the model that commands print nothing; as output they
    still tell it what this user works on. The `pwd` grounds where the
    session is and keeps the indented history lines from being the last
    thing the model copies."""
    lines = []
    if sys.platform == "darwin":
        lines.append(time.strftime("Last login: %a %b %e %H:%M:%S on ttys000"))
    recent = history[-n:] if n > 0 else []
    if recent:
        lines.append(f"{prompt}history | tail -{len(recent)}")
        first = len(history) - len(recent) + 1
        for i, cmd in enumerate(recent, first):
            lines.append(f"{i:5d}  " + cmd.replace("\n", "\\n"))
    lines += [f"{prompt}pwd", cwd]
    return "".join(line + "\n" for line in lines)


def git_branch(cwd: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    branch = out.stdout.strip()
    return branch if out.returncode == 0 and branch and branch != "HEAD" else None


def fallback_prompt() -> str:
    cwd = os.getcwd()
    home = os.path.expanduser("~")
    if cwd == home or cwd.startswith(home + os.sep):
        cwd = "~" + cwd[len(home):]
    return f"{os.environ.get('USER', 'user')}@{socket.gethostname().split('.')[0]}:{cwd}$ "


def _bash_supports_param_transform(shell: str) -> bool:
    """`${PS1@P}` needs bash >= 4.4; macOS ships /bin/bash 3.2."""
    try:
        out = subprocess.run([shell, "-c", "echo ${BASH_VERSINFO[0]} ${BASH_VERSINFO[1]}"],
                             capture_output=True, text=True, timeout=2)
        major, minor = (int(x) for x in out.stdout.split()[:2])
    except (OSError, subprocess.SubprocessError, ValueError):
        return False
    return (major, minor) >= (4, 4)


def capture_prompt() -> str | None:
    """The user's real PS1, rendered by an interactive shell (colors included),
    or None if it can't be expanded.

    Prompt hooks run first (zsh `precmd`, bash `PROMPT_COMMAND`) so themes
    that build the prompt there (vcs_info, starship, powerlevel10k) render.
    Startup chatter is ignored by extracting between sentinels."""
    shell = os.environ.get("SHELL", "/bin/sh")
    name = os.path.basename(shell)
    if "zsh" in name:
        cmd = ("(( $+functions[precmd] )) && precmd >/dev/null 2>&1; "
               "for f in $precmd_functions; do $f >/dev/null 2>&1; done; "
               f'print -rn -- "{_BEGIN}${{(%%)PS1}}{_END}"')
    elif "bash" in name and _bash_supports_param_transform(shell):
        cmd = ('for c in "${PROMPT_COMMAND[@]}"; do eval "$c" >/dev/null 2>&1; done; '
               f'printf "%s" "{_BEGIN}${{PS1@P}}{_END}"')
    else:
        print(f"[kenoma: can't expand PS1 with {shell} (zsh or bash >= 4.4 needed); "
              "pass --prompt to set one]", file=sys.stderr)
        return None
    try:
        out = subprocess.run([shell, "-i", "-c", cmd], capture_output=True, text=True,
                             timeout=5, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None
    i = out.stdout.find(_BEGIN)
    j = out.stdout.find(_END, i + len(_BEGIN)) if i >= 0 else -1
    if i < 0 or j < 0:
        return None
    prompt = out.stdout[i + len(_BEGIN):j]
    return prompt if prompt.strip() else None
