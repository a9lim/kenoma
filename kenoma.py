#!/usr/bin/env python3
"""LLM fake terminal: raw completion, no chat template. The real shell's PS1
is captured once and used as both the seed and the stop string, so the model
hallucinates command output until it emits the next prompt."""
import argparse
import os
import re
import readline
import socket
import subprocess
import sys
import threading

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)


def capture_tmux_pane(lines: int) -> str:
    """Return the last `lines` lines of the current tmux pane's scrollback,
    which includes real commands AND their outputs. Empty string if not in tmux
    or the capture fails."""
    if lines <= 0 or not os.environ.get("TMUX"):
        return ""
    try:
        out = subprocess.run(
            ["tmux", "capture-pane", "-p", "-J", "-S", f"-{lines}"],
            capture_output=True, text=True, timeout=3,
        )
        # Drop the tail: the command that launched fake_term, the "[loading ...]"
        # message, and the current live prompt.
        rows = out.stdout.rstrip("\n").split("\n")
        return "\n".join(rows[:-3]) if len(rows) > 3 else ""
    except Exception:
        return ""


def read_history(n: int) -> list:
    """Return the last n commands from the user's shell history file.
    Handles zsh extended-history format (`: <ts>:<dur>;<cmd>`) and plain bash."""
    if n <= 0:
        return []
    shell = os.environ.get("SHELL", "")
    histfile = os.environ.get("HISTFILE")
    if not histfile:
        home = os.path.expanduser("~")
        histfile = os.path.join(home, ".zsh_history" if "zsh" in shell else ".bash_history")
    try:
        with open(histfile, "r", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return []
    cmds = []
    pending = ""
    for raw in lines:
        line = raw.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        if line.startswith(": ") and ";" in line[:32]:
            line = line.split(";", 1)[1]
        # zsh multi-line commands end with a trailing backslash
        if line.endswith("\\"):
            pending += line[:-1] + "\n"
            continue
        cmd = pending + line
        pending = ""
        if cmd.strip():
            cmds.append(cmd)
    return cmds[-n:]


def capture_prompt() -> str:
    """Capture the user's real rendered PS1.

    Interactive shell startup (rc files, plugins like zsh-autosuggestions,
    'Restored session: ...' banners, etc.) can print to stdout before our
    command runs, so we wrap the prompt in sentinels and extract.
    """
    shell = os.environ.get("SHELL", "/bin/zsh")
    begin, end = "\x1e<<PS1_BEGIN>>\x1e", "\x1e<<PS1_END>>\x1e"
    try:
        if "zsh" in shell:
            cmd = f'print -rn -- "{begin}${{(%)PS1}}{end}"'
        else:
            cmd = f'printf "%s" "{begin}${{PS1@P}}{end}"'
        out = subprocess.run(
            [shell, "-i", "-c", cmd],
            capture_output=True, text=True, timeout=3,
        )
        blob = out.stdout
        i = blob.find(begin)
        j = blob.find(end, i + len(begin)) if i >= 0 else -1
        if i >= 0 and j > i:
            p = blob[i + len(begin):j]
            if p.strip():
                return p if p.endswith(" ") else p + " "
    except Exception:
        pass
    return f"{os.getenv('USER', 'user')}@{socket.gethostname().split('.')[0]}:{os.getcwd()} $ "


PROMPT_LIKE = re.compile(r"\n[^\n]{1,200}[\$#%>] ")
PROMPT_LIKE_MAX_LINE = 200


def safe_flush_point(produced: str, stop_str: str) -> int:
    """Index up to which `produced` can be flushed to stdout without risking
    that the held-back portion turns into a prompt match on the next token."""
    n = len(produced)
    # Regex holdback: the latest \n could still anchor a prompt-shaped line
    # if the chars since it are all non-newline and within the line budget.
    regex_safe = n
    last_nl = produced.rfind("\n")
    if last_nl >= 0:
        tail = produced[last_nl + 1:]
        if "\n" not in tail and len(tail) <= PROMPT_LIKE_MAX_LINE:
            regex_safe = last_nl
    # Exact holdback: longest suffix of produced that's a prefix of stop_str.
    exact_safe = n
    for k in range(min(len(stop_str), n), 0, -1):
        if produced.endswith(stop_str[:k]):
            exact_safe = n - k
            break
    return min(regex_safe, exact_safe)


class StopOnPromptLike(StoppingCriteria):
    """Stop when the generated tail contains the exact captured prompt OR
    anything structurally prompt-shaped (newline + line ending in $/#/%/> + space)."""

    def __init__(self, tokenizer, exact: str, prompt_len: int):
        self.tok = tokenizer
        self.exact = exact
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kw) -> bool:
        text = self.tok.decode(input_ids[0][self.prompt_len:], skip_special_tokens=True)
        if self.exact in text:
            return True
        return PROMPT_LIKE.search(text) is not None

    def match_end(self, text: str) -> int:
        """Return index in `text` where the prompt-like region begins, or -1."""
        i = text.find(self.exact)
        if i >= 0:
            return i
        m = PROMPT_LIKE.search(text)
        if m:
            return m.start() + 1  # skip the leading \n, keep it in body
        return -1


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM fake terminal")
    ap.add_argument("model", nargs="?", default="Qwen/Qwen2.5-0.5B",
                    help="HF model id or local path (base/completion model, not chat-tuned)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    ap.add_argument("--context-chars", type=int, default=6000,
                    help="Max rolling buffer size in chars")
    ap.add_argument("--prompt", default=None,
                    help="Override captured prompt string")
    ap.add_argument("--history", type=int, default=20,
                    help="Seed with last N commands from shell history (0 = disabled). Ignored if tmux capture succeeds.")
    ap.add_argument("--tmux-lines", type=int, default=300,
                    help="If inside tmux, seed with the last N lines of the current pane's scrollback (real commands + outputs). 0 = disabled.")
    args = ap.parse_args()

    prompt_tmpl = args.prompt if args.prompt is not None else capture_prompt()
    stop_str = prompt_tmpl.rstrip() or prompt_tmpl

    print(f"[loading {args.model} ...]", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else "auto",
        device_map=args.device,
    )
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    # Seed readline history so arrow-up recalls previous commands.
    history_cmds = read_history(args.history)
    for cmd in history_cmds:
        readline.add_history(cmd)

    tmux_seed = capture_tmux_pane(args.tmux_lines)
    buf = ""
    if tmux_seed:
        # Real transcript from the live pane — already has its own prompts and outputs.
        buf = tmux_seed.rstrip() + "\n"
    else:
        for cmd in history_cmds:
            buf += prompt_tmpl + cmd + "\n"
    buf += prompt_tmpl
    # Display only the live prompt, not the seeded transcript.
    sys.stdout.write(prompt_tmpl)
    sys.stdout.flush()

    while True:
        try:
            line = input()
        except EOFError:
            sys.stdout.write("\n")
            return
        except KeyboardInterrupt:
            sys.stdout.write("\n")
            return

        buf += line + "\n"
        if len(buf) > args.context_chars:
            buf = buf[-args.context_chars:]

        enc = tok(buf, return_tensors="pt").to(model.device)
        prompt_len = enc.input_ids.shape[1]

        streamer = TextIteratorStreamer(
            tok, skip_prompt=True, skip_special_tokens=True
        )
        stopper = StopOnPromptLike(tok, stop_str, prompt_len)
        stopping = StoppingCriteriaList([stopper])
        gen_kwargs = dict(
            **enc,
            streamer=streamer,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            stopping_criteria=stopping,
            pad_token_id=tok.pad_token_id,
        )

        t = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        t.start()

        produced = ""
        shown = 0
        try:
            for chunk in streamer:
                produced += chunk
                idx = stopper.match_end(produced)
                if idx >= 0:
                    if idx > shown:
                        sys.stdout.write(produced[shown:idx])
                        sys.stdout.flush()
                    shown = idx
                    break
                safe_upto = safe_flush_point(produced, stop_str)
                if safe_upto > shown:
                    sys.stdout.write(produced[shown:safe_upto])
                    sys.stdout.flush()
                    shown = safe_upto
        except KeyboardInterrupt:
            sys.stdout.write("\n")
        t.join()

        idx = stopper.match_end(produced)
        if idx >= 0:
            body = produced[:idx]
        else:
            body = produced
            sys.stdout.write(produced[shown:])

        if not body.endswith("\n"):
            sys.stdout.write("\n")
            body += "\n"
        sys.stdout.write(prompt_tmpl)
        sys.stdout.flush()

        buf += body + prompt_tmpl


if __name__ == "__main__":
    main()
