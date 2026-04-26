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
from pathlib import Path
from typing import Any, cast

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    tomllib = None

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

__version__ = "0.1.0"


# ---------------------------------------------------------------------------
# Config loading: file + env, merged beneath argparse defaults.
# Precedence: CLI flags > KENOMA_* env vars > config file > hardcoded default.
# ---------------------------------------------------------------------------

CONFIG_KEYS = {
    "model": str,
    "device": str,
    "max_new_tokens": int,
    "temperature": float,
    "top_p": float,
    "repetition_penalty": float,
    "context_chars": int,
    "prompt": str,
    "history": int,
    "tmux_lines": int,
    "quantize": str,
    "kv_cache": bool,
}

ENV_PREFIX = "KENOMA_"


def config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return Path(base) / "kenoma" / "config.toml"


def load_config_file() -> dict:
    """Return a flat dict of recognized keys from the TOML config, or {}.
    Silently returns {} on 3.9/3.10 (no tomllib) or if file missing."""
    if tomllib is None:
        return {}
    p = config_path()
    if not p.is_file():
        return {}
    try:
        with open(p, "rb") as f:
            data = tomllib.load(f)
    except Exception as e:
        print(f"[kenoma: failed to read {p}: {e}]", file=sys.stderr)
        return {}
    return {k: v for k, v in data.items() if k in CONFIG_KEYS}


def load_env_overrides() -> dict:
    out = {}
    for k, typ in CONFIG_KEYS.items():
        name = ENV_PREFIX + k.upper()
        if name not in os.environ:
            continue
        raw = os.environ[name]
        try:
            if typ is bool:
                out[k] = raw.strip().lower() in ("1", "true", "yes", "y", "on")
            else:
                out[k] = typ(raw)
        except (ValueError, TypeError):
            print(f"[kenoma: ignoring invalid {name}={raw!r}]", file=sys.stderr)
    return out


def merged_defaults() -> dict:
    d = load_config_file()
    d.update(load_env_overrides())
    return d


# ---------------------------------------------------------------------------
# Seed transcript capture.
# ---------------------------------------------------------------------------

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
        # Drop the tail: the command that launched kenoma, the "[loading ...]"
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


# ---------------------------------------------------------------------------
# Stop detection.
#
# We build a *structural skeleton* of the captured prompt at startup by taking
# the rendered prompt verbatim and substituting the cwd (and its `~`-relative
# form) with `[^\s]+`. The result is a regex that matches the captured prompt
# and any cwd-drifted variant of it, but not arbitrary shell-prompt-shaped
# garbage like `50 % done`. This is the *only* stop signal beyond EOS.
#
# Skeleton-only matches are strict enough that adoption is safe: when the
# model emits a matching line, we treat it as the new canonical prompt for
# both display and stop matching going forward (cwd in the prompt actually
# tracks the model's fake cd's). See the REPL loop below.
# ---------------------------------------------------------------------------

HOLDBACK_LINE_BUDGET = 200  # streaming-flush heuristic, see safe_flush_point


def build_skeleton(prompt: str) -> "re.Pattern":
    """Return a compiled regex that matches the captured prompt with its cwd
    portion turned into a wildcard. Anchored at \\n. The wildcard is
    `[^\\s]+`, so cwd substitutions can't bleed across a space (which keeps
    false positives like `\\n50 % done` from matching).

    Limitations: only the cwd (full path or `~`-relative) is wildcarded.
    Prompts that show only the cwd basename (bash `\\W`), or that include a
    git branch / exit-status segment, won't match drifted versions — they'll
    only match exact reproductions of the captured prompt. Generation in
    those cases runs to `--max-new-tokens` instead of stopping early."""
    cwd = os.getcwd()
    home = os.path.expanduser("~")
    candidates = [cwd]
    if cwd == home:
        candidates.append("~")
    elif cwd.startswith(home + os.sep):
        candidates.append("~" + cwd[len(home):])
    pat = re.escape(prompt)
    for c in candidates:
        ec = re.escape(c)
        if ec and ec in pat:
            pat = pat.replace(ec, r"[^\s]+", 1)
            break
    return re.compile(r"\n" + pat)


def safe_flush_point(produced: str, current_prompt: str) -> int:
    """Index up to which `produced` can be flushed to stdout without risking
    that the held-back portion turns into a prompt match on the next token."""
    n = len(produced)
    # Regex holdback: the latest \n could still anchor a skeleton match if
    # the chars since it are all non-newline and within the line budget.
    regex_safe = n
    last_nl = produced.rfind("\n")
    if last_nl >= 0:
        tail = produced[last_nl + 1:]
        if "\n" not in tail and len(tail) <= HOLDBACK_LINE_BUDGET:
            regex_safe = last_nl
    # Exact holdback: longest suffix of produced that's a prefix of the
    # current prompt — those chars might complete into a verbatim repro.
    # Subsumed by the regex holdback in most cases, but cheap insurance.
    exact_safe = n
    stop_str = current_prompt.rstrip() or current_prompt
    for k in range(min(len(stop_str), n), 0, -1):
        if produced.endswith(stop_str[:k]):
            exact_safe = n - k
            break
    return min(regex_safe, exact_safe)


class StopOnPromptLike(StoppingCriteria):
    """Stop when the generated tail contains a line matching the prompt
    skeleton (newline + literal-prefix + cwd-wildcard + literal-suffix)."""

    def __init__(self, tokenizer, skeleton: "re.Pattern", prompt_len: int):
        self.tok = tokenizer
        self.skeleton = skeleton
        self.prompt_len = prompt_len

    # Transformers accepts a plain bool here even though the base signature
    # declares BoolTensor; the override is intentional.
    def __call__(self, input_ids, scores, **kw) -> bool:  # type: ignore[reportIncompatibleMethodOverride]
        text = self.tok.decode(input_ids[0][self.prompt_len:], skip_special_tokens=True)
        return self.skeleton.search(text) is not None

    def match_span(self, text: str):
        """Return (body_end, prompt_end) where text[body_end:prompt_end] is
        the matched prompt and text[:body_end] is everything before it
        (including the leading \\n). Returns (-1, -1) if no match."""
        m = self.skeleton.search(text)
        if not m:
            return -1, -1
        return m.start() + 1, m.end()  # +1 keeps the leading \n in body


# ---------------------------------------------------------------------------
# Device / dtype / quantization.
# ---------------------------------------------------------------------------

def pick_device(requested: str) -> str:
    """Resolve --device auto to the best available backend."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available() and mps.is_built():
        return "mps"
    return "cpu"


def pick_dtype(device: str, quantize: str) -> "torch.dtype":
    if quantize in ("4bit", "8bit"):
        # bnb compute dtype; weights are quantized separately.
        return torch.float16
    if device.startswith("cuda") or device == "mps":
        return torch.float16
    return torch.float32


def build_quant_config(kind: str):
    """Return a BitsAndBytesConfig, or None. Exits with a clear error if the
    user asked for quantization but bitsandbytes isn't installed."""
    if kind in (None, "none", ""):
        return None
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        print("[kenoma: this transformers install lacks BitsAndBytesConfig]", file=sys.stderr)
        sys.exit(2)
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        print("[kenoma: --quantize requires bitsandbytes — `pip install kenoma[quantize]` or `pip install bitsandbytes`]", file=sys.stderr)
        sys.exit(2)
    if kind == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    if kind == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    print(f"[kenoma: unknown --quantize {kind!r}]", file=sys.stderr)
    sys.exit(2)


def load_model(args):
    """Resolve device, build quant config, load tokenizer + model. Returns
    (tokenizer, model, device)."""
    device = pick_device(args.device)
    quant = None if args.quantize == "none" else args.quantize
    if quant and not device.startswith("cuda"):
        print(f"[kenoma: --quantize {quant} requires CUDA; resolved device is {device!r}]", file=sys.stderr)
        sys.exit(2)
    dtype = pick_dtype(device, quant or "none")
    kwargs: dict[str, Any] = dict(torch_dtype=dtype)
    quant_config = build_quant_config(quant) if quant else None
    if quant_config is not None:
        kwargs["quantization_config"] = quant_config
        # bnb needs accelerate's placement; don't fight it.
        kwargs["device_map"] = "auto"

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if quant_config is None:
        # Not using accelerate's device_map; place the whole model ourselves.
        model.to(device)  # type: ignore[reportArgumentType]
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok, model, device


# ---------------------------------------------------------------------------
# KV cache reuse across turns.
#
# We keep (cached_ids, cached_kv) where cached_ids is the token sequence the
# cache covers. On each turn we tokenize the new buffer, find the longest
# token-level common prefix with cached_ids, truncate the cache to that
# prefix, and feed only the delta. After generation we truncate the returned
# cache back to len(target_ids) — i.e. drop the generated-tokens portion.
# That matters because we don't actually store the model's generated tail as-
# is; we strip the prompt-shaped suffix and append our own canonical
# prompt_tmpl. Dropping the cached generated tail keeps the cache aligned
# with what we'll retokenize next turn.
# ---------------------------------------------------------------------------

def lcp_len(a, b) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def truncate_cache(cache, n_tokens: int):
    if cache is None or n_tokens <= 0:
        return None
    if hasattr(cache, "crop"):
        cache.crop(n_tokens)
        return cache
    # Legacy tuple-of-tuples: each (key, value) is [batch, heads, seq, head_dim].
    return tuple(
        (k[..., :n_tokens, :], v[..., :n_tokens, :])
        for (k, v) in cache
    )


# ---------------------------------------------------------------------------
# REPL.
# ---------------------------------------------------------------------------

def main() -> None:
    defaults = merged_defaults()

    ap = argparse.ArgumentParser(description="LLM fake terminal")
    ap.add_argument("model", nargs="?",
                    default=defaults.get("model", "Qwen/Qwen2.5-0.5B"),
                    help="HF model id or local path (base/completion model, not chat-tuned)")
    ap.add_argument("--device", default=defaults.get("device", "auto"),
                    help="'auto' picks cuda → mps → cpu")
    ap.add_argument("--max-new-tokens", type=int,
                    default=defaults.get("max_new_tokens", 2048))
    ap.add_argument("--temperature", type=float,
                    default=defaults.get("temperature", 1.0))
    ap.add_argument("--top-p", type=float,
                    default=defaults.get("top_p", 0.95))
    ap.add_argument("--repetition-penalty", type=float,
                    default=defaults.get("repetition_penalty", 1.05))
    ap.add_argument("--context-chars", type=int,
                    default=defaults.get("context_chars", 6000),
                    help="Max rolling buffer size in chars")
    ap.add_argument("--prompt", default=defaults.get("prompt"),
                    help="Override captured prompt string")
    ap.add_argument("--history", type=int,
                    default=defaults.get("history", 20),
                    help="Seed with last N commands from shell history (0 = disabled). Ignored if tmux capture succeeds.")
    ap.add_argument("--tmux-lines", type=int,
                    default=defaults.get("tmux_lines", 300),
                    help="If inside tmux, seed with the last N lines of the current pane's scrollback (real commands + outputs). 0 = disabled.")
    ap.add_argument("--quantize", choices=["none", "4bit", "8bit"],
                    default=defaults.get("quantize", "none"),
                    help="Load model with bitsandbytes quantization (CUDA only).")
    ap.add_argument("--no-kv-cache", action="store_true",
                    default=not defaults.get("kv_cache", True),
                    help="Disable KV cache reuse across turns. Slower; mostly for debugging.")
    args = ap.parse_args()

    prompt_tmpl = args.prompt if args.prompt is not None else capture_prompt()
    # Skeleton is derived once from the captured prompt's structure; only
    # `prompt_tmpl` itself drifts (gets reassigned to whatever the model
    # emits when its output matches the skeleton).
    skeleton = build_skeleton(prompt_tmpl)

    print(f"[loading {args.model} ...]", file=sys.stderr)
    tok, model, device = load_model(args)
    print(f"[kenoma: device={device} dtype={next(model.parameters()).dtype} "
          f"quantize={args.quantize} kv_cache={not args.no_kv_cache}]",
          file=sys.stderr)

    # Seed readline history so arrow-up recalls previous commands.
    history_cmds = read_history(args.history)
    for cmd in history_cmds:
        readline.add_history(cmd)

    tmux_seed = capture_tmux_pane(args.tmux_lines)
    buf = ""
    if tmux_seed:
        buf = tmux_seed.rstrip() + "\n"
    else:
        for cmd in history_cmds:
            buf += prompt_tmpl + cmd + "\n"
    buf += prompt_tmpl
    # Display only the live prompt, not the seeded transcript.
    sys.stdout.write(prompt_tmpl)
    sys.stdout.flush()

    cached_ids = None
    cached_kv = None

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

        target_ids = tok(buf, return_tensors="pt").input_ids[0].tolist()

        # Figure out how much of the cache we can reuse, and what delta to feed.
        if args.no_kv_cache or cached_ids is None:
            feed_ids = target_ids
            kv = None
            lcp = 0
        else:
            lcp = lcp_len(cached_ids, target_ids)
            # Drop one more token so the final cached position is regenerated;
            # avoids edge cases where the last cached token's logits are stale
            # under some model configs.
            lcp = max(0, lcp - 1) if lcp == len(target_ids) else lcp
            kv = truncate_cache(cached_kv, lcp) if lcp > 0 else None
            feed_ids = target_ids[lcp:]
            if not feed_ids:  # buf didn't grow; shouldn't happen, but be defensive
                feed_ids = target_ids[-1:]
                lcp = len(target_ids) - 1
                kv = truncate_cache(cached_kv, lcp) if lcp > 0 else None

        input_ids = torch.tensor([feed_ids], dtype=torch.long, device=model.device)
        attention_mask = torch.ones((1, lcp + len(feed_ids)), dtype=torch.long, device=model.device)
        # The stopper slices the generated tail out of `input_ids` inside the
        # generation loop — that's feed_ids + generated, so skip len(feed_ids).
        prompt_len = len(feed_ids)

        streamer = TextIteratorStreamer(
            tok, skip_prompt=True, skip_special_tokens=True
        )
        stopper = StopOnPromptLike(tok, skeleton, prompt_len)
        stopping = StoppingCriteriaList([stopper])
        gen_kwargs: dict[str, Any] = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            streamer=streamer,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            stopping_criteria=stopping,
            pad_token_id=tok.pad_token_id,
            use_cache=True,
            return_dict_in_generate=True,
        )
        if kv is not None:
            gen_kwargs["past_key_values"] = kv

        gen_result: dict[str, Any] = {}
        def run_gen():
            try:
                gen_result["out"] = cast(Any, model).generate(**gen_kwargs)
            except Exception as e:
                gen_result["err"] = e

        t = threading.Thread(target=run_gen)
        t.start()

        produced = ""
        shown = 0
        try:
            for chunk in streamer:
                produced += chunk
                body_end, _ = stopper.match_span(produced)
                if body_end >= 0:
                    if body_end > shown:
                        sys.stdout.write(produced[shown:body_end])
                        sys.stdout.flush()
                    shown = body_end
                    break
                safe_upto = safe_flush_point(produced, prompt_tmpl)
                if safe_upto > shown:
                    sys.stdout.write(produced[shown:safe_upto])
                    sys.stdout.flush()
                    shown = safe_upto
        except KeyboardInterrupt:
            sys.stdout.write("\n")
        t.join()

        if "err" in gen_result:
            print(f"[kenoma: generation failed: {gen_result['err']}]", file=sys.stderr)
            cached_ids = None
            cached_kv = None
            sys.stdout.write(prompt_tmpl)
            sys.stdout.flush()
            continue

        body_end, prompt_end = stopper.match_span(produced)
        if body_end >= 0:
            body = produced[:body_end]
            # Adopt the model's emitted prompt as the new canonical. By
            # construction (skeleton match) it has the same structure as the
            # original captured prompt, just with whatever cwd the model
            # decided to put there. This keeps the fake terminal internally
            # consistent across `cd`s.
            prompt_tmpl = produced[body_end:prompt_end]
        else:
            body = produced
            sys.stdout.write(produced[shown:])

        if not body.endswith("\n"):
            sys.stdout.write("\n")
            body += "\n"
        sys.stdout.write(prompt_tmpl)
        sys.stdout.flush()

        buf += body + prompt_tmpl

        # Update the cache for next turn. We truncate back to len(target_ids)
        # — the point where cache covers exactly the text we fed (pre-
        # generation). The generated tail is discarded because we've just
        # replaced it with `body + prompt_tmpl`, which will be retokenized
        # next turn anyway; any LCP past len(target_ids) would be a lie.
        if args.no_kv_cache:
            cached_ids = None
            cached_kv = None
        else:
            out = gen_result.get("out")
            new_kv = getattr(out, "past_key_values", None) if out is not None else None
            if new_kv is None:
                cached_ids = None
                cached_kv = None
            else:
                cached_kv = truncate_cache(new_kv, len(target_ids))
                cached_ids = target_ids


if __name__ == "__main__":
    main()
