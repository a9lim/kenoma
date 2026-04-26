#!/usr/bin/env python3
"""LLM fake terminal: raw completion, no chat template. The real shell's PS1
is captured once and used as both the seed and the stop string, so the model
hallucinates command output until it emits the next prompt."""

# Kept above the heavy imports so setuptools' ``attr: kenoma.__version__``
# can AST-read it without paying the torch/transformers import cost.
__version__ = "1.0.0"

import argparse
import os
import queue
import re
import readline
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, cast

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)


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

# Single source of truth for default values (#14). Argparse pulls from
# `defaults.get(k, DEFAULTS[k])` so a missing config-file key falls back
# here instead of duplicating the literal in argparse. Update both
# CONFIG_KEYS (for type) and DEFAULTS (for value) when adding a flag —
# CONTRIBUTING.md mentions both.
DEFAULTS: "dict[str, Any]" = {
    "model": "Qwen/Qwen2.5-0.5B",
    "device": "auto",
    "max_new_tokens": 2048,
    "temperature": 1.0,
    "top_p": 0.95,
    "repetition_penalty": 1.05,
    "context_chars": 6000,
    "prompt": None,
    "history": 20,
    "tmux_lines": 300,
    "quantize": "none",
    "kv_cache": True,
}

ENV_PREFIX = "KENOMA_"


def config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return Path(base) / "kenoma" / "config.toml"


def load_config_file() -> dict[str, Any]:
    """Return a flat dict of recognized keys from the TOML config, or {}.
    Returns {} if the file is missing.

    Validates each key against CONFIG_KEYS (#13): unknown keys and values
    that fail type coercion are reported to stderr and dropped. The env-var
    path already does this; matching the file path keeps typos from being
    silently ignored.

    Warns loudly (#20) only if a config file exists but no TOML reader is
    available — the silent no-op was confusing for developers running from
    a checkout on 3.9/3.10 without `tomli` installed."""
    p = config_path()
    if not p.is_file():
        return {}
    if tomllib is None:
        print(
            f"[kenoma: {p} exists but no TOML reader available — install "
            "`tomli` (Python 3.9/3.10) or run on Python 3.11+]",
            file=sys.stderr,
        )
        return {}
    try:
        with open(p, "rb") as f:
            data = tomllib.load(f)
    except Exception as e:
        print(f"[kenoma: failed to read {p}: {e}]", file=sys.stderr)
        return {}

    out: dict[str, Any] = {}
    for k, v in data.items():
        if k not in CONFIG_KEYS:
            print(f"[kenoma: unknown config key {k!r} in {p} — ignored]",
                  file=sys.stderr)
            continue
        typ = CONFIG_KEYS[k]
        try:
            if typ is bool and isinstance(v, str):
                out[k] = v.strip().lower() in ("1", "true", "yes", "y", "on")
            else:
                out[k] = typ(v)
        except (ValueError, TypeError):
            print(
                f"[kenoma: invalid {k}={v!r} in {p} (expected "
                f"{typ.__name__}) — ignored]",
                file=sys.stderr,
            )
    return out


def load_env_overrides() -> dict[str, Any]:
    out: dict[str, Any] = {}
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


def merged_defaults() -> dict[str, Any]:
    d = load_config_file()
    d.update(load_env_overrides())
    return d


# ---------------------------------------------------------------------------
# Seed transcript capture.
# ---------------------------------------------------------------------------

def capture_tmux_pane(lines: int) -> str:
    """Return the last `lines` lines of the current tmux pane's scrollback,
    which includes real commands AND their outputs. Empty string if not in tmux
    or the capture fails.

    Caller must invoke this BEFORE writing any of our own stderr messages
    (`[loading]`, `[kenoma: device=]`, etc.) — otherwise those land in the
    pane and the trim heuristic has to count them. With early capture, the
    only line we drop is the `kenoma <model>` invocation itself."""
    if lines <= 0 or not os.environ.get("TMUX"):
        return ""
    try:
        out = subprocess.run(
            ["tmux", "capture-pane", "-p", "-J", "-S", f"-{lines}"],
            capture_output=True, text=True, timeout=3,
        )
        rows = out.stdout.rstrip("\n").split("\n")
        # Drop the last row: the command that launched kenoma.
        return "\n".join(rows[:-1]) if len(rows) > 1 else ""
    except Exception:
        return ""


_ZSH_EXT_PREFIX = re.compile(r"^: \d+:\d+;")


def read_history(n: int) -> list[str]:
    """Return the last n commands from the user's shell history file.
    Handles zsh extended-history format (`: <ts>:<dur>;<cmd>`) and plain bash.

    #15: detect zsh extended-history with a tight regex (the old check
    `";" in line[:32]` was a magic-number proxy). Treat trailing `\\`
    continuations as zsh-only — bash doesn't fold them. Don't drop
    `#`-leading lines: bash history doesn't use `#` for metadata, so a
    pasted command starting with `#` is a real command."""
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

    is_zsh = "zsh" in shell
    cmds = []
    pending = ""
    for raw in lines:
        line = raw.rstrip("\n")
        if not line:
            continue
        if is_zsh:
            m = _ZSH_EXT_PREFIX.match(line)
            if m:
                line = line[m.end():]
            # Trailing-`\` is a zsh extended-history record continuation.
            # Bash newline-separates and doesn't fold these.
            if line.endswith("\\"):
                pending += line[:-1] + "\n"
                continue
        cmd = pending + line
        pending = ""
        if cmd.strip():
            cmds.append(cmd)
    return cmds[-n:]


def constructed_fallback_prompt() -> str:
    """Fallback prompt used when PS1 capture fails or is rejected (#4
    multi-line, #16 bash 3.2). Reasonable approximation of a default
    `user@host:cwd $ ` shell prompt."""
    return f"{os.getenv('USER', 'user')}@{socket.gethostname().split('.')[0]}:{os.getcwd()} $ "


def _bash_supports_param_transform(shell: str) -> bool:
    """Return True if `shell` is bash >= 4.4 — the version that introduced
    `${PARAM@P}` parameter transformation, which we use to expand PS1
    client-side. macOS ships /bin/bash 3.2, so users with bash as $SHELL
    fall through this check and use the constructed-prompt fallback."""
    try:
        out = subprocess.run(
            [shell, "-c", "echo $BASH_VERSINFO"],
            capture_output=True, text=True, timeout=2,
        )
        parts = out.stdout.strip().split()
        if not parts:
            return False
        major = int(parts[0])
        if major >= 5:
            return True
        if major == 4 and len(parts) >= 2 and int(parts[1]) >= 4:
            return True
        return False
    except Exception:
        return False


def capture_prompt() -> str:
    """Capture the user's real rendered PS1.

    Interactive shell startup (rc files, plugins like zsh-autosuggestions,
    'Restored session: ...' banners, etc.) can print to stdout before our
    command runs, so we wrap the prompt in sentinels and extract.

    On bash < 4.4 (notably macOS's stock /bin/bash 3.2), we can't expand
    PS1 client-side, so we warn and fall through to the constructed
    fallback (#16). The user can pass --prompt to override."""
    shell = os.environ.get("SHELL", "/bin/zsh")
    begin, end = "\x1e<<PS1_BEGIN>>\x1e", "\x1e<<PS1_END>>\x1e"
    if "zsh" in shell:
        cmd = f'print -rn -- "{begin}${{(%)PS1}}{end}"'
    elif _bash_supports_param_transform(shell):
        cmd = f'printf "%s" "{begin}${{PS1@P}}{end}"'
    else:
        print(
            f"[kenoma: {shell} doesn't support PS1 expansion (needs bash "
            ">= 4.4); using constructed fallback. Pass --prompt to override.]",
            file=sys.stderr,
        )
        return constructed_fallback_prompt()
    try:
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
    return constructed_fallback_prompt()


def reject_multiline_prompt(prompt: str) -> str:
    """If `prompt` contains an embedded newline, warn and fall through to
    the constructed fallback (#4). Otherwise pass through unchanged.

    Multi-line PS1 defeats `safe_flush_point`'s holdback (which only
    protects from the latest \\n), so earlier prompt fragments can leak to
    the screen before the skeleton match completes. Must run before
    `build_skeleton` — otherwise `re.escape` happily escapes the embedded
    newline into the skeleton and poisons it."""
    if "\n" in prompt or "\r" in prompt:
        print(
            "[kenoma: multi-line PS1 not supported; falling back to "
            "constructed prompt. Pass --prompt '<single line>' to override.]",
            file=sys.stderr,
        )
        return constructed_fallback_prompt()
    return prompt


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

HOLDBACK_LINE_BUDGET_FLOOR = 200  # floor for the dynamic holdback budget
WINDOW_TOKENS_FLOOR = 64          # min trailing-token window for stop check
WINDOW_TOKENS_MARGIN = 16         # extra tokens past the prompt span so a
                                  # match anchored at the window's left edge
                                  # isn't lost to truncation
ADOPTED_PROMPT_MAX = 512          # hard cap on adopted prompt length
DRAIN_DEADLINE_SECONDS = 30.0     # wall-clock cap on the cancel-path drain
                                  # loop. The streamer's per-get timeout
                                  # bounds individual blocks, but a slow
                                  # prefill on a large model can keep the
                                  # gen thread alive past many timeouts;
                                  # this caps the total wait.

# Path-shape gate for adoption (#6/#7/#21). The wildcarded cwd span must be
# made of plausible path characters only — no shell metachars, no control
# bytes, no whitespace. Rejecting non-conforming spans prevents the model
# from poisoning prompt_tmpl with ANSI escapes or control chars.
_PATH_SHAPED = re.compile(r"^[A-Za-z0-9._/~\-]+$")


def line_budget_for(prompt: str) -> int:
    """Holdback budget in chars, derived from the current prompt length.
    A constant cap (#3) leaks the in-progress line for fancy powerline
    prompts; doubling the prompt length absorbs cwd drift growth, with a
    floor for short prompts."""
    return max(HOLDBACK_LINE_BUDGET_FLOOR, 2 * len(prompt))


def build_skeleton(prompt: str) -> "tuple[re.Pattern[str], bool]":
    """Return (compiled regex, substituted) where the regex matches the
    captured prompt with its cwd portion turned into a wildcard, and
    `substituted` is True iff a cwd-form was actually substituted.

    Anchored at \\n. The wildcard is `(?P<cwd>[^\\s]+)` — named so
    `match_span` can extract the matched cwd for the adoption gate. Can't
    bleed across a space, which keeps false positives like
    `\\n50 % done` from matching.

    Limitations: only the cwd (full path or `~`-relative) is wildcarded.
    Prompts that show only the cwd basename (bash `\\W`), or that include a
    git branch / exit-status segment, won't match drifted versions — they'll
    only match exact reproductions of the captured prompt. Generation in
    those cases runs to `--max-new-tokens` instead of stopping early. The
    `substituted` return lets the caller warn the user when this happens."""
    cwd = os.getcwd()
    home = os.path.expanduser("~")
    candidates = [cwd]
    if cwd == home:
        candidates.append("~")
    elif cwd.startswith(home + os.sep):
        candidates.append("~" + cwd[len(home):])
    pat = re.escape(prompt)
    substituted = False
    for c in candidates:
        ec = re.escape(c)
        if ec and ec in pat:
            pat = pat.replace(ec, r"(?P<cwd>[^\s]+)", 1)
            substituted = True
            break
    return re.compile(r"\n" + pat), substituted


def safe_flush_point(produced: str, current_prompt: str) -> int:
    """Index up to which `produced` can be flushed to stdout without risking
    that the held-back portion turns into a prompt match on the next token."""
    n = len(produced)
    budget = line_budget_for(current_prompt)
    # Regex holdback: the latest \n could still anchor a skeleton match if
    # the chars since it are all non-newline and within the line budget.
    regex_safe = n
    last_nl = produced.rfind("\n")
    if last_nl >= 0:
        tail = produced[last_nl + 1:]
        if "\n" not in tail and len(tail) <= budget:
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


def adoptable(prompt: str, matched_cwd: str) -> bool:
    """Gate for #6/#7/#21: the matched span looks like a real prompt with a
    plausible cwd, not a false-positive in the model's output. Reject:

    - over-long prompts (model went off the rails)
    - control bytes (ANSI escapes, etc.)
    - newlines (multi-line model emission, not real prompts)
    - cwd spans containing shell metachars

    On reject, the caller stops the turn but does NOT mutate prompt_tmpl —
    the model's matching span is silently dropped and we keep the canonical
    prompt we had."""
    if len(prompt) > ADOPTED_PROMPT_MAX:
        return False
    if "\n" in prompt or "\r" in prompt:
        return False
    if any(0x00 <= ord(c) < 0x20 and c != "\t" for c in prompt):
        return False
    if matched_cwd and not _PATH_SHAPED.match(matched_cwd):
        return False
    return True


def sanitize_prompt(p: str) -> str:
    """Strip control bytes from an adopted prompt. Defense in depth:
    `adoptable()` should already have rejected anything containing them,
    but if the regex escaped past the gate we still don't want the bytes
    on the user's terminal."""
    return "".join(c for c in p if c == "\t" or ord(c) >= 0x20)


class StopOnPromptLike(StoppingCriteria):
    """Stop when the generated tail contains a line matching the prompt
    skeleton (newline + literal-prefix + cwd-wildcard + literal-suffix).

    Decodes only a trailing window of tokens (#5) — the skeleton can match
    at most `len(prompt) + cwd_drift` chars, anchored at \\n, so re-decoding
    the full growing tail every step is wasteful for long generations."""

    def __init__(self, tokenizer: Any, skeleton: "re.Pattern[str]",
                 prompt_len: int, prompt_chars: int):
        self.tok = tokenizer
        self.skeleton = skeleton
        self.prompt_len = prompt_len
        # Heuristic: 4 chars/token worst case for ASCII prompts; halve to
        # convert from chars to tokens (we want ~2x the prompt-char span in
        # tokens), then add a margin so a match anchored at the window's
        # left edge isn't lost to truncation. Reconstructed each turn so it
        # tracks the adopted prompt's length.
        derived = (prompt_chars // 2) + WINDOW_TOKENS_MARGIN
        self.window = max(WINDOW_TOKENS_FLOOR, derived)

    # Transformers accepts a plain bool here even though the base signature
    # declares BoolTensor; the override is intentional.
    def __call__(self, input_ids, scores, **kw) -> bool:  # type: ignore[reportIncompatibleMethodOverride]
        gen_len = input_ids.shape[1] - self.prompt_len
        if gen_len <= 0:
            return False
        start = max(self.prompt_len, input_ids.shape[1] - self.window)
        text = self.tok.decode(input_ids[0][start:], skip_special_tokens=True)
        return self.skeleton.search(text) is not None

    def match_span(self, text: str) -> "tuple[int, int, str]":
        """Return (body_end, prompt_end, cwd) where text[body_end:prompt_end]
        is the matched prompt, text[:body_end] is everything before it
        (including the leading \\n), and `cwd` is the captured wildcard span
        (empty string if the skeleton has no `cwd` group). Returns
        (-1, -1, "") if no match."""
        m = self.skeleton.search(text)
        if not m:
            return -1, -1, ""
        # `cwd` named group only exists when build_skeleton substituted; if
        # no substitution fired the skeleton is exact-match-only and we
        # treat the cwd as empty (adoptable() handles empty-cwd gracefully).
        try:
            cwd = m.group("cwd") or ""
        except IndexError:
            cwd = ""
        return m.start() + 1, m.end(), cwd  # +1 keeps the leading \n in body


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
        import bitsandbytes  # noqa: F401  # pyright: ignore[reportMissingImports]
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


def load_model(args: argparse.Namespace):
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
    # Inference-only: disable grads so we don't accumulate autograd state
    # across turns (#18). Belt-and-braces alongside torch.inference_mode in
    # run_gen / warm_kv_cache.
    model.requires_grad_(False)
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

def lcp_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _reset_cache() -> "tuple[None, None]":
    """Single source of truth for "the KV cache is now invalid". Used by:
    truncation failure (#9), generation error, Ctrl-C cancel (#11), warmup
    failure (Bonus A). Caller assigns the return tuple to (cached_ids,
    cached_kv) and is responsible for forcing lcp=0 / full prefill on the
    same turn — a None cache without a same-turn lcp reset means the next
    feed_ids slice misses the prefix the model expects."""
    return None, None


def truncate_cache(cache: Any, n_tokens: int):
    """Crop a KV cache to the first `n_tokens` positions.

    Returns the (possibly mutated in place) cache on success, or None on
    failure. None means: caller MUST also reset cached_ids and force the
    current turn's lcp=0 / feed_ids=target_ids — the helper just signals
    that the cache state is invalid; it does not mutate caller state."""
    if cache is None or n_tokens <= 0:
        return None
    try:
        if hasattr(cache, "crop"):
            cache.crop(n_tokens)
            return cache
        # Legacy tuple-of-tuples: each (key, value) is [batch, heads, seq, head_dim].
        return tuple(
            (k[..., :n_tokens, :], v[..., :n_tokens, :])
            for (k, v) in cache
        )
    except Exception as e:
        # Soft failure: caller falls back to full prefill. Once per session
        # in the common case (cache class becomes known-unsupported, every
        # subsequent turn lands in the "no cached_kv" branch).
        print(
            f"[kenoma: KV cache truncate failed ({e!r}); falling back to "
            "full prefill]",
            file=sys.stderr,
        )
        return None


def trim_buf(buf: str, max_chars: int) -> str:
    """Trim `buf` to ~max_chars from the right, aligned to a \\n boundary
    so the buffer never starts mid-line. Falls back to a hard char trim if
    no newline is found ahead of the cut.

    Used at turn boundaries (Bonus B) — after `body + prompt_tmpl` lands —
    so we never present the model with a half-trimmed turn at its leading
    edge."""
    if len(buf) <= max_chars:
        return buf
    cut = len(buf) - max_chars
    nl = buf.find("\n", cut)
    if nl == -1:
        # No newline ahead; keep the whole tail as-is. Worst case: a single
        # turn whose body alone exceeds max_chars, in which case we honor
        # the budget rather than corrupting the buffer further.
        return buf[cut:]
    return buf[nl + 1:]


class CancelOnFlag(StoppingCriteria):
    """Stop generation when an external `threading.Event` is set. Used to
    plumb Ctrl-C from the main thread's streamer loop into the generation
    thread (#11). The event is checked once per token, so cancellation
    takes at most one token of compute to land."""

    def __init__(self, event: threading.Event):
        self.event = event

    def __call__(self, input_ids, scores, **kw) -> bool:  # type: ignore[reportIncompatibleMethodOverride]
        return self.event.is_set()


def warm_kv_cache(model: Any, tok: Any, buf: str) -> "tuple[list[int] | None, Any]":
    """Run one forward pass through the seeded `buf` so the first user turn
    starts with a populated KV cache instead of paying the seed's prefill
    cost mid-stream (Bonus A).

    Returns (cached_ids, cached_kv) on success, (None, None) on any
    failure — the caller's "both must be non-None" check then ensures
    we never ship a partial-state cache. We use `max_new_tokens=1` because
    some transformers versions reject 0; the generated token is discarded
    by truncating the cache back to seed-only length."""
    seed_ids = tok(buf, return_tensors="pt").input_ids[0].tolist()
    if not seed_ids:
        return None, None
    input_ids = torch.tensor([seed_ids], dtype=torch.long, device=model.device)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        out = cast(Any, model).generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1,
            do_sample=False,  # generated token is discarded; no point rolling RNG
            pad_token_id=tok.pad_token_id,
            use_cache=True,
            return_dict_in_generate=True,
        )
    new_kv = getattr(out, "past_key_values", None)
    if new_kv is None:
        return None, None
    truncated = truncate_cache(new_kv, len(seed_ids))
    if truncated is None:
        return None, None
    return seed_ids, truncated


# ---------------------------------------------------------------------------
# REPL.
# ---------------------------------------------------------------------------

def main() -> None:
    defaults = merged_defaults()

    ap = argparse.ArgumentParser(description="LLM fake terminal")
    ap.add_argument("--version", action="version",
                    version=f"%(prog)s {__version__}")
    # All argparse defaults pull from DEFAULTS (#14) so adding a flag only
    # needs CONFIG_KEYS + DEFAULTS + the argparse line — no literals to
    # keep in sync across multiple sites.
    ap.add_argument("model", nargs="?",
                    default=defaults.get("model", DEFAULTS["model"]),
                    help="HF model id or local path (base/completion model, not chat-tuned)")
    ap.add_argument("--device", default=defaults.get("device", DEFAULTS["device"]),
                    help="'auto' picks cuda → mps → cpu")
    ap.add_argument("--max-new-tokens", type=int,
                    default=defaults.get("max_new_tokens", DEFAULTS["max_new_tokens"]))
    ap.add_argument("--temperature", type=float,
                    default=defaults.get("temperature", DEFAULTS["temperature"]))
    ap.add_argument("--top-p", type=float,
                    default=defaults.get("top_p", DEFAULTS["top_p"]))
    ap.add_argument("--repetition-penalty", type=float,
                    default=defaults.get("repetition_penalty", DEFAULTS["repetition_penalty"]))
    ap.add_argument("--context-chars", type=int,
                    default=defaults.get("context_chars", DEFAULTS["context_chars"]),
                    help="Max rolling buffer size in chars")
    ap.add_argument("--prompt", default=defaults.get("prompt", DEFAULTS["prompt"]),
                    help="Override captured prompt string")
    ap.add_argument("--history", type=int,
                    default=defaults.get("history", DEFAULTS["history"]),
                    help="Seed with last N commands from shell history (0 = disabled). Ignored if tmux capture succeeds.")
    ap.add_argument("--tmux-lines", type=int,
                    default=defaults.get("tmux_lines", DEFAULTS["tmux_lines"]),
                    help="If inside tmux, seed with the last N lines of the current pane's scrollback (real commands + outputs). 0 = disabled.")
    ap.add_argument("--quantize", choices=["none", "4bit", "8bit"],
                    default=defaults.get("quantize", DEFAULTS["quantize"]),
                    help="Load model with bitsandbytes quantization (CUDA only).")
    ap.add_argument("--no-kv-cache", action="store_true",
                    default=not defaults.get("kv_cache", DEFAULTS["kv_cache"]),
                    help="Disable KV cache reuse across turns. Slower; mostly for debugging.")
    args = ap.parse_args()

    # Capture tmux scrollback BEFORE any stderr writes (#10). Our own
    # `[loading ...]` / `[kenoma: device=...]` lines land in the same pane,
    # and the trim heuristic that strips them was brittle (off-by-N on any
    # extra stderr line, e.g. HF download progress). Capturing early means
    # we only need to drop the kenoma-invocation line itself.
    tmux_seed = capture_tmux_pane(args.tmux_lines)

    prompt_tmpl = args.prompt if args.prompt is not None else capture_prompt()
    # Multi-line check (#4) MUST run before build_skeleton — re.escape would
    # otherwise quietly bake an embedded \n into the skeleton, defeating
    # safe_flush_point's holdback (which only protects from the latest \n).
    prompt_tmpl = reject_multiline_prompt(prompt_tmpl)
    # Skeleton is derived once from the captured prompt's structure; only
    # `prompt_tmpl` itself drifts (gets reassigned to whatever the model
    # emits when its output matches the skeleton).
    skeleton, substituted = build_skeleton(prompt_tmpl)
    if not substituted:
        print(
            "[kenoma: prompt has no detectable cwd; stop will only fire on "
            "exact reproductions. Each fake `cd` will run to "
            "--max-new-tokens. Pass --prompt '<prompt with $PWD>' to fix.]",
            file=sys.stderr,
        )
    else:
        # Whitespace-in-cwd subcase: the wildcard is \S+, so a cwd
        # containing spaces ('/Users/a9lim/My Documents') will substitute
        # but never match a drifted version. Same warning, distinct cause.
        cwd = os.getcwd()
        home = os.path.expanduser("~")
        cwd_forms = [cwd]
        if cwd.startswith(home):
            cwd_forms.append("~" + cwd[len(home):])
        if any(any(c.isspace() for c in form) for form in cwd_forms):
            print(
                "[kenoma: cwd contains whitespace; the prompt skeleton's "
                "wildcard (\\S+) cannot match it. Stop will only fire on "
                "exact reproductions. Pass --prompt '<prompt without cwd>' "
                "to silence.]",
                file=sys.stderr,
            )

    print(f"[loading {args.model} ...]", file=sys.stderr)
    tok, model, device = load_model(args)
    print(f"[kenoma: device={device} dtype={next(model.parameters()).dtype} "
          f"quantize={args.quantize} kv_cache={not args.no_kv_cache}]",
          file=sys.stderr)

    # Seed readline history so arrow-up recalls previous commands.
    history_cmds = read_history(args.history)
    for cmd in history_cmds:
        readline.add_history(cmd)

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

    cached_ids, cached_kv = _reset_cache()
    # Warm the KV cache from the seeded transcript (Bonus A) so the first
    # user turn sees a near-empty delta instead of paying the seed's full
    # prefill cost mid-stream. Both ids AND kv must be non-None for the
    # warmup to be useful — a partial cache + valid ids breaks the next
    # turn's lcp_len-driven delta feed.
    if not args.no_kv_cache:
        try:
            ids, kv = warm_kv_cache(model, tok, buf)
            if ids is not None and kv is not None:
                cached_ids, cached_kv = ids, kv
                print("[kenoma: kv cache warmed]", file=sys.stderr)
            else:
                print(
                    "[kenoma: kv cache warmup produced no usable state; "
                    "cold first turn]",
                    file=sys.stderr,
                )
        except Exception as e:
            print(
                f"[kenoma: kv cache warmup failed ({e!r}); cold first turn]",
                file=sys.stderr,
            )
            cached_ids, cached_kv = _reset_cache()

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
        # Buffer trim moved to turn boundary (Bonus B) — happens after the
        # full `body + prompt_tmpl` lands, not after the user's input. Means
        # the model never sees a half-trimmed turn at its leading edge.

        target_ids = tok(buf, return_tensors="pt").input_ids[0].tolist()

        # Figure out how much of the cache we can reuse, and what delta to
        # feed. If truncate_cache fails (#9), force lcp=0 / full prefill —
        # a None cache with lcp>0 would feed the delta against nothing.
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
            if lcp > 0 and kv is None:
                # Truncation failed — full prefill, not delta. Cache state
                # is now invalid; reset both ids and kv via the helper.
                cached_ids, cached_kv = _reset_cache()
                feed_ids = target_ids
                lcp = 0
            else:
                feed_ids = target_ids[lcp:]
                if not feed_ids:  # buf didn't grow; shouldn't happen, but be defensive
                    feed_ids = target_ids[-1:]
                    lcp = len(target_ids) - 1
                    kv = truncate_cache(cached_kv, lcp) if lcp > 0 else None
                    if lcp > 0 and kv is None:
                        cached_ids, cached_kv = _reset_cache()
                        feed_ids = target_ids
                        lcp = 0

        input_ids = torch.tensor([feed_ids], dtype=torch.long, device=model.device)
        attention_mask = torch.ones((1, lcp + len(feed_ids)), dtype=torch.long, device=model.device)
        # The stopper slices the generated tail out of `input_ids` inside the
        # generation loop — that's feed_ids + generated, so skip len(feed_ids).
        prompt_len = len(feed_ids)

        # timeout=5.0 prevents the drain loop from hanging if generation
        # raises before enqueuing the sentinel (#11). Without this, a cancel
        # arriving during prefill would block the consumer forever.
        streamer = TextIteratorStreamer(
            tok, skip_prompt=True, skip_special_tokens=True, timeout=5.0
        )
        stopper = StopOnPromptLike(tok, skeleton, prompt_len, len(prompt_tmpl))
        cancel = threading.Event()
        stopping = StoppingCriteriaList([stopper, CancelOnFlag(cancel)])
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
                # inference_mode is slightly faster than no_grad and is the
                # right tool here — we never backprop through generate().
                with torch.inference_mode():
                    gen_result["out"] = cast(Any, model).generate(**gen_kwargs)
            except Exception as e:
                gen_result["err"] = e

        # daemon=True so a hard process exit doesn't hang on the gen thread.
        t = threading.Thread(target=run_gen, daemon=True, name="kenoma-gen")
        t.start()

        produced = ""
        shown = 0
        try:
            for chunk in streamer:
                produced += chunk
                body_end, _, _ = stopper.match_span(produced)
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
            cancel.set()
            sys.stdout.write("\n[kenoma: cancelled]\n")
            sys.stdout.flush()
            # Drain with a wall-clock deadline so Ctrl-C during a slow
            # prefill on a large model can't block the main thread for the
            # full prefill duration. Per-`get` timeout (5s) yields
            # queue.Empty; deadline check between iterations bounds the
            # outer loop. Empty + dead thread means gen raised before
            # yielding the sentinel; bail immediately.
            deadline = time.monotonic() + DRAIN_DEADLINE_SECONDS
            while t.is_alive() and time.monotonic() < deadline:
                try:
                    for _ in streamer:
                        pass
                    break
                except queue.Empty:
                    continue
                except Exception:
                    break
        t.join(timeout=10.0)

        # Cancel path: discard everything; cache is unreliable.
        if cancel.is_set():
            cached_ids, cached_kv = _reset_cache()
            sys.stdout.write(prompt_tmpl)
            sys.stdout.flush()
            continue

        if "err" in gen_result:
            print(f"[kenoma: generation failed: {gen_result['err']}]", file=sys.stderr)
            cached_ids, cached_kv = _reset_cache()
            sys.stdout.write(prompt_tmpl)
            sys.stdout.flush()
            continue

        body_end, prompt_end, matched_cwd = stopper.match_span(produced)
        if body_end >= 0:
            body = produced[:body_end]
            candidate_prompt = produced[body_end:prompt_end]
            # Adoption gate (#6/#7/#21): only mutate `prompt_tmpl` if the
            # match looks structurally like a real prompt with a plausible
            # cwd. Reject ANSI escapes, control bytes, multi-line spans,
            # over-long prompts, and shell-metachar cwds. On reject we
            # silently keep the previous `prompt_tmpl` and drop the model's
            # match — the body is still the body, the discarded suffix is
            # just hallucinated noise.
            if adoptable(candidate_prompt, matched_cwd):
                prompt_tmpl = sanitize_prompt(candidate_prompt)
            else:
                print(
                    "[kenoma: rejected adopted prompt (failed path/length/"
                    "control-byte gate); keeping previous]",
                    file=sys.stderr,
                )
        else:
            body = produced
            sys.stdout.write(produced[shown:])

        if not body.endswith("\n"):
            sys.stdout.write("\n")
            body += "\n"
        sys.stdout.write(prompt_tmpl)
        sys.stdout.flush()

        buf += body + prompt_tmpl
        # Trim at turn boundary, line-aligned (Bonus B + #12).
        buf = trim_buf(buf, args.context_chars)

        # Update the cache for next turn. We truncate back to len(target_ids)
        # — the point where cache covers exactly the text we fed (pre-
        # generation). The generated tail is discarded because we've just
        # replaced it with `body + prompt_tmpl`, which will be retokenized
        # next turn anyway; any LCP past len(target_ids) would be a lie.
        # All failure paths route through _reset_cache() to keep cached_ids
        # and cached_kv coherent (cross-cutting invariant).
        if args.no_kv_cache:
            cached_ids, cached_kv = _reset_cache()
        else:
            out = gen_result.get("out")
            new_kv = getattr(out, "past_key_values", None) if out is not None else None
            if new_kv is None:
                cached_ids, cached_kv = _reset_cache()
            else:
                truncated = truncate_cache(new_kv, len(target_ids))
                if truncated is None:
                    cached_ids, cached_kv = _reset_cache()
                else:
                    cached_kv = truncated
                    cached_ids = target_ids


if __name__ == "__main__":
    main()
