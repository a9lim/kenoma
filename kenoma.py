#!/usr/bin/env python3
"""LLM fake terminal: raw completion, no chat template. The real shell's PS1
is captured once and used as both the seed and the stop string, so the model
hallucinates command output until it emits the next prompt."""

# Kept above the heavy imports so setuptools' ``attr: kenoma.__version__``
# can AST-read it without paying the torch/transformers import cost.
__version__ = "1.1.0"

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
    "compile": bool,
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
    "compile": False,
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
# Backtrack window for the main-loop regex search cursor: any match must end
# within MAX_SKELETON_MATCH_LEN chars of where it starts (literal prefix +
# wildcard cwd + literal suffix), so re-searching from `len(produced) -
# MAX_SKELETON_MATCH_LEN` covers any match completed by the latest chunk
# without rescanning the full buffer. ADOPTED_PROMPT_MAX is the hard cap on
# adoptable prompts; +64 is slack for the wildcard cwd region.
MAX_SKELETON_MATCH_LEN = ADOPTED_PROMPT_MAX + 64

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


def safe_flush_point(produced: str, current_prompt: str, last_nl_hint: int = -1) -> int:
    """Index up to which `produced` can be flushed to stdout without risking
    that the held-back portion turns into a prompt match on the next token.

    `last_nl_hint` is the caller-known absolute position of the latest \\n in
    `produced`, or -1 if unknown. With the hint we skip a full rfind on the
    growing buffer each chunk; without it we fall back to scanning."""
    n = len(produced)
    budget = line_budget_for(current_prompt)
    # Regex holdback: the latest \n could still anchor a skeleton match if
    # the chars since it are all non-newline and within the line budget.
    regex_safe = n
    last_nl = last_nl_hint if last_nl_hint >= 0 else produced.rfind("\n")
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


def _pick_attn_impl(device: str, quantized: bool) -> str:
    """Pick the best available attention implementation for `device`.

    On CUDA without quantization we try `flash_attention_2` if the
    `flash_attn` package is importable; otherwise we ask for `sdpa`
    (PyTorch scaled-dot-product attention, which itself dispatches to
    flash / memory-efficient kernels under the hood). MPS and CPU stick
    with `sdpa`. We avoid `flash_attention_2` under bitsandbytes quantization
    because bnb's matmul interception doesn't compose cleanly with FA2.

    Caller passes the result as ``attn_implementation=`` to
    ``from_pretrained``. If the model architecture rejects the choice (some
    models don't support FA2), we fall back to `sdpa` at load time."""
    if device.startswith("cuda") and not quantized:
        try:
            import flash_attn  # noqa: F401  # pyright: ignore[reportMissingImports]
            return "flash_attention_2"
        except ImportError:
            return "sdpa"
    return "sdpa"


def load_model(args: argparse.Namespace):
    """Resolve device, build quant config, load tokenizer + model. Returns
    (tokenizer, model, device)."""
    device = pick_device(args.device)
    quant = None if args.quantize == "none" else args.quantize
    if quant and not device.startswith("cuda"):
        print(f"[kenoma: --quantize {quant} requires CUDA; resolved device is {device!r}]", file=sys.stderr)
        sys.exit(2)
    dtype = pick_dtype(device, quant or "none")
    # `low_cpu_mem_usage=True` streams shards into final-place tensors instead
    # of building a full FP32 copy first; less peak RSS, faster startup.
    kwargs: dict[str, Any] = dict(torch_dtype=dtype, low_cpu_mem_usage=True)
    quant_config = build_quant_config(quant) if quant else None
    if quant_config is not None:
        kwargs["quantization_config"] = quant_config
        # bnb needs accelerate's placement; don't fight it.
        kwargs["device_map"] = "auto"
    attn_impl = _pick_attn_impl(device, quantized=quant_config is not None)
    kwargs["attn_implementation"] = attn_impl

    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    except (ValueError, ImportError, RuntimeError) as e:
        # The model may not support flash_attention_2 (architecture-specific)
        # or the local install may be missing wheels. Retry with sdpa, which
        # every modern transformers backend supports.
        if kwargs.get("attn_implementation") == "flash_attention_2":
            print(
                f"[kenoma: flash_attention_2 unavailable ({e!s}); "
                "falling back to sdpa]",
                file=sys.stderr,
            )
            kwargs["attn_implementation"] = "sdpa"
            model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
        else:
            raise
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

# How many chars of the previous buf to re-tokenize on the suffix-encode
# fast path, to absorb BPE/SP merges that span the buf-boundary. 64 is enough
# to cover any realistic single-token span in a byte-level BPE vocab; we
# verify the overlap re-tokenizes to the cached tail and fall back on
# mismatch.
_INCREMENTAL_TOK_OVERLAP = 64


def encode_with_cache(
    tok: Any,
    buf: str,
    prev_buf: "str | None",
    prev_ids: "list[int] | None",
) -> "tuple[list[int], str]":
    """Encode `buf` to token ids, reusing `prev_ids` as a prefix when `buf`
    strictly extends `prev_buf`.

    Returns ``(ids, snapshot)`` where ``snapshot`` is the buf string the
    caller should remember for next turn's cache lookup. Falls back to a
    full ``tok.encode(buf)`` whenever the cache is empty, the prefix
    relation is broken, or the boundary tokenization differs (BPE merges
    can span the join point).

    The boundary verification re-tokenizes the last ``_INCREMENTAL_TOK_OVERLAP``
    characters of ``prev_buf`` together with the suffix and compares the
    overlap region to the cached tail. If they match, the suffix tokenization
    is stable and we can append. If not, the join-point merge differs and we
    bail to a full re-encode (correct, no faster than before)."""
    if prev_buf is None or prev_ids is None or not buf.startswith(prev_buf) or len(buf) == len(prev_buf):
        return tok.encode(buf), buf
    suffix = buf[len(prev_buf):]
    # Re-tokenize a small back-overlap together with the new suffix to detect
    # BPE merges across the join. add_special_tokens=False to avoid a stray BOS.
    overlap_chars = min(_INCREMENTAL_TOK_OVERLAP, len(prev_buf))
    overlap_text = prev_buf[len(prev_buf) - overlap_chars:] + suffix
    overlap_ids = tok.encode(overlap_text, add_special_tokens=False)
    # Find the cached id sequence that corresponds to the overlap region by
    # walking back from the end of prev_ids and re-tokenizing prefixes; cheap
    # because we only need a probe. Simpler: just re-tokenize the overlap text
    # alone (no suffix) and compare to the cached tail.
    overlap_only_ids = tok.encode(prev_buf[len(prev_buf) - overlap_chars:], add_special_tokens=False)
    if not overlap_only_ids or len(overlap_only_ids) > len(prev_ids):
        return tok.encode(buf), buf
    cached_tail = prev_ids[-len(overlap_only_ids):]
    if cached_tail != overlap_only_ids:
        # Cached tail doesn't tokenize as expected — earlier merges differ.
        # Bail to a full re-encode rather than risk corruption.
        return tok.encode(buf), buf
    # The overlap region must match across both encodings, otherwise BPE
    # merges across the join differ and we can't safely concat.
    if overlap_ids[:len(overlap_only_ids)] != overlap_only_ids:
        return tok.encode(buf), buf
    suffix_ids = overlap_ids[len(overlap_only_ids):]
    return list(prev_ids) + suffix_ids, buf


def lcp_len(a: list[int], b: list[int]) -> int:
    """Longest common prefix length of two int lists.

    Fast path: a slice equality check at the C level handles the common
    case where one list is a strict prefix of the other (cache fully
    reused across turns). The Python-level loop only runs on real
    mismatches, where it terminates early at the first divergence."""
    n = min(len(a), len(b))
    if n == 0:
        return 0
    # Slice-equality fast path (C-level memcmp on small-int lists).
    if a[:n] == b[:n]:
        return n
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
    # `tok.encode` skips the BatchEncoding dict + tensor + .tolist() round-trip
    # that `tok(..., return_tensors='pt').input_ids[0].tolist()` does. Same
    # add_special_tokens default (True), same result for every tokenizer we
    # care about.
    seed_ids = tok.encode(buf)
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
    ap.add_argument("--compile", action="store_true",
                    default=defaults.get("compile", DEFAULTS["compile"]),
                    help="torch.compile the model with a static KV cache for faster decode. "
                         "Disables cross-turn KV cache reuse (incompatible with the static cache). "
                         "First turn pays a compile cost; best on CUDA.")
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

    # --compile mode: static KV cache + torch.compile for fast decode.
    # The static cache is allocated once per generate() call to a fixed
    # size; it doesn't expose .crop(), so cross-turn LCP cache reuse is
    # off the table here. We trade prefill cost (every turn re-prefills
    # the buffer from scratch) for ~2-3x faster decode on CUDA. On
    # CPU/MPS, torch.compile gains are smaller and sometimes negative;
    # warn but still try.
    if args.compile:
        if not args.no_kv_cache:
            print(
                "[kenoma: --compile disables cross-turn KV cache reuse "
                "(static cache doesn't support .crop()); ignoring "
                "--no-kv-cache=False]",
                file=sys.stderr,
            )
            args.no_kv_cache = True
        if not device.startswith("cuda"):
            print(
                f"[kenoma: --compile on device={device!r} may be slower "
                "than uncompiled; CUDA is the supported target]",
                file=sys.stderr,
            )
        try:
            cast(Any, model.generation_config).cache_implementation = "static"
            # Compile only `forward`. Compiling the whole module hits more
            # of `generate()`'s control flow, which still trips graph breaks
            # in current torch/transformers; forward-only is robust.
            model.forward = torch.compile(  # type: ignore[reportAttributeAccessIssue]
                model.forward, mode="reduce-overhead", fullgraph=False,
            )
            print(
                "[kenoma: torch.compile enabled (first turn will be slow "
                "while CUDA graphs warm up)]",
                file=sys.stderr,
            )
        except Exception as e:
            print(
                f"[kenoma: torch.compile setup failed ({e!r}); continuing "
                "uncompiled]",
                file=sys.stderr,
            )

    print(f"[kenoma: device={device} dtype={next(model.parameters()).dtype} "
          f"quantize={args.quantize} kv_cache={not args.no_kv_cache} "
          f"compile={args.compile}]",
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

    # Cross-turn tokenization cache: prev_buf is the text we last tokenized,
    # prev_ids the corresponding ids. encode_with_cache uses this to skip
    # re-tokenizing the whole growing buffer when the next turn's buf
    # strictly extends the previous (the common case). On trim, the prefix
    # relation breaks, encode_with_cache silently falls back to a full
    # encode, and prev_buf/prev_ids re-anchor on the next turn.
    prev_buf: "str | None" = None
    prev_ids: "list[int] | None" = None

    # Preallocated all-ones attention_mask buffer — a slice of this is
    # passed to generate() each turn. Avoids reallocating the tensor every
    # time. Grows geometrically when a turn outsizes it.
    amask_capacity = 0
    amask_buf: "torch.Tensor | None" = None

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

        # Incremental encode: when buf strictly extends prev_buf (the common
        # case: previous turn's body was appended, then user typed a line),
        # re-tokenize only the suffix and concat. Falls back to a full
        # tok.encode(buf) on trim/divergence.
        target_ids, prev_buf = encode_with_cache(tok, buf, prev_buf, prev_ids)
        prev_ids = target_ids

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
        # Reuse the preallocated all-ones buffer when it's big enough; grow
        # geometrically + add headroom for `max_new_tokens` worth of generated
        # positions when we resize. The slice is a view, so the actual tensor
        # passed to generate() needs no new allocation in steady state.
        need_len = lcp + len(feed_ids)
        if amask_buf is None or amask_capacity < need_len:
            amask_capacity = max(2 * need_len, need_len + args.max_new_tokens, 4096)
            amask_buf = torch.ones((1, amask_capacity), dtype=torch.long, device=model.device)
        attention_mask = amask_buf[:, :need_len]
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
            stopping_criteria=stopping,
            pad_token_id=tok.pad_token_id,
            use_cache=True,
            return_dict_in_generate=True,
        )
        # 1.0 is a no-op for the repetition-penalty logits processor, but
        # passing it still constructs the processor and runs it every token.
        # Skip when neutral.
        if args.repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = args.repetition_penalty
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
        # Search-position cursor: skeleton.search starts from search_anchor
        # each iteration so we don't rescan the entire growing `produced`.
        # Backtrack of MAX_SKELETON_MATCH_LEN ensures any match completed
        # by the latest chunk is still found even if its anchor \n was
        # written in an earlier chunk.
        search_anchor = 0
        # Latest-newline cursor for safe_flush_point — avoids a full rfind
        # on the buffer each chunk.
        last_nl_in_produced = -1
        # Captured match details — reused after the loop instead of running
        # match_span on `produced` a second time.
        matched_body_end = -1
        matched_prompt_end = -1
        matched_cwd_text = ""
        # Distinguishes Ctrl-C (discard the turn) from a normal skeleton
        # match that also sets `cancel` to stop the gen thread. Without
        # this, the post-loop `if cancel.is_set()` branch would fire on
        # *every* successful match and skip the body append + prev_buf
        # update, drifting the transcript across turns.
        user_cancelled = False
        try:
            for chunk in streamer:
                # Update last-newline cursor before the concat so the chunk
                # offset is straightforward.
                chunk_nl = chunk.rfind("\n")
                if chunk_nl >= 0:
                    last_nl_in_produced = len(produced) + chunk_nl
                produced += chunk
                m = skeleton.search(produced, search_anchor)
                if m:
                    matched_body_end = m.start() + 1  # +1 keeps the leading \n in body
                    matched_prompt_end = m.end()
                    try:
                        matched_cwd_text = m.group("cwd") or ""
                    except IndexError:
                        matched_cwd_text = ""
                    if matched_body_end > shown:
                        sys.stdout.write(produced[shown:matched_body_end])
                        sys.stdout.flush()
                    shown = matched_body_end
                    # Tell the gen thread to stop. The gen-side stopper
                    # usually fires on the same token, but signaling here is
                    # defense in depth — without it, a slow gen-side decode
                    # path could keep producing tokens nobody reads, piling
                    # them in the streamer's unbounded queue until the
                    # 10s join timeout.
                    cancel.set()
                    break
                # Advance the cursor: never search anything more than
                # MAX_SKELETON_MATCH_LEN back from the latest position.
                new_anchor = max(0, len(produced) - MAX_SKELETON_MATCH_LEN)
                if new_anchor > search_anchor:
                    search_anchor = new_anchor
                safe_upto = safe_flush_point(produced, prompt_tmpl, last_nl_in_produced)
                if safe_upto > shown:
                    sys.stdout.write(produced[shown:safe_upto])
                    sys.stdout.flush()
                    shown = safe_upto
        except KeyboardInterrupt:
            user_cancelled = True
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

        # Cancel path: discard generated tokens; KV cache is unreliable.
        # Only the user-cancelled path (Ctrl-C) takes this branch — a
        # normal-match exit also sets `cancel` to stop the gen thread, but
        # we want to keep its body and adopt its prompt. The tokenization
        # cache (prev_buf/prev_ids) does NOT need a reset: it tracks `buf`,
        # which only grew by the user's input line this turn — no
        # generated text was appended — so the prefix relation still holds
        # for the next turn.
        if user_cancelled:
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

        # Reuse the match captured inside the streamer loop. If no match
        # fired in the loop (max_new_tokens, EOS, or natural end of stream
        # without a prompt-shaped line) we fall back to a single match_span
        # call here so we still catch a tail-only match the streaming
        # search-cursor might have advanced past.
        if matched_body_end >= 0:
            body_end = matched_body_end
            prompt_end = matched_prompt_end
            matched_cwd = matched_cwd_text
        else:
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
