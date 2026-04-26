# Audit fixes — implementation plan

> **Status: implemented.** All four PR groups (release-unblock,
> correctness, robustness, polish + perf) shipped together as one push to
> `main` after Codex plan review and a code-reviewer pass on the diff. Plus
> issue 17 (README truncated sentence) was fixed in the same docs sweep.
> This file is kept as a record of the design rationale and the
> Codex-flagged edge cases. CLAUDE.md is the up-to-date design doc.

Fixes for findings 1–16 and 18–21 from the April 2026 audit, plus two
opportunistic improvements (seed KV cache warmup, turn-boundary buffer
trim). Issue 17 (README typo) was originally out of scope but landed as
part of the docs audit.

Numbering matches the audit. Each entry: **what**, **where**, **how**, and
any user-visible behavior change.

The work is grouped into landable chunks; the suggested order is at the end.

## Cross-cutting invariant: cache state is binary

A shared concern hits #9, #11, and Bonus A: if any code path leaves
`cached_kv` truncated or partially-built while `cached_ids` still implies
a longer prefix, the next turn feeds only a delta against an invalid
cache and either crashes or silently generates from a wrong prefix.

To prevent four code sites from drifting out of sync, all cache mutation
goes through a single helper:

```python
def _reset_cache() -> tuple[None, None]:
    """Clear the KV cache. Caller assigns the return tuple to
    (cached_ids, cached_kv). Used on: truncation failure (#9), generation
    error, Ctrl-C cancel (#11), warmup failure (Bonus A), and post-trim
    fallback when cache class can't crop."""
    return None, None
```

Caller pattern (used in every cache-touching site):

```python
cached_ids, cached_kv = _reset_cache()
```

Any time a cache mutation fails or returns `None`, the *caller* is
responsible for also forcing `lcp = 0` and `feed_ids = target_ids` for the
current turn. The truncate helpers (#9) only signal failure; they don't
mutate caller state. The Group 3 PR introduces this helper; #9, #11, and
Bonus A all consume it.

---

## 1. Release workflow KeyError on dynamic version

**Where:** `.github/workflows/release.yml`, the "Read version from
pyproject.toml" step.

**What:** `tomllib.load(...)["project"]["version"]` will `KeyError` because
`version` is now `dynamic`. Same root cause as #2.

**How:** Read `__version__` from `kenoma.py` directly, mirroring the regex
already used by the CI version job:

```yaml
- name: Read version from kenoma.py
  id: ver
  run: |
    python - <<'PY' >> "$GITHUB_OUTPUT"
    import pathlib, re
    text = pathlib.Path("kenoma.py").read_text()
    m = re.search(r'^__version__ = "([^"]+)"', text, re.M)
    if m is None:
        raise SystemExit("kenoma.py has no __version__")
    print(f"version={m.group(1)}")
    PY
```

Keep the rest of the workflow (tag check, build, publish) unchanged.

---

## 2. CI version-consistency job is obsolete

**Where:** `.github/workflows/ci.yml`, the `version` job.

**What:** It compared `pyproject["project"]["version"]` against
`kenoma.__version__`. The pyproject side now raises, and conceptually the
job is moot since there's a single source of truth.

**How:** Replace the job with one that validates the dynamic resolution
actually works end-to-end — install the package and confirm
`importlib.metadata.version("kenoma")` matches `kenoma.__version__`:

```yaml
version:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v6
    - uses: actions/setup-python@v6
      with:
        python-version: "3.12"
    - run: pip install .
    - name: Verify dynamic version resolves
      run: |
        python - <<'PY'
        import importlib.metadata, kenoma
        installed = importlib.metadata.version("kenoma")
        if installed != kenoma.__version__:
            raise SystemExit(
                f"dynamic version mismatch: installed={installed}, "
                f"__version__={kenoma.__version__}"
            )
        PY
```

This catches `attr:` regressions (e.g., someone introduces a top-of-module
import that breaks setuptools' AST read).

---

## 3. Long prompts leak before the skeleton matches

**Where:** `kenoma.py`, `safe_flush_point()` and the `HOLDBACK_LINE_BUDGET`
constant.

**What:** A 200-char hard cap can be smaller than a fancy powerline prompt,
so partial prompt text streams to screen before `match_span` fires.

**How:** Make the budget a function of the *current* prompt length, not a
constant:

```python
HOLDBACK_LINE_BUDGET_FLOOR = 200

def line_budget_for(prompt: str) -> int:
    return max(HOLDBACK_LINE_BUDGET_FLOOR, 2 * len(prompt))
```

Plumb `prompt_tmpl` into `safe_flush_point` (already a parameter — just use
its length to compute the budget instead of the module constant). The
factor of 2 absorbs cwd drift growth.

---

## 4. Multi-line prompts defeat the holdback

**Where:** `capture_prompt()` and the `--prompt` parsing path.

**What:** `safe_flush_point` only protects the latest `\n`; a `PS1` with
embedded newlines can leak earlier prompt fragments before the skeleton
match completes.

**How:** Reject-and-warn at capture/parse time. After `capture_prompt()`
and after `--prompt` is read, check `"\n" in prompt_tmpl`. If yes, emit:

```
[kenoma: multi-line PS1 not supported; falling back to constructed prompt.
 Pass --prompt '<single line>' to override.]
```

…and use the constructed `user@host:cwd $ ` fallback (the existing one at
the bottom of `capture_prompt`). Document the limitation in README and
SECURITY.md.

**Ordering (Codex review):** the multi-line check must run *before*
`build_skeleton(prompt_tmpl)` is called. If a multi-line prompt slips
through to `build_skeleton`, `re.escape` will happily escape the `\n` and
poison the skeleton with a multi-line anchor. Concretely, the sequence
in `main()` is: capture/parse → multi-line check + fallback assignment →
`build_skeleton(prompt_tmpl)`. Same sequence applies to `--prompt`.

---

## 5. O(n²) tail decoding in StopOnPromptLike

**Where:** `kenoma.py`, `StopOnPromptLike.__call__`.

**What:** `self.tok.decode(input_ids[0][self.prompt_len:], …)` decodes the
entire growing tail every step.

**How:** Decode only the trailing window large enough to cover the
skeleton's possible match length. The skeleton matches at most
`len(prompt_tmpl) + cwd_drift` chars, anchored at `\n`. A safe window in
*tokens* is `ceil((len(prompt_tmpl) * 4) / min_chars_per_token) + slack`;
for typical BPE, 64–128 tokens covers any reasonable prompt.

```python
WINDOW_TOKENS_FLOOR = 64
WINDOW_TOKENS_MARGIN = 16  # extra tokens past prompt span so an anchor
                           # at the left edge isn't lost to truncation

class StopOnPromptLike(StoppingCriteria):
    def __init__(self, tokenizer, skeleton, prompt_len, prompt_chars):
        self.tok = tokenizer
        self.skeleton = skeleton
        self.prompt_len = prompt_len
        # Heuristic: 4 chars/token worst case for ASCII prompts; double the
        # char-derived size to absorb cwd drift, then add a token margin so
        # a match anchored at the left edge of the window isn't missed.
        derived = (prompt_chars // 2) + WINDOW_TOKENS_MARGIN
        self.window = max(WINDOW_TOKENS_FLOOR, derived)

    def __call__(self, input_ids, scores, **kw) -> bool:
        gen_len = input_ids.shape[1] - self.prompt_len
        if gen_len <= 0:
            return False
        start = max(self.prompt_len, input_ids.shape[1] - self.window)
        text = self.tok.decode(input_ids[0][start:], skip_special_tokens=True)
        return self.skeleton.search(text) is not None
```

**Window-sizing edge case (Codex review):** if the matching `\n` lies at
the very first decoded token of the window, `re.search` only finds it
when at least one prior token's worth of context has been emitted before
the `\n`. The `WINDOW_TOKENS_MARGIN` guarantees this: the window covers
*more* than the maximum possible match span, so the anchor is never the
first byte of decoded text.

When `prompt_tmpl` is adopted (#6) and grows or shrinks, the stopper is
reconstructed for the next turn — no in-flight resize needed.

Note: the *full* `match_span` call after stop is unaffected — that runs on
`produced` (the streamer's text), not on tokens. So we keep correctness for
the post-stop slice and just speed up the per-token check.

---

## 6. False-positive skeleton matches truncate real output

**Where:** `kenoma.py`, REPL loop's adoption branch (around line 576–584).

**What:** Skeleton fires on legitimate prompt-shaped lines in real output
(e.g., a `cat /etc/issue`-like response from the model). We currently
adopt that span as the new canonical prompt and discard the rest.

**How:** Decouple stopping from adoption. The stopper still fires on any
match (we need a budget cap). Adoption gates on the wildcarded cwd span
being "path-shaped":

```python
_PATH_SHAPED = re.compile(r"^[A-Za-z0-9._/~\-]+$")
ADOPTED_PROMPT_MAX = 512

def adoptable(prompt: str, matched_cwd_span: str) -> bool:
    if len(prompt) > ADOPTED_PROMPT_MAX:
        return False
    if "\n" in prompt or "\r" in prompt:
        return False
    # Strip ANSI / control bytes check (also see #7)
    if any(0x00 <= ord(c) < 0x20 and c != "\t" for c in prompt):
        return False
    if not _PATH_SHAPED.match(matched_cwd_span):
        return False
    return True
```

To extract the matched cwd span we need the regex group it sits in.
Refactor `build_skeleton()` to wrap the wildcard in a named group:

```python
pat = pat.replace(ec, r"(?P<cwd>[^\s]+)", 1)
```

…and `match_span` returns `(body_end, prompt_end, cwd)` (third element is
`m.group("cwd") or ""`).

When `adoptable()` is False:
- Stop the turn (we already broke out of the streamer loop).
- Do **not** mutate `prompt_tmpl`.
- Body is `produced[:body_end]` as today.
- Re-emit the existing `prompt_tmpl` and append `body + prompt_tmpl` to
  `buf`. The garbage-shaped match from the model is silently dropped.

Optionally: if adoption was rejected, log `[kenoma: rejected adopted prompt]`
to stderr to give the user a debugging signal.

---

## 7. Adopted prompts aren't sanitized

**Where:** `kenoma.py`, the adoption branch (line 584).

**What:** `[^\s]+` matches ANSI escapes and control bytes happily. Adopted
text gets written to stdout and glued into `buf`.

**How:** Folded into #6 — `adoptable()` already rejects control bytes
(`< 0x20` except `\t`) and over-length spans. Belt-and-braces, also strip
any remaining control bytes from the adopted prompt before assignment:

```python
def sanitize_prompt(p: str) -> str:
    return "".join(c for c in p if c == "\t" or ord(c) >= 0x20)
```

Apply on adoption *only after* `adoptable()` passes (so this is mostly a
no-op, just defense in depth).

---

## 8. Bash `\W` / git-branch prompts degrade silently

**Where:** `kenoma.py`, `build_skeleton()`.

**What:** When neither the full cwd nor `~/...` form is a literal substring
of the captured prompt, no wildcard substitution fires. The skeleton then
only matches exact reproductions of the launch prompt; after the model's
first fake `cd`, no match fires until `--max-new-tokens` runs out, every
turn.

**How:** Detect when no substitution happened and warn at startup. Per
agreement, *no* basename fallback (CLAUDE.md explicitly rejects it):

```python
def build_skeleton(prompt: str) -> tuple["re.Pattern[str]", bool]:
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
```

In `main()`, after building the skeleton:

```python
skeleton, substituted = build_skeleton(prompt_tmpl)
if not substituted:
    print(
        "[kenoma: prompt has no detectable cwd; stop will only fire on exact "
        "reproductions. Each fake `cd` will run to --max-new-tokens. "
        "Pass --prompt '<prompt with $PWD>' to fix this.]",
        file=sys.stderr,
    )
```

**Whitespace-in-cwd subcase (Codex review):** the wildcard is `[^\s]+`,
so a cwd containing spaces (`/Users/a9lim/My Documents`) would substitute
but never match a drifted version, since `\s` will reject the space. Same
warning, same triggering condition, distinct cause. Detect this in
`build_skeleton` and report:

```python
if any(any(c.isspace() for c in cand) for cand in candidates) and substituted:
    # We did substitute, but the wildcard can't match the literal cwd.
    print(
        "[kenoma: cwd contains whitespace; the prompt skeleton's wildcard "
        "(\\S+) cannot match it. Stop will only fire on exact reproductions. "
        "Pass --prompt '<prompt without cwd>' to silence.]",
        file=sys.stderr,
    )
```

CLAUDE.md should grow a one-line note about both warnings.

---

## 9. KV cache `.crop()` is model-dependent

**Where:** `kenoma.py`, `truncate_cache()` and its callers.

**What:** Some sliding-window cache classes don't support arbitrary
truncation; they raise. We currently let the exception propagate, killing
the turn.

**How:** Make truncation soft: on any exception in `truncate_cache()`,
return `None` and let the caller fall back to a full retokenize/prefill.
Same for `truncate_cache(new_kv, len(target_ids))` in the post-generation
update — on failure, set `cached_ids = cached_kv = None`.

```python
def truncate_cache(cache, n_tokens):
    if cache is None or n_tokens <= 0:
        return None
    try:
        if hasattr(cache, "crop"):
            cache.crop(n_tokens)
            return cache
        return tuple(
            (k[..., :n_tokens, :], v[..., :n_tokens, :])
            for (k, v) in cache
        )
    except Exception as e:
        # Soft failure: caller will retokenize from scratch.
        print(f"[kenoma: KV cache truncate failed ({e!r}); falling back to full prefill]",
              file=sys.stderr)
        return None
```

The print only fires once per session in the common case (after the cache
class becomes known-unsupported, every subsequent turn won't have a cache
to truncate).

**Caller responsibility (Codex review):** the helper's `None` return is
*just a signal*. The caller has to also force a full prefill for the
current turn — `truncate_cache` returning `None` while the caller still
holds `lcp > 0` and feeds only `target_ids[lcp:]` is exactly the broken
state we're trying to prevent. Pattern in the REPL loop:

```python
if not args.no_kv_cache and cached_ids is not None:
    lcp = lcp_len(cached_ids, target_ids)
    lcp = max(0, lcp - 1) if lcp == len(target_ids) else lcp
    kv = truncate_cache(cached_kv, lcp) if lcp > 0 else None
    if lcp > 0 and kv is None:
        # Truncation failed — full prefill, not delta.
        cached_ids, cached_kv = _reset_cache()
        feed_ids = target_ids
        lcp = 0
    else:
        feed_ids = target_ids[lcp:]
else:
    feed_ids = target_ids
    kv = None
    lcp = 0
```

Same pattern at the post-generation update site:

```python
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
```

Both sites use `_reset_cache()` from the cross-cutting invariant section.

---

## 10. tmux scrollback `-3` heuristic is brittle

**Where:** `kenoma.py`, `capture_tmux_pane()` and its caller in `main()`.

**What:** The `[loading ...]`, model-load output, and `[kenoma: device=]`
stderr messages are written *before* tmux capture runs. Trim of `-3` rows
assumes a fixed line count which any HF download progress or future stderr
print breaks.

**How:** Capture *early* — move `capture_tmux_pane()` to the top of
`main()`, before any `print(file=sys.stderr)` calls. Since we no longer
have any of our own stderr noise in the pane, trim only the bottom row
(the `kenoma <model>` invocation line):

```python
def capture_tmux_pane(lines: int) -> str:
    if lines <= 0 or not os.environ.get("TMUX"):
        return ""
    try:
        out = subprocess.run(
            ["tmux", "capture-pane", "-p", "-J", "-S", f"-{lines}"],
            capture_output=True, text=True, timeout=3,
        )
        rows = out.stdout.rstrip("\n").split("\n")
        # Drop the last row: the kenoma invocation line itself.
        return "\n".join(rows[:-1]) if len(rows) > 1 else ""
    except Exception:
        return ""
```

Reorder in `main()`:

```python
def main():
    defaults = merged_defaults()
    ap = argparse.ArgumentParser(...)
    args = ap.parse_args()

    # Capture BEFORE any stderr output lands in the pane.
    tmux_seed = capture_tmux_pane(args.tmux_lines)

    prompt_tmpl = args.prompt if args.prompt is not None else capture_prompt()
    # ... existing prompt validation, multi-line check (#4) ...
    skeleton, substituted = build_skeleton(prompt_tmpl)
    if not substituted:
        print(...)  # see #8

    print(f"[loading {args.model} ...]", file=sys.stderr)
    tok, model, device = load_model(args)
    ...
```

---

## 11. No graceful Ctrl-C cancel during generation

**Where:** `kenoma.py`, REPL loop's streamer iteration (lines 549–566).

**What:** Ctrl-C during streaming propagates up and most likely kills the
process. CLAUDE.md acknowledges this.

**How:** Flag-driven `StoppingCriteria`. Add a cancel event, install a
second criterion that returns `event.is_set()`, set the event from the
KeyboardInterrupt handler, drain the streamer with a timeout, redraw the
prompt:

```python
class CancelOnFlag(StoppingCriteria):
    def __init__(self, event: threading.Event):
        self.event = event
    def __call__(self, input_ids, scores, **kw) -> bool:
        return self.event.is_set()
```

In the REPL loop:

```python
cancel = threading.Event()
stopping = StoppingCriteriaList([stopper, CancelOnFlag(cancel)])
streamer = TextIteratorStreamer(
    tok, skip_prompt=True, skip_special_tokens=True,
    timeout=5.0,  # see "drain hang" note below
)
...
try:
    for chunk in streamer:
        ...
except KeyboardInterrupt:
    cancel.set()
    sys.stdout.write("\n[kenoma: cancelled]\n")
    # Drain only while the generation thread is alive AND not raising.
    # An empty queue + dead thread means the streamer never got a sentinel
    # (gen raised before yielding), which would otherwise hang forever.
    while t.is_alive():
        try:
            for _ in streamer:
                pass
            break
        except queue.Empty:
            # 5s timeout fired and gen is still alive — keep waiting.
            continue
        except Exception:
            # Some other streamer error; stop draining.
            break
    t.join(timeout=10.0)

if cancel.is_set():
    # Discard whatever was produced; reset cache (we don't trust it).
    cached_ids, cached_kv = _reset_cache()
    sys.stdout.write(prompt_tmpl)
    sys.stdout.flush()
    continue
```

**Drain-hang failure mode (Codex review):** `TextIteratorStreamer` is
backed by a `queue.Queue` with no built-in timeout in older transformers.
If `model.generate()` raises *before* enqueuing the sentinel (e.g., the
cancel arrives during prefill, or a partially-warmed cache from Bonus A
breaks generation immediately), the consumer loop blocks indefinitely
waiting for a sentinel that never lands. Two safeguards together prevent
this:

1. Pass `timeout=5.0` to `TextIteratorStreamer` so its blocking get raises
   `queue.Empty` instead of hanging forever.
2. Drain only while `t.is_alive()` — if the thread is dead and the queue
   stayed empty, we know the sentinel won't come and bail out.

The existing `gen_result["err"]` capture path (which runs *inside*
`run_gen`) already handles the case where generation raises and the
streamer iterator blocks on the empty queue — but only on the success
path, where we'd never enter this drain. The timeout + alive check covers
the cancel path.

Need `import queue` at top of module (or catch a more general exception
class — `Exception` from streamer iteration would suffice if we don't
want the import).

Note: setting `cancel.set()` doesn't stop the generation thread *instantly*;
the criterion is checked once per token, so we wait at most one token's
worth of compute. That's acceptable.

The first Ctrl-C cancels the turn; a second Ctrl-C at the input prompt
exits the program (existing behavior). Tests: cancel mid-stream, cancel
right at start, cancel during long generation, cancel when warmup-bound
generation immediately raises.

---

## 12. Buffer trim splits mid-line

**Where:** `kenoma.py`, REPL loop's `buf` trim (lines 486–487).

**What:** `buf = buf[-args.context_chars:]` chops mid-line.

**How:** Folded into the turn-boundary trim work below ("Buffer trim on
turn boundaries"). After the trim, advance the cut to the next `\n` to
guarantee the buffer never starts mid-line:

```python
def trim_buf(buf: str, max_chars: int) -> str:
    if len(buf) <= max_chars:
        return buf
    cut = len(buf) - max_chars
    nl = buf.find("\n", cut)
    if nl == -1:
        # No newline found ahead of the cut — keep the whole tail.
        return buf[cut:]
    return buf[nl + 1:]
```

Combined with "Buffer trim on turn boundaries" — see below.

---

## 13. Config validation too forgiving

**Where:** `kenoma.py`, `load_config_file()`.

**What:** Filters by key but doesn't typecheck values; ignores unknown
keys silently.

**How:** Apply `CONFIG_KEYS[k]` coercion (mirroring `load_env_overrides`)
and warn on unknown keys + coercion failures:

```python
def load_config_file() -> dict[str, Any]:
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
            print(f"[kenoma: invalid {k}={v!r} in {p} (expected {typ.__name__}) — ignored]",
                  file=sys.stderr)
    return out
```

---

## 14. Numeric defaults duplicated

**Where:** `kenoma.py`, the `argparse` block in `main()`. README and
CLAUDE.md repeat the values.

**What:** `history=20`, `tmux_lines=300`, `context_chars=6000`,
`max_new_tokens=2048`, `temperature=1.0`, `top_p=0.95`,
`repetition_penalty=1.05`, `quantize="none"`, `kv_cache=True`,
`device="auto"` are all literals in argparse and again in docs.

**How:** Add a `DEFAULTS` dict alongside `CONFIG_KEYS`:

```python
DEFAULTS: dict[str, Any] = {
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
```

Argparse then pulls from it:

```python
ap.add_argument("--temperature", type=float,
                default=defaults.get("temperature", DEFAULTS["temperature"]))
```

…and so on for every flag. CONTRIBUTING.md's "Adding a new feature flag"
section grows a fourth bullet for `DEFAULTS`. README's example config block
becomes generated from `DEFAULTS` if we feel like it; not in scope here.

---

## 15. `read_history` edge cases

**Where:** `kenoma.py`, `read_history()`.

**What:** `";" in line[:32]` for zsh extended-history detection is brittle;
unconditional drop of `#`-leading lines loses bash comment-commands;
trailing-`\` continuation conflates record-format with command content.

**How:** Tighter parsing:

```python
_ZSH_EXT_PREFIX = re.compile(r"^: \d+:\d+;")

def read_history(n: int) -> list[str]:
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
            # Only treat trailing `\` as continuation in zsh extended format.
            if line.endswith("\\"):
                pending += line[:-1] + "\n"
                continue
        # bash: don't drop `#`-leading lines (could be a real command).
        cmd = pending + line
        pending = ""
        if cmd.strip():
            cmds.append(cmd)
    return cmds[-n:]
```

Notes:
- `#` filtering removed — bash history doesn't use `#` for metadata; zsh
  doesn't either (extended-history uses `: <ts>:<dur>;`).
- Trailing-`\` continuation is now zsh-only, since that's where the
  extended-history record convention lives.

---

## 16. `${PS1@P}` requires bash 4.4+

**Where:** `kenoma.py`, `capture_prompt()`.

**What:** macOS ships `/bin/bash` 3.2; users with bash as `$SHELL` silently
fall through to the constructed-prompt fallback, losing their real PS1
structure.

**How:** Detect bash version; on <4.4, skip the `${PS1@P}` attempt and
emit a warning:

```python
def _bash_supports_param_transform(shell: str) -> bool:
    """Return True if `shell` is bash >= 4.4 (needed for ${PS1@P})."""
    try:
        out = subprocess.run(
            [shell, "-c", "echo $BASH_VERSINFO"],
            capture_output=True, text=True, timeout=2,
        )
        major = int(out.stdout.strip().split()[0])
        # BASH_VERSINFO[0]=major; we need >= 4 for @-transforms, >= 4.4 for @P.
        # Cheap to be slightly conservative: require >= 5 to skip the 4.0–4.3 grey zone.
        return major >= 5 or (
            major == 4 and len(out.stdout.split()) >= 2
            and int(out.stdout.split()[1]) >= 4
        )
    except Exception:
        return False
```

…then in `capture_prompt()`:

```python
if "zsh" in shell:
    cmd = f'print -rn -- "{begin}${{(%)PS1}}{end}"'
elif _bash_supports_param_transform(shell):
    cmd = f'printf "%s" "{begin}${{PS1@P}}{end}"'
else:
    print(
        f"[kenoma: {shell} doesn't support PS1 expansion (needs bash >= 4.4); "
        "using constructed fallback. Pass --prompt to override.]",
        file=sys.stderr,
    )
    return _constructed_fallback_prompt()
```

…where `_constructed_fallback_prompt()` is the existing
`f"{os.getenv('USER', 'user')}@…"` formatting, factored out.

---

## 18. Quick wins in the model load path

**Where:** `kenoma.py`, `load_model()` and the `run_gen()` thread.

**What:** Inference-mode wrap, gradient disable, daemon thread.

**How:**

- In `load_model()`, after model construction:
  ```python
  model.requires_grad_(False)
  ```

- In the REPL loop, wrap `model.generate(**gen_kwargs)`:
  ```python
  def run_gen():
      try:
          with torch.inference_mode():
              gen_result["out"] = cast(Any, model).generate(**gen_kwargs)
      except Exception as e:
          gen_result["err"] = e
  ```

- Mark the thread daemon:
  ```python
  t = threading.Thread(target=run_gen, daemon=True, name="kenoma-gen")
  ```

---

## 19. `--version` flag

**Where:** `kenoma.py`, the `argparse` block.

**How:** One-liner:

```python
ap.add_argument("--version", action="version",
                version=f"%(prog)s {__version__}")
```

Place it right after `ap = argparse.ArgumentParser(...)` so it's the first
flag listed in `--help`.

---

## 20. `tomli` fallback path is functionally dead

**Where:** `kenoma.py`, top-of-module `tomllib` import.

**What:** `pyproject.toml` already pins `tomli>=1.1.0; python_version <
'3.11'` as a hard dep. The `tomllib = None` branch only fires for someone
running `python kenoma.py` from a checkout on 3.9/3.10 without the dep.

**How:** Keep the fallback (a checkout-developer footgun is a real use
case — `pip install -e .` solves it but we should be helpful) but warn
loudly *only when a config file exists and would have been read*:

```python
def load_config_file() -> dict[str, Any]:
    p = config_path()
    if not p.is_file():
        return {}
    if tomllib is None:
        print(
            f"[kenoma: {p} exists but no TOML reader available — install tomli "
            "or run on Python 3.11+]",
            file=sys.stderr,
        )
        return {}
    ...
```

This way the silent-no-op only happens when there's nothing to read
anyway.

---

## 21. Sanity-cap adopted prompts

**Where:** Folded into #6 — `adoptable()` enforces
`len(prompt) <= ADOPTED_PROMPT_MAX` (512) and rejects control bytes.

No additional work beyond #6.

---

## Bonus A: KV cache warmup for the seed

**Where:** `kenoma.py`, end of `main()` setup, just before the REPL loop.

**What:** First user turn pays the prefill cost for the entire seeded
buffer (history or tmux scrollback). Warm the cache once at startup so
that cost shifts to launch time, where it's user-acceptable.

**How:** Run a single forward pass through the seeded buffer, capture the
returned KV cache, drop the generated token:

```python
def warm_kv_cache(model, tok, buf: str, device: str):
    """Pre-populate cached_ids / cached_kv by running one no-op generation
    pass through the seeded buffer. Returns (cached_ids, cached_kv)."""
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
            do_sample=False,  # deterministic; we discard the token anyway
            pad_token_id=tok.pad_token_id,
            use_cache=True,
            return_dict_in_generate=True,
        )
    new_kv = getattr(out, "past_key_values", None)
    if new_kv is None:
        return None, None
    # Truncate cache back to seed-only length (drop the one generated token).
    return seed_ids, truncate_cache(new_kv, len(seed_ids))
```

In `main()`, *after* writing the live prompt to stdout but *before* the
REPL loop:

```python
cached_ids, cached_kv = _reset_cache()
if not args.no_kv_cache:
    try:
        ids, kv = warm_kv_cache(model, tok, buf, device)
        # Both must be valid for the warmup to be useful — otherwise we'd
        # ship a cached_ids without a coherent cache, breaking the next turn.
        if ids is not None and kv is not None:
            cached_ids, cached_kv = ids, kv
            print("[kenoma: kv cache warmed]", file=sys.stderr)
        else:
            print("[kenoma: kv cache warmup produced no usable state; cold first turn]",
                  file=sys.stderr)
    except Exception as e:
        print(f"[kenoma: kv cache warmup failed ({e!r}); cold first turn]",
              file=sys.stderr)
        cached_ids, cached_kv = _reset_cache()
```

**Cache-coherence guard (Codex review):** the explicit "both ids and kv
must be non-None" check prevents the partial-state failure mode shared
with #9 and #11 — a cache that says "I cover N tokens" alongside a
truncated/None tensor is worse than no cache at all because the next
turn computes `lcp_len` against `cached_ids`, gets `lcp > 0`, and feeds
a delta into nothing. Same `_reset_cache()` helper used everywhere.

Notes:
- `do_sample=False` because the generated token is discarded; no point
  rolling RNG.
- We use `--max-new-tokens 1` (not 0) because some transformers versions
  reject 0. The truncate-back drops it.
- Wraps in try/except so a model that rejects warmup (rare cache class,
  device-map weirdness) still launches.
- Skipped entirely under `--no-kv-cache` for orthogonality.
- The warmup runs on the same `inference_mode` path as turn generation,
  so it benefits from #18.

User-visible: launch is slower by roughly one prefill of the seed
(depends on seed size and device); first turn is faster by the same
amount. On a 0.5B model with 6KB of seed on M-series MPS, this is
~hundreds of ms shifted from "first prompt feels laggy" to "loading time".

---

## Bonus B: Buffer trim on turn boundaries

**Where:** `kenoma.py`, REPL loop. Currently `buf` is trimmed right after
the user's input is appended (lines 486–487).

**What:** Trimming after `line` lands but before the model runs means the
model can see a partially-trimmed turn at its leading edge. Trimming after
the *full* turn (`body + prompt_tmpl`) keeps turn integrity, and the
line-boundary fix from #12 prevents partial-line leaks.

**How:** Remove the in-loop trim after `buf += line + "\n"`. Add the trim
at the bottom of the loop after `buf += body + prompt_tmpl`, using
`trim_buf` from #12:

```python
# OLD:
buf += line + "\n"
if len(buf) > args.context_chars:
    buf = buf[-args.context_chars:]
# NEW:
buf += line + "\n"
# (no trim here)
...
buf += body + prompt_tmpl
buf = trim_buf(buf, args.context_chars)
```

Edge case: a single turn whose `body` blows past `context_chars` will
still get trimmed mid-line by the `if nl == -1` branch in `trim_buf`. We
keep the whole tail in that case rather than corrupting the buffer
further.

Interaction with KV cache: the next turn's `lcp_len(cached_ids,
target_ids)` will produce a smaller LCP after a trim (cached prefix is
longer than the current `buf`'s prefix). That's already handled — `lcp`
just becomes shorter, more tokens get fed.

---

## Suggested smoke tests

For Group 3 in particular, before merge:

1. **Streamer-timeout-with-bogus-cache.** Construct a `cached_kv` of the
   wrong sequence length, run one REPL turn, confirm it doesn't hang —
   the truncate fails, falls through to full prefill, and the turn
   completes normally. This catches regressions in the #9/#11/Bonus A
   triangle without needing a GPU.
2. **Ctrl-C cancel mid-prefill.** Use a model large enough that prefill
   takes ≥2s. Send Ctrl-C *during* prefill (before any token streams).
   Confirm process recovers, prompt redraws, next turn works.
3. **Multi-line PS1 reject.** Set `--prompt $'a\nb $ '`, confirm warn +
   fall through to constructed prompt, no skeleton corruption.
4. **`build_skeleton` no-substitution warn.** Run with `--prompt 'sh$ '`
   in `/tmp`, confirm the warning fires.
5. **Adoption gating.** Pipe a prompt-shaped output line into the model's
   context, observe stop fires but `prompt_tmpl` is *not* mutated when
   the matched cwd contains `;` or control bytes.

These can live as a `scripts/smoke.sh` checklist (manual) until we have
a real test harness; running them is part of the Group 3 PR description.

## Implementation order

Group 1 — release-blocking (one PR):
- #1 (release.yml fix)
- #2 (CI version job replacement)
- #19 (`--version` flag, since it's adjacent)

Group 2 — correctness (one PR):
- #5 (windowed decoding)
- #3 (dynamic holdback budget)
- #4 (multi-line PS1 reject)
- #6 + #7 + #21 (adoption gating + sanitization)
- #8 (warn on no substitution)

Group 3 — robustness (one PR):
- #9 (KV cache try/except)
- #10 (capture tmux early)
- #11 (Ctrl-C cancel)
- #12 (line-boundary trim) + Bonus B (turn-boundary trim — they share
  `trim_buf`)

Group 4 — polish (one PR):
- #13 (config validation)
- #14 (DEFAULTS dict) — also update CONTRIBUTING.md
- #15 (history parsing)
- #16 (bash 3.2 detection)
- #18 (inference_mode + requires_grad_ + daemon)
- #20 (tomli warn-when-needed)
- Bonus A (KV cache warmup)

Splitting this way means:
- Group 1 unblocks the next release.
- Group 2 fixes user-visible misbehavior; small enough to review carefully.
- Group 3 is the "harden" pass.
- Group 4 is the "polish + perf" pass.

If any group needs a manual smoke test, Qwen/Qwen2.5-0.5B on CPU is
sufficient. For #11 specifically, please test against a model large
enough that generation takes >2s so Ctrl-C can be timed mid-stream.

---

## Out of scope for this plan

- #17 (README typo) — fix in a doc-only PR.
- Speculative decoding via `assistant_model` — separate feature.
- Skeleton learning across adoptions — separate feature.
- Full multi-line PS1 support — explicitly rejected per #4 decision.
- Basename-fallback wildcarding for `\W` prompts — explicitly rejected
  per CLAUDE.md.
