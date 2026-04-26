# kenoma

A fake shell powered by raw LLM completion. No chat template, no system prompt,
no instruction following. The real shell's `PS1` is captured once at startup
and turned into a structural skeleton; the model hallucinates command output
until it emits a line matching that skeleton, at which point we stop, adopt
its prompt as the new canonical, and hand control back to the user.

Name is from Gnostic cosmology — *kenoma* is the deficient, illusory world of
appearances, as opposed to *pleroma*, the fullness of reality. Fits a terminal
that only pretends to do anything.

## Layout

- `kenoma.py` — single-module entry point. CLI, prompt capture, history/tmux seeding, readline input, REPL, streaming generation, stop logic.
- `pyproject.toml` — packaging. Declares `kenoma` console script (`kenoma = kenoma:main`) and deps: `torch`, `transformers>=4.44`, `accelerate`, `sentencepiece`.

## Design

**Raw completion, not chat.** We call `AutoModelForCausalLM.generate()` on a
plain string buffer. No `apply_chat_template`. The model is playing the role
of an entire terminal (prompt + user + output), not an assistant responding to
a turn. Chat-tuned models break character; prefer base/completion models
(`google/gemma-3n-E4B` base, `Qwen/Qwen2.5-0.5B` base, `Llama-3.2-1B` base,
`gpt2`, `pythia-*`). `-it` variants will ignore the framing and try to be
helpful.

**Prompt capture.** `capture_prompt()` spawns `$SHELL -i` and expands `PS1`
(`${(%)PS1}` in zsh, `${PS1@P}` in bash 4.4+). Interactive startup can print
to stdout before our command runs (`.zshrc`, `Restored session: ...` banners,
plugins), so the real prompt is wrapped in `\x1e<<PS1_BEGIN>>\x1e` /
`\x1e<<PS1_END>>\x1e` sentinels and extracted from the blob. Captured once
at startup. The captured value seeds `prompt_tmpl`, but `prompt_tmpl` itself
is *not* frozen — see "Stop detection" for adoption semantics. `--prompt`
overrides.

`${PS1@P}` is bash 4.4+. `_bash_supports_param_transform()` probes
`BASH_VERSINFO`; on bash <4.4 (notably macOS's stock `/bin/bash` 3.2) we
warn and fall through to `constructed_fallback_prompt()` (a default
`user@host:cwd $ ` shape). Same fallback fires if the captured PS1
contains an embedded newline — `reject_multiline_prompt()` warns and
swaps it out before `build_skeleton` ever sees the multi-line value.
Multi-line PS1 would otherwise corrupt the skeleton (re.escape happily
escapes `\n`) and defeat the streaming holdback.

**Seed transcript (the model sees real history).** The buffer is seeded
before the REPL starts so the model has grounding:

1. **`--tmux-lines N`** (default 300). If `$TMUX` is set, run
   `tmux capture-pane -p -J -S -N` to grab the current pane's scrollback —
   real commands and real outputs. **Captured at the very top of `main()`,
   before any of our own stderr writes** (`[loading ...]`,
   `[kenoma: device=...]`, `[kenoma: kv cache warmed]`); we then only need
   to drop the last row of the capture (the `kenoma <model>` invocation
   line itself). The previous "drop last 3 rows" heuristic was brittle:
   one extra stderr message or a wrapped HF download progress bar broke
   the trim count.
2. **`--history N`** (default 20). Fallback if tmux capture is unavailable
   or empty. Reads `$HISTFILE` or `~/.zsh_history` / `~/.bash_history`,
   handles zsh extended-history format via the `^: \d+:\d+;` regex, folds
   trailing-`\` continuations *only on zsh* (bash newline-separates),
   accepts `#`-leading lines as real commands (bash history doesn't reserve
   `#` for metadata), and seeds commands back-to-back with our captured
   prompt between them (no outputs — commands look like they produced
   nothing).

The seeded transcript is *not* echoed to the user's real terminal; only the
live prompt is displayed. The model sees it all in context.

**Rolling buffer.** One string, `buf`, holds the whole fake-session
transcript: `<seed> → PS1 → user line → model output → PS1 → ...`. Trimmed
from the left to `--context-chars` (default 6000) to cap memory. Each turn
re-tokenizes the whole buffer — simple, no KV-cache plumbing. Fine for small
models; revisit if latency matters. Note: `--max-new-tokens` (default 2048)
is per-turn; neither cap respects the model's actual context window.

The trim is **turn-boundary + line-aligned**: `trim_buf()` runs after the
full `body + prompt_tmpl` lands on the buffer (not after the user's input,
which is the obvious-but-wrong place — the model would see a half-trimmed
turn at the leading edge), and the cut is advanced to the next `\n` so the
buffer never starts mid-line. A degenerate single turn whose body alone
exceeds `--context-chars` falls back to a hard char trim.

**Defaults.** All argparse default values live in a single `DEFAULTS` dict
next to `CONFIG_KEYS`. Argparse pulls from `defaults.get(k, DEFAULTS[k])`
so adding a flag means: update `CONFIG_KEYS` (key + type), `DEFAULTS`
(value), the argparse line, and the README. Env-var (`KENOMA_<KEY>`) and
config-file (`$XDG_CONFIG_HOME/kenoma/config.toml`) paths are automatic.
The config-file path typechecks values against `CONFIG_KEYS` and warns on
unknown keys, so typos surface instead of being silently dropped.

## Stop detection

Two things can halt generation:

1. **`eos_token_id`** — `generate()` honors it by default. Always active.
2. **Skeleton match** — `\n` + literal-prefix + `(?P<cwd>[^\s]+)` (cwd
   wildcard, named) + literal-suffix, derived from the captured prompt by
   `build_skeleton()`. See "Skeleton derivation" below.

A third stopper, `CancelOnFlag`, is installed alongside the skeleton
stopper. It's a flag-driven `StoppingCriteria` checked once per token; the
main thread sets the flag from a Ctrl-C handler and the gen thread exits
on the next token. See "Controls".

These live in `StopOnPromptLike`, a custom `StoppingCriteria` subclass.
Transformers ≥4.38 has a native `stop_strings=` kwarg backed by
`StopStringCriteria`, but we roll our own because the matched span has to
flow back to the REPL (we *adopt* what the model emitted as the new prompt).

**Skeleton derivation.** `build_skeleton(prompt)` does
`re.escape(prompt)`, then substitutes the first occurrence of `os.getcwd()`
or its `~`-relative form with `(?P<cwd>[^\s]+)`, then anchors at `\n`. The
result matches the captured prompt verbatim *and* any cwd-drifted variant
of it (e.g., after the model's fake `cd /tmp`, a continuation like
`\na9lim@host:/tmp % ` matches), but rejects unrelated prompt-shaped lines
like `\n50 % done` (the literal prefix doesn't match) or
`\nroot@host /tmp #` (different user/host). Returns
`(skeleton, substituted)`; the bool flags whether a cwd form was found in
the prompt at all (see Limitations).

**Adoption.** When the skeleton matches, we slice the matched substring
out of the model's output and *make that the new `prompt_tmpl`* — for
display, for the next turn's `safe_flush_point` heuristic, and for the
next turn's buffer glue. The skeleton itself stays static (it encodes the
user's prompt *structure*, not its current cwd value). Net effect: the
fake terminal's prompt cwd actually tracks the model's fake `cd`s, instead
of forever showing the cwd kenoma was launched in.

**Adoption gating.** Stopping and adopting are *decoupled*. The stopper
fires on any skeleton match (we need the budget cap), but `adoptable()`
gates whether `prompt_tmpl` actually mutates. The matched span must:

- Fit within `ADOPTED_PROMPT_MAX = 512` chars.
- Contain no embedded newlines or carriage returns.
- Contain no control bytes (ANSI escapes, etc.) except `\t`.
- Have a "path-shaped" cwd capture: `[A-Za-z0-9._/~\-]+` only, so a model
  emitting `\nuser@host:/tmp;rm -rf $ ` doesn't poison `prompt_tmpl` with
  shell metachars.

If `adoptable()` returns False, we still stop the turn (treating the body
through `body_end` as the model's actual output) but keep the previous
`prompt_tmpl`. `sanitize_prompt()` strips remaining control bytes as
defense in depth on the success path.

**Limitations of the skeleton.**

- Only the cwd portion is wildcarded. Prompts that show only the basename
  (bash `\W`-style) won't have anything substituted, so the skeleton
  becomes exact-match-only and adoption only fires on perfect reproduction
  of the launch prompt. Same for prompts with a git branch / exit-status
  segment that drifts independently of cwd. **We now warn at startup** when
  `build_skeleton`'s `substituted` flag is False ("prompt has no detectable
  cwd; stop will only fire on exact reproductions") so the user knows
  every fake `cd` will run to `--max-new-tokens` until they pass `--prompt`.
- A separate warning fires when the cwd contains whitespace: the wildcard
  is `\S+`, so `/Users/a9lim/My Documents` substitutes but never matches
  any drifted variant. Same effective failure mode, distinct cause.
- We deliberately *don't* substitute the cwd basename as a fallback. It's
  too footgun-prone: a prompt like `a9lim % ` with cwd `/Users/a9lim` would
  reduce to `[^\s]+ % `, which false-positives on `50 % done`.
- When nothing matches, generation runs to `--max-new-tokens`. There's no
  loose-regex safety net anymore. The user opted into "stop only on
  strict" — bound is `max_new_tokens`, worst case is one wasted turn.

**Why no `$` end-of-string anchor?** Tokenization would foil it. The model
emits `work % ls` as one chunk, so we'd never see a state where the decoded
tail ended exactly at `% `. The skeleton has no end anchor for the same
reason — it fires the instant a new prompt-shaped line appears mid-stream,
even if the model continues past it. Anything after the match in the same
chunk is discarded (see "Mechanics").

**Mechanics.** `generate()` calls the criterion after each new token. The
stopper decodes only a *trailing window* of tokens, not the full growing
tail — `WINDOW_TOKENS_FLOOR = 64`, plus a `WINDOW_TOKENS_MARGIN = 16`-token
slack past `len(prompt_tmpl) // 2` so a match anchored at the window's
left edge isn't lost to truncation. The full-tail decode used to be
O(n²) for long generations; the window cap collapses it to O(n). Decoded-
text matching (rather than token-level) is still necessary because
identical characters tokenize differently depending on neighbors — "`$ `"
can merge with a preceding newline into one token. After stop,
`match_span(produced)` returns `(body_end, prompt_end, cwd)`:
`produced[:body_end]` is the body (keeps the leading `\n`),
`produced[body_end:prompt_end]` is the candidate prompt fed into
`adoptable()`, `cwd` is the captured wildcard span, anything past
`prompt_end` is dropped.

## Streaming output

Generation runs on a background thread; the main thread iterates a
`TextIteratorStreamer` and flushes chunks to stdout. `skip_prompt=True`
suppresses the input echo.

**Smart holdback.** We can't flush everything immediately — the held-back
tail might still turn into a skeleton match on the next token, and we don't
want a partial prompt to leak to the screen before the stop fires and
retracts it. But a *constant* holdback makes output arrive in fixed-size
bursts, which feels laggy.

`safe_flush_point(produced, current_prompt)` computes the earliest of two
positions:

- **Regex holdback.** A future skeleton match can only be anchored at an
  existing `\n`, extending forward. Only the *latest* `\n` in `produced`
  can still anchor one, and only if the chars since it are all non-newline
  and within `line_budget_for(current_prompt)`. The budget is *dynamic* —
  `max(HOLDBACK_LINE_BUDGET_FLOOR=200, 2 * len(current_prompt))` — because
  the previous fixed 200-char cap could be smaller than a fancy powerline
  prompt, leaking partial prompt text before the skeleton match could
  fire.
- **Exact holdback.** Longest suffix of `produced` that's a prefix of
  `current_prompt.rstrip()` — those chars might complete into a verbatim
  reproduction of the current prompt. Subsumed by the regex holdback in
  most realistic cases, but kept as cheap insurance. The `current_prompt`
  parameter is *the prompt as of this turn*, not the original captured
  one — adoption can change it.

Net effect: output streams line-by-line instead of in fixed-size chunks.
Each line flushes the instant its terminating `\n` arrives and the next
line starts; the in-progress line is held back until it either completes
(with `\n`), grows past the dynamic budget, or hits a stop.

**After the stop fires**, we flush `produced[shown:body_end]`, break out,
and feed the matched span through the adoption gate (see "Adoption
gating" above). On accept, `prompt_tmpl` becomes the matched span (after
`sanitize_prompt()`); on reject, we keep the previous `prompt_tmpl` and
drop the model's match silently. Either way we redraw the canonical
prompt to the screen.

**Search cursor (cancel + reuse).** Two cooperating optimizations live in
the streamer loop:

- `skeleton.search(produced, search_anchor)` advances `search_anchor` past
  what we've already searched, with a `MAX_SKELETON_MATCH_LEN` backtrack
  so a match anchored in an earlier chunk but completed by the latest
  chunk isn't missed. Without this, each chunk re-scanned the whole
  growing `produced` (O(n²) over a turn).
- `last_nl_in_produced` is updated from each `chunk.rfind("\n")` and fed
  to `safe_flush_point` as a hint, so the holdback check doesn't rfind
  the full buffer per chunk.
- On a main-thread match we set `cancel.set()` before `break`, so the gen
  thread stops producing into the streamer's unbounded queue. The
  gen-side `StopOnPromptLike` *usually* fires on the same token, but the
  cancel is defense in depth.
- The match details (`body_end`, `prompt_end`, captured cwd) are remembered
  in loop locals and reused after the loop instead of re-running
  `match_span` on `produced`.

## Controls

- **Enter** — submit command, model generates fake output.
- **Up/Down arrows** — cycle through command history (readline).
- **Left/Right arrows** — move cursor within the current line.
- **Ctrl-C** during generation — cancels the turn, drops the in-progress
  output, invalidates the KV cache, redraws the prompt. Bounded by a 30s
  wall-clock deadline (`DRAIN_DEADLINE_SECONDS`) so a slow prefill on a
  large model can't block forever.
- **Ctrl-C** at the input prompt — exit.
- **Ctrl-D / EOF** — exit.

## Tokenization cache

`encode_with_cache(tok, buf, prev_buf, prev_ids)` skips re-tokenizing the
whole growing buffer when `buf` strictly extends `prev_buf` (the common
case: previous turn's body landed, then user typed a line). It encodes
only the suffix with `add_special_tokens=False` and concats. To guard
against BPE/SP merges that span the join point, it re-tokenizes a
`_INCREMENTAL_TOK_OVERLAP=64`-char back-overlap and verifies it tokenizes
to the cached tail — if not, falls back to a full `tok.encode(buf)`.
After buffer trim, `buf` no longer starts with `prev_buf`, the prefix
check fails, and the next turn full-encodes. `lcp_len` has a slice-equality
fast path that handles the very common "full prefix match" case at C
speed; the Python loop only runs on real divergence.

## KV cache

KV cache reuse is opt-in (`kv_cache = true`, default; disable with
`--no-kv-cache`). Mutually exclusive with `--compile` (the static cache
doesn't expose `crop()`; `--compile` forces `--no-kv-cache=True`). On
each turn we tokenize the new buffer, find the longest token-level
common prefix with the cached `cached_ids`, truncate `cached_kv` to that
prefix, and feed only the delta. After generation we truncate the
returned cache back to `len(target_ids)` — i.e. drop the generated-tokens
portion — because we replace the model's tail with our own
`body + prompt_tmpl` and any LCP past `len(target_ids)` would be a lie.

**Cache state is binary.** A partial-state cache (cached_ids saying "I
cover N tokens" alongside a None/wrong-length kv) breaks the next turn:
`lcp_len` returns N, we feed the delta, and the model regenerates
against a stale prefix. To prevent this from drifting across multiple
sites, *every* cache mutation routes through `_reset_cache()` which
returns `(None, None)`. Truncation failures (some sliding-window cache
classes can't `crop()`), generation errors, Ctrl-C cancel, and warmup
failure all use the same helper. The caller pairs the reset with
`lcp = 0` / `feed_ids = target_ids` for the *current* turn — a None
cache without a same-turn lcp reset is the broken state the helper
exists to prevent.

`truncate_cache()` is soft: any exception (including from cache classes
that don't support `crop`) prints a one-time stderr message and returns
None. The caller's "if `lcp > 0` and `kv is None`: full prefill" branch
recovers gracefully; subsequent turns simply fall through the
"`cached_ids is None`" branch and don't try to reuse cache at all.

**Warmup.** At startup, after the seed buffer is built and before the
first `input()`, `warm_kv_cache()` runs one forward pass through the
seeded buffer with `max_new_tokens=1` and discards the generated token.
The returned cache is truncated back to seed-only length; both
`cached_ids` and `cached_kv` are required to be non-None before adoption
(the partial-state guard). On any failure the cache stays empty and the
first user turn pays the seed's prefill cost — the original behavior.
Trade-off: launch is slower by one prefill of the seed; first turn is
faster by the same amount.

## Backend selection

`load_model()` resolves device, picks a dtype, and asks for the best
attention implementation we can use:

- CUDA without bnb quantization: try `flash_attention_2` (requires the
  `flash_attn` package); on `from_pretrained` failure, fall back to
  `sdpa` and reload.
- Anywhere else (CUDA-with-bnb, MPS, CPU): `sdpa`. PyTorch's SDPA
  dispatches to flash / memory-efficient kernels under the hood when
  available.

`low_cpu_mem_usage=True` is always set; weights stream into final-place
tensors instead of an FP32 copy first.

`--compile` (default off) wires up `torch.compile(model.forward,
mode="reduce-overhead")` and flips the generation config to
`cache_implementation="static"`. This trades cross-turn KV cache reuse
(the static cache can't `crop()`) for ~2-3x faster decode on CUDA. First
turn pays the compile + CUDA-graph capture cost. On MPS/CPU the gain is
smaller and sometimes negative; we warn but still try.

## Per-turn allocations

The all-ones `attention_mask` tensor is preallocated once, sized
geometrically (with `max_new_tokens` headroom), and sliced on each turn
— no per-turn `torch.ones()` allocation. `repetition_penalty` is dropped
from `gen_kwargs` when the user sets it to exactly 1.0 (HF would
otherwise still construct a no-op processor and run it every token).

## Running

```
pip install .
kenoma google/gemma-3n-E4B
```

Or `pip install -e .` for editable install during development. The model is a
positional argument (defaults to `Qwen/Qwen2.5-0.5B` when omitted).

Flags worth knowing: `--prompt` (override captured PS1), `--temperature`
(default 1.0), `--top-p` (default 0.95), `--repetition-penalty` (default
1.05; set to 1.0 to skip the processor entirely), `--max-new-tokens`
(default 2048), `--context-chars`, `--history`, `--tmux-lines`,
`--device`, `--quantize {none,4bit,8bit}`, `--no-kv-cache`, `--compile`
(static KV cache + torch.compile, mutually exclusive with cross-turn KV
cache reuse), `--version`.

## Non-goals

- **Actually executing anything.** This is pure hallucination; nothing
  touches the real filesystem. Don't add real `subprocess` execution — the
  point is the model *is* the terminal.
- **Multi-turn "conversation" semantics.** No user/assistant distinction.
  Everything is one text stream.
- **Chat-model support.** `-it` variants technically load but behave poorly;
  not worth special-casing.
