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
(`${(%)PS1}` in zsh, `${PS1@P}` in bash). Interactive startup can print to
stdout before our command runs (`.zshrc`, `Restored session: ...` banners,
plugins), so the real prompt is wrapped in `\x1e<<PS1_BEGIN>>\x1e` /
`\x1e<<PS1_END>>\x1e` sentinels and extracted from the blob. Captured once
at startup. The captured value seeds `prompt_tmpl`, but `prompt_tmpl` itself
is *not* frozen — see "Stop detection" for adoption semantics. `--prompt`
overrides.

**Seed transcript (the model sees real history).** The buffer is seeded
before the REPL starts so the model has grounding:

1. **`--tmux-lines N`** (default 300). If `$TMUX` is set, run
   `tmux capture-pane -p -J -S -N` to grab the current pane's scrollback —
   real commands and real outputs. We drop the last 3 rows of the capture
   (the `python kenoma.py` line, the `[loading ...]` stderr line, the
   live prompt) to avoid feeding the launch noise back to the model.
2. **`--history N`** (default 20). Fallback if tmux capture is unavailable
   or empty. Reads `$HISTFILE` or `~/.zsh_history` / `~/.bash_history`,
   handles zsh extended-history format (`: <ts>:<dur>;cmd`), folds
   `\`-continued multi-line entries, and seeds commands back-to-back with
   our captured prompt between them (no outputs — commands look like they
   produced nothing).

The seeded transcript is *not* echoed to the user's real terminal; only the
live prompt is displayed. The model sees it all in context.

**Rolling buffer.** One string, `buf`, holds the whole fake-session
transcript: `<seed> → PS1 → user line → model output → PS1 → ...`. Trimmed
from the left to `--context-chars` (default 6000) to cap memory. Each turn
re-tokenizes the whole buffer — simple, no KV-cache plumbing. Fine for small
models; revisit if latency matters. Note: `--max-new-tokens` (default 2048)
is per-turn; neither cap respects the model's actual context window.

## Stop detection

Two things can halt generation:

1. **`eos_token_id`** — `generate()` honors it by default. Always active.
2. **Skeleton match** — `\n` + literal-prefix + `[^\s]+` (cwd wildcard) +
   literal-suffix, derived from the captured prompt by `build_skeleton()`.
   See "Skeleton derivation" below.

These live in `StopOnPromptLike`, a custom `StoppingCriteria` subclass.
Transformers ≥4.38 has a native `stop_strings=` kwarg backed by
`StopStringCriteria`, but we roll our own because the matched span has to
flow back to the REPL (we *adopt* what the model emitted as the new prompt).

**Skeleton derivation.** `build_skeleton(prompt)` does
`re.escape(prompt)`, then substitutes the first occurrence of `os.getcwd()`
or its `~`-relative form with `[^\s]+`, then anchors at `\n`. The result
matches the captured prompt verbatim *and* any cwd-drifted variant of it
(e.g., after the model's fake `cd /tmp`, a continuation like
`\na9lim@host:/tmp % ` matches), but rejects unrelated prompt-shaped lines
like `\n50 % done` (the literal prefix doesn't match) or
`\nroot@host /tmp #` (different user/host).

**Adoption.** When the skeleton matches, we slice the matched substring out
of the model's output and *make that the new `prompt_tmpl`* — for display,
for the next turn's `safe_flush_point` heuristic, and for the next turn's
buffer glue. The skeleton itself stays static (it encodes the user's prompt
*structure*, not its current cwd value). Net effect: the fake terminal's
prompt cwd actually tracks the model's fake `cd`s, instead of forever
showing the cwd kenoma was launched in.

**Limitations of the skeleton.**

- Only the cwd portion is wildcarded. Prompts that show only the basename
  (bash `\W`-style) won't have anything substituted, so the skeleton
  becomes exact-match-only and adoption only fires on perfect reproduction
  of the launch prompt. Same for prompts with a git branch / exit-status
  segment that drifts independently of cwd.
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

**Mechanics.** `generate()` calls the criterion after each new token. We
slice off the seed (`input_ids[0][self.prompt_len:]`), decode the generated
tail to text, run `skeleton.search(text)`. Decoded-text matching (rather
than token-level) is necessary because identical characters tokenize
differently depending on neighbors — "`$ `" can merge with a preceding
newline into one token. The tail is short, so re-decoding every step is
cheap. After stop, `match_span(produced)` returns
`(body_end, prompt_end)`: `produced[:body_end]` is the body (keeps the
leading `\n`), `produced[body_end:prompt_end]` becomes the new
`prompt_tmpl`, anything past `prompt_end` is dropped.

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
  and ≤`HOLDBACK_LINE_BUDGET` (200). If so, we hold back from that `\n`;
  otherwise no regex holdback is needed.
- **Exact holdback.** Longest suffix of `produced` that's a prefix of
  `current_prompt.rstrip()` — those chars might complete into a verbatim
  reproduction of the current prompt. Subsumed by the regex holdback in
  most realistic cases, but kept as cheap insurance. The `current_prompt`
  parameter is *the prompt as of this turn*, not the original captured
  one — adoption can change it.

Net effect: output streams line-by-line instead of in fixed-size chunks.
Each line flushes the instant its terminating `\n` arrives and the next
line starts; the in-progress line is held back until it either completes
(with `\n`), grows past 200 chars, or hits a stop.

**After the stop fires**, we flush `produced[shown:body_end]`, break out,
extract the matched prompt as the new `prompt_tmpl`, and write *that* to
the screen. The display follows the model — `cd /tmp` actually changes the
displayed prompt's cwd next turn.

## Controls

- **Enter** — submit command, model generates fake output.
- **Up/Down arrows** — cycle through command history (readline).
- **Left/Right arrows** — move cursor within the current line.
- **Ctrl-C** at the input prompt — exit.
- **Ctrl-D / EOF** — exit.

Ctrl-C during generation isn't specially handled; it'll propagate out of the
streamer loop and most likely kill the process. If interrupt-to-cancel-turn
becomes worth it, wrap the streamer loop and signal the generation thread.

## Running

```
pip install .
kenoma google/gemma-3n-E4B
```

Or `pip install -e .` for editable install during development. The model is a
positional argument (defaults to `Qwen/Qwen2.5-0.5B` when omitted).

Flags worth knowing: `--prompt` (override captured PS1), `--temperature`
(default 1.0), `--top-p` (default 0.95), `--repetition-penalty` (default
1.05), `--max-new-tokens` (default 2048), `--context-chars`, `--history`,
`--tmux-lines`, `--device`.

## Non-goals

- **Actually executing anything.** This is pure hallucination; nothing
  touches the real filesystem. Don't add real `subprocess` execution — the
  point is the model *is* the terminal.
- **Multi-turn "conversation" semantics.** No user/assistant distinction.
  Everything is one text stream.
- **Chat-model support.** `-it` variants technically load but behave poorly;
  not worth special-casing.
