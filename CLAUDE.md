# kenoma

A fake shell powered by raw LLM completion. No chat template, no system prompt,
no instruction following. The real shell's `PS1` is captured once at startup
and used as the stop string; the model hallucinates command output until it
emits the next prompt, at which point we stop and hand control back to the
user.

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
at startup, frozen thereafter. `--prompt` overrides.

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

Three things can halt generation:

1. **`eos_token_id`** — `generate()` honors it by default. Always active.
2. **Exact match on the captured prompt** — substring check against
   `stop_str` (`prompt_tmpl.rstrip()`).
3. **Prompt-shaped regex** — `\n[^\n]{1,200}[\$#%>] `. A newline, at least
   one non-newline char, then a shell prompt terminator (`$` / `#` / `%` /
   `>`) and a space.

(2) and (3) live in `StopOnPromptLike`, a custom `StoppingCriteria` subclass.
Transformers ≥4.38 has a native `stop_strings=` kwarg backed by
`StopStringCriteria`, but we roll our own so both paths live in one explicit
place and behavior is stable across versions.

**Why not just exact-match the captured prompt?** The user's prompt contains
variable parts (cwd, git branch, `$?`). After `cd foo`, a realistic model
continuation is `user@host foo %` — same shape, different cwd, no exact
match. Generation would run until EOS or `--max-new-tokens`, producing a wall
of runaway fake commands. The regex catches all prompt-shaped lines.

**Why no `$` end-of-string anchor in the regex?** Earlier versions had one,
but tokenization foiled it: the model would emit `work % ls` as one chunk,
and we'd never see a state where the decoded tail ended exactly at `% `. The
trailing-prompt match eventually fired further down, *after* the runaway
commands had already been generated. Dropping the end anchor makes the regex
fire the instant a new prompt-shaped line appears, even mid-stream, so
`\na9lim@...work % ` matches regardless of what follows.

**Trade-off.** The loose regex false-positives on legitimate output
containing `<word> % ` (e.g. `50 % done`). The `{1,200}` minimum rules out
bare `\n% `; most real output doesn't hit this. Tighten if it bites.

**Mechanics.** `generate()` calls the criterion after each new token. We
slice off the seed (`input_ids[0][self.prompt_len:]`), decode the generated
tail to text, check `exact in text`, then run the regex. Decoded-text
matching (rather than token-level) is necessary because identical characters
tokenize differently depending on neighbors — "`$ `" can merge with a
preceding newline into one token. The tail is short, so re-decoding every
step is cheap.

## Streaming output

Generation runs on a background thread; the main thread iterates a
`TextIteratorStreamer` and flushes chunks to stdout. `skip_prompt=True`
suppresses the input echo.

**Smart holdback.** We can't flush everything immediately — the held-back
tail might still turn into a prompt match on the next token, and we don't
want a partial prompt to leak to the screen before the stop fires and
retracts it. But a *constant* holdback (we tried `max(len(stop_str), 220)`)
makes output arrive in 220-char bursts, which feels laggy.

`safe_flush_point(produced, stop_str)` computes the earliest of two
positions:

- **Regex holdback.** A future regex match can only be anchored at an
  existing `\n`, extending forward. Only the *latest* `\n` in `produced` can
  still anchor one, and only if the chars since it are all non-newline and
  ≤200 (the regex's line budget). If so, we hold back from that `\n`;
  otherwise no regex holdback is needed.
- **Exact holdback.** Longest suffix of `produced` that's a prefix of
  `stop_str` — those chars might complete into the captured prompt, so we
  hold them back.

Net effect: output streams line-by-line instead of in fixed-size chunks.
Each line flushes the instant its terminating `\n` arrives and the next line
starts; the in-progress line is held back until it either completes (with
`\n`), grows past 200 chars (the regex can't anchor here anymore), or hits
a stop.

**After the stop fires**, we flush `produced[shown:idx]`, break out, and
write a fresh canonical `prompt_tmpl` — the display stays anchored on our
captured prompt even if the model generated a slightly different one.

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
