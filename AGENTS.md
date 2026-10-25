# kenoma

A fake shell powered by raw LLM completion. No chat template, no system
prompt, no instruction following: a base model continues a terminal
transcript, and kenoma stops it when it writes the user's next prompt. Every
turn is one sample, kept in a tree the user can re-roll, rewind and branch
(loom-style), and the model's continuation past the prompt is shown as a
guess at the user's next command.

Name is from Gnostic cosmology: *kenoma* is the deficient, illusory world of
appearances, as opposed to *pleroma*, the fullness of reality. Fits a terminal
that only pretends to do anything, and many illusory worlds branching from it.

## Layout

- `kenoma/template.py`: the user's prompt as a `Template` (literals plus typed holes) and the streaming `Recognizer`. Pure; no model.
- `kenoma/session.py`: `Node` (one turn) and `Session` (the tree, the context window, turns, ghost text). Talks to an engine through the `Engine` protocol only.
- `kenoma/engine.py`: `MLXEngine` (in-process, Apple silicon) and `RemoteEngine` (OpenAI-compatible `/v1/completions`).
- `kenoma/ui.py`: prompt_toolkit front end: the inline prompt, loom keys, redraws, the full-screen tree explorer.
- `kenoma/shell.py`: what is read from the real environment at startup (rendered PS1, tmux scrollback, history, git branch).
- `kenoma/cli.py`: config (CLI > `KENOMA_*` env > `config.toml` > `DEFAULTS`) and startup wiring.
- `tests/`: model-free tests. `FakeEngine` in `test_session.py` scripts token streams.

## Design

**Raw completion.** The model plays the whole terminal (prompts, commands,
outputs) as one text stream. Use base models; `-it`/chat models answer
instead. Even "base" checkpoints now contain instruction data, so the MLX
engine forbids sampling the tokenizer's added tokens (`<think>`,
`<|im_start|>`, tool and vision markers; EOS stays allowed): sampling one is
how a base model slips into assistant mode, and a terminal never prints
them.

**Prompt as template.** `shell.capture_prompt()` runs `$SHELL -i -c` once,
runs prompt hooks (zsh `precmd`/`precmd_functions`, bash `PROMPT_COMMAND`) so
themes that build the prompt there render, and expands PS1 (`${(%%)PS1}` in
zsh, `${PS1@P}` in bash ≥ 4.4) between sentinels. `Template.from_render`
splits the result into visible text and escapes. SGR colors are kept for
display only; OSC titles, readline `\x01`/`\x02` markers and other escapes
are dropped. Leading newlines are dropped. Holes are found by value, not by
parsing PS1 syntax: the cwd (absolute, `~`-relative, or a trailing component
suffix like `Work/kenoma`/`kenoma`; the rightmost bounded occurrence wins),
the git branch, a clock. Trailing spaces/tabs become `trail`, the place the
command starts. Multi-line prompts are fine: `"\n"` is just literal text.

**Hole matching.** Holes match printable characters; `time` only digits and
`:`. A hole with literal text before it on its line may contain spaces and
ends where the next literal fits (shortest first), so `a9lim@Mac kenoma src
%` after a fake `cd src` still matches. A hole that starts a line can't
contain spaces (it would swallow every output line), and a hole that ends the
prompt is greedy and space-free. `Template.anchored` is False for a prompt
with no literal text (`PS1='%~'`), which can't be recognized; the CLI falls
back to a default prompt then.

**Loose fallback.** `Template.loose` is `[first literal, any, last literal's
final word]`, built when it differs from the strict template. It catches
prompts whose optional parts come and go (a `(main)` segment vanishing after
`cd /tmp`). A loose match ends the turn, adopts no hole values, and the
node's `prompt` keeps the model's own prompt text for display.

**Recognizer.** Every line start is a candidate, *including the start of the
stream*: a silent command's output is just the next prompt. Text is held back
only while some candidate is still a viable prefix of the template (strict
or loose), so ordinary output streams token by token; only a line that
begins like the prompt waits. `Template.scan` explores every hole split and
returns the match that ends earliest; the recognizer takes the earliest end
across candidates and across strict/loose (strict wins ties). That is what
makes the result independent of chunking (tokens, SSE frames): once any
match completes, no path that still needs text can end before it. Keep that
invariant (there's a test feeding whole vs char-by-char). A match whose line
ends where the trailing space belongs (`a9lim@Mac kenoma %\n`) counts, with
`trailed=False` (the model predicted an empty command).

**Transcript as the model's tokens.** A node stores `text`, its token ids
(local engine), and `leftover`: the text the node owes to the start of the
next segment. After a match, the node keeps the longest prefix of the model's
own sampled tokens whose text lies within the prompt (`Session._covered`,
verified by decoding); the rest of the prompt, often just the trailing space
fused into the token `" ls"`, goes into `leftover` and is tokenized together
with the next command. So the model's output is never re-encoded, and the
`prompt ␣ + command` join tokenizes the way it would in natural text. When
kenoma has to supply the prompt (EOS, `max_new_tokens`, or ^C), it goes into
`leftover` too (`forced=True`); it repeats the parent's prompt, including a
loose one (`node.prompt` is inherited). An interrupted turn keeps exactly
what was shown, then `^C`. A ^C that lands after the prompt was already
found is a normal finish. The remote path drops the prompt's trailing space
from `text` into `leftover` the same way.

**Context window.** `Session.window(node, extra)` is a pure function of the
path, so a re-roll always conditions on the same history. The limit is
`context_tokens` minus `max_new_tokens` (at least half the budget). Costs are
tokens (chars/3 remotely). Past the limit, the window starts at the first
node beyond a cut that advances in steps of half the limit: consecutive turns
keep the same start (and the engine's cached prefix), and a full re-prefill
happens once per half-limit of new text. The last node is always kept even
if it alone is over the limit (one turn is bounded by `max_new_tokens` plus
the command). The seed is trimmed to its newest lines within half the limit.

## Engines

`Engine.stream(segments, max_tokens)` yields `(text, token_id | None)`. The
last segment is where generation starts. Engines keep caches across calls,
so any call sharing a prefix with the previous one (next turn, re-roll,
another branch) pays only for the difference.

**MLXEngine.** Owns the decode loop (mlx-lm's `generate_step` with
`max_tokens=-1`; it feeds each token into the cache *before* yielding it, so
after a yield the cache has consumed it). `self.tokens` is exactly what the
cache has consumed. `_sync` rewinds to the longest common prefix with the new
target and prefills the rest, always leaving one token to step on (reserved
independently of segment boundaries, so an empty last segment works).
Attention caches rewind by `trim`. Recurrent layers (Qwen3.5's gated-delta
`ArraysCache`) can't, so they are snapshotted at turn boundaries (end of the
context, at most 16) and at the latest line start while decoding; MLX arrays
are values, so a snapshot is a list of references. A rewind restores the
nearest snapshot at or below the target and prefills forward. If the cache
can't be restored (e.g. a wrapped sliding-window `RotatingKVCache`), it
resets and prefills from scratch: correct, slower. `_recover` puts every
layer back on a known position after an interrupted model call, and resets
rather than restore anything that would grow a cache; `_reset` empties the
bookkeeping before swapping the cache so an interrupt there errs safe. Verified:
after arbitrary rewinds, next-token logits match a fresh prefill to float
noise: in float32, Qwen3.5 max |Δ| ≈ 0.011 vs 0.012 for merely splitting the
prefill (a deliberately broken restore gives 3.3); Gemma 4 E4B 0.023 vs
0.037; Qwen2.5 0.014 vs 0.025.

Weights are quantized at load (`--bits`, default 4; mlx-lm's
`quantize_model`, under a second) unless the checkpoint is already quantized:
decode is memory-bound, so 4-bit is ~3x faster. M5 Max, 2k-token context:
Qwen3.5-0.8B 435 tok/s, 2B 283, 4B 142 (bf16: 4B 48). `_cached` loads an
already-downloaded repo from its local snapshot so startup prints no download
bars.

`_Detok` decodes each token together with an anchor token on its left
(SentencePiece strips a leading space at the start of a decode; mlx-lm's
streaming detokenizer also drops the first token's leading space and holds
lone spaces) and re-anchors every 32 tokens so decodes stay short.

**RemoteEngine.** POSTs the concatenated segment texts to
`<url>/v1/completions` with `stream: true` and reads SSE `data:` lines. Text
only (`tokenized=False`); the server's prefix cache (vLLM APC, SGLang radix,
llama-server `cache_prompt`) does the reuse. Closing the generator closes the
connection, which servers treat as an abort.

## Loom and ghost text

A turn is `Session.turn(parent, command)`, a generator that yields safe
output and returns the new child. Re-roll is just another `turn` from the
same parent; `Node.last_child` remembers where Alt-↓ goes. `Session.visit`
sets `current` and the `last_child` chain.

Ghost text: right after a local turn, the same stream is still open just past
the prompt, so `Session.ghost` keeps reading it until a newline (≤ 48
tokens). Otherwise (a forced prompt, a remote engine, or a node revisited
before its guess finished) it starts a new `interactive` stream from the
prompt minus the trailing space, if the transcript left that space out, and
strips it from the result. `Session.ghost` yields after every piece, and the
UI drives it as a prompt_toolkit background task with `await
asyncio.sleep(0)` after each: single-threaded, keystrokes at most a token
behind (plus the initial rewind of a restarted guess), and the engine is
never used from two places. prompt_toolkit cancels and awaits the task when
the prompt exits, before the next turn touches the engine. A remote
interactive stream yields on every SSE line, keepalives included, gives up
on a read after 1 s of silence, and ends 5 s after it began, so a slow or
chatty server can delay keystrokes by at most one read.

## UI

The prompt is drawn by prompt_toolkit with `ANSI(display_prompt(node))`.
Model output is written straight to stdout through `sanitize` (drops C0/C1
controls, ESC and CR; keeps tab and newline). Navigation clears the screen
and redraws the tail of the current path (`_tail` counts wrapped rows). Loom
keys exit the prompt with a `Nav` result carrying the typed text, which is
restored. ^C during a turn arrives either inside `next(gen)` (the turn
catches it) or between pieces (the UI throws it into the generator); both
end in an interrupted node. If the turn already died of it, there's no node
and the UI just moves on. ^C at the prompt clears the line; ^D exits.

The tree explorer (`tree_view`) is a full-screen prompt_toolkit app. Chains
of single children render at one indentation; only branch points indent.

## Seeds

Inside tmux, the pane's last `--tmux-lines` lines (captured before kenoma
prints anything; the invocation row is dropped). Otherwise
`shell.history_seed`: a macOS-style `Last login:` line, the last `--history`
commands as the *output of* `history | tail -N`, then `pwd` and the real cwd.
Bare commands back to back (v1) taught the model that commands print
nothing. The `pwd` both grounds the cwd and keeps the indented history lines
from being the last thing the model copies.

## Testing

- `pytest`: templates, the recognizer (including chunking invariance), turns,
  leftover, ghost, interrupts, the window, loose prompts, the remote path.
- Engine changes: check rewinds against a fresh prefill on a recurrent model
  (Qwen3.5) and an attention-only one; compare the max logit difference to
  the split-prefill baseline.
- UI changes: drive a real session in a private tmux server (`tmux -L
  <name> new-session -d`, `send-keys`, `capture-pane -p`).

## Limitations

- Prompt segments that appear and disappear (an exit-status marker after a
  failure) aren't holes; the loose fallback catches some of them.
- Sliding-window models (Gemma) re-prefill from scratch once their window
  wraps, since a rotating cache can't be rewound.
- The model sees the seed and the current path only. Nothing persists between
  runs.

## Non-goals

- **Executing anything.** Nothing the user types touches the real system.
  Startup reads (PS1, history, tmux pane, git branch) are the only contact.
- **Chat semantics.** No user/assistant roles; one text stream.
- **Chat models.** They load, but they answer instead of playing a terminal.
