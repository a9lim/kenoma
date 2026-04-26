# kenoma

[![CI](https://github.com/a9lim/kenoma/actions/workflows/ci.yml/badge.svg)](https://github.com/a9lim/kenoma/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-%3E%3D3.9-blue)](https://www.python.org/)
[![License: AGPL v3](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue)](https://www.gnu.org/licenses/agpl-3.0)

A fake shell powered by raw LLM completion. No chat template, no system prompt. The real shell's `PS1` is captured once at startup and turned into a structural skeleton; the model hallucinates command output until it emits a line matching that skeleton, at which point generation stops and that emitted prompt becomes the new canonical for the next turn.

The name is from Gnostic cosmology: *kenoma* is the deficient, illusory world of appearances, as opposed to *pleroma*, the fullness of reality. It fits a terminal that only pretends to do anything.

## Install

```
pip install kenoma
```

Or from source:

```
git clone https://github.com/a9lim/kenoma
cd kenoma
pip install -e .
```

For bitsandbytes quantization (CUDA only):

```
pip install kenoma[quantize]
```

## Usage

```
kenoma                                # defaults to Qwen/Qwen2.5-0.5B
kenoma google/gemma-3n-E4B
kenoma /path/to/local/model
```

The model argument is any HuggingFace model id or a local path. Please use a base or completion model, not a chat-tuned one. Instruction-tuned variants technically load but break character and try to be helpful, which defeats the whole point.

Type a command and hit Enter; the model generates fake output until it emits something prompt-shaped, then gives you a fresh prompt. Up and Down cycle command history via readline, left and right move the cursor within the current line, Ctrl-C at the input prompt exits, and Ctrl-D exits too.

## How it works

kenoma seeds the model with your real shell context so the hallucinations are grounded:

1. `PS1` is captured once at startup by spawning an interactive shell and expanding it. Sentinels are wrapped around the expansion so we can pull it out of the startup banner noise.
2. If we're inside tmux, the current pane's scrollback is grabbed via `tmux capture-pane` and fed into the buffer. The model sees real commands with their real outputs.
3. Otherwise, the last N commands from `$HISTFILE` (or `~/.zsh_history` or `~/.bash_history`) are replayed as seed, with the captured `PS1` between them. The model sees the commands but no outputs, so it thinks they produced nothing.

Generation runs on a rolling text buffer that holds the whole fake session transcript. Each turn, the user's line is appended, the buffer is retokenized (with KV cache reuse across turns for speed), and `model.generate()` streams tokens until one of two things happens:

- `eos_token_id` fires.
- A line matches the prompt skeleton — `\n` plus the captured prompt with its cwd portion turned into a `[^\s]+` wildcard (full path or `~`-relative form, whichever appears in the prompt). So if your captured prompt was `a9lim@host:/Users/a9lim/Work/kenoma %`, the skeleton matches `\na9lim@host:/tmp %` after a fake `cd`, but rejects `\n50 % done`.

When the skeleton matches, kenoma adopts the matched substring as the new canonical prompt for display, the next turn's stop check, and the buffer glue. The fake terminal's prompt cwd actually tracks the model's fake `cd`s instead of forever showing the cwd kenoma was launched in. The trade-off: prompts that show only the cwd basename (bash `\W`-style), or that include a git branch / exit-status segment, won't match drifted versions — those generations run to `--max-new-tokens` instead of stopping early.

Output streams line by line. A small holdback keeps a partial prompt from leaking to the screen before the stop fires; once a line either completes, grows past the holdback budget, or hits a stop, its visible portion is flushed.

## Configuration

Three sources, in order of precedence: CLI flags, then `KENOMA_*` environment variables, then a TOML config file at `$XDG_CONFIG_HOME/kenoma/config.toml` (falls back to `~/.config/kenoma/config.toml`).

Example config:

```toml
model = "google/gemma-3n-E4B"
device = "auto"
temperature = 1.0
top_p = 0.95
repetition_penalty = 1.05
max_new_tokens = 2048
context_chars = 6000
history = 20
tmux_lines = 300
quantize = "none"
kv_cache = true
```

The env var for any key is `KENOMA_<KEY>` uppercased, so `KENOMA_MODEL=gpt2 kenoma` works. Config file support requires Python 3.11+ for `tomllib`; on 3.9 and 3.10 the file is silently ignored but env vars and CLI flags still work.

Flags worth knowing:

- `--prompt`: override the captured `PS1`.
- `--device {auto,cuda,mps,cpu}`: `auto` resolves to cuda, then mps, then cpu.
- `--quantize {none,4bit,8bit}`: bitsandbytes quantization. Requires CUDA and the `quantize` extra.
- `--no-kv-cache`: disable KV cache reuse across turns. Slower, mostly for debugging.
- `--history N`: seed with the last N commands from shell history (0 disables).
- `--tmux-lines N`: if inside tmux, seed with the last N lines of pane scrollback (0 disables).
- `--context-chars N`: cap the rolling buffer at N chars.
- `--max-new-tokens N`: per-turn cap on generated tokens.

Neither `--context-chars` nor `--max-new-tokens` respects the model's actual context window. Please pick values that fit.

## What it does not do

- Execute anything. Nothing touches your real filesystem; every command output is hallucinated. Please do not rely on kenoma for anything real.
- Support chat-tuned models meaningfully. Instruction-tuned and RLHF'd variants will not play a terminal; they'll try to answer you. Please use a base model.
- Handle Ctrl-C gracefully during generation. It will most likely kill the process. If interrupt-to-cancel-turn becomes important, please open an issue.
- Respect the model's context window automatically. The caps are blunt.
- Run on Windows. `readline`, tmux capture, and the `PS1` expansion trick all assume POSIX.

## License

AGPL-3.0-or-later. See `LICENSE`.
