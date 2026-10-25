# kenoma

[![CI](https://github.com/a9lim/kenoma/actions/workflows/ci.yml/badge.svg)](https://github.com/a9lim/kenoma/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/kenoma)](https://pypi.org/project/kenoma/)
[![Downloads](https://img.shields.io/pypi/dm/kenoma)](https://pypi.org/project/kenoma/)
[![License: GPL v3](https://img.shields.io/badge/License-GPL_v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://pypi.org/project/kenoma/)

A shell powered by raw LLM completion. You type commands at your own prompt; a base model hallucinates what they print. Nothing is executed.

kenoma captures your real prompt and seeds the model with your terminal (tmux scrollback, or your shell history), so the fake session starts out looking like yours. Every output is one sample of many, and kenoma keeps them all as a tree you can re-roll, rewind and branch, loom-style.

## Install

```
pip install kenoma
```

On Apple silicon the model runs locally with [MLX](https://github.com/ml-explore/mlx-lm). Elsewhere, kenoma talks to any OpenAI-compatible completion server (see [Using a server](#using-a-server)).

## Usage

```
kenoma                          # Qwen/Qwen3.5-4B-Base, quantized to 4 bits at load
kenoma Qwen/Qwen3.5-2B-Base     # about twice as fast, a little less convincing
kenoma /path/to/local/model
```

Use a base (pretrained) model. Instruction-tuned models answer you instead of playing a terminal.

| Key | |
|---|---|
| Enter | run the command |
| → or Ctrl-E | accept the gray suggestion (the model's guess at what you'll type next) |
| Alt-r | re-roll: run the last command again for a different output |
| Alt-← / Alt-→ | previous / next alternative output of the last command |
| Alt-↑ / Alt-↓ | step back a turn / forward again. Typing after stepping back starts a branch. |
| Alt-t | tree explorer: browse every branch and jump anywhere |
| Ctrl-C | interrupt the running command (or clear the line) |
| Ctrl-D | exit |

Alt is Option on a Mac keyboard ("Use Option as Meta key" in Terminal.app, "Esc+" in iTerm2), or press Esc and then the key.

On an M5 Max the default model streams about 140 tokens a second, and a command that prints nothing gets its prompt back in about a tenth of a second.

## Using a server

```
kenoma --url http://localhost:8000
```

Any server with a streaming `/v1/completions` endpoint works: vLLM, SGLang, llama.cpp's `llama-server`, or `mlx_lm.server`. kenoma uses the first model the server lists, or the one you name (`kenoma --url ... <model>`; `mlx_lm.server` lists every cached model, so name it there). Your seed transcript (scrollback or history) is sent to that server.

## Configuration

By precedence: CLI flags, then `KENOMA_*` environment variables, then a TOML file at `$XDG_CONFIG_HOME/kenoma/config.toml` (falls back to `~/.config/kenoma/config.toml`).

```toml
model = "Qwen/Qwen3.5-4B-Base"
bits = 4
temperature = 1.0
top_p = 0.95
max_new_tokens = 2048
context_tokens = 16384
history = 20
tmux_lines = 300
ghost = true
```

The environment variable for any key is `KENOMA_<KEY>` uppercased, so `KENOMA_MODEL=Qwen/Qwen3.5-2B-Base kenoma` works.

- `--url URL`, `--api-key KEY`: use a completion server instead of the local engine.
- `--bits {3,4,5,6,8,16}`: quantize the local model's weights at load (16 keeps them as they are; already-quantized checkpoints are left alone).
- `--prompt STR`: use this prompt instead of your rendered `PS1`.
- `--history N`: without tmux, seed with your last N shell commands (0 disables).
- `--tmux-lines N`: inside tmux, seed with the last N lines of the pane (0 disables).
- `--context-tokens N`: transcript budget; past it, the oldest turns are dropped down to half.
- `--max-new-tokens N`: per-command cap on generated tokens.
- `--no-ghost`: don't show the model's guess at your next command.

## License

GPL-3.0-or-later. See `LICENSE`.
