# kenoma

[![CI](https://github.com/a9lim/kenoma/actions/workflows/ci.yml/badge.svg)](https://github.com/a9lim/kenoma/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/kenoma)](https://pypi.org/project/kenoma/)
[![Downloads](https://img.shields.io/pypi/dm/kenoma)](https://pypi.org/project/kenoma/)
[![License: GPL v3](https://img.shields.io/badge/License-GPL_v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://pypi.org/project/kenoma/)

A shell powered by LLM completion.

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

The model argument is any HuggingFace model id or a local path. This is meant for base/completion models, instruction-tuned models may not work properly.

## Configuration

By precedence: CLI flags, then `KENOMA_*` environment variables, then a TOML config file at `$XDG_CONFIG_HOME/kenoma/config.toml` (falls back to `~/.config/kenoma/config.toml`).

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
compile = false
```

The env var for any key is `KENOMA_<KEY>` uppercased, so `KENOMA_MODEL=gpt2 kenoma` works.

Flags:

- `--version`: print version and exit.
- `--prompt`: override the captured `PS1`. Multi-line prompts are not supported and fall back to a constructed `user@host:cwd $ `.
- `--device {auto,cuda,mps,cpu}`: `auto` resolves to cuda, then mps, then cpu.
- `--quantize {none,4bit,8bit}`: bitsandbytes quantization. Requires CUDA and the `quantize` extra.
- `--no-kv-cache`: disable KV cache reuse across turns.
- `--compile`: `torch.compile` the model with a static KV cache for faster decode (best on CUDA). The first turn pays a compile cost; cross-turn KV cache reuse is forfeited because the static cache doesn't expose `crop()`.
- `--history N`: seed with the last N commands from shell history (0 disables).
- `--tmux-lines N`: if inside tmux, seed with the last N lines of pane scrollback (0 disables).
- `--context-chars N`: cap the rolling buffer at N chars. 
- `--max-new-tokens N`: per-turn cap on generated tokens.

**Cancelling a turn.** Ctrl-C during generation cancels the current turn, invalidates the KV cache, and redraws the prompt. Ctrl-C at the input prompt exits.

## License

GPL-3.0-or-later. See `LICENSE`.
