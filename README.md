# kenoma

[![CI](https://github.com/a9lim/kenoma/actions/workflows/ci.yml/badge.svg)](https://github.com/a9lim/kenoma/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/kenoma)](https://pypi.org/project/kenoma/)
[![Downloads](https://img.shields.io/pypi/dm/kenoma)](https://pypi.org/project/kenoma/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
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

The model argument is any HuggingFace model id or a local path. This is meant for base/completion models — instruction-tuned (`-it`) variants technically load but break character constantly, since they're trained to respond to a turn instead of pretending to be the entire terminal.

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
```

The env var for any key is `KENOMA_<KEY>` uppercased, so `KENOMA_MODEL=gpt2 kenoma` works. Config files use stdlib `tomllib` on 3.11+ and the `tomli` backport on 3.9 / 3.10 (pulled in automatically).

Config-file values are typechecked against the expected type and unknown keys are warned about, so typos surface instead of being silently dropped.

Flags:

- `--version`: print version and exit.
- `--prompt`: override the captured `PS1`. Multi-line prompts are not supported and fall back to a constructed `user@host:cwd $ `.
- `--device {auto,cuda,mps,cpu}`: `auto` resolves to cuda, then mps, then cpu.
- `--quantize {none,4bit,8bit}`: bitsandbytes quantization. Requires CUDA and the `quantize` extra.
- `--no-kv-cache`: disable KV cache reuse across turns. Mostly for debugging; reuse is the default and includes a startup warmup pass through the seeded buffer.
- `--history N`: seed with the last N commands from shell history (0 disables).
- `--tmux-lines N`: if inside tmux, seed with the last N lines of pane scrollback (0 disables).
- `--context-chars N`: cap the rolling buffer at N chars. Trim happens at turn boundaries and is line-aligned.
- `--max-new-tokens N`: per-turn cap on generated tokens.

**Cancelling a turn.** Ctrl-C during generation cancels the current turn, invalidates the KV cache, and redraws the prompt. Ctrl-C at the input prompt exits.

**Bash 4.4+ required for PS1 capture.** macOS ships `/bin/bash` 3.2, which can't expand `${PS1@P}`; users with bash as `$SHELL` will get a one-line warning at startup and a constructed fallback prompt. Pass `--prompt` to override, or use zsh.

## License

AGPL-3.0-or-later. See `LICENSE`.
