# Security Policy

## Reporting a vulnerability

If you found a security issue in kenoma, please report it privately instead of filing a public issue.

- **Email:** mx@a9l.im
- **GitHub:** use [private security advisories](https://github.com/a9lim/kenoma/security/advisories/new)

Please include a description, steps to reproduce, and the version you are on. I will respond in a few days and try to have a fix as soon as possible.

## Supported versions

Only the latest release on PyPI has security fixes. If you're on an older version, the fix is to upgrade.

## What kenoma does at startup

A few things kenoma does may touch your real environment, even though no command you type is actually executed:

- **Interactive shell spawns** (`$SHELL -i -c ...`) once, to expand `PS1`. This runs your shell's RC files (`.zshrc`, `.bashrc`, plugins, etc.) as opening a new terminal does.
- **Shell history files read** (`$HISTFILE` or `~/.zsh_history` or `~/.bash_history`) fills the buffer with the last N commands as context for the model. The history is given to the model but is not echoed back to your screen.
- **Captures the current tmux pane scrollback** if `$TMUX` is set, via `tmux capture-pane`. This includes recent commands and their real outputs, and gets given to the model as context.

If you don't want this, please pass `--history 0`, `--tmux-lines 0`, or `--prompt '...'` to disable them. The model runs against whatever is in its context, so don't give it anything you don't want it to see.

## What kenoma does not do

- Execute anything; nothing touches your real filesystem.
- Sandbox the loaded model.

## Model and checkpoint trust

Kenoma loads HuggingFace checkpoints via `transformers`, which executes code from the checkpoint repo in some cases. Kenoma does not set `trust_remote_code=True`, but if you pass a model that requires it, please be aware that you are executing arbitrary code from that repo. Please only load models from publishers you trust.

The base and completion models that kenoma is meant for (Qwen base, Gemma base, Llama base, GPT-2, Pythia) do not require `trust_remote_code`.
