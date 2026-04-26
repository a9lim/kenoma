# Security Policy

## Reporting a vulnerability

If you've found a security issue in kenoma, please report it privately rather than filing a public issue.

- **Email:** a9lim@protonmail.com
- **GitHub:** use [private security advisories](https://github.com/a9lim/kenoma/security/advisories/new)

Please include a description, steps to reproduce, and the version you are on. I'll respond within a few days and aim to have a fix as soon as possible.

## Supported versions

Only the latest release on PyPI receives security fixes. If you're on an older version, the fix is to upgrade.

## What kenoma does at startup

A few things kenoma does on launch are worth understanding because they touch your real environment, even though no fake-shell command actually executes:

- **Spawns an interactive shell** (`$SHELL -i -c ...`) once, just to expand `PS1`. This runs your shell's RC files (`.zshrc`, `.bashrc`, plugins, etc.) the same way opening a new terminal would.
- **Reads your shell history file** (`$HISTFILE` or `~/.zsh_history` or `~/.bash_history`) to seed the buffer with the last N commands as fake context for the model. The history content is fed into the model but is not echoed back to your screen.
- **Captures the current tmux pane scrollback** if `$TMUX` is set, via `tmux capture-pane`. This includes recent commands and their real outputs, and gets fed into the model as seed context.

If you don't want any of these, please pass `--history 0`, `--tmux-lines 0`, or `--prompt '...'` to disable them individually. The model still runs in completion mode against whatever is in its context, so anything you don't want it to see, please don't seed.

## What kenoma does not do

- Execute anything the model emits. Every command output is hallucinated; nothing touches your real filesystem. Please do not pipe kenoma output into a real shell, and please don't trust file paths or commands it suggests as if they were real.
- Send anything off your machine. Model loading goes through HuggingFace's `transformers` library, which fetches weights from the Hub on first use; after that, generation is local.
- Sandbox the loaded model.

## Model and checkpoint trust

Kenoma loads HuggingFace checkpoints via `transformers`, which executes code from the checkpoint repo in some cases. Kenoma does not set `trust_remote_code=True`, but if you pass a model that requires it, please be aware that you are executing arbitrary code from that repo. Please only load models from publishers you trust.

The base and completion models that kenoma is meant for (Qwen base, Gemma base, Llama base, GPT-2, Pythia) do not require `trust_remote_code`; if a model asks for it, that's a signal to be more careful, not less.

## Hallucinated output

Kenoma's whole point is that the model fakes shell output. Please do not rely on anything it produces. A fake `ls` may invent files that don't exist, a fake `cat` may invent file contents, a fake `curl` may invent network responses. The model is playing a character, not querying reality.
