# Security Policy

## Reporting a vulnerability

If you found a security issue in kenoma, please report it privately instead of filing a public issue.

- **Email:** mx@a9l.im
- **GitHub:** use [private security advisories](https://github.com/a9lim/kenoma/security/advisories/new)

Please include a description, steps to reproduce, and the version you are on. I will respond in a few days and try to have a fix as soon as possible.

## Supported versions

Only the latest release on PyPI has security fixes. If you're on an older version, the fix is to upgrade.

## What kenoma does at startup

No command you type is executed. A few things at startup do touch your real environment:

- **Spawns an interactive shell** (`$SHELL -i -c ...`) once to render `PS1`. This runs your shell's RC files (`.zshrc`, `.bashrc`, plugins) as opening a new terminal does, and runs your prompt hooks (zsh `precmd`, bash `PROMPT_COMMAND`) so themes that build the prompt there render.
- **Reads your shell history** (`$HISTFILE`, `~/.zsh_history` or `~/.bash_history`). The last `--history` commands become part of the model's context (shown to it as the output of `history`); all of it feeds arrow-key recall.
- **Captures the current tmux pane** if `$TMUX` is set, via `tmux capture-pane`: recent commands and their real outputs, given to the model as context.
- **Runs `git rev-parse --abbrev-ref HEAD`** in the current directory to find a branch name in your prompt.

To avoid these, pass `--history 0`, `--tmux-lines 0`, or `--prompt '...'`. The model sees whatever is in its context, so don't give it anything you don't want it to see.

## With `--url`, your context leaves the machine

The remote engine sends the whole transcript (seed included: scrollback or history, plus everything you type) to the server you name. Only point it at servers you trust with that.

## Model output is sanitized

Model output goes to your real terminal, so control bytes are dropped before display: C0 and C1 controls, ESC (and with it every escape sequence) and carriage returns. Tab and newline are kept. The prompt kenoma draws is your own rendered PS1 with only its SGR color codes kept; a prompt the model wrote that doesn't fit your template is shown as plain text.

## Model and checkpoint trust

The local engine loads checkpoints with mlx-lm, which does not execute code from the repository (no `trust_remote_code`). Weights from an untrusted publisher can still produce any text, which kenoma only ever displays.
