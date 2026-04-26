# Contributing to kenoma

Thank you very much for wanting to contribute! I really appreciate any contribution you would like to make, whether it's a PR or a bug report.

## Dev setup

```bash
git clone https://github.com/a9lim/kenoma
cd kenoma
pip install -e ".[dev]"
```

The `dev` extra pulls in ruff, pyright, build, and twine. The `quantize` extra pulls bitsandbytes if you want to try the 4bit or 8bit paths (CUDA only).

## Lint and typecheck

CI runs `ruff check .` and `pyright` on every PR. Please run them locally first:

```bash
ruff check .
ruff check . --fix    # auto-fix what's fixable
pyright
```

Pyright is scoped to `kenoma.py` and runs in standard mode; the warnings from torch and transformers are downgraded so the only errors that appear are real.

## Manual testing

There is no automated test suite yet because kenoma is a wrapper around `model.generate()` and the meaningful behavior is in the streaming and stop-detection which is hard to unit-test without a real model. If you touch anything, please run a manual session and confirm:

1. The prompt streams line by line, not in big bursts.
2. Generation stops on a prompt-shaped line and adopts it (try a fake `cd /tmp` and check the next prompt's cwd).
3. Ctrl-D and Ctrl-C at the input prompt both exit cleanly.

```bash
kenoma Qwen/Qwen2.5-0.5B
```

is small enough to run on CPU for a smoke test.

## Adding a new feature flag

Three places need an update if you add something:

1. `CONFIG_KEYS` at the top of `kenoma.py` (key plus type).
2. The `argparse` block in `main()` with a CLI flag.
3. The README's Configuration section.

The env var version is automatic from `CONFIG_KEYS` (`KENOMA_<KEY>` uppercased), and the config file picks it up too as long as it's in the dict.

## PRs

- Please don't bump the version in your PR unless you want a new release; the PyPI publish workflow is triggered by a version update on `main`.
- If you add a flag, please mention which model you tested it against in the PR description.
- Small PRs are easier to review than big ones. If you're unsure, please open an issue first so we can sketch it together.

## Questions

Please open an issue or reach out on Discord. For anything security-sensitive, please see [SECURITY.md](SECURITY.md).
