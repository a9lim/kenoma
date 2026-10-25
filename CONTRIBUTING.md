# Contributing to kenoma

Thank you very much for wanting to contribute! I really appreciate any contribution you would like to make, whether it's a PR or a bug report.

## Dev setup

```bash
git clone https://github.com/a9lim/kenoma
cd kenoma
pip install -e ".[dev]"
```

The `dev` extra pulls in pytest, ruff, pyright, build, and twine. The local engine needs Apple silicon (mlx-lm); on other machines, run any OpenAI-compatible completion server and pass `--url`.

[AGENTS.md](AGENTS.md) is the design doc: how the prompt template, the recognizer, the transcript and the engine caches fit together.

## Tests, lint, typecheck

CI runs all three on every PR. Please run them locally first:

```bash
pytest
ruff check .
pyright
```

The tests need no model: `FakeEngine` in `tests/test_session.py` scripts token streams, so turns, leftover handling, ghost text and interrupts are covered without weights.

## Manual testing

If you touch anything user-facing, please run a session and confirm:

1. Output streams as it's generated, and a command with no output gets its prompt back immediately.
2. A fake `cd` changes the prompt's cwd.
3. Alt-r re-rolls, Alt-↑ steps back (typing there branches), Alt-t opens the tree, Ctrl-C interrupts, Ctrl-D exits.

```bash
kenoma Qwen/Qwen3.5-2B-Base
```

is quick to load. If you changed the engine's cache handling, check that next-token logits after a rewind match a fresh prefill on a Qwen3.5 model (recurrent layers) and an attention-only one.

## Adding a new setting

Four places need an update:

1. `CONFIG_KEYS` in `kenoma/cli.py` (key plus type).
2. `DEFAULTS` right below it: the default value. Argparse pulls from it.
3. The argparse block in `parse_args()` (use `d["key"]` for the default).
4. The README's Configuration section.

The env var (`KENOMA_<KEY>` uppercased) and the config file pick it up automatically.

## PRs

- Please don't bump the version in your PR unless you want a new release; the PyPI publish workflow is triggered by a version update on `main`.
- If you add a setting, please mention which model you tested it against in the PR description.
- Small PRs are easier to review than big ones. If you're unsure, please open an issue first so we can sketch it together.

## Questions

Please open an issue or reach out on Discord. For anything security-sensitive, please see [SECURITY.md](SECURITY.md).
