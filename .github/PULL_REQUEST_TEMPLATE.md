## What

<!-- One or two sentences on what changed -->

## Why

<!-- What problem does this solve? Please link issues with "Fixes #N" if applicable -->

## Test plan

- [ ] `pytest`, `ruff check .` and `pyright` pass
- [ ] Manual session against a base model (e.g. `Qwen/Qwen3.5-2B-Base`): output streams, a fake `cd` changes the prompt, Alt-r / Alt-↑ / Alt-t work, Ctrl-D exits
- [ ] If you touched the template, recognizer or engine cache: please note which model and shell you tested against

## Notes

<!-- Anything reviewers should know: design choices, followups, known limitations -->
