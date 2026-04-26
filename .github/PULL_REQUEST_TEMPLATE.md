## What

<!-- One or two sentences on what changed -->

## Why

<!-- What problem does this solve? Please link issues with "Fixes #N" if applicable -->

## Test plan

- [ ] `ruff check .` passes
- [ ] `pyright` passes
- [ ] Manual session against a small base model (e.g. `Qwen/Qwen2.5-0.5B`) still streams line by line, stops on a prompt-shaped line, and exits cleanly on Ctrl-D
- [ ] If you touched the skeleton, holdback, or KV cache: please note which model and shell you tested against

## Notes

<!-- Anything reviewers should know: design choices, followups, known limitations -->
