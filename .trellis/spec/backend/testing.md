# Backend Testing

## Defaults

- Use `pytest`.
- Use `ruff` and `mypy` as gates, not optional polish.
- New behavior needs tests.
- Bug fixes need at least one regression test.

## Test Shape

- Pure logic gets unit tests.
- Configuration behavior gets environment-isolated tests.
- File, process, network, and script behavior gets focused integration tests
  when the behavior matters.
- Avoid tests that only assert implementation details.

## Public API Tests

Keep public import tests for stable package surfaces. They catch accidental
renames and disappearing exports earlier than downstream users will.

## Before Completion

Run the smallest relevant check first while working, then the full required
backend or full-stack gate before claiming the task is complete.
