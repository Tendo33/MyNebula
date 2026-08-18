# Backend Testing

## Defaults

- Use `pytest`.
- Use `ruff check` and `ruff format --check` as gates, not optional polish.
  There is no `mypy` in this project: it is not a dependency, not a CI job, and
  not in `shared/verification.md`. Do not cite it as a gate.
- New behavior needs tests.
- Bug fixes need at least one regression test.

## A Guard Must Be Verified To Fail

A regression test that passes against the buggy implementation is not a
regression test. After writing one, revert the fix, confirm the test fails with
a message that names the defect, then restore the fix. The same applies to
static guards (i18n parity, accessibility, lint scope): temporarily introduce
the violation and confirm the guard reports it with the offending
file/line/key.

This is not a formality. During the 2026-08-18 hardening epic, the first
version of the edge auto-load regression test passed against both the fixed and
the broken hook, and the accessibility guard silently stopped covering the
command palette when its focus trap moved into a shared hook.

## Test Shape

- Pure logic gets unit tests.
- Configuration behavior gets environment-isolated tests. Clear the relevant
  `lru_cache` (`get_app_settings`, `get_sync_settings`, …) in both directions,
  or the next test inherits the override.
- File, process, network, and script behavior gets focused integration tests
  when the behavior matters.
- Avoid tests that only assert implementation details.

### Faking the database session

Most backend tests fake `AsyncSession`. Two traps that have already cost time:

- Discriminating statements by substring is fragile. `select(StarredRepo)`
  renders column names including `open_issues_count`, so a `"count" in sql`
  check matches the row query as well as the aggregate. Match `"count("`.
- A fake session must model transaction boundaries the way the code under test
  relies on them. Statements take effect immediately; a checkpoint is taken at
  `commit()` and restored at `rollback()`. Deferring writes until commit makes
  bulk-update-then-ORM-assign sequences behave differently from production.

## Public API Tests

Keep public import tests for stable package surfaces. They catch accidental
renames and disappearing exports earlier than downstream users will.

## Before Completion

Run the smallest relevant check first while working, then the full required
backend or full-stack gate before claiming the task is complete.

Report which tiers actually ran. "Tests pass" without saying what was executed
is not a completion report.
