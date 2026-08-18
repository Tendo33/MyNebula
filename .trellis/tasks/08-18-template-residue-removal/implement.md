# Implementation plan — Remove template residue and broken dead modules

Ordering: this is child **1 of 5** in
`.trellis/tasks/08-18-comprehensive-hardening`. It runs before
`08-18-pipeline-correctness`, which depends on `async_retry_decorator`
surviving.

## Step 0 — Prove the deletion set is safe

- [ ] Grep every symbol slated for deletion, not just module names, across
      `src`, `tests`, `scripts`, `alembic`:
      `grep -rn "<symbol>" src tests scripts alembic --include="*.py"` for each
      name in the current `nebula.utils.__all__` minus the five survivors.
- [ ] Grep for dynamic access that a plain import grep would miss:
      `grep -rn "importlib\|__import__\|getattr(utils\|nebula\.utils\." src tests scripts alembic`
- [ ] Record the baseline test count and coverage of the modules being kept:
      `uv run pytest -q 2>&1 | tail -3`
- [ ] **Gate:** if any grep finds a live reference, stop and revise
      `design.md` before deleting anything.

## Step 1 — Delete `src/nebula/models/`

- [ ] `git rm -r src/nebula/models`
- [ ] `uv run python -c "import nebula.main"` still succeeds.
- [ ] `uv run python -c "import nebula.models"` now raises
      `ModuleNotFoundError`.

## Step 2 — Delete the zero-call-site utility modules

- [ ] `git rm src/nebula/utils/common_utils.py src/nebula/utils/file_utils.py
      src/nebula/utils/json_utils.py src/nebula/utils/date_utils.py`
- [ ] Do not touch `utils/__init__.py` yet; the import errors it now raises are
      the checklist for step 4.

## Step 3 — Trim `decorator_utils.py` to `async_retry_decorator`

- [ ] Remove `timing_decorator`, `retry_decorator`, `catch_exceptions`,
      `log_calls`, `deprecated`, `singleton`, `ContextTimer`,
      `async_timing_decorator`, `async_catch_exceptions`, `async_log_calls`,
      `AsyncContextTimer`.
- [ ] Keep `async_retry_decorator` byte-identical in body, signature, and log
      strings.
- [ ] Remove imports that become unused (`time`, `Any`, and any others), keep
      `asyncio`, `traceback`, `wraps`, `Callable`, `Coroutine`, and the module
      logger.
- [ ] `uv run ruff check --fix src/nebula/utils/decorator_utils.py`

## Step 4 — Rewrite `src/nebula/utils/__init__.py`

- [ ] Reduce to three imports and a five-name `__all__`:
      `async_retry_decorator`, `compute_content_hash`, `compute_topics_hash`,
      `get_logger`, `setup_logging`.
- [ ] Update the module docstring so it describes the surviving three modules,
      not the deleted seven.
- [ ] `uv run python -c "from nebula.utils import async_retry_decorator,
      compute_content_hash, compute_topics_hash, get_logger, setup_logging"`

## Step 5 — Trim the tests

- [ ] `git rm tests/test_json_utils.py tests/test_date_utils.py`
- [ ] Reduce `tests/test_decorator_utils.py` to only the tests exercising
      `async_retry_decorator`; delete tests for every removed decorator.
- [ ] Leave `tests/test_main_health.py` and
      `tests/core/test_sync_execution_support.py` untouched — they are the
      regression signal.
- [ ] Prune `tests/conftest.py`. Its `temp_dir`, `temp_file`, `temp_json_file`,
      `sample_dict`, `sample_nested_dict`, and `sample_datetime` fixtures exist
      only for the deleted `file_utils` / `json_utils` / `date_utils` /
      `common_utils` tests. Grep each fixture name across `tests/` first and
      remove only those with no remaining consumer.
- [ ] `uv run pytest -q` passes. Expect the pass count to drop by exactly the
      number of removed dead-code tests; any other delta is a regression to
      investigate before continuing.

## Step 6 — Update the spec

- [ ] `.trellis/spec/backend/directory-structure.md:14` — remove the
      `models/  # Legacy/internal models` line.
- [ ] `.trellis/spec/backend/python-package.md:14` — remove the `models/` entry.
- [ ] If either file enumerates `utils/` submodules, reduce that list to
      `logger_util`, `decorator_utils`, `hash_utils`.

## Step 7 — Verify

Backend scope plus docs link check.

```bash
uv run ruff check src tests scripts alembic
uv run ruff format --check src tests scripts alembic
uv run pytest -q
uv run python -c "import nebula.main"
```

Then the documentation and link check block from
`.trellis/spec/shared/verification.md`.

## Review gates

- **Gate A** (after step 0): deletion set proven unreferenced. Blocking.
- **Gate B** (after step 4): `import nebula.main` and the five-name import both
  succeed. Blocking — do not touch tests until the package imports cleanly.
- **Gate C** (after step 5): test-count delta equals the number of intentionally
  removed tests. Blocking.
- **Gate D** (after step 7): full backend verification green.

## Rollback points

- After step 1: `git checkout src/nebula/models` restores the package.
- After steps 2–4: revert `src/nebula/utils/` as a directory; the surviving
  modules were never edited except `__init__.py` and `decorator_utils.py`.
- After step 7: single-commit `git revert`. No schema, data, or config state is
  involved, so revert is complete and immediate.

## Expected diff shape

- Deleted: ~1,400 LOC across four utility modules, ~2 files under `models/`,
  2 test files.
- Edited: `utils/__init__.py` (209 → ~25 LOC), `decorator_utils.py`
  (541 → ~70 LOC), `tests/test_decorator_utils.py` (trimmed), 2 spec files.
- Net: roughly −1,800 LOC, zero behaviour change.
