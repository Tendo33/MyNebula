# Remove template residue and broken dead modules

## Goal

Delete the generic-Python-template code that survived MyNebula's evolution into
a product repository, so that `src/nebula/` contains only code the application
actually uses. This directly serves the repository guardrail: "MyNebula turns
GitHub Stars into a semantic knowledge graph; it is not a generic Python
template."

This is finding group **B** of parent task
`.trellis/tasks/08-18-comprehensive-hardening`.

## Background

A full-repository scan on 2026-08-18 established the exact usage surface of
`nebula.utils` by parsing every import in `src/`:

```
async_retry_decorator      -> core/embedding.py, core/llm.py
compute_content_hash       -> application/services/sync_execution_support.py
compute_topics_hash        -> application/services/sync_execution_support.py
get_logger                 -> 20 modules
setup_logging              -> main.py
```

Those five names are the entire application-facing surface. The package's
`__all__` exports 77 names. The remaining 67 have no call site outside the
`utils` package itself and its own dedicated tests.

Separately, `src/nebula/models/` does not import at all.

## Requirements

### B1 — Remove the broken `nebula.models` package

- `import nebula.models` currently raises
  `ImportError: cannot import name 'TimestampMixin' from 'nebula.models.base'`.
- `src/nebula/models/__init__.py` also imports `.examples`, which does not exist
  on disk.
- Nothing in `src/`, `tests/`, `scripts/`, or `alembic/` imports it. The real
  ORM models live in `src/nebula/db/models.py` and the real Pydantic schemas in
  `src/nebula/schemas/`.
- The package must be deleted, not repaired. Repairing it would reintroduce a
  second, unused base-model hierarchy alongside `nebula/schemas/`.

### B2 — Remove unused utility modules

- Delete modules with no application call site:
  `utils/common_utils.py`, `utils/file_utils.py`, `utils/json_utils.py`,
  `utils/date_utils.py`.
- Reduce `utils/decorator_utils.py` to `async_retry_decorator` plus whatever it
  needs to run.
- Keep `utils/logger_util.py` and `utils/hash_utils.py`.
- Reduce `utils/__init__.py` `__all__` to the names that are actually imported.

### B3 — Remove tests that exist only to cover deleted code

- `tests/test_json_utils.py`, `tests/test_date_utils.py`, and
  `tests/test_decorator_utils.py` test the deleted surface. Coverage for
  `async_retry_decorator` must survive; everything else in those files goes.
- `tests/test_main_health.py` imports `nebula.utils.logger_util` and
  `tests/core/test_sync_execution_support.py` imports the two hash helpers. Both
  must keep passing untouched.
- `tests/conftest.py` carries six fixtures (`temp_dir`, `temp_file`,
  `temp_json_file`, `sample_dict`, `sample_nested_dict`, `sample_datetime`) that
  exist only for the deleted tests. Remove those with no remaining consumer.

### B4 — Keep the public import path stable for surviving names

- `from nebula.utils import get_logger, setup_logging, async_retry_decorator,
  compute_content_hash, compute_topics_hash` must keep working unchanged. No
  call site in `src/` may need editing for import reasons.

## Constraints

- This task runs **first** in the parent's ordering. Nothing else may reference
  the deleted modules afterwards.
- `async_retry_decorator` must survive because sibling task
  `08-18-pipeline-correctness` (A1) reapplies it at per-call granularity.
- No behaviour change in logging, retry semantics, or hashing.
- No new dependency.

## Non-goals

- Not rewriting or reformatting `logger_util.py` or `hash_utils.py`.
- Not changing retry defaults (`max_retries=3, delay=1.0, backoff=2.0`).
- Not touching `src/nebula/schemas/`, which is the live Pydantic layer.

## Acceptance criteria

- [ ] `src/nebula/models/` no longer exists, and
      `uv run python -c "import nebula.models"` fails with `ModuleNotFoundError`
      rather than `ImportError` from a half-broken package.
- [ ] `src/nebula/utils/` contains only `__init__.py`, `logger_util.py`,
      `hash_utils.py`, and `decorator_utils.py`.
- [ ] `nebula.utils.__all__` lists exactly the names with live call sites.
- [ ] `uv run python -c "from nebula.utils import get_logger, setup_logging,
      async_retry_decorator, compute_content_hash, compute_topics_hash"` succeeds.
- [ ] `uv run python -c "import nebula.main"` succeeds.
- [ ] `uv run ruff check src tests scripts alembic` passes.
- [ ] `uv run ruff format --check src tests scripts alembic` passes.
- [ ] `uv run pytest -q` passes with no import errors and no collection errors.
- [ ] Retry behaviour of `async_retry_decorator` is still covered by at least one
      test.
- [ ] `.trellis/spec/backend/directory-structure.md` no longer describes the
      removed modules, if it currently does.

## Verification

Backend-only scope. Run the "Backend" section of
`.trellis/spec/shared/verification.md`, plus the documentation link check,
because spec files change.
