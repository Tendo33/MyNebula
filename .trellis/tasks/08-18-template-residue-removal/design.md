# Design — Remove template residue and broken dead modules

## Scope boundary

Deletion only, inside `src/nebula/utils/`, `src/nebula/models/`, `tests/`, and
two `.trellis/spec/` files. No call site in `src/nebula/` outside
`utils/__init__.py` changes, because every surviving symbol keeps its import
path.

## Current state

### `src/nebula/models/`

```
src/nebula/models/__init__.py   # imports .base (TimestampMixin missing) and .examples (file missing)
src/nebula/models/base.py       # BaseModel only; no TimestampMixin
```

`__init__.py` fails on its first import line. `examples.py` was never committed.
No importer exists anywhere in the repository. The live equivalents are
`src/nebula/db/models.py` (SQLAlchemy ORM) and `src/nebula/schemas/` (Pydantic
request/response models).

### `src/nebula/utils/`

| Module | LOC | Coverage | Application call sites |
| --- | --- | --- | --- |
| `logger_util.py` | 291 | 49% | `get_logger` (20 modules), `setup_logging` (`main.py`) |
| `decorator_utils.py` | 541 | 55% | `async_retry_decorator` (`core/embedding.py`, `core/llm.py`) |
| `hash_utils.py` | 18 | 55% | `compute_content_hash`, `compute_topics_hash` (`sync_execution_support.py`) |
| `common_utils.py` | 377 | 10% | none |
| `file_utils.py` | 432 | 11% | none |
| `json_utils.py` | 306 | 50% | none |
| `date_utils.py` | 292 | 84% | none |

The four zero-call-site modules import only `.logger_util`, so nothing in the
surviving set depends on them. `decorator_utils.py` likewise imports only
`.logger_util`, `asyncio`, `time`, `traceback`, `functools.wraps`, and
`typing.Any`.

`async_retry_decorator` (`decorator_utils.py:329`) is self-contained: it needs
`asyncio`, `traceback`, `wraps`, `Callable`/`Coroutine`, and the module logger.
It does not call any other decorator in the file.

## Target state

```
src/nebula/utils/
├── __init__.py          # re-exports exactly 5 names
├── logger_util.py       # unchanged
├── decorator_utils.py   # async_retry_decorator only
└── hash_utils.py        # unchanged
```

`src/nebula/models/` is gone.

## Decisions and rationale

### D1 — Delete `nebula.models` rather than fix it

Fixing means writing a `TimestampMixin` and an `examples.py` that no code will
call, and standing up a second Pydantic base hierarchy next to
`src/nebula/schemas/`. That adds a maintenance surface with no consumer.
Deletion also converts a confusing `ImportError` into an honest
`ModuleNotFoundError`.

### D2 — Delete whole modules, trim only `decorator_utils.py`

`common_utils.py`, `file_utils.py`, `json_utils.py`, and `date_utils.py` have no
surviving symbol, so whole-file deletion is the smaller and more reviewable
change. `decorator_utils.py` is the only module with a mixed population, so it is
the only one edited in place.

### D3 — Keep `logger_util.py` and `hash_utils.py` untouched

`logger_util.py` is the logging backbone; `tests/test_main_health.py` reaches
into it directly. Some of its module-level convenience wrappers
(`configure_json_logging`, `log_function_calls`, `debug`/`info`/`warning`/
`error`/`critical`/`exception`) have no call site, but trimming them changes the
logging layer during a deletion-only task and buys ~100 LOC. They stay, and drop
out of `__all__` only if they were exported for no reason. Recorded as a
deliberate deferral so a future reader does not read the omission as an
oversight.

### D4 — Shrink `__all__` to the live surface

`nebula.utils.__all__` currently advertises 77 names. After deletion it lists
exactly the five importable-by-the-app names. This makes the package's contract
self-describing and makes any future reintroduction of template helpers an
explicit, reviewable act.

### D5 — Trim tests rather than delete all three files

`tests/test_decorator_utils.py` covers `async_retry_decorator` among others.
That coverage must survive, so the file is reduced to its retry tests rather
than removed. `tests/test_json_utils.py` and `tests/test_date_utils.py` cover
only deleted code and are removed outright.

## Contracts

Unchanged import contract for every surviving name:

```python
from nebula.utils import (
    async_retry_decorator,
    compute_content_hash,
    compute_topics_hash,
    get_logger,
    setup_logging,
)
```

`async_retry_decorator(max_retries=3, delay=1.0, backoff=2.0, exceptions=(Exception,))`
keeps its signature, its exponential backoff, its warn-per-attempt and
error-on-exhaustion logging, and its re-raise of the last exception.

## Blast radius

- 20 modules import `get_logger`; none change.
- 2 modules import `async_retry_decorator`; neither changes.
- 1 module imports the hash helpers; it does not change.
- `tests/test_main_health.py` and `tests/core/test_sync_execution_support.py`
  must pass without edits — they are the regression signal that D3 and D4 did
  not break the surviving surface.

## Compatibility

`nebula.utils` is internal. `pyproject.toml` exposes only the
`mynebula = "nebula.main:run"` console script, and `hatch` packages
`src/nebula` wholesale. No published API is affected, so no deprecation window
is needed.

## Rollback

Single-commit revert. The task performs no schema migration, no data change, and
no config change, so revert restores the prior state exactly.

## Risks

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| A dynamic import or string-based reference to a deleted helper exists | Low | Grep for each deleted symbol name across `src`, `tests`, `scripts`, `alembic` before deleting, not just for module names |
| `ruff` flags newly unused imports in trimmed `decorator_utils.py` | Medium | Run `ruff check --fix` on the trimmed file and re-verify |
| Coverage percentage moves and masks a real regression | Medium | Compare absolute pass count, not coverage percentage; 243 tests minus the removed dead-code tests is the expected count |
