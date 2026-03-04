# Graph Refactor Stability Report

Date: 2026-03-04  
Branch: `codex/graph-full-refactor`

## Target gates
- 50 consecutive pipeline runs with no data corruption
- 10 rollback drills with 100% success

## Command checklist
- `uv run pytest -q`
- `uv run alembic upgrade head`
- `uv run alembic downgrade -1`
- `uv run alembic upgrade head`

## Drill template
1. Trigger pipeline (`/api/v2/sync/start`) and wait for completion.
2. Validate active snapshot version changed.
3. Rollback to previous snapshot version.
4. Re-query graph and verify version + node count consistency.

## Current status
- Automated functional tests: `DONE` (`uv run pytest -q`, `npm --prefix frontend run test`)
- Soak test runs: `DONE (mock mode)` (`50/50`, success `100%`)
- Rollback drills: `DONE (mock mode)` (`10/10`, success `100%`)
- Migration smoke: `DONE` (`uv run alembic upgrade head && uv run alembic downgrade -1 && uv run alembic upgrade head`)
- Browser E2E critical flows: `DONE` (`RUN_E2E=1 npm --prefix frontend run test:e2e`)
- Soak test runs: `DONE (live api mode)` (`50/50`, success `100%`)
- Rollback drills: `DONE (live service mode)` (`10/10`, success `100%`)

## Environment prerequisites
- PostgreSQL is reachable from local runtime (`DATABASE_HOST`, `DATABASE_PORT`).
- Test fixture data is loaded (stable repo count and snapshot history).
- External dependencies are mocked/stubbed for soak reliability.

## Additional notes
- Commands used:
  - `uv run python scripts/perf/soak_pipeline.py --runs 50 --mode mock`
  - `uv run python scripts/perf/rollback_drill.py --runs 10 --mode mock`
  - `uv run python scripts/perf/soak_pipeline.py --runs 50 --mode api --base-url http://127.0.0.1:8071`
  - `uv run python scripts/perf/rollback_drill.py --runs 10 --mode service`
  - `uv run alembic upgrade head && uv run alembic downgrade -1 && uv run alembic upgrade head`
  - `RUN_E2E=1 npm --prefix frontend run test:e2e`
