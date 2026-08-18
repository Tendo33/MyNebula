# Verification

This file is MyNebula's canonical verification reference.

Current version example: `1.3.0`.

## Backend

```bash
uv sync --all-extras
uv run ruff check src tests scripts alembic
uv run ruff format --check src tests scripts alembic
uv run pytest -q -m "not integration"
uv export --frozen --all-extras --no-hashes --no-emit-project --format requirements-txt --output-file /tmp/mynebula-requirements.txt
uvx pip-audit -r /tmp/mynebula-requirements.txt --disable-pip --no-deps
```

### Integration tier (real PostgreSQL + pgvector)

`TEST_DATABASE_URL` is **required** and must point at a disposable database.
There is deliberately no fallback to `DATABASE_URL`: this tier runs
`alembic upgrade head`, drops and recreates the `public` schema, and truncates
tables. A developer `.env` may point `DATABASE_URL` at a real remote database,
so falling back to it would run destructive migrations against production. A
non-local host is refused unless `MYNEBULA_ALLOW_REMOTE_TEST_DB=true`.

```bash
docker run -d --name mynebula-test-db \
  -e POSTGRES_USER=mynebula_test -e POSTGRES_PASSWORD=mynebula_test \
  -e POSTGRES_DB=mynebula_test -p 127.0.0.1:5433:5432 pgvector/pgvector:pg16

export TEST_DATABASE_URL=postgresql://mynebula_test:mynebula_test@127.0.0.1:5433/mynebula_test
uv run pytest -q -m integration
```

Without `TEST_DATABASE_URL` the tier skips and prints those commands. Schema
comes from Alembic, never `metadata.create_all`: two migrations create indexes
the models alone would not produce (the ivfflat cosine index behind vector
search, and the pg_trgm GIN indexes behind Data-page search), so building from
models would hide migration drift — the exact class of bug this tier exists to
catch.

## Frontend

```bash
pnpm --prefix frontend run lint
pnpm --prefix frontend exec tsc --noEmit -p tsconfig.json
pnpm --prefix frontend run test
pnpm --prefix frontend run build
pnpm --prefix frontend audit --prod
```

### End-to-end (Playwright)

The suite is gated behind `RUN_E2E=1` and mocks its own network payloads, so no
backend is needed. `playwright.config.ts` starts a Vite server on 4173 itself.

```bash
node frontend/node_modules/@playwright/test/cli.js install --with-deps chromium
RUN_E2E=1 pnpm --prefix frontend run test:e2e
```

Point it at an already-running app instead with `E2E_BASE_URL`, which disables
the managed server.

## Container

Assert against the built image; `docker build` succeeding proves neither that
the copied virtualenv works nor that the toolchain is gone.

```bash
docker build --tag mynebula:ci .
test "$(docker run --rm mynebula:ci sh -c 'command -v gcc || echo absent')" = absent
test "$(docker run --rm mynebula:ci sh -c 'command -v g++ || echo absent')" = absent
docker run --rm mynebula:ci sh -c 'command -v curl'
test "$(docker run --rm mynebula:ci id -un)" = appuser
docker run --rm mynebula:ci python -c 'import nebula.main'
```

## Full Stack

```bash
uv sync --all-extras
uv run ruff check src tests scripts alembic
uv run ruff format --check src tests scripts alembic
uv run pytest -q
uv export --frozen --all-extras --no-hashes --no-emit-project --format requirements-txt --output-file /tmp/mynebula-requirements.txt
uvx pip-audit -r /tmp/mynebula-requirements.txt --disable-pip --no-deps
pnpm --prefix frontend run lint
pnpm --prefix frontend exec tsc --noEmit -p tsconfig.json
pnpm --prefix frontend run test
pnpm --prefix frontend run build
pnpm --prefix frontend audit --prod
```

## CI Gate

`.github/workflows/ci.yml` is the authority. Keep this section command-for-command
identical to it; a stale CI Gate section is worse than none, because it reads as
authoritative.

Jobs: `backend-lint` -> `backend-test`; `frontend-lint` -> `frontend-test` /
`frontend-build`; `frontend-build` -> `e2e`; and `container-build` gated on
`backend-test`, `frontend-test`, `frontend-build`.

```bash
# backend-lint
uv sync --frozen --extra dev
uv run ruff check src tests scripts alembic
uv run ruff format --check src tests scripts alembic

# backend-test (pgvector/pgvector:pg16 service on 5432)
uv sync --frozen --all-extras
uv run alembic upgrade head
uv run pytest -q -m "not integration"
TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/mynebula_test \
  uv run pytest -q -m integration
uv export --frozen --all-extras --no-hashes --no-emit-project --format requirements-txt --output-file /tmp/mynebula-requirements.txt
uvx pip-audit -r /tmp/mynebula-requirements.txt --disable-pip --no-deps

# frontend-lint / frontend-test / frontend-build
pnpm --prefix frontend exec tsc --noEmit
pnpm --prefix frontend run lint
pnpm --prefix frontend audit --prod
pnpm --prefix frontend run test
pnpm --prefix frontend run build

# e2e
node frontend/node_modules/@playwright/test/cli.js install --with-deps chromium
RUN_E2E=1 pnpm --prefix frontend run test:e2e

# container-build
docker build --tag mynebula:ci .
# plus the image assertions in the Container section above
```

The integration tier runs as its **own** CI step rather than folded into
`pytest -q`. It auto-skips when no database is reachable, and a skip buried in
the unit run would read as a pass; as a separate step, pytest's exit code 5 on
"no tests collected" fails the job.

## Documentation and Link Checks

```bash
python3 - <<'PY'
from pathlib import Path
import re
import sys

docs = [
    *Path(".trellis/spec").rglob("*.md"),
    *Path("doc").glob("*.md"),
    Path("README.md"),
    Path("README.zh.md"),
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
]
pattern = re.compile(r"\[[^\]]+\]\(([^)#]+)")
missing = []

for doc in docs:
    if not doc.exists():
        continue
    text = doc.read_text(encoding="utf-8")
    for rel in pattern.findall(text):
        if "://" in rel or rel.startswith("#") or rel.startswith("/"):
            continue
        target = (doc.parent / rel).resolve()
        if not target.exists():
            missing.append(f"{doc}: {rel}")

if missing:
    print("\n".join(missing))
    sys.exit(1)
PY
```

## Hotspot Regression Matrix

Graph / Search hotspots:

```bash
uv run pytest -q tests/api/test_v2_auth_access.py -k "repo_search"
pnpm --prefix frontend run test -- src/components/ui/__tests__/CommandPalette.test.tsx
pnpm --prefix frontend run test -- src/pages/__tests__/GraphPage.url-state.test.tsx
pnpm --prefix frontend run test -- src/features/data/hooks/useDataReposQuery.test.tsx
pnpm --prefix frontend exec tsc --noEmit -p tsconfig.json
```

Sync / Scheduler hotspots:

```bash
uv run pytest -q tests/core/test_scheduler_service.py
uv run pytest -q tests/core/test_pipeline_state_machine.py
uv run pytest -q tests/core/test_pipeline_service_orchestration.py
uv run pytest -q tests/core/test_embedding_chunked_resume.py tests/core/test_embedding_batch_retry.py
uv run pytest -q tests/core/test_cluster_swap_atomicity.py
uv run pytest -q tests/api/test_v2_sync_pipeline_api.py tests/api/test_v2_settings_routes.py
pnpm --prefix frontend run test -- src/pages/__tests__/Settings.partial-failed.test.tsx src/pages/__tests__/settings.polling.test.tsx
pnpm --prefix frontend exec tsc --noEmit -p tsconfig.json
```

Graph render hotspots (canvas painting, adjacency, edge paging):

```bash
pnpm --prefix frontend run test -- src/components/graph/__tests__/graph2dPainters.test.ts src/components/graph/__tests__/graph2dStyles.test.ts src/components/graph/__tests__/graph2dLayout.test.ts
pnpm --prefix frontend run test -- src/contexts/adjacencyIndex.test.ts src/contexts/__tests__/adjacencyIndexSharing.test.tsx
pnpm --prefix frontend run test -- src/features/graph/__tests__/edgeAutoLoadBudget.test.tsx src/features/graph/__tests__/useGraphEdgesInfiniteQuery.test.tsx
```

Contract guards (fail on drift, not on taste):

```bash
pnpm --prefix frontend run test -- src/locales/zh/translation.test.ts
pnpm --prefix frontend run test -- src/__tests__/accessibility.baseline.test.ts
```

## Rule

- Backend-only changes run backend checks.
- Frontend-only changes run frontend checks.
- Cross-boundary, scripts, or docs changes run full stack.
- Hotspot changes run the relevant matrix in addition to broad checks.
