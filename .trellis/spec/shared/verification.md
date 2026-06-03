# Verification

This file is MyNebula's canonical verification reference.

Current version example: `1.2.11`.

## Backend

```bash
uv sync --all-extras
uv run ruff check src tests scripts alembic
uv run ruff format --check src tests scripts alembic
uv run pytest -q
```

## Frontend

```bash
pnpm --prefix frontend run lint
pnpm --prefix frontend exec tsc --noEmit -p tsconfig.json
pnpm --prefix frontend run test
pnpm --prefix frontend run build
```

## Full Stack

```bash
uv sync --all-extras
uv run ruff check src tests scripts alembic
uv run ruff format --check src tests scripts alembic
uv run pytest -q
pnpm --prefix frontend run lint
pnpm --prefix frontend exec tsc --noEmit -p tsconfig.json
pnpm --prefix frontend run test
pnpm --prefix frontend run build
```

## CI Gate

GitHub Actions are the authority. The local equivalent is:

```bash
uv sync --frozen --extra dev
uv run ruff check src/
uv run ruff format --check src/
uv sync --frozen --all-extras
uv run pytest -q
pnpm --prefix frontend exec tsc --noEmit
pnpm --prefix frontend run lint
pnpm --prefix frontend run test
pnpm --prefix frontend run build
```

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
uv run pytest -q tests/api/test_v2_sync_pipeline_api.py tests/api/test_v2_settings_routes.py
pnpm --prefix frontend run test -- src/pages/__tests__/Settings.partial-failed.test.tsx src/pages/__tests__/settings.polling.test.tsx
pnpm --prefix frontend exec tsc --noEmit -p tsconfig.json
```

## Rule

- Backend-only changes run backend checks.
- Frontend-only changes run frontend checks.
- Cross-boundary, scripts, or docs changes run full stack.
- Hotspot changes run the relevant matrix in addition to broad checks.
