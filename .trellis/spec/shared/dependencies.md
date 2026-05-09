# Dependencies

Use this when adding or updating MyNebula dependencies.

## Baseline

| Area | Tooling |
| --- | --- |
| Python runtime | Python 3.10+ |
| Python package manager | `uv` |
| Python quality | `ruff`, `pytest`, optional `mypy` where configured |
| Python API | FastAPI |
| Python persistence | SQLAlchemy async, PostgreSQL, pgvector, Alembic |
| Background jobs | APScheduler |
| AI and data processing | OpenAI-compatible clients, NumPy, scikit-learn |
| GitHub integration | GitHub API / GraphQL |
| Frontend package manager | npm with `frontend/package-lock.json` |
| Frontend runtime | React, TypeScript, Vite, Tailwind CSS 4 |
| Frontend data/graph | React Query, react-force-graph-2d, Zustand |
| Frontend testing | Vitest, Testing Library, Playwright |

## Rules

- Check existing dependencies before adding a new one.
- Keep frontend package management on npm for this repository unless a separate
  migration is requested.
- Do not introduce `pnpm-lock.yaml` into `frontend/` during this docs migration.
- Do not expose backend-only secrets through `VITE_*` variables.
- Update specs and verification commands when a dependency changes project
  setup, build, or runtime behavior.

## Search Before Adding

```bash
rg "\"dependency-name\"" pyproject.toml frontend/package.json
rg "from dependency_name|import dependency_name" src tests scripts
rg "dependency-name" frontend/src
```
