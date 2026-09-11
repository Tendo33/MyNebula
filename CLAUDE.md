# Claude Code Project Instructions

This file is Claude Code's root entrypoint for MyNebula.

## Read order

1. Start at [AGENTS.md](AGENTS.md)

## Claude-specific notes

- Use [AGENTS.md](AGENTS.md) as the shared project entrypoint.

## Project guardrails

- MyNebula turns GitHub Stars into a semantic knowledge graph; it is not a
  generic Python template.
- Preserve FastAPI `/api` and `/api/v2`, SQLAlchemy async,
  PostgreSQL/pgvector, Alembic, sync pipeline state, graph snapshots, and
  APScheduler boundaries.
- Preserve admin authentication, CSRF protection, trusted proxy rules, and
  `READ_ACCESS_MODE` demo/authenticated read behavior.
- Frontend work must preserve Dashboard, Data, Graph, and Settings flows,
  including React Query, GraphContext, shared search utilities, progressive
  edge loading, and Settings polling lifecycle.
- Frontend package management is pnpm with `frontend/pnpm-lock.yaml`.

## Commands

```bash
uv run pytest -q -m "not integration"
uv run ruff check src tests scripts alembic
pnpm --prefix frontend test
pnpm --prefix frontend lint
```

## Claude execution style

- State assumptions explicitly when they shape the solution.
- Keep diffs tightly scoped to the task.
- Match existing style even when you would normally choose differently.
- Visual work follows DESIGN.md and PRODUCT.md; do not introduce a second palette.
