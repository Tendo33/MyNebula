# Claude Code Project Instructions

This file is Claude Code's root entrypoint for MyNebula. Keep it aligned with
AGENTS.md, but keep detailed project facts in `.trellis/spec/`.

## Read order

1. Start at [AGENTS.md](AGENTS.md)
2. Use [.trellis/spec/README.md](.trellis/spec/README.md) for the Trellis spec overview
3. Use [.trellis/spec/shared/index.md](.trellis/spec/shared/index.md) for repository-wide facts
4. Use [.trellis/spec/backend/index.md](.trellis/spec/backend/index.md) before backend work
5. Use [.trellis/spec/frontend/index.md](.trellis/spec/frontend/index.md) before frontend work
6. Run the relevant section in [.trellis/spec/shared/verification.md](.trellis/spec/shared/verification.md)

## Claude-specific notes

- Use [AGENTS.md](AGENTS.md) as the shared project entrypoint.
- Route task-specific work through `.trellis/spec/`.
- Do not reintroduce any parallel AI-docs tree; `.trellis/spec/` is the
  detailed project contract.
- Keep this file thin. If this file and `.trellis/spec/` disagree, update this
  file or follow the spec before changing code.

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
- Frontend package management is currently npm with `frontend/package-lock.json`;
  do not document or migrate it to pnpm in this docs migration.

## Claude execution style

- State assumptions explicitly when they shape the solution.
- Keep diffs tightly scoped to the task.
- Match existing style even when you would normally choose differently.
- Update `.trellis/spec/` when behavior, structure, scripts, public APIs, or
  verification commands change.
- Before declaring success, run the relevant commands in [.trellis/spec/shared/verification.md](.trellis/spec/shared/verification.md).
