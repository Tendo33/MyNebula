# MyNebula Trellis Spec

MyNebula turns GitHub Stars into a semantic knowledge graph. It is a real
full-stack app with FastAPI, PostgreSQL/pgvector, graph snapshots, sync
pipelines, scheduling, admin controls, and a React + Vite frontend.

## Structure

### [Backend](./backend/index.md)

MyNebula backend implementation rules:

- [Directory Structure](./backend/directory-structure.md)
- [Database Guidelines](./backend/database-guidelines.md)
- [Error Handling](./backend/error-handling.md)
- [Logging Guidelines](./backend/logging-guidelines.md)
- [Quality Guidelines](./backend/quality-guidelines.md)
- [HTTP API](./backend/http-api-when-added.md)
- [Testing](./backend/testing.md)

### [Frontend](./frontend/index.md)

MyNebula frontend implementation rules:

- [Directory Structure](./frontend/directory-structure.md)
- [Query and Filtering](./frontend/query-and-filtering.md)
- [DESIGN.md Workflow](./frontend/design-md.md)
- [Components](./frontend/components.md)
- [Quality](./frontend/quality.md)

### [Shared](./shared/index.md)

Cross-cutting rules:

- [Code Quality](./shared/code-quality.md)
- [Dependencies](./shared/dependencies.md)
- [Project Docs](./shared/project-docs.md)
- [Verification](./shared/verification.md)

### [Guides](./guides/index.md)

Thinking and handoff guides:

- [Code Reuse Thinking Guide](./guides/code-reuse-thinking-guide.md)
- [Cross-Layer Thinking Guide](./guides/cross-layer-thinking-guide.md)
- [Pre-Implementation Checklist](./guides/pre-implementation-checklist.md)
- [Review Checklist](./guides/review-checklist.md)

## Read Order

1. `shared/index.md`
2. `backend/index.md` before backend work
3. `frontend/index.md` before frontend work
4. `guides/pre-implementation-checklist.md` before non-trivial changes
5. `shared/verification.md` before claiming completion

## Baseline Stack

- Python 3.10+
- `uv`
- FastAPI
- SQLAlchemy async
- PostgreSQL + pgvector
- Alembic
- APScheduler
- OpenAI-compatible embedding and LLM providers
- GitHub API / GraphQL
- React + TypeScript + Vite
- Tailwind CSS 4
- React Query
- react-force-graph-2d
- Zustand
- i18n
- Frontend package manager: pnpm with `frontend/pnpm-lock.yaml`

## Project Bias

- Snapshot-backed reads are preferred for graph, dashboard, and data views.
- Long-running sync/pipeline state is persisted in database rows, not memory.
- Current runtime is effectively single-user unless a task explicitly widens the
  auth model end to end.
- `READ_ACCESS_MODE=demo` allows anonymous read paths; authenticated mode
  requires a valid admin session.
- Do not trust forwarded headers unless `TRUST_PROXY_HEADERS=true` and the
  request source matches `TRUSTED_PROXY_IPS`.
