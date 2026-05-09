# MyNebula Shared Index

MyNebula is a concrete product repository, not a generic template.

```text
src/nebula/             # FastAPI, sync pipeline, graph snapshots, auth, DB
frontend/               # React + Vite graph/data/admin frontend
tests/                  # Python tests
scripts/                # Reset, eval, perf, version, maintenance helpers
alembic/                # Database migrations
.trellis/spec/          # Current facts, standards, and references
```

## Source of Truth

- Start with `.trellis/spec/README.md`.
- Use `backend/index.md` before backend work.
- Use `frontend/index.md` before frontend work.
- Use `shared/verification.md` before claiming completion.
- Use `guides/pre-implementation-checklist.md` for non-trivial changes.

## Core Rules

- No untyped public Python APIs.
- No frontend `any` in new TypeScript code.
- No secrets in logs or Vite public environment variables.
- Do not import Python modules through `src.nebula...`.
- Do not document future SDK, multi-user auth, or CLI product surfaces as
  current implementation.
- Keep frontend and backend contracts explicit. If the frontend calls a backend
  endpoint, document request, response, error, auth, and cache assumptions.
- Frontend uses npm and `frontend/package-lock.json`.

## Documentation Files

| File | Description | When to Read |
| --- | --- | --- |
| [code-quality.md](./code-quality.md) | Mandatory quality rules | Always |
| [dependencies.md](./dependencies.md) | Stack and dependency constraints | Adding or updating dependencies |
| [project-docs.md](./project-docs.md) | Trellis spec conventions | Changing docs or project structure |
| [verification.md](./verification.md) | Verification commands and hotspot matrix | Before completion |
