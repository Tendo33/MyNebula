# Backend Directory Structure

## Current Layout

```text
src/nebula/
├── api/                 # FastAPI routes, dependencies, v1/v2 boundaries
├── application/         # Service orchestration and use-case flows
├── core/                # Config, auth, scheduler, embeddings, clustering
├── db/                  # ORM models, sessions, database lifecycle
├── domain/              # Sync/snapshot lifecycle types
├── infrastructure/      # Snapshot repositories and persistence adapters
├── jobs/                # Background job entrypoints
├── models/              # Legacy/internal models
├── schemas/             # API schemas and v2 aggregate responses
└── utils/               # Shared utilities
```

## Placement Rules

| New thing | Default location |
| --- | --- |
| API route or HTTP dependency | `src/nebula/api/` |
| v2 route/schema contract | `src/nebula/api/v2/` or `src/nebula/schemas/v2/` |
| Sync, pipeline, dashboard, graph, data orchestration | `src/nebula/application/services/` |
| Config/auth/scheduler/embedding/clustering helper | `src/nebula/core/` |
| ORM model or DB session lifecycle | `src/nebula/db/` |
| Snapshot persistence adapter | `src/nebula/infrastructure/` |
| Shared utility with 2+ real users | `src/nebula/utils/` |

## Rules

- Route handlers validate access and translate HTTP; they do not own business
  orchestration.
- Do not add new service modules that duplicate existing sync/pipeline/data
  services.
- Scope user-owned records by `user_id`.
- Persist background status transitions.
