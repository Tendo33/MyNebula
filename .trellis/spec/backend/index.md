# Backend Development Guidelines

## Overview

MyNebula uses FastAPI + SQLAlchemy async + PostgreSQL/pgvector. The backend is
split into:

- `api` and `api/v2`: HTTP contracts, access control, settings/sync/data/graph
  routes, and compatibility endpoints.
- `application/services`: sync execution, pipeline orchestration, snapshot
  building, ranking, dashboard/data query shaping, and related recommendations.
- `core`: config, auth, logging, scheduler, clustering, embeddings, LLM helpers,
  proxy trust, and runtime settings.
- `db`: ORM models, database session lifecycle, and migrations via Alembic.
- `domain`: sync and snapshot lifecycle enums.
- `infrastructure`: snapshot persistence repositories.
- `schemas`: common API schemas and `schemas/v2/*` aggregate responses.

The current runtime is effectively single-user. Many services resolve one
default user and scope data access by `user_id`. Any new backend work must
preserve that assumption unless the change explicitly widens the auth model
end-to-end.

## Actual Project Conventions

- Route handlers stay thin. They validate access, translate HTTP payloads, and
  delegate to service functions.
- Orchestration belongs in `application/services`, not in `api/*`.
- Long-running work is represented with `SyncTask` or `PipelineRun` rows and
  terminal statuses are persisted.
- Snapshot-backed reads are preferred for graph/dashboard/data views. Realtime
  rebuilding should stay exceptional.
- Admin write paths use session cookie + CSRF validation. Read paths may still
  run in `demo` mode depending on `READ_ACCESS_MODE`.
- `api/v2/access.py` is the explicit single-user access boundary for Settings,
  Sync, Data, and Graph style flows.
- Default user bootstrap has concurrency protection and conflict recovery; do
  not assume first access is naturally serial.
- Graph historical version lookup should return explicit errors for missing
  versions instead of silently falling back to active.

## Current Source Of Truth

- Startup and middleware: `src/nebula/main.py`
- Config model: `src/nebula/core/config.py`
- Sync pipeline lifecycle: `src/nebula/application/services/pipeline_service.py`
- Full refresh orchestration: `src/nebula/application/services/sync_ops_service.py`
- Sync execution helpers: `src/nebula/application/services/sync_execution_support.py`
- Scheduler behavior: `src/nebula/core/scheduler.py`
- Admin auth boundary: `src/nebula/api/v2/auth.py`, `src/nebula/core/auth.py`
- ORM model and naming conventions: `src/nebula/db/models.py`

## Non-Negotiable Rules

- Do not trust forwarded headers unless `TRUST_PROXY_HEADERS=true` and the
  request comes from `TRUSTED_PROXY_IPS`.
- Do not add new background flows that only report progress in memory; persist
  observable state in the database.
- Do not introduce route-level business logic that duplicates service
  orchestration.
- Do not hide partial failures. If a flow completes with degraded outcome,
  persist `partial_failed` or task metadata so the UI can show warning state.
- Scheduler advisory locks are PostgreSQL-only; avoid noisy errors in non-Postgres
  local environments.
