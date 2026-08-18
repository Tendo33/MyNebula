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
- Sync stage execution: `src/nebula/application/services/sync_execution_service.py`
- Scheduler behavior: `src/nebula/core/scheduler.py`
- Admin auth boundary: `src/nebula/api/v2/auth.py`, `src/nebula/core/auth.py`
- ORM model and naming conventions: `src/nebula/db/models.py`

## Pipeline Stage Contracts

These are behavioral contracts, not implementation details. Changing them
changes what a user loses when something fails mid-run.

### Embedding stage is chunked and resumable

- `compute_embeddings_task` processes repos in `SYNC_BATCH_SIZE` chunks and
  commits each chunk before starting the next.
- `StarredRepo.is_embedded` is the durable resume marker. A re-run must embed
  only repos still marked `False`.
- Chunk iteration uses a keyset cursor on `StarredRepo.id` and advances the
  cursor **before** processing. A chunk that keeps failing must not be
  re-selected within the same pass, or the loop never terminates.
- A failing chunk is rolled back, added to `failed_items`, and skipped. It must
  not abort the remaining chunks.
- Status matrix: any chunk succeeded → `completed` with `failed_items` carrying
  the partial failure; zero chunks succeeded with at least one failure →
  `failed`; nothing to embed → `completed`.
- `EmbeddingService.embed_batch` retries each provider request individually.
  Do not wrap it in an outer retry: that re-sends every slice that already
  succeeded and re-bills a metered API.

### Cluster swap is atomic

- `run_clustering_task` must complete all expensive side-effect-free work —
  clustering, projection, LLM naming, deduplication — **before** any
  destructive statement.
- Delete-old, insert-new, and reassign land in a single transaction. An
  interrupted run must leave the previous clusters intact.
- Detach repos (`UPDATE starred_repos SET cluster_id = NULL`) before deleting
  clusters, so the `starred_repos.cluster_id → clusters.id` foreign key stays
  satisfied.
- The bulk `UPDATE` bypasses the identity map. Realign loaded ORM objects
  (`repo.cluster_id = None`) before assigning new ids, or an assignment that
  matches the stale in-session value emits no `UPDATE` and silently leaves the
  row `NULL`.
- Both statements are set-based. Do not reintroduce per-row ORM deletes: each
  one issues its own child-nullification query.

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
