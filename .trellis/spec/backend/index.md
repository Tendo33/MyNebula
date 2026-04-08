# Backend Development Guidelines

## Overview

MyNebula uses FastAPI + SQLAlchemy async + PostgreSQL/pgvector. The backend is split into:

- `api/v2`: HTTP contracts and access control
- `application/services`: orchestration, ranking, snapshot building
- `core`: config, auth, scheduler, clustering, embeddings
- `db`: ORM models and session lifecycle

This project is effectively single-user in its current runtime model. Many services resolve one default user and then scope all data access by `user_id`. Any new backend work must preserve that assumption unless the change explicitly widens the auth model end-to-end.

## Actual Project Conventions

- Route handlers stay thin. They validate access, translate HTTP payloads, and delegate to service functions.
- Orchestration belongs in `application/services`, not in `api/*`.
- Long-running work is represented with `SyncTask` or `PipelineRun` rows and terminal statuses are persisted, not kept only in memory.
- Snapshot-backed reads are preferred for graph/dashboard style views. Realtime rebuilding should stay exceptional.
- Admin write paths use session cookie + CSRF validation. Read paths may still run in `demo` mode depending on `READ_ACCESS_MODE`.

## Current Source Of Truth

- Sync pipeline lifecycle: [`src/nebula/application/services/pipeline_service.py`](/Users/simonsun/.codex/worktrees/bcde/MyNebula/src/nebula/application/services/pipeline_service.py)
- Full refresh orchestration: [`src/nebula/application/services/sync_ops_service.py`](/Users/simonsun/.codex/worktrees/bcde/MyNebula/src/nebula/application/services/sync_ops_service.py)
- Scheduler behavior: [`src/nebula/core/scheduler.py`](/Users/simonsun/.codex/worktrees/bcde/MyNebula/src/nebula/core/scheduler.py)
- Admin auth boundary: [`src/nebula/api/v2/auth.py`](/Users/simonsun/.codex/worktrees/bcde/MyNebula/src/nebula/api/v2/auth.py), [`src/nebula/core/auth.py`](/Users/simonsun/.codex/worktrees/bcde/MyNebula/src/nebula/core/auth.py)

## Non-Negotiable Rules

- Do not trust forwarded headers unless `TRUST_PROXY_HEADERS=true` and the request comes from `TRUSTED_PROXY_IPS`.
- Do not add new background flows that only report progress in memory; persist observable state in the database.
- Do not introduce route-level business logic that duplicates service orchestration.
- Do not hide partial failures. If a flow completes with degraded outcome, persist that in `PipelineStatus.partial_failed` or task metadata.
