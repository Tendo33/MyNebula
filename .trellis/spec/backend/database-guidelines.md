# Database Guidelines

## Overview

The project uses SQLAlchemy 2 async ORM with Alembic migrations and PostgreSQL extensions, including `pgvector`.

## Query Patterns

- Always scope user-owned records by `user_id`.
- Prefer batch reads over per-row ORM lookups in loops.
- For semantic search, keep database ordering on `embedding.cosine_distance(...)`; do not fetch all embeddings into Python for ranking.
- Snapshot-oriented reads should query the active snapshot tables instead of reconstructing graph state ad hoc.

## Transactions And Task State

- Background jobs must persist status transitions (`pending`, `running`, `completed`, `failed`, `partial_failed`, `interrupted`) in the database.
- Active orchestration owners (`PipelineRun` and `SyncTask.task_type ==
  "full_refresh"`) must own a non-expired lease, heartbeat, and unique
  per-execution fencing token in `worker_id`. Pipeline/full-refresh child
  `SyncTask` rows are progress records and do not claim independent leases.
  Renewal is one conditional update over id, active status, exact token, and
  unexpired lease. A worker that loses this update must stop at the next phase
  boundary and must not write terminal state for a newer owner.
- Startup and online status/launch reconciliation change expired active owners
  to retryable `interrupted`; status alone never proves activity.
- Launch serialization uses PostgreSQL advisory locks for full refresh creation, pipeline creation, and scheduler ticks.
- If an async workflow fails after partially updating state, the terminal error must still be committed so the UI can report it.

## Snapshot Lifecycle

- Snapshot versions are immutable and create-only; never replace payload rows
  when a version collides.
- Build and validate before atomic activation. Reads never reconstruct a live
  graph as a fallback for snapshot hydration failure.
- Retention protects the active snapshot and the newest 30 successful snapshots,
  and may delete only other user-scoped snapshots older than 90 days.
- Retention runs only after successful activation and must not touch repository,
  user, task, or current active data.

## Migrations

- Migrations live in `alembic/versions`.
- New migrations must be additive and idempotent where practical.
- pgvector schema changes should explicitly preserve the `vector` extension and any ANN index requirements.
- Semantic-search-related migrations must mention whether they change runtime query assumptions.
- Text search uses concurrently-created `pg_trgm` indexes. Topic/tag substring
  search uses native `unnest` semantics; ordinary array GIN indexes do not serve
  that query shape and must not be added without a matching operator/query-plan
  contract. Do not cast arrays to text for search.

## Scenario: Job leases, search indexes, and deployment migrations

### 1. Scope / Trigger

- Use this contract when changing persisted background-job ownership or search
  indexes. These changes cross model, service, migration, API status, and deploy
  boundaries.

### 2. Signatures

- Claim: `apply_job_lease(record) -> execution_token`.
- Renew: `renew_job_lease(kind, record_id, execution_token) -> bool`.
- Reconcile one: `interrupt_expired_job(db, kind, record_id, user_id=...)`.
- DB lease fields: `heartbeat_at timestamptz`, `lease_expires_at timestamptz`,
  `worker_id varchar(255)`.

### 3. Contracts

- `worker_id` is a fencing token, not a reusable process label.
- Only the exact unexpired owner may renew or advance phases.
- Search index creation is a dedicated Alembic revision using
  `autocommit_block()` plus `postgresql_concurrently=True`.
- `MYNEBULA_IMAGE` is required by Compose and names an already-published
  immutable tag or digest; source changes do not imply that image exists.

### 4. Validation & Error Matrix

- Token mismatch or expired lease -> `JobLeaseLostError`; no stale-owner write.
- Expired active row observed through status/launch -> persisted `interrupted`.
- Snapshot hydration or metadata persistence failure -> bounded API `503` with
  request ID; no live rebuild fallback.
- Missing `MYNEBULA_IMAGE` -> Compose configuration failure before startup.

### 5. Good/Base/Bad Cases

- Good: heartbeat renews the exact token and the terminal transition clears it.
- Base: restart reconciliation makes pre-lease/expired active rows retryable.
- Bad: checking only `status='running'`, reusing `host:pid` as ownership, or
  adding an array GIN index while querying `unnest(... ) ILIKE`.

### 6. Tests Required

- Assert unique claim tokens, conditional renewal SQL, lease-loss cancellation,
  online expiry transition, and terminal lease cleanup.
- Assert migration chain, concurrent trigram DDL, and absence of unused array
  GIN indexes; generate offline Alembic SQL.
- Assert Compose has no stale default image.

### 7. Wrong vs Correct

#### Wrong

```sql
UPDATE pipeline_runs SET heartbeat_at = now() WHERE id = :id;
```

#### Correct

```sql
UPDATE pipeline_runs
SET heartbeat_at = now(), lease_expires_at = :next_expiry
WHERE id = :id AND worker_id = :token
  AND status IN ('pending', 'running') AND lease_expires_at > now();
```

## Naming Conventions

- Table/index/constraint naming follows the SQLAlchemy naming convention in
  `src/nebula/db/models.py`.
- Composite user scoping indexes should start with `ix_<table>_user_...`.
- ANN indexes should include the embedding column and distance family in the name, for example `ix_starred_repos_embedding_cosine_ann`.

## Common Mistakes To Avoid

- Do not return unscoped records and filter them in Python.
- Do not materialize full candidate sets in memory when pgvector ordering can do the first-stage narrowing in SQL.
- Do not add migrations without a matching test or assertion for the expected query/index contract.
