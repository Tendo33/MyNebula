# Backend hardening and frontend refinement — design

## 1. Design intent

This refactor keeps MyNebula's current product architecture: one effective
user, FastAPI + SQLAlchemy async + PostgreSQL/pgvector, APScheduler, persisted
pipeline state, immutable graph snapshots, React Query, and a graph-only full
snapshot owner. The change repairs lifecycle ownership and performance at those
existing boundaries instead of introducing Redis, Celery, another API version,
or a new frontend design system.

First principle: a persisted job may outlive a web process, so database state —
not an in-memory task — decides whether work is active and recoverable.

## 2. Canonical owners and retirement decisions

| Concern | Canonical owner | Retired behavior |
| --- | --- | --- |
| Job activity and recovery | `PipelineRun` / `SyncTask` lease fields plus application services | Status-only permanent activity |
| Startup reconciliation | Application lifecycle service called by `main.lifespan` | Orphan rows surviving indefinitely |
| Snapshot identity | Snapshot builder creates collision-resistant immutable ID | Second-level count-based version and overwrite-on-collision |
| Snapshot retention | Snapshot repository/service with configured policy | Unlimited accumulation |
| Snapshot read errors | Graph query service finite error contract | Generic exception → full live rebuild |
| Login throttling | Auth service transaction/advisory-lock boundary | Separate count/check/insert race |
| Graph query state | Existing GraphContext and graph feature hooks | Rebuilding stable node model for every edge page |
| Accessible interaction | Existing graph/layout/UI components | Pointer-only controls and visual-only progress |

Anti-entropy classification:

- Internal fallback and duplicate behavior: delete-first after regression proof.
- Public `/api/v2` contracts: compatibility boundary; additive fields only.
- Snapshot cleanup: approved persistent-data mutation with the exact policy in
  the PRD. No repository or user source data is deleted.
- Existing Alembic history is never rewritten or deleted.

## 3. Persisted job lifecycle

### 3.1 Schema

Add nullable/additive lifecycle columns to both `pipeline_runs` and
`sync_tasks`; leases are claimed only where the record independently represents
orchestration work (`PipelineRun` and full-refresh `SyncTask`):

- `heartbeat_at timestamptz`
- `lease_expires_at timestamptz`
- `worker_id varchar`

Add partial indexes for active user-scoped lookups. Existing rows remain valid
during migration. `interrupted` becomes an explicit terminal status understood
by schemas and frontend progress mapping.

### 3.2 Claim and heartbeat

The existing per-user PostgreSQL advisory lock remains the launch serializer.
Creation and claim set the initial lease in the same transaction. A small
service-owned heartbeat coroutine renews the lease while the orchestration is
alive. Each claim receives a unique execution fencing token. Renewal and phase
transitions require the exact token, active status, and an unexpired lease;
lease loss cancels orchestration and prevents an old worker from overwriting a
new owner. Cleanup cancels the heartbeat in `finally`.

Activity queries require both an active status and a non-expired lease. During
the deployment compatibility window, a newly-created active row must always
receive a lease before it can block work.

### 3.3 Reconciliation

After database initialization and before scheduler start, startup reconciliation
marks expired `pending`/`running` rows as `interrupted`, records a safe diagnostic
reason and completion timestamp, and leaves them retryable. Reconciliation is
idempotent and user-agnostic, but never modifies terminal rows.

The HTTP status response remains backward compatible and adds enough state for
the frontend to display interruption/retry semantics. It never reports a stale
run as actively progressing. Status and launch paths reconcile observed expired
owners online, so correctness does not depend on another process restart.

## 4. Snapshot lifecycle

### 4.1 Identity and writes

Snapshot versions use a collision-resistant value containing a UTC timestamp
with microseconds plus random/UUID entropy. `save_snapshot_payload` creates only;
a version conflict raises instead of deleting children.

Nodes and edges are inserted in bounded chunks through SQLAlchemy bulk inserts.
The snapshot stays non-active until payload validation succeeds; activation is
committed atomically after validation. A failed build cannot replace the active
snapshot.

### 4.2 Retention

After successful activation, a user-scoped retention operation:

1. Always protects the active snapshot.
2. Protects the 30 newest successful snapshots.
3. Selects only remaining snapshots older than 90 days.
4. Deletes selected snapshots and cascading payload rows transactionally.
5. Logs counts and oldest/newest timestamps without payload contents.

Retention has no broad delete command, no table truncation, and no effect on
repositories, users, tasks, or current active data.

### 4.3 Read errors

Initial no-snapshot creation remains under the existing advisory rebuild lock.
Historical-version misses remain explicit 404-style errors. Hydration failure,
database timeout, corruption, or validation failure no longer runs a full live
build. The service emits a typed unavailable error mapped to a stable 503 API
response with a request ID; detailed exception data remains server-side.

The old `SNAPSHOT_READ_FALLBACK_ON_ERROR` setting and branch are retired after
tests and documentation are migrated. No compatibility exception is retained
because the project baseline already states that realtime rebuild is exceptional
and historical reads must not silently fall back.

## 5. Authentication and operational security

Login attempts are reserved before password verification in one serialized
database boundary. PostgreSQL uses stable advisory transaction locks for both
IP and username buckets; the non-PostgreSQL test path uses the same service
contract without issuing PostgreSQL SQL. A successful login clears its buckets;
a rejected attempt remains counted. A blocked response contains `Retry-After`.

Public `/health` reports only component state. Scheduler error details remain in
logs. Logout increments or validates a server-side session version so a stolen
cookie can be revoked; existing cookies fail closed after the migration without
exposing credentials.

Set a conservative CSP compatible with the current same-origin SPA, API calls,
fonts/images, and graph canvas. Docker Compose requires an explicit database
password, binds the development database to loopback, requires the operator to
name a published immutable application image tag/digest, and adds production-safe
privilege restrictions where compatible.

Dependency upgrades stay within framework-compatible ranges. An advisory that
cannot be fixed without breaking FastAPI/React contracts receives a documented,
time-bounded exception rather than a forced incompatible major upgrade.

## 6. Database/query performance

Add `pg_trgm` indexes concurrently for the proven text-search fields. Retain
native `unnest` substring semantics for tags/topics instead of casting arrays to
text, but do not add ordinary array GIN indexes that this query shape cannot
use. Keep shared search semantics, including `stars:>N`, unchanged.

Dashboard topic counts are produced in one aggregate query or snapshot metadata
owner instead of two separate full unnests. Data-page counts and rows are
consolidated where SQL remains readable; cluster and snapshot metadata stay
lightweight. Offset pagination remains API-compatible for this refactor; cursor
pagination is deferred because it changes the public contract.

Timeline aggregation moves as much counting as practical into SQL while
preserving the existing response shape and capped repository samples.

## 7. Frontend refinement

### 7.1 State and graph rendering

Preserve `GraphContext`'s public shape initially, per project spec. Internally,
memoize stable node normalization independently from staged edge normalization.
Graph filter indexes rebuild only when node/snapshot data changes, not for each
edge page. If consumer profiling still shows broad rerenders, split internal
providers behind the same exported `useGraph` facade rather than forcing a
repository-wide caller rewrite.

`Graph2D` receives stable nodes and incrementally changing links. Physics reset
and auto-fit are tied to meaningful snapshot/node changes, not every edge batch.

### 7.2 Job recovery UX

API types and progress mapping add `interrupted`. Settings and sync progress show
a concise interruption explanation and retry action; they do not pretend an
expired job is still running. Existing `partial_failed` warning behavior remains.

### 7.3 Accessibility and route recovery

- `SyncProgress` becomes a labelled modal dialog with focus entry/return,
  Escape behavior where closing is allowed, and live progress announcements.
- Timeline ranges and sidebar resizing support keyboard input and expose value
  semantics.
- The graph provides a keyboard-accessible repository list/search alternative;
  canvas remains the visual exploration surface.
- Form labels/errors are programmatically associated.
- Language changes update `<html lang>`.
- Page headings and skip navigation are corrected.
- A tested catch-all route displays recovery links without interfering with the
  backend SPA fallback or API 404 behavior.

Visual work reuses existing semantic tokens, spacing, typography, and dark mode.
No visual rebrand or graph-library replacement is included.

## 8. API and compatibility contracts

- Existing `/api/v2` routes and required response fields remain.
- Status responses may add lease/interruption metadata; existing clients can
  ignore it.
- `interrupted` is a new terminal value and is documented across backend and
  frontend types.
- README references to nonexistent legacy `/api` routes are corrected to the
  actual v2-only implementation confirmed by tests.
- Unknown `/api/*` paths continue to return API errors, never SPA HTML.

## 9. Rollout and rollback

This is one final change set, implemented through internal checkpoints:

1. Additive schema and lifecycle behavior.
2. Snapshot/query/security hardening.
3. Frontend refinement.
4. Dependency, container, CI, and documentation closure.

Before any checkpoint proceeds, targeted tests pass. The final result is
accepted only after the full stack gate. Source rollback is a normal Git revert;
the additive columns and indexes may remain harmlessly if application rollback
is required. No down migration automatically restores snapshots deleted by the
approved retention policy, so retention is invoked only after new snapshot
activation and is separately tested before final enablement.

## 10. Rejected alternatives

- Celery/Redis or another queue: unnecessary new operational owner for the
  current single-user workload; database lease recovery is sufficient.
- Keep live snapshot fallback “for safety”: hides corruption and amplifies
  failure, contradicting the snapshot source of truth.
- Replace GraphContext and graph library wholesale: high regression cost with no
  requirement evidence.
- Restore undocumented legacy API routes: tests and current implementation are
  v2-only; documentation should follow the active contract.
