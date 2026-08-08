# Backend hardening and frontend refinement

## Goal

Make MyNebula's persisted sync pipeline recoverable across process failure,
remove the highest-risk backend security and data-growth defects, and refine
the existing frontend so core workflows remain responsive, accessible, and
recoverable without changing the product's single-user, snapshot-backed model.

The outcome is a production-harder release whose correctness can be proven by
tests and migration checks, not a visual rewrite or a new product architecture.

## Background and confirmed facts

- The authoritative project model is single-user, PostgreSQL/pgvector-backed,
  with persisted pipeline state and snapshot-backed graph reads.
- Pipeline and full-refresh records are persisted, but their execution uses
  process-local background tasks. After a restart, `pending`/`running` records
  have no heartbeat, lease, or startup reconciliation and can permanently
  block later work (`src/nebula/api/v2/sync.py:52`,
  `src/nebula/application/services/pipeline_service.py:429`,
  `src/nebula/application/services/sync_ops_service.py:468`,
  `src/nebula/main.py:29`).
- Snapshot versions have second-level precision and node/edge counts. A
  collision rewrites existing snapshot children despite the model describing
  snapshots as immutable (`src/nebula/application/services/graph_snapshot_service.py:328`,
  `src/nebula/infrastructure/repositories/snapshot_repository.py:51`).
- Snapshot payloads are inserted as full in-memory ORM collections, and no
  retention policy exists (`snapshot_repository.py:100`).
- Snapshot read exceptions can trigger an expensive live full-graph rebuild
  (`src/nebula/application/services/graph_query_service.py:107`).
- Login rate limiting performs separate count and insert operations and is not
  atomic under concurrent attempts (`src/nebula/api/v2/auth.py:153`).
- The public health response exposes the scheduler's last error text
  (`src/nebula/main.py:160`).
- Data search uses leading-wildcard `ILIKE` across multiple fields and casts
  arrays to text (`src/nebula/api/v2/data.py:125`); dashboard topic statistics
  repeatedly unnest all topics (`src/nebula/api/v2/dashboard.py:56`).
- The frontend progressively loads graph edges, but the monolithic graph
  context and `Graph2D` data conversion repeat broad work as pages arrive
  (`frontend/src/contexts/GraphContext.tsx:215`,
  `frontend/src/components/graph/Graph2D.tsx:141`).
- Canvas graph, timeline range selection, sidebar resizing, and sync progress
  modal have keyboard or dialog-semantics gaps.
- Current baseline checks pass: 221 backend tests, 57 frontend tests, Ruff,
  ESLint, TypeScript, and production frontend build. Critical execution and
  snapshot modules nevertheless have low coverage.
- Locked Python and frontend production dependencies both contain known
  vulnerability advisories and need compatible upgrades plus regression proof.
- The working tree contains Trellis upgrade changes. Product implementation
  must preserve them and must not mix or revert them accidentally.

## Requirements

### R1. Recoverable persisted jobs

- Pipeline and full-refresh activity must be based on a renewable lease, not
  status text alone.
- Startup must reconcile expired `pending`/`running` work into an explicit,
  retryable terminal state.
- A restart or worker loss must not permanently block manual or scheduled sync.
- Existing per-user concurrency protection must remain in force.

### R2. Immutable and bounded snapshots

- New snapshots must receive collision-resistant versions and must never
  mutate an already-persisted historical version.
- Snapshot activation must remain atomic from readers' perspective.
- Snapshot writes must avoid building an unbounded number of ORM objects in one
  transaction.
- Retention must preserve the active snapshot and a documented rollback/history
  window; cleanup must be safe and observable.

### R3. Predictable read failure behavior

- An absent first snapshot may be built under the existing rebuild lock.
- Corrupt, timed-out, or failed snapshot reads must not silently trigger an
  unlimited live graph rebuild.
- Clients must receive a stable retryable response while operators receive a
  useful diagnostic signal without sensitive data disclosure.

### R4. Authentication and operational security

- Login throttling must be atomic under concurrent requests and emit a useful
  `Retry-After` response.
- Public health output must expose state, not raw internal error messages.
- Session logout/revocation behavior, CSP defaults, Docker credentials,
  database exposure, image pinning, and dependency advisories must be hardened
  without exposing secrets or breaking local development.

### R5. Query and persistence performance

- Search and topic statistics must have database-native query/index strategies
  suitable for a growing starred-repository collection.
- Data pagination and aggregate metadata must avoid unnecessary sequential
  full-table operations.
- Index additions require migration coverage and plan-level `EXPLAIN` evidence
  or a documented reason when production-scale evidence is unavailable.

### R6. Frontend refinement

- Preserve the current visual language and information architecture; this is a
  refinement, not a redesign.
- Graph node processing and progressive edge loading must avoid repeated full
  graph conversion where possible.
- Graph state ownership must reduce unrelated consumer rerenders while keeping
  shared filters, URL state, and current public behavior.
- Core controls must be keyboard accessible and expose appropriate dialog,
  status, label, focus, and language semantics.
- Unknown routes must render a useful recovery page.
- Interrupted/retryable jobs must have a clear user-facing recovery path.

### R7. Compatibility and evidence

- Existing database data and active snapshot reads must survive migrations.
- Existing `/api/v2` clients remain compatible unless a change is explicitly
  documented and approved.
- README and Trellis specs must be synchronized with actual API and runtime
  behavior.
- CI must cover dependency auditing, migrations, the restart/recovery path,
  frontend tests/type checks/build, and a minimal core user journey where the
  environment permits.

## Acceptance Criteria

- [ ] Killing/restarting the application during a running pipeline does not
      leave future syncs permanently blocked; the prior run becomes visibly
      interrupted/retryable.
- [ ] Concurrent requests cannot create two active jobs for one user.
- [ ] Snapshot versions remain unique under same-second, same-size rebuilds,
      and an existing snapshot payload cannot be overwritten.
- [ ] Retention never deletes the active snapshot and has regression coverage.
- [ ] Snapshot read corruption/failure returns a bounded retryable error rather
      than launching an unbounded live build.
- [ ] Login throttle concurrency tests prove the configured limit cannot be
      bypassed by simultaneous attempts.
- [ ] `/health` contains no scheduler exception message or stack detail.
- [ ] Relevant migration upgrade tests pass against PostgreSQL with pgvector.
- [ ] Representative search/topic queries have an indexed or precomputed plan
      and retain existing response semantics.
- [ ] Progressive edge loading no longer rebuilds the stable node model for
      every edge page; this is covered by a focused test or instrumentation.
- [ ] Graph/timeline/resize/sync modal workflows are keyboard operable and
      expose correct accessible names, focus behavior, and live status.
- [ ] Unknown routes render a tested recovery page.
- [ ] Backend lint/tests, frontend lint/typecheck/tests/build, dependency audit,
      and documentation-contract checks pass, or any unavoidable advisory has
      a narrowly documented, time-bounded exception.
- [ ] No real secret, token, DSN, or credential-bearing response is committed
      or printed by tests and tooling.

## Out of scope

- Multi-user tenancy or role-based access control.
- A wholesale visual redesign or replacement of the graph library.
- Replacing PostgreSQL/pgvector.
- Introducing Redis/Celery merely as convention; a new worker owner is only
  justified if the database-backed lease design cannot meet the requirements.
- Deploying to production or changing live infrastructure credentials.

## Approved product and data decisions

- Deliver the work as one comprehensive refactor and one final change set,
  while retaining internal checkpoints and rollback points during execution.
- Automatic snapshot retention is authorized to preserve the active snapshot
  and the 30 most recent successful snapshots, and to delete only other
  snapshots older than 90 days. Deletion must remain user-scoped, transactional,
  observable, and covered by tests.

## Notes

- This is a complex task and requires `design.md` and `implement.md` before
  activation.
- The user selected one comprehensive refactor delivered as one change set.
  Implementation may still use internal checkpoints and rollback points, but
  there will be no separately released intermediate slices.
- Planning and implementation run inline in the main session; no subagents are
  permitted by project rules.
