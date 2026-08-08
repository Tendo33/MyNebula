# Backend hardening and frontend refinement — implementation plan

## Plan basis

- PRD: `prd.md`
- Design: `design.md`
- Required baselines: `.trellis/spec/shared/*`, `.trellis/spec/backend/*`,
  `.trellis/spec/frontend/*`, and `.trellis/spec/shared/verification.md`
- Execution mode: Codex inline; no subagents
- Delivery: one comprehensive change set with internal rollback checkpoints

## Execution result

- Completed: database-backed leases/heartbeat/reconciliation for pipeline and
  full-refresh jobs, including unique execution fencing tokens, conditional
  renewal, online expiry reconciliation, and retryable `interrupted` state.
- Completed: immutable collision-resistant snapshots, bounded bulk persistence,
  active/latest-30/90-day retention, activation/retention serialization, and
  bounded snapshot-read 503 responses without live fallback.
- Completed: atomic login-attempt reservation, database-backed session
  revocation, health/CSP/container hardening, and dependency remediation.
- Completed: PostgreSQL-native array search SQL, consolidated aggregates,
  additive lease/auth-state SQL, and a dedicated concurrent trigram-index
  migration. Unused array GIN indexes were deliberately removed.
- Completed: graph node/edge memoization, stable node layout across edge pages,
  interrupted-job UX, keyboard/dialog/form/language semantics, and route recovery.
- Completed: README/Trellis contract sync and CI migration/audit/container gates.
- Verified locally: Ruff check/format, 243 backend tests, TypeScript, ESLint,
  62 frontend tests, production build, both production dependency audits, offline
  Alembic SQL, documentation links, YAML parsing, and desktop/mobile theme review.
- Runtime boundary: this workstation has no Docker or local PostgreSQL binaries,
  so fresh-database migration, container build, and runtime `EXPLAIN` remain CI/
  deployment checks. The configured external Neon database was read only and was
  deliberately not upgraded by this refactor.

## TDD route

- Mode: hybrid
- Decision: the initial broad refactor used regression tests alongside changes;
  review-discovered lease/auth/snapshot/focus/index defects used strict red/green
  tests before production repair.
- Strict authority: not applicable
- Test posture: focused diagnostic/regression tests alongside each behavior,
  followed by hotspot and full-stack verification
- Reason: strict TDD was applied where review produced falsifiable regressions;
  the whole historical refactor was not retroactively represented as test-first

## Compatibility and change necessity

- Code change is necessary because configuration/docs alone cannot recover
  orphaned persisted jobs, make snapshots immutable, close the login race, or
  make pointer-only UI operable.
- Minimum owner boundary: existing models/migrations, application services,
  v2 schemas/routes, graph/settings frontend owners, deployment/CI/docs.
- Preserve single-user semantics, current graph/search behavior, `/api/v2`,
  current visual system, URL filters, and snapshot-backed reads.
- Do not introduce a second task engine, state owner, search grammar, frontend
  package manager, or design system.

## Execution checklist

### 0. Establish a protected baseline

- [ ] Record the exact pre-existing Trellis-upgrade worktree paths and avoid
      reverting or staging them as product work.
- [ ] Run targeted baseline tests for pipeline state, scheduler, snapshot reads,
      auth, Graph URL state, Settings polling, and progressive edge loading.
- [ ] Inspect Alembic heads and confirm one additive migration chain.

Rollback point: no product changes.

### 1. Persisted job recovery

- [ ] Add lease/heartbeat/worker fields and active partial indexes to ORM models
      and a new additive Alembic migration.
- [ ] Add typed job lifecycle helpers for claim, heartbeat, expiry, terminal
      transition, and startup reconciliation using existing service ownership.
- [ ] Integrate heartbeat cleanup into pipeline and full-refresh orchestration.
- [ ] Make activity queries ignore expired leases while preserving per-user
      advisory-lock serialization.
- [ ] Run startup reconciliation before scheduler start.
- [ ] Extend v2 schemas/status mapping with `interrupted` and retry metadata.
- [ ] Add regression tests for restart/orphan recovery, idempotent reconciliation,
      concurrent creation, heartbeat renewal, and terminal-state preservation.

Targeted verification:

```bash
uv run pytest -q tests/core/test_pipeline_state_machine.py tests/core/test_scheduler_service.py
uv run pytest -q tests/api/test_v2_sync_pipeline_api.py tests/api/test_v2_settings_routes.py
uv run ruff check src tests alembic
```

Rollback point: revert service usage while leaving additive nullable columns.

### 2. Immutable snapshots, bounded storage, and read failures

- [ ] Replace count/time snapshot IDs with collision-resistant immutable IDs.
- [ ] Make snapshot save create-only and use bounded bulk insertion.
- [ ] Preserve build → validate → atomic activate ordering.
- [ ] Implement the approved retention selection: active + latest 30 successful
      protected, only other snapshots older than 90 days deletable.
- [ ] Trigger retention only after successful activation; log aggregate results.
- [ ] Replace broad live-build fallback with a typed 503 contract and request ID.
- [ ] Remove the obsolete fallback config after all references/tests/docs migrate.
- [ ] Move timeline/topic aggregation toward database-native queries without
      changing response shapes.
- [ ] Add collision, immutability, chunking, retention-boundary, active-protection,
      hydration-failure, and atomic-activation regression tests.

Targeted verification:

```bash
uv run pytest -q tests -k "snapshot or graph_edges or timeline"
uv run ruff check src tests alembic
```

Rollback point: source rollback retains created snapshots; never run an inverse
cleanup that deletes user repository data.

### 3. Auth and operational security

- [ ] Refactor login throttling into one serialized reserve/check operation and
      add `Retry-After`.
- [ ] Add server-side session-version revocation with additive migration and
      preserve CSRF/session cookie protections.
- [ ] Remove raw scheduler error details from public health output.
- [ ] Add a compatible default CSP and regression tests for security headers.
- [ ] Harden Docker defaults: required password, loopback DB exposure, pinned
      configurable app image, non-root/no-new-privilege settings where valid.
- [ ] Upgrade Python dependencies to compatible fixed versions; document only
      unavoidable, time-bounded advisories.
- [ ] Add auth concurrency, logout revocation, health disclosure, proxy-header,
      CSRF, and generic-error tests.

Targeted verification:

```bash
uv run pytest -q tests/api/test_v2_auth_access.py tests -k "auth or csrf or health"
uv run ruff check src tests alembic
```

Rollback point: additive session version remains backward compatible; Docker
changes are independently revertible without touching database contents.

### 4. Search and database performance

- [ ] Add `pg_trgm` and user-scoped text/array indexes through Alembic without
      disturbing pgvector/ANN indexes.
- [ ] Remove array-to-text search casts and keep shared literal search semantics.
- [ ] Consolidate Dashboard topic aggregation and avoid duplicate unnest scans.
- [ ] Reduce Data API sequential aggregate queries without changing pagination
      or response fields.
- [ ] Add query semantics tests and PostgreSQL migration/index assertions.
- [ ] Capture `EXPLAIN` evidence when a local PostgreSQL runtime is available;
      otherwise record the unverified runtime boundary without claiming speedup.

Targeted verification:

```bash
uv run pytest -q tests/api/test_v2_auth_access.py -k repo_search
uv run pytest -q tests -k "dashboard or data or migration or index"
```

Rollback point: drop only newly added indexes in down migration; never mutate
repository content.

### 5. Frontend state and rendering refinement

- [ ] Separate stable node normalization/indexing from staged edge updates while
      preserving the exported GraphContext contract.
- [ ] Prevent edge-page arrival from resetting node physics/layout unnecessarily.
- [ ] Add focused tests or counters proving stable nodes are not reconverted for
      each edge page.
- [ ] Add `interrupted` job state and retry UI while preserving `partial_failed`.
- [ ] Refactor large components only where extraction creates a real owner:
      graph renderer/interaction and settings sync controller; do not perform
      cosmetic file splitting.

Targeted verification:

```bash
pnpm --prefix frontend run test -- src/pages/__tests__/GraphPage.url-state.test.tsx
pnpm --prefix frontend run test -- src/pages/__tests__/Settings.partial-failed.test.tsx src/pages/__tests__/settings.polling.test.tsx
pnpm --prefix frontend exec tsc --noEmit -p tsconfig.json
```

Rollback point: GraphContext facade allows internal optimizations to revert
without rewriting callers.

### 6. Frontend accessibility and route recovery

- [ ] Add skip navigation, correct page heading hierarchy, and synchronized
      document language.
- [ ] Convert SyncProgress to an accessible modal with focus lifecycle and live
      progress status.
- [ ] Add keyboard/value semantics to timeline selection and sidebar resizing.
- [ ] Provide a keyboard-accessible graph repository alternative.
- [ ] Associate Settings/Data form labels and errors with controls.
- [ ] Add tested `*` route recovery while preserving API/static fallback rules.
- [ ] Verify mobile and desktop layouts and both themes for changed surfaces.

Targeted verification:

```bash
pnpm --prefix frontend run test
pnpm --prefix frontend run lint
pnpm --prefix frontend exec tsc --noEmit -p tsconfig.json
pnpm --prefix frontend run build
```

Rollback point: interaction changes are component-local and must preserve
mouse/touch behavior while adding keyboard behavior.

### 7. Dependency, CI, contract, and documentation closure

- [ ] Upgrade frontend dependencies to compatible fixed versions and re-run the
      full frontend gate.
- [ ] Add production dependency audits to CI with narrowly documented exception
      handling; do not ignore the command globally.
- [ ] Add PostgreSQL migration verification, static/API fallback regression,
      Docker build, and minimal browser/core-flow CI coverage where feasible.
- [ ] Correct README v2 endpoints and legacy-route claims.
- [ ] Update `.trellis/spec` for job leases, snapshot retention/error behavior,
      security, frontend accessibility, and new verification commands.
- [ ] Run lingering-reference checks for retired fallback config, status-only
      active queries, old API docs, and raw health errors.

### 8. Full integration and completion evidence

- [ ] Run Alembic upgrade against a fresh PostgreSQL/pgvector database and, when
      safely available, an existing-data fixture; verify no source data loss.
- [ ] Exercise restart during an active job and prove a subsequent job can run.
- [ ] Exercise snapshot activation plus retention boundaries.
- [ ] Run dependency audits and classify remaining results by reachable path.
- [ ] Run canonical full-stack verification:

```bash
uv sync --all-extras
uv run ruff check src tests scripts alembic
uv run ruff format --check src tests scripts alembic
uv run pytest -q
pnpm --prefix frontend run lint
pnpm --prefix frontend exec tsc --noEmit -p tsconfig.json
pnpm --prefix frontend run test
pnpm --prefix frontend run build
```

- [ ] Review the complete diff for secret exposure, accidental Trellis-upgrade
      mixing, migration reversibility, API compatibility, and stale/dead paths.
- [ ] Do not claim Docker, PostgreSQL plan, browser, or production proof when the
      corresponding runtime is unavailable; report the residual boundary.

## Architecture and review gates

- After lifecycle work: no status-only active owner remains.
- After snapshot work: no create path can overwrite a historical version and no
  read error can launch an unbounded live build.
- After frontend work: GraphContext public behavior, shared search, URL filters,
  progressive-edge threshold, and `partial_failed` remain intact.
- Before completion: specs and README describe only behavior that exists and
  the full-stack gate has fresh output.

## Stop/rewind rules

- Re-enter planning if implementation requires multi-user auth, a new task
  broker, an API version break, graph-library replacement, or destructive data
  cleanup beyond the approved snapshot policy.
- Stop for scoped confirmation before deleting any persistent data other than
  the exact approved historical snapshot set.
- Rewind a slice if its targeted regression tests fail or if compatibility
  requires a second active owner/fallback rather than repairing the canonical
  owner.
