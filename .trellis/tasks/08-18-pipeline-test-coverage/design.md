# Design — Pipeline and snapshot test coverage against real Postgres

## Scope boundary

Additive. New files under `tests/integration/`, extensions to existing files
under `tests/core/` and `tests/api/`, one new fixture module, and edits to
`pyproject.toml`, `.trellis/spec/backend/testing.md`, and
`.trellis/spec/shared/verification.md`. No production source file changes.

## Two-tier test architecture

### Tier 1 — unit (default, no database)

Everything that exists today, plus the new pure-logic coverage for
`sync_execution_service`. Uses fakes for the database session, as the existing
tests do. Stays seconds-scale so it remains the inner-loop command.

### Tier 2 — integration (opt-in, real database)

Marked `@pytest.mark.integration`. Talks to a real PostgreSQL with pgvector.
Providers stay faked.

### Selection

`pyproject.toml` already declares both markers with `--strict-markers`:

```toml
markers = [
    "slow: ...",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
]
```

Reuse them. Selection is `-m integration` and `-m "not integration"`. The
default `uv run pytest -q` runs both, with the integration tier auto-skipping
when no database is reachable — see the skip contract below.

Rationale for auto-skip rather than default-deselect: a developer who *does*
have `docker compose up -d db` running gets the stronger signal for free, and CI
does not need a special invocation. The cost is that "the tests passed" means
different things in different environments, which the skip reason must make
loud.

## Database provisioning

### Connection

A session-scoped fixture resolves the test database URL in this order:

1. `TEST_DATABASE_URL` if set.
2. `DATABASE_URL` if set.
3. The split `DATABASE_*` settings via `get_database_settings()`.

### Skip contract

The session fixture attempts one short-timeout connection. On failure it calls
`pytest.skip` with a reason naming the resolved host and database, for example:

```
integration tier skipped: cannot reach postgresql://mynebula@localhost:5432/mynebula_test
```

A silent skip is worse than a failure, because it reads as a pass. The reason
string is part of the acceptance criteria for that reason.

### Schema

Schema is applied by `alembic upgrade head` against the test database, not by
`Base.metadata.create_all`. This is a deliberate choice:

- CI already runs `alembic upgrade head`, so the tiers agree.
- `create_all` would silently mask migration drift — exactly the class of bug an
  integration tier should catch. Two migrations in this repo
  (`20260408_..._embedding_ann_index`, `20260808_1800_add_search_indexes`) create
  indexes that `create_all` would not produce, including the ivfflat cosine index
  the vector search path depends on.
- `CREATE EXTENSION IF NOT EXISTS vector` and `pg_trgm` are handled by
  `init_db()` and the migrations respectively.

### Isolation between tests

Per-test isolation uses an outer transaction that is rolled back at teardown,
binding the session to that connection. This is faster than truncating tables
and keeps tests order-independent.

Two cases cannot use the outer-transaction trick and get explicit
truncate-based cleanup instead:

- Tests asserting commit-visibility across separate sessions — notably the
  atomic cluster swap, whose whole point is what survives a rollback.
- Tests exercising `pg_advisory_xact_lock`, since the lock's scope is the
  transaction under test.

This split is the main complexity in the fixture layer and is called out here so
it is not rediscovered during implementation.

## Provider fakes

Three seams, all already injectable:

| Provider | Seam | Fake behaviour |
| --- | --- | --- |
| GitHub | `GitHubClient` async context manager | yields a fixed repo list; supports `stop_before` truncation |
| Embedding | `get_embedding_service()` | returns deterministic unit vectors of `EMBEDDING_DIMENSIONS`, with a controllable per-slice failure hook |
| LLM | `get_llm_service()` | returns fixed summaries and tags, with a controllable failure hook |

Deterministic embeddings matter: clustering assertions must be reproducible.
Vectors are derived from a hash of the repo full name so that semantically
"related" fixtures land near each other and cluster counts are stable.

## Coverage plan for `sync_execution_service.py`

4% → 60% is 442 statements → roughly 265 covered. The three entry points split
roughly 340 statements between them, so the target is reachable without
contorted tests.

| Path | Tier | Why |
| --- | --- | --- |
| `compute_embeddings_task` chunk success / failure / resume / status matrix | unit | pure control flow over a faked session; sibling task A ships the core cases, this task fills the branches |
| LLM-fallback seeding `ai_tags` from `repo.topics[:5]` | unit | branch reachable with a failing LLM fake |
| `run_clustering_task` early returns (no repos, <5 embedded) | unit | no persistence semantics involved |
| `run_clustering_task` incremental branch and its fallback | unit | numeric assertions on coordinate assignment |
| `run_clustering_task` atomic swap | integration | the property under test *is* transactional |
| `sync_stars_task` new/existing/reprocess/hash paths | unit | fakeable |
| `sync_stars_task` unstarred deletion under both settings | integration | large `NOT IN` and cascade behaviour are database-real |
| `total_stars` / `synced_stars` accounting per mode | unit | arithmetic |

## Integration test inventory

`tests/integration/`:

- `test_pipeline_end_to_end.py` — full run through
  `SyncPipelineService.start_pipeline`, asserting `PipelineRun` phase
  transitions, `SyncTask` rows per phase, and a `ready` activated snapshot.
- `test_cluster_swap_atomicity.py` — injected failure between delete and insert;
  asserts prior clusters and every `StarredRepo.cluster_id` survive.
- `test_snapshot_persistence.py` — chunked node/edge inserts, timeline payload,
  `ix_graph_snapshots_user_version` uniqueness, activation, and
  `prune_snapshots` retention with its `with_for_update` row lock.
- `test_vector_search.py` — `POST /api/v2/repos/search` end to end, asserting
  ordering by cosine distance and the active-snapshot node filter.
- `test_pipeline_advisory_lock.py` — two concurrent `create_pipeline_run` calls
  for one user; exactly one succeeds. This path returns early on non-PostgreSQL
  dialects, so it has never executed in the current suite.

## Interaction with sibling tasks

- `08-18-pipeline-correctness` (A) ships tests for the behaviours it introduces.
  This task does not duplicate them; it extends coverage to the branches A did
  not need and adds the integration tier A could not have.
- `08-18-delivery-hardening` (D) wires `-m integration` into the CI gate. This
  task defines the tier; D makes CI run it. The two must agree on the exact
  invocation, which is recorded in
  `.trellis/spec/shared/verification.md` by this task and consumed by D.

## Risks

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Auto-skip hides a broken integration tier in CI | Medium | False green | D adds an explicit `-m integration` CI step that must collect a non-zero test count; a zero-collection run fails the job |
| Alembic-based setup makes the tier slow | Medium | Developers stop running it | Migrations run once per session, not per test; per-test isolation is transaction rollback |
| Deterministic fake embeddings produce degenerate clusters | Medium | Flaky cluster-count assertions | Derive vectors from a name hash with controlled separation; assert cluster membership relationships, not exact cluster counts |
| Transaction-rollback isolation conflicts with code that calls `commit()` | High | Confusing failures | Explicitly documented above: commit-visibility and advisory-lock tests use truncate-based cleanup instead |
| Coverage target invites low-value tests | Medium | Maintenance burden, the exact problem this epic removes | The path table above is the contract; coverage is the by-product, not the goal |
