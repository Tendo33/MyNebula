# Pipeline and snapshot test coverage against real Postgres

## Goal

Move backend test coverage to where the business risk actually is. The suite is
not thin — 33 test files, 243 passing tests — but its coverage is inverted: the
module that owns the entire sync pipeline is at 4%, while deleted-in-this-epic
template utilities had dedicated test files.

This is finding group **C** of parent task
`.trellis/tasks/08-18-comprehensive-hardening`.

## Background

Coverage from the 2026-08-18 scan (`uv run pytest -q`, branch coverage on):

| Module | Statements | Coverage |
| --- | --- | --- |
| `application/services/sync_execution_service.py` | 442 | **4%** |
| `application/services/graph_snapshot_service.py` | 100 | 19% |
| `infrastructure/repositories/snapshot_repository.py` | 128 | 42% |
| `application/services/pipeline_service.py` | 218 | 52% |
| `application/services/sync_execution_support.py` | 87 | 40% |
| Total | 5,500 | 57% |

The whole suite finishes in ~3 seconds. That is the tell: nothing touches a
database. Every persistence path — transaction boundaries, foreign key
behaviour, `ON CONFLICT` handling, pgvector operators, advisory locks, the
partial indexes on `pipeline_runs` and `sync_tasks` — is asserted only against
mocks, or not at all.

CI already provisions `pgvector/pgvector:pg16` as a service and runs
`alembic upgrade head` before the tests. The infrastructure exists and is
unused.

## Requirements

### C1 — Cover the sync pipeline execution paths

Raise `sync_execution_service.py` from 4% to at least 60% line coverage, with
tests that assert behaviour rather than chase the number. Priority order:

1. `compute_embeddings_task` — chunk success, chunk failure, resume, status
   matrix. Sibling task `08-18-pipeline-correctness` introduces these behaviours
   and ships its own tests; this task extends them to the branches that task did
   not need, including the LLM-fallback path that seeds `ai_tags` from
   `repo.topics`.
2. `run_clustering_task` — the incremental branch and its fallback to full
   clustering, the "fewer than 5 embedded repos" early return, the
   missing-embedding repo reset, and `repo_count` recomputation.
3. `sync_stars_task` — new versus existing repo handling, the content-hash
   reprocess trigger, the truncated incremental fetch, unstarred detection under
   both `SYNC_DETECT_UNSTARRED_ON_INCREMENTAL` settings, and the
   `total_stars` / `synced_stars` accounting difference between incremental and
   full mode.

### C2 — Add a real-Postgres integration layer

- Introduce an integration test tier that runs against a live
  PostgreSQL + pgvector instance, marked so it can be selected or deselected.
- The tier must be skipped cleanly, not failed, when no database is reachable,
  so `uv run pytest -q` stays usable on a laptop without Docker running.
- CI must run the tier, since CI already has the service and the migration step.
- Schema for the tier comes from `alembic upgrade head`, not
  `metadata.create_all`, so migrations are exercised as part of the test setup
  rather than drifting from the models.

At minimum the integration tier must cover:

- A full pipeline run: stars → embedding → clustering → snapshot, with GitHub,
  embedding, and LLM providers faked but every database interaction real.
- The atomic cluster swap from `08-18-pipeline-correctness`, asserting the
  foreign key from `starred_repos.cluster_id` is never violated and that an
  interrupted swap leaves prior clusters intact.
- Snapshot persistence and activation, including the chunked node and edge
  inserts and the `ix_graph_snapshots_user_version` uniqueness constraint.
- The pgvector cosine-distance search path used by `POST /api/v2/repos/search`.
- The advisory-lock serialization in
  `SyncPipelineService._acquire_pipeline_creation_lock`, which is skipped
  entirely on non-PostgreSQL dialects and therefore never runs today.

### C3 — Keep the fast tier fast

- The default `uv run pytest -q` invocation must stay a seconds-scale unit run.
- Integration tests are additive; no existing unit test is converted into an
  integration test.

## Constraints

- Runs after `08-18-pipeline-correctness`, so tests encode corrected behaviour
  rather than freezing today's defects.
- Runs after `08-18-template-residue-removal`, so no effort goes into covering
  deleted modules.
- `pyproject.toml` already declares `integration` and `unit` markers under
  `[tool.pytest.ini_options]` with `--strict-markers`. Reuse them; do not invent
  a parallel marker vocabulary.
- No new production dependency. Test-only dependencies must go in the `test` or
  `dev` extra.
- Provider calls to GitHub, the embedding API, and the LLM API must stay faked.
  This task adds database realism, not network realism.

## Non-goals

- Not chasing a global coverage percentage target. `common_utils`-style
  coverage-for-its-own-sake is what this epic is removing.
- Not adding frontend tests; those belong to `08-18-frontend-hardening`.
- Not building a performance or load-test harness. `scripts/perf/` already
  exists for that and is out of scope.
- Not testing against a real GitHub account or a real embedding provider.

## Acceptance criteria

- [ ] `sync_execution_service.py` line coverage is at least 60%.
- [ ] `graph_snapshot_service.py` line coverage is at least 60%.
- [ ] `snapshot_repository.py` line coverage is at least 70%.
- [ ] An integration tier exists, is marked `integration`, and runs against
      PostgreSQL + pgvector with schema applied by `alembic upgrade head`.
- [ ] With no database reachable, `uv run pytest -q` passes with the integration
      tier skipped, and prints a skip reason that names the missing database.
- [ ] With a database reachable, `uv run pytest -q -m integration` passes.
- [ ] The default `uv run pytest -q` unit run stays under 15 seconds on a
      developer machine.
- [ ] An integration test proves the cluster swap is atomic under an injected
      mid-swap failure.
- [ ] An integration test exercises the pgvector cosine-distance search path.
- [ ] An integration test exercises the pipeline creation advisory lock.
- [ ] `.trellis/spec/backend/testing.md` and
      `.trellis/spec/shared/verification.md` document how to run each tier.

## Verification

Backend scope, plus the full-stack docs link check because spec files change.
Both tiers must be run and reported: the unit tier without a database, and the
integration tier with one.

## Progress notes (2026-08-18)

### C1 complete — all targets exceeded

| Module | Before | After | Target |
| --- | --- | --- | --- |
| `sync_execution_service.py` | 4% | **91%** | 60% |
| `graph_snapshot_service.py` | 19% | **98%** | 60% |
| `snapshot_repository.py` | 42% | **80%** | 70% |
| Backend total | 57% | **73%** | — |

New test files:

- `tests/core/test_sync_stars_task.py` (14 tests)
- `tests/core/test_clustering_task_branches.py` (7 tests)
- `tests/core/test_graph_snapshot_payload.py` (10 tests)
- `tests/core/test_snapshot_repository_persistence.py` (16 tests)

Plus, from `08-18-pipeline-correctness`:
`test_embedding_batch_retry.py` (6), `test_embedding_chunked_resume.py` (9),
`test_cluster_swap_atomicity.py` (5).

### C2 blocked on environment

The PostgreSQL + pgvector integration tier is **not started**. Docker Desktop is
installed at `/Applications/Docker.app` with a working client at
`/usr/local/bin/docker`, but the daemon is not running, and there is no local
`postgres`, `psql`, `podman`, or `colima`.

Deliberately not shipped as untested code: writing five integration test files
that have never executed against a database would put unverified tests into the
CI gate, which is worse than an honest gap. The tier is designed in
`design.md` and ready to implement once a database is reachable.

To unblock: start Docker Desktop, then `docker compose up -d db`.

## C2 complete (2026-08-18)

Integration tier built and **run against a real PostgreSQL 16 + pgvector**
container. 11 integration tests pass.

### Safety correction to `design.md`

The original design resolved `TEST_DATABASE_URL` → `DATABASE_URL` → split
`DATABASE_*` settings. **That order is unsafe and was changed.** This
repository's `.env` points `DATABASE_URL` at a live remote Neon database; the
tier runs `alembic upgrade head`, drops and recreates the `public` schema, and
truncates tables. Falling back to the application's own connection string would
have run destructive migrations against production.

Now: `TEST_DATABASE_URL` is required with no fallback, and a non-local host is
refused unless `MYNEBULA_ALLOW_REMOTE_TEST_DB=true`. Both the missing-variable
skip and the remote-host refusal were verified to fire with actionable messages.

### What the tier covers that a fake session cannot

- `tests/integration/test_pipeline_end_to_end.py`
  - full run stars → embedding → clustering → snapshot, ending in an activated
    snapshot with `total_nodes == 12`
  - chunked embedding: chunk 1 fails, chunk 2 still commits, re-run embeds only
    the remaining 10
  - **`pg_advisory_xact_lock` serialization** — two concurrent
    `create_pipeline_run` calls, exactly one wins. This branch returns early on
    non-PostgreSQL dialects and had therefore never executed in any test
  - no repo left pointing at a deleted cluster after the atomic swap
- `tests/integration/test_snapshot_and_search.py`
  - chunked node/edge inserts over the 1000-row batch boundary (1250 nodes)
  - `ix_graph_snapshots_user_version` duplicate rejection
  - activation demoting the previous active snapshot
  - consistency validator against real persisted counts
  - retention pruning protecting the active snapshot
  - **pgvector `<=>` cosine ordering**, the operator behind
    `POST /api/v2/repos/search`

Schema is applied by `alembic upgrade head`, verified by asserting that a
migration-only index (`ix_starred_repos_full_name_trgm`) exists.

Local run: `TEST_DATABASE_URL=... uv run pytest -q -m integration` → 11 passed.
