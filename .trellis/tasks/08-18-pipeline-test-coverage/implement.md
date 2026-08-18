# Implementation plan — Pipeline and snapshot test coverage against real Postgres

Ordering: child **3 of 5** in `.trellis/tasks/08-18-comprehensive-hardening`.
Requires `08-18-template-residue-removal` and `08-18-pipeline-correctness` to be
merged first.

## Step 0 — Baseline

- [ ] Record current per-module coverage:
      `uv run pytest -q 2>&1 | grep -E "sync_execution_service|graph_snapshot_service|snapshot_repository|TOTAL"`
- [ ] Record unit-tier wall time: `time uv run pytest -q`
- [ ] Confirm markers already exist in `pyproject.toml`
      (`integration`, `unit`, `--strict-markers`).

## Step 1 — Integration fixture layer

- [ ] Create `tests/integration/__init__.py` and `tests/integration/conftest.py`.
- [ ] Session fixture `integration_database_url`: resolve
      `TEST_DATABASE_URL` → `DATABASE_URL` → `get_database_settings()`.
- [ ] Session fixture `integration_engine`: attempt one short-timeout connection;
      on failure `pytest.skip` with a reason naming the resolved host and
      database name. Do not swallow the reason.
- [ ] Session fixture applying schema via `alembic upgrade head` against the
      resolved URL, once per session. Ensure the `vector` extension exists first,
      matching `init_db()`.
- [ ] Function fixture `integration_db`: outer transaction bound to a
      connection, session bound to it, rollback at teardown.
- [ ] Function fixture `integration_db_committed`: truncate-based cleanup, for
      the commit-visibility and advisory-lock tests called out in `design.md`.
- [ ] Auto-apply `pytest.mark.integration` to everything under
      `tests/integration/` via a `pytest_collection_modifyitems` hook, so no test
      can forget the marker.

**Gate 1:** with the database down, `uv run pytest -q` passes and the skip
reason is visible in `-rs` output. With `docker compose up -d db` running,
`uv run pytest -q -m integration` collects and passes a trivial smoke test.
Blocking — do not write real integration tests until both directions work.

## Step 2 — Provider fakes

- [ ] `tests/integration/fakes.py` with the three fakes from `design.md`.
- [ ] Embedding fake: deterministic unit vectors derived from a hash of
      `full_name`, dimensioned from `EMBEDDING_DIMENSIONS`, with an injectable
      per-slice failure hook.
- [ ] LLM fake: fixed summary and tags, with an injectable failure hook.
- [ ] GitHub fake: async context manager yielding a fixed repo list, honouring
      `stop_before`.
- [ ] A fixture that seeds a user plus N starred repos with controlled semantic
      grouping, so cluster-membership assertions are stable.

**Gate 2:** a smoke test drives `SyncPipelineService.start_pipeline` to
completion against the real database with all three fakes, and produces an
activated snapshot.

## Step 3 — Integration tests

Write in this order; each is independently valuable and independently
revertable.

- [ ] `test_pipeline_end_to_end.py` — phase transitions on `PipelineRun`, one
      `SyncTask` per phase, activated snapshot in `ready` status.
- [ ] `test_cluster_swap_atomicity.py` — inject a failure between delete and
      insert; assert prior `Cluster` rows and every `StarredRepo.cluster_id`
      survive unchanged. Uses `integration_db_committed`.
- [ ] `test_snapshot_persistence.py` — chunked node and edge inserts, timeline
      payload round-trip, `ix_graph_snapshots_user_version` uniqueness violation
      on duplicate version, activation, and `prune_snapshots` retention.
- [ ] `test_vector_search.py` — `POST /api/v2/repos/search` ordered by cosine
      distance, with and without an active snapshot filter.
- [ ] `test_pipeline_advisory_lock.py` — two concurrent `create_pipeline_run`
      calls for one user; exactly one succeeds, the other raises. Uses
      `integration_db_committed`.

**Gate 3:** `uv run pytest -q -m integration` green, and the same run with the
database stopped skips cleanly.

## Step 4 — Unit coverage for `sync_execution_service.py`

Follow the path table in `design.md`. Extend existing files where a natural home
exists (`tests/core/test_sync_execution_support.py`,
`tests/core/test_incremental_drift_guard.py`) rather than creating parallel
files.

- [ ] `compute_embeddings_task`: LLM-fallback `ai_tags` seeding from
      `repo.topics[:5]`; branches not already covered by
      `08-18-pipeline-correctness`.
- [ ] `run_clustering_task`: no-repos early return; fewer-than-5-embedded early
      return; missing-embedding repo reset clearing `cluster_id` and coords;
      incremental branch coordinate assignment; incremental fallback to full
      clustering when no positioned repos exist; `repo_count` recomputation.
- [ ] `sync_stars_task`: new repo creation; existing repo update; content-hash
      change triggering reprocess flags; truncated incremental fetch;
      unstarred detection with `SYNC_DETECT_UNSTARRED_ON_INCREMENTAL` both true
      and false; `total_stars` / `synced_stars` accounting per mode; missing
      `GITHUB_TOKEN` failure path.

**Gate 4:** `sync_execution_service.py` at 60% or better; unit tier still under
15 seconds.

## Step 5 — Spec and verification docs

- [ ] `.trellis/spec/backend/testing.md`: document the two tiers, the marker
      vocabulary, the skip contract, the `alembic upgrade head`-based schema
      decision, and when a test belongs in which tier.
- [ ] `.trellis/spec/shared/verification.md`: add the integration invocation to
      the Backend and Full Stack sections, and to the CI Gate section. Record the
      exact command string — `08-18-delivery-hardening` consumes it verbatim.

## Step 6 — Verify

```bash
uv run ruff check src tests scripts alembic
uv run ruff format --check src tests scripts alembic
uv run pytest -q                    # database down: integration skipped
docker compose up -d db
uv run alembic upgrade head
uv run pytest -q -m integration     # database up
uv run pytest -q                    # database up: everything runs
```

Then the documentation link check from
`.trellis/spec/shared/verification.md`.

Report both the with-database and without-database results. A single "tests
pass" without saying which tier ran is not an acceptable completion report for
this task.

## Review gates summary

- **Gate 1** — skip/run both directions work. Blocking.
- **Gate 2** — end-to-end pipeline reaches an activated snapshot. Blocking.
- **Gate 3** — integration tier green both with and without a database.
- **Gate 4** — coverage targets met, unit tier still fast.

## Rollback points

- Steps 1–3 are purely additive: delete `tests/integration/` to revert.
- Step 4 extends existing test files; revert per file.
- Step 5 is documentation; revert independently.
- No production code changes, so no runtime rollback risk exists for this task.
