# Sync pipeline correctness and cost control

## Goal

Make the embedding and clustering stages of the sync pipeline safe and
resumable at real data volume, and make `SYNC_BATCH_SIZE` mean what the
documentation says it means. Today a single transient failure from a paid
embedding provider can cost 3× the tokens, discard an entire run's work, or
leave the cluster graph destroyed.

This is finding group **A** of parent task
`.trellis/tasks/08-18-comprehensive-hardening`.

## Background

MyNebula's target user is someone with a large, growing star collection. That is
exactly the regime where these four defects surface: they are invisible with 50
repos and severe with 5,000.

## Requirements

### A1 — Retry the embedding API call, not the whole batch loop

`src/nebula/core/embedding.py:99` applies
`@async_retry_decorator(max_retries=3, delay=1.0, backoff=2.0)` to `embed_batch`,
which internally loops over slices of `texts` and issues one
`client.embeddings.create` call per slice.

Consequence: if slice 40 of 100 fails, the decorator replays `embed_batch` from
slice 0. All 39 successful slices are re-sent to a metered API. With
`max_retries=3` and a persistent fault, the run bills up to 4× the successful
prefix before failing anyway.

Requirement: retry must wrap the individual provider call. A failing slice must
be retried in place, and slices that already succeeded must never be re-sent.

`src/nebula/core/llm.py:58` decorates `complete`, which is a single API call.
That usage is already correct and must not change.

### A2 — Make `SYNC_BATCH_SIZE` real

`SyncSettings.batch_size` is declared in `src/nebula/core/config.py:149` with a
validated range of 10–500 and a default of 100. `README.md` documents
`SYNC_BATCH_SIZE` under "Controls throughput and cost". No code reads it.
`sync_execution_service.py:488` passes a hardcoded `batch_size=32`.

Requirement: the embedding stage must derive its batch size from
`get_sync_settings().batch_size`. A user who sets `SYNC_BATCH_SIZE` must observe
a change in provider request sizing.

### A3 — Make the embedding stage incremental and resumable

`compute_embeddings_task` (`sync_execution_service.py:374`) currently:

1. loads every repo with `is_embedded == False` as full ORM objects, including
   `readme_content` up to `SYNC_README_MAX_LENGTH` (default 10,000 chars);
2. runs LLM enhancement across all of them;
3. builds every embedding text into one in-memory list;
4. issues one `embed_batch` call for the entire set;
5. on any exception sets `failed_items = len(repos)`, `status = "failed"`, and
   commits **no** embeddings.

Consequences: peak memory scales with the full un-embedded set times README
size; there is no progress to resume from; and a fault at 95% completion
discards 95% of paid work.

Requirements:

- Process repos in chunks sized by `SYNC_BATCH_SIZE`.
- Persist and commit each successful chunk before starting the next, so
  `is_embedded` acts as a durable resume marker.
- A failed chunk must not abort the whole task. Advance past it, count it in
  `failed_items`, and keep going.
- Re-running the task after a failure must pick up only the repos that are still
  un-embedded, without re-embedding completed ones.
- Chunk iteration must not loop forever on a chunk that keeps failing.
- `task.processed_items` must advance per chunk so the Settings page progress
  poll reflects real progress rather than jumping 0 → total.

### A4 — Make the cluster swap atomic

`run_clustering_task` (`sync_execution_service.py:792-799`) deletes every
`Cluster` row for the user and commits, then performs LLM cluster naming, then
inserts new clusters and commits, then reassigns `StarredRepo.cluster_id` and
commits again.

Consequences:

- The destructive commit lands *before* the slow LLM naming calls, so the window
  where the user has zero clusters spans network latency measured in minutes. A
  crash, lease loss, or container restart inside that window leaves every repo
  with `cluster_id = NULL` and no cluster rows — the Graph and Data pages lose
  all grouping with no automatic recovery.
- The delete is an ORM loop, so SQLAlchemy nullifies each cluster's children one
  cluster at a time.

Requirements:

- All expensive, side-effect-free work — clustering, coordinate projection, LLM
  naming, deduplication — must complete before any destructive database write.
- Delete-old, insert-new, and reassign must land in a single transaction, so an
  interrupted run leaves the previous clusters intact.
- Replace the per-row ORM delete with set-based statements.
- The existing foreign key from `starred_repos.cluster_id` to `clusters.id` must
  never be violated during the swap.

## Constraints

- Runs after `08-18-template-residue-removal`, which preserves
  `async_retry_decorator`.
- No Alembic migration. No change to `StarredRepo`, `Cluster`, `SyncTask`, or
  `PipelineRun` columns.
- `SyncPipelineService` calls `compute_embeddings_task(user_id, task_id)` and
  `run_clustering_task(...)` with unchanged signatures; the orchestrator in
  `pipeline_service.py` must not need editing.
- `_inspect_task_outcome` treats `failed_items > 0` as a partial failure and
  `status == "failed"` as a hard failure. A3's per-chunk failure accounting must
  keep producing `partial_failed` rather than `failed` when some chunks succeed.
- Preserve the incremental clustering branch (`incremental=True`) behaviour.
- No new runtime dependency.

## Non-goals

- Not changing the clustering algorithm, its parameters, or
  `derive_clustering_params_for_max_clusters`.
- Not changing embedding model selection, `build_repo_text`, or tag
  normalization.
- Not adding a job queue, worker pool, or retry ledger table.
- Not changing the LLM enhancement concurrency model
  (`SYNC_LLM_ENHANCEMENT_CONCURRENCY` stays as is).

## Acceptance criteria

- [ ] A failing provider call inside `embed_batch` retries only that call;
      a test asserts that slices which already succeeded are not re-sent.
- [ ] `SYNC_BATCH_SIZE` changes observable request sizing; a test asserts the
      embedding stage uses `get_sync_settings().batch_size` and not `32`.
- [ ] A test asserts that when chunk *k* fails, chunks before it are committed
      with `is_embedded = True` and chunks after it still run.
- [ ] A test asserts a re-run after partial failure embeds only the remaining
      repos.
- [ ] A test asserts chunk iteration terminates when a chunk fails repeatedly.
- [ ] `task.processed_items` increases monotonically across chunks, and a task
      with some failed and some successful chunks ends as a partial failure, not
      a hard failure.
- [ ] A test asserts that an exception raised during cluster naming leaves the
      pre-existing clusters and every `StarredRepo.cluster_id` unchanged.
- [ ] The cluster swap issues set-based delete and update statements rather than
      one statement per row.
- [ ] `uv run ruff check src tests scripts alembic` and
      `uv run ruff format --check src tests scripts alembic` pass.
- [ ] `uv run pytest -q` passes.
- [ ] `README.md` and `.trellis/spec/` describe the now-real `SYNC_BATCH_SIZE`
      behaviour and the resumable embedding stage.

## Verification

Backend scope plus the Sync/Scheduler hotspot matrix from
`.trellis/spec/shared/verification.md`, because this task changes pipeline
stage behaviour that the Settings polling tests observe.
