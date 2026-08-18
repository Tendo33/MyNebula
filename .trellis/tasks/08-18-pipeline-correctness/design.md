# Design — Sync pipeline correctness and cost control

## Scope boundary

Two files carry the behaviour change:

- `src/nebula/core/embedding.py` — A1, and the batch-size parameter surface for A2.
- `src/nebula/application/services/sync_execution_service.py` — A2 wiring, A3, A4.

`pipeline_service.py` is deliberately untouched: it stays the orchestrator, and
both task entry points keep their signatures.

## A1 — Retry granularity

### Current

```python
@async_retry_decorator(max_retries=3, delay=1.0, backoff=2.0)
async def embed_batch(self, texts, batch_size=32):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = await self.client.embeddings.create(...)   # <- the failure point
        all_embeddings.extend(...)
    return all_embeddings
```

The decorator sits one level too high. `all_embeddings` is rebuilt from empty on
every retry, so the successful prefix is both discarded and re-billed.

### Target

Extract the single provider call into a private coroutine and decorate that:

```python
@async_retry_decorator(max_retries=3, delay=1.0, backoff=2.0)
async def _embed_one_batch(self, batch: list[str]) -> list[list[float]]:
    ...one client.embeddings.create call, index-sorted, placeholder handling...

async def embed_batch(self, texts, batch_size: int | None = None):
    ...loop, calling self._embed_one_batch(batch) per slice...
```

Properties this gives us:

- A transient fault costs at most 4 attempts at *one* slice.
- Slices already returned are never re-sent.
- Backoff still applies, at the granularity where waiting actually helps.
- A permanent fault still surfaces as an exception from `embed_batch`, so A3's
  chunk-level handling is the only new failure path.

The empty-string placeholder substitution and the `sorted(response.data,
key=lambda x: x.index)` ordering guarantee move into `_embed_one_batch`
unchanged — they are per-call concerns.

`embed_text` and `llm.py`'s `complete` already decorate a single call. Neither
changes.

## A2 — `SYNC_BATCH_SIZE` wiring

`embed_batch`'s `batch_size` default changes from `32` to `None`, resolved
inside as `batch_size or get_sync_settings().batch_size`. Resolving the default
inside the service rather than at the call site means every caller gets the
configured value, and an explicit argument still wins for tests.

`sync_execution_service.compute_embeddings_task` stops passing `batch_size=32`
and instead uses `sync_settings.batch_size` as its chunk size (see A3), passing
it through so provider request sizing and database chunk sizing agree.

Range is already validated at 10–500 by `SyncSettings`, so no additional
clamping is needed.

## A3 — Incremental, resumable embedding

### Why the current shape fails

The task is a single straight-line pass: select-all, enhance-all, build-all,
embed-all, commit-once. Every one of those "all"s is a memory multiplier and a
single point of total loss.

### Chunking strategy: keyset cursor, not offset

The obvious loop is "repeatedly select the first N rows where
`is_embedded == False`". That is self-resuming when chunks succeed, but it
**infinite-loops** when a chunk fails: the failed rows stay `is_embedded =
False` and are re-selected forever.

The design therefore uses a keyset cursor on `StarredRepo.id`:

```
last_id = 0
while True:
    chunk = SELECT ... WHERE user_id = :uid
                        AND is_embedded IS FALSE
                        AND id > :last_id
                    ORDER BY id
                    LIMIT :batch_size
    if not chunk: break
    last_id = chunk[-1].id          # advance BEFORE processing
    process(chunk)
```

Advancing the cursor before processing means a failing chunk is skipped on this
pass rather than retried forever, while remaining `is_embedded = False` so the
*next task run* picks it up. Termination is guaranteed because `last_id`
strictly increases and `id` is the primary key.

### Per-chunk sequence

For each chunk, inside its own transaction:

1. LLM enhancement for the chunk's repos that lack `ai_summary` or `ai_tags`,
   reusing `generate_repo_enhancements_in_parallel` with the existing
   `llm_enhancement_concurrency`.
2. `build_repo_text` per repo, assigned to `embedding_text`.
3. `embed_batch(texts, batch_size=sync_settings.batch_size)`.
4. Length check, then assign `embedding` and `is_embedded = True`.
5. `task.processed_items += len(chunk)`; commit.

On exception in steps 1–4: roll back the chunk, add `len(chunk)` to
`failed_items`, log with the chunk's id range, and continue the loop. No repo in
a failed chunk is marked embedded, so nothing is silently lost.

### Failure accounting and pipeline status

`_inspect_task_outcome` in `pipeline_service.py` raises on
`status == "failed"` and reports partial failure on `failed_items > 0`. To keep
that contract:

- Any successful chunk ⇒ final `status = "completed"`, with `failed_items`
  carrying the count. The pipeline reports `partial_failed`.
- Zero successful chunks and at least one failed chunk ⇒ `status = "failed"`,
  preserving today's hard-failure behaviour for a fully broken provider.
- Zero repos to embed ⇒ `status = "completed"`, unchanged from today.

This is the one place where A3 could accidentally change pipeline semantics, so
it gets its own test.

### Memory

Peak resident set becomes O(`SYNC_BATCH_SIZE` × README size) instead of
O(total un-embedded × README size). At the default of 100 that is a bounded
~1 MB of README text per chunk regardless of collection size.

## A4 — Atomic cluster swap

### Why the current shape fails

The ordering is destructive-write → slow-network-work → constructive-write:

```
fit_transform()
DELETE all clusters; COMMIT        # <- graph is now empty
for each cluster: await LLM naming # <- minutes of network latency
INSERT clusters; COMMIT
assign repo.cluster_id; COMMIT
```

Any interruption after the first commit — crash, lease loss, container
restart, OOM — leaves a persistent state where the user has no clusters and
every repo has `cluster_id = NULL`. Nothing restores it automatically.

### Target ordering

Compute everything first, then swap in one transaction:

```
Phase 1 (no DB writes):
  fit_transform()
  for each label: build naming inputs, call LLM, collect entry
  deduplicate_cluster_entries()

Phase 2 (single transaction):
  UPDATE starred_repos SET cluster_id = NULL WHERE user_id = :uid
  DELETE FROM clusters WHERE user_id = :uid
  INSERT new clusters; flush to obtain ids
  UPDATE starred_repos SET cluster_id, coord_x, coord_y, coord_z  (grouped)
  update task row
  COMMIT
```

The null-then-delete order satisfies the `starred_repos.cluster_id →
clusters.id` foreign key without needing a schema change or an `ON DELETE`
clause. Both statements are set-based, replacing the per-row ORM delete and its
child-nullification queries.

Phase 1 already exists in the current code — it is merely interleaved with
phase 2. The change is sequencing, not new logic. The existing per-cluster
`try/except` that falls back from `generate_cluster_name_llm` to the heuristic
`generate_cluster_name` stays exactly as is, so LLM unavailability still
degrades rather than fails.

### Coordinate assignment

Repos are grouped by target cluster and coordinates are applied with grouped
statements rather than one `UPDATE` per repo. Repos whose label is `-1` or
missing from `cluster_map` get `cluster_id = NULL`, matching today's
`unassigned_count` branch.

### Interruption semantics after the change

An interruption during phase 1 costs LLM tokens and leaves the database exactly
as it was. An interruption during phase 2 rolls back. Either way the previous
clusters survive, which is the property A4 exists to establish.

### Incremental branch

The `incremental=True` path (`sync_execution_service.py:662-759`) does not delete
clusters; it assigns new repos to existing ones. It is out of scope for the swap
change and must keep its current behaviour, including the fallback to full
clustering when no positioned repos exist.

## Contracts

Unchanged public signatures:

```python
async def compute_embeddings_task(user_id: int, task_id: int) -> None
async def run_clustering_task(user_id, task_id, use_llm=True,
                              max_clusters=8, min_clusters=None,
                              incremental=False) -> None
```

Changed internal signature:

```python
async def EmbeddingService.embed_batch(
    texts: list[str], batch_size: int | None = None
) -> list[list[float]]
```

`batch_size=None` resolves to `get_sync_settings().batch_size`. Existing callers
that pass an explicit value keep working.

## Compatibility and rollout

- No migration, no schema change, no new column.
- No API contract change; `/api/v2/sync/*` responses are unaffected.
- Settings-page polling sees *more* granular progress, never less.
- A user upgrading mid-collection benefits immediately: the first post-upgrade
  run resumes from whatever `is_embedded` already records.

## Rollback

Single-commit revert restores prior behaviour. The one asymmetry: repos embedded
under the new chunked path stay embedded after a revert, which is desirable — no
cleanup needed.

## Risks

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Keyset cursor skips rows if ids are not monotonic per user | Low | Repos silently never embedded | `id` is an autoincrement primary key; test asserts every un-embedded repo is visited exactly once |
| Per-chunk commits interleave badly with the job lease heartbeat | Medium | Lease lost mid-task | Chunk transactions are short; heartbeat runs on its own session via `JobLeaseHeartbeat`; test a multi-chunk run under an active lease |
| Partial-failure status change flips a passing pipeline to `failed` | Medium | Settings page shows failure where it used to show success | Explicit status matrix above, with a dedicated test per branch |
| Larger `SYNC_BATCH_SIZE` exceeds a provider's per-request input limit | Medium | Provider 400s | Range already capped at 500 by `SyncSettings`; chunk failure is now non-fatal and logged with its id range, so the fault is diagnosable and bounded |
| Grouped coordinate updates mis-map repo to coordinate | Low | Wrong graph layout | Assign by explicit repo id, never by list position across a re-query; test asserts coordinate round-trip for a multi-cluster fixture |
