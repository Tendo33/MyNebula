# Implementation plan — Sync pipeline correctness and cost control

Ordering: child **2 of 5** in `.trellis/tasks/08-18-comprehensive-hardening`.
Requires `08-18-template-residue-removal` to be merged first, because A1
reapplies `async_retry_decorator` and that task decides which utils survive.

Each step is independently revertable and leaves the pipeline runnable.

## Step 1 — A1: move retry onto the provider call

- [ ] In `src/nebula/core/embedding.py`, extract the body of the `for` loop's
      API call into `_embed_one_batch(self, batch: list[str]) -> list[list[float]]`.
- [ ] Move into it, unchanged: empty-string placeholder substitution, the
      `client.embeddings.create` call, `sorted(response.data, key=...index)`,
      and zero-vector replacement for placeholder slots.
- [ ] Decorate `_embed_one_batch` with
      `@async_retry_decorator(max_retries=3, delay=1.0, backoff=2.0)`.
- [ ] Remove the decorator from `embed_batch`.
- [ ] Leave `embed_text` and `src/nebula/core/llm.py` untouched.

**Test:** a fake client whose 2nd of 3 slices fails once then succeeds. Assert
the total number of `embeddings.create` calls is 4, not 6, and that slice 1 is
requested exactly once.

**Gate 1:** `uv run pytest -q tests/` green. Blocking.

## Step 2 — A2: wire `SYNC_BATCH_SIZE`

- [ ] Change `embed_batch(self, texts, batch_size: int | None = None)`.
- [ ] Resolve inside: `size = batch_size or get_sync_settings().batch_size`.
- [ ] Import `get_sync_settings` in `embedding.py`.
- [ ] Delete the hardcoded `batch_size=32` at
      `sync_execution_service.py:488`.

**Test:** set `SYNC_BATCH_SIZE` via a settings override, call `embed_batch` with
no explicit size, assert observed slice sizes match the configured value. Clear
the `lru_cache` on `get_sync_settings` in the fixture.

**Gate 2:** `uv run pytest -q` green.

## Step 3 — A3: chunked, resumable embedding

Rewrite `compute_embeddings_task` in
`src/nebula/application/services/sync_execution_service.py`.

- [ ] Compute the total un-embedded count once, up front, for
      `task.total_items` — preserving today's "0 repos ⇒ complete immediately"
      early return.
- [ ] Replace the select-all with the keyset loop from `design.md`:
      `WHERE user_id AND is_embedded IS FALSE AND id > last_id ORDER BY id LIMIT size`,
      advancing `last_id` to the chunk's last id **before** processing it.
- [ ] Per chunk, in order: LLM enhancement for repos missing summary or tags →
      `build_repo_text` → `embed_batch` → length check → assign `embedding` and
      `is_embedded = True` → `task.processed_items += len(chunk)` → commit.
- [ ] Wrap each chunk in `try/except`: on failure roll back, add `len(chunk)` to
      `task.failed_items`, log the chunk id range and the exception, continue.
- [ ] Keep the existing per-repo LLM fallback that seeds `ai_tags` from
      `repo.topics[:5]` when generation fails.
- [ ] Apply the final status matrix:
      - any chunk succeeded → `completed` (partial failure surfaces via
        `failed_items`)
      - no chunk succeeded and at least one failed → `failed`
      - nothing to do → `completed`
- [ ] Set `completed_at` on every terminal path.
- [ ] Keep the outer `except` that marks the task failed on an unexpected error.

**Tests:**
- chunk 2 of 3 fails: chunks 1 and 3 committed with `is_embedded = True`,
  `failed_items == len(chunk2)`, status `completed`.
- re-run after that failure embeds only chunk 2's repos.
- a chunk that always fails does not loop; the task terminates.
- `processed_items` is monotonic and ends at the number actually embedded.
- all chunks fail ⇒ status `failed`, so `_inspect_task_outcome` still raises.
- zero un-embedded repos ⇒ `completed`, `total_items == 0`.

**Gate 3:** `uv run pytest -q tests/core tests/api` green, and
`uv run pytest -q tests/core/test_pipeline_state_machine.py` still passes
unmodified. Blocking — if the state machine test needs editing, the status
matrix is wrong.

## Step 4 — A4: atomic cluster swap

Restructure the non-incremental branch of `run_clustering_task`.

- [ ] Move the whole naming loop — `build_cluster_naming_inputs`,
      `generate_cluster_name_llm` / `generate_cluster_name` fallback,
      `assigned_names` accumulation, `cluster_entries` build, and
      `deduplicate_cluster_entries` — so it completes **before** any destructive
      statement. Do not change its logic or its per-cluster `try/except`.
- [ ] Delete the current `for cluster in existing_clusters: await db.delete(...)`
      block and its `await db.commit()`.
- [ ] Add a single transaction that performs, in order:
      1. `update(StarredRepo).where(StarredRepo.user_id == user_id).values(cluster_id=None)`
      2. `delete(Cluster).where(Cluster.user_id == user_id)`
      3. insert the new `Cluster` rows, `flush()` to obtain ids
      4. grouped `update(StarredRepo)` per target cluster for `cluster_id`, and
         coordinate assignment keyed by explicit repo id
      5. task fields (`processed_items`, `status`, `completed_at`)
      6. one `commit()`
- [ ] Leave the `incremental=True` branch (lines ~662–759) untouched, including
      its fallback to full clustering when no positioned repos exist.
- [ ] Move the `import math` currently inside the node-size loop to module
      scope.

**Tests:**
- LLM naming raises for every cluster: pre-existing `Cluster` rows and every
  `StarredRepo.cluster_id` are unchanged after the task returns.
- an exception injected between delete and insert leaves the prior clusters
  intact (transaction rollback).
- happy path: old clusters replaced, `repo_count` correct, coordinates match the
  fit result for a multi-cluster fixture, no repo left pointing at a deleted
  cluster id.
- incremental branch regression: existing behaviour unchanged.

**Gate 4:** `uv run pytest -q` green, including
`tests/test_clustering_improvements.py` unmodified. Blocking.

## Step 5 — Docs and spec

- [ ] `README.md`: `SYNC_BATCH_SIZE` now genuinely controls embedding request
      and chunk sizing; state the 10–500 range and the default of 100.
- [ ] `.trellis/spec/backend/` — record the resumable embedding contract
      (`is_embedded` as the durable resume marker, keyset chunk iteration,
      per-chunk failure accounting) and the atomic cluster-swap ordering, since
      both are now behavioural contracts future changes must not break.
- [ ] Note in the spec that `embed_batch` retries per provider call, so callers
      must not add an outer retry.

## Step 6 — Verify

```bash
uv run ruff check src tests scripts alembic
uv run ruff format --check src tests scripts alembic
uv run pytest -q
```

Then the Sync / Scheduler hotspot matrix from
`.trellis/spec/shared/verification.md`:

```bash
uv run pytest -q tests/core/test_scheduler_service.py
uv run pytest -q tests/core/test_pipeline_state_machine.py
uv run pytest -q tests/api/test_v2_sync_pipeline_api.py tests/api/test_v2_settings_routes.py
pnpm --prefix frontend run test -- src/pages/__tests__/Settings.partial-failed.test.tsx src/pages/__tests__/settings.polling.test.tsx
pnpm --prefix frontend exec tsc --noEmit -p tsconfig.json
```

The frontend Settings tests are in scope because A3 changes how
`processed_items` and `failed_items` evolve, and those tests assert on partial
failure and polling behaviour.

Finally the documentation link check, because spec files change.

## Rollback points

- After step 1 or 2: revert `embedding.py` alone; `sync_execution_service.py` is
  untouched until step 2's one-line deletion.
- After step 3: revert `compute_embeddings_task` only. Repos already embedded
  under the chunked path stay embedded and are correctly skipped by the old
  code, so no data cleanup is needed.
- After step 4: revert `run_clustering_task` only. The swap is transactional, so
  no partially-applied cluster state can exist to clean up.
- Whole task: single `git revert`. No migration, no config change, no data
  fixup.
