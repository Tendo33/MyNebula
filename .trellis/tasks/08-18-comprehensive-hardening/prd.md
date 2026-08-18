# Comprehensive hardening

## Goal

Close the technical debt found in the 2026-08-18 full-repository scan without
changing MyNebula's product surface. The scan established that the baseline is
healthy (ruff clean, 243 backend tests passing, frontend lint + `tsc` clean, no
`any`, no TODO markers, `.env` untracked, docs already consolidated into
`.trellis/spec/`). The debt is therefore not firefighting: it is pipeline
correctness under real data volume, template residue, misplaced test coverage,
delivery gaps, and frontend structure.

## Source requirement set

The user asked for a full scan followed by comprehensive improvement, then
confirmed "全面修复" over the complete finding set (A–E below) and chose a
Trellis parent/child task tree.

## Findings inventory

Each finding is owned by exactly one child task.

### A. Pipeline correctness and cost — `08-18-pipeline-correctness`

- A1 `src/nebula/core/embedding.py:99` — `@async_retry_decorator` wraps the whole
  `embed_batch` loop, so a failure in batch N replays batches 0..N-1 against a
  paid embedding API. Retry granularity belongs on the single API call.
- A2 `SYNC_BATCH_SIZE` is a dead setting. It is declared and range-validated in
  `src/nebula/core/config.py:149`, documented in `README.md` as controlling
  throughput and cost, and read by nothing;
  `sync_execution_service.py:488` hardcodes `batch_size=32`.
- A3 `compute_embeddings_task` is all-or-nothing. It loads every un-embedded repo
  (including README text up to `SYNC_README_MAX_LENGTH`) into memory, builds all
  embedding texts, then issues one `embed_batch` call. Any failure sets
  `failed_items = len(repos)` and persists zero embeddings, so a large first sync
  has no incremental progress and no resume point.
- A4 `run_clustering_task` swaps clusters non-atomically
  (`sync_execution_service.py:792-799`): it deletes every `Cluster` row and
  commits, then rebuilds clusters and reassigns `cluster_id` in later commits. A
  crash in that window leaves every repo with `cluster_id = NULL` and no
  clusters. The delete is also an ORM loop that nullifies children row by row.

### B. Template residue — `08-18-template-residue-removal`

- B1 `src/nebula/models/` is broken: `import nebula.models` raises
  `ImportError: cannot import name 'TimestampMixin'`, and the package also
  imports a non-existent `.examples` module.
- B2 67 of the 77 names exported from `nebula.utils` are unused by the
  application. Only `get_logger`, `setup_logging`, `async_retry_decorator`,
  `compute_content_hash`, and `compute_topics_hash` have real call sites.
  `common_utils.py` (377 LOC, 10% covered) and `file_utils.py` (432 LOC, 11%)
  are effectively entirely dead, and three test files exist solely to exercise
  dead code. This contradicts the repository's own guardrail that MyNebula is
  not a generic Python template.

### C. Test coverage placement — `08-18-pipeline-test-coverage`

- C1 Overall backend coverage is 57%, but the risk is inverted: the largest and
  most business-critical module, `sync_execution_service.py` (442 statements),
  sits at 4%. `graph_snapshot_service.py` is 19%,
  `snapshot_repository.py` 42%, `pipeline_service.py` 52%.
- C2 The suite finishes in ~3s because nothing touches a database. CI already
  provisions a `pgvector/pgvector:pg16` service and runs
  `alembic upgrade head`, but no test asserts real pipeline or snapshot
  behaviour against it.

### D. Delivery — `08-18-delivery-hardening`

- D1 `Dockerfile` installs `gcc`, `g++`, and `python3-dev` into the final
  runtime image and never removes them, inflating image size and CVE surface.
- D2 CI runs `ruff check src/` while
  `.trellis/spec/shared/verification.md` mandates
  `ruff check src tests scripts alembic`. CI gate and canonical verification
  reference disagree.
- D3 `frontend/e2e/graph-sync-flow.spec.ts` and a Playwright config exist, but
  no CI job ever runs them.

### E. Frontend — `08-18-frontend-hardening`

- E1 God components: `Graph2D.tsx` (910 LOC), `Settings.tsx` (821),
  `CommandPalette.tsx` (784), `DataPage.tsx` (735). Flagged in the 2026-05-09
  health check and still unsplit.
- E2 `useNodeNeighbors` (`frontend/src/contexts/GraphContext.tsx:287`) rebuilds a
  full O(E) adjacency map inside every consuming component instead of once in
  the provider.
- E3 i18n drift: `repoDetails.similar` exists only in the `zh` bundle.
- E4 No accessibility baseline: 70 `<button>` elements with sparse aria coverage
  and no automated a11y check.

## Non-goals

- No product, API contract, or database-schema behaviour changes visible to
  users, except where a finding is itself a contract defect (A2 makes a
  documented setting real; A4 makes an existing operation atomic).
- No redesign of the clustering algorithm, relevance scoring, or graph layout.
- No move away from the snapshot-backed read model, persisted pipeline state,
  single-user runtime assumption, or the pnpm/uv toolchain.
- No new runtime dependency unless a child task justifies it in its `design.md`.
- No visual redesign of Graph, Data, Dashboard, or Settings. E1 is a structural
  refactor that must keep rendered output equivalent.

## Constraints

- Preserve admin authentication, CSRF protection, trusted-proxy rules, and
  `READ_ACCESS_MODE` demo/authenticated behaviour.
- Preserve FastAPI `/api` and `/api/v2` routes, SQLAlchemy async usage,
  PostgreSQL/pgvector, Alembic, sync pipeline state, graph snapshots, and
  APScheduler boundaries.
- Preserve Dashboard, Data, Graph, and Settings flows, including React Query,
  GraphContext, shared search utilities, progressive edge loading, and the
  Settings polling lifecycle.
- Update `.trellis/spec/` whenever behaviour, structure, scripts, public APIs, or
  verification commands change.

## Child task map and ordering

Parent/child is not a dependency system, so the required ordering is stated
here and repeated in each child's `implement.md`.

1. `08-18-template-residue-removal` (B) — first, so later work does not test,
   refactor, or document code that is about to be deleted. Must preserve
   `get_logger`, `setup_logging`, `async_retry_decorator`,
   `compute_content_hash`, `compute_topics_hash`.
2. `08-18-pipeline-correctness` (A) — depends on B keeping
   `async_retry_decorator`, since A1 reapplies it at per-call granularity.
3. `08-18-pipeline-test-coverage` (C) — after A, so the new tests encode the
   corrected batching, resume, and atomic-swap behaviour rather than the old
   behaviour.
4. `08-18-delivery-hardening` (D) — after C, because D2/D3 wire the new test
   surface into the CI gate.
5. `08-18-frontend-hardening` (E) — independent of A–D; may run in parallel but
   is sequenced last to keep the review queue linear.

## Cross-child acceptance criteria

- [ ] Every finding A1–A4, B1–B2, C1–C2, D1–D3, E1–E4 is either resolved in its
      owning child task or explicitly recorded as deferred with a reason in that
      child's task notes.
- [ ] `import nebula.models` no longer fails, because the package is gone.
- [ ] `uv run ruff check src tests scripts alembic` and
      `uv run ruff format --check src tests scripts alembic` pass.
- [ ] `uv run pytest -q` passes, and backend line coverage for
      `src/nebula/application/services/sync_execution_service.py` is at least
      60%, up from 4%.
- [ ] `pnpm --prefix frontend run lint`,
      `pnpm --prefix frontend exec tsc --noEmit -p tsconfig.json`,
      `pnpm --prefix frontend run test`, and
      `pnpm --prefix frontend run build` pass.
- [ ] The documentation link check in
      `.trellis/spec/shared/verification.md` passes.
- [ ] CI lint scope matches `.trellis/spec/shared/verification.md`.
- [ ] No user-visible change in Graph, Data, Dashboard, or Settings behaviour,
      confirmed by the existing frontend test suite plus the Playwright flow.

## Verification

Full-stack, because the change set crosses backend, frontend, scripts, CI, and
docs. Run the "Full Stack" section of
`.trellis/spec/shared/verification.md`, plus both hotspot matrices
(Graph/Search and Sync/Scheduler), since A and E touch both hotspots.

## Newly discovered finding (2026-08-18, during execution)

### F1 — `useGraphEdgesInfiniteQuery` has a flaky test over a real auto-load race

`frontend/src/features/graph/__tests__/useGraphEdgesInfiniteQuery.test.tsx`
→ "does not consume auto-load budget when a page fetch fails transiently"
fails roughly 1 run in 5:

```
expected [ Array(2) ] to have a length of 3 but got 2
  at useGraphEdgesInfiniteQuery.test.tsx:238
```

Pre-existing; not introduced by this epic. Nothing in
`useGraphEdgesInfiniteQuery.ts` or its test was modified here.

Suspected cause in
`frontend/src/features/graph/hooks/useGraphEdgesInfiniteQuery.ts`:

- `pagesLoadedRef.current += 1` runs *after* `await fetchNextPage()`. Resolving
  that promise updates `data.pages`, which changes the `attemptLoadNextPage`
  callback identity and re-runs the auto-load effect. The budget guard
  `pagesLoadedRef.current >= maxAutoPages` can therefore observe a stale count.
- `retryEdgeLoading()` resets `autoLoadEnabledRef`, `autoLoadHalted`,
  `edgesError`, and `seenNextCursorsRef`, but **not** `pagesLoadedRef`, so the
  post-retry budget depends on how far the pre-failure run got.
- `void attemptLoadNextPage()` in the effect discards the rejection, so a failed
  auto-load surfaces only through React Query's `error`.

Not fixed in this epic: it is outside all five children's scope, and the fix is
a change to the auto-load state machine that needs its own design rather than a
timeout bump in the test. A flaky test in the CI gate erodes the gate, so this
should be scheduled next.

Reproduce:

```bash
for i in $(seq 1 8); do
  pnpm --prefix frontend run test -- src/features/graph/__tests__/useGraphEdgesInfiniteQuery.test.tsx \
    2>&1 | grep "Tests  "
done
```

## F1 resolved (2026-08-18)

Root cause was **not** the `pagesLoadedRef` race hypothesised above. Two
separate problems were tangled together.

### The flakiness was a test-timing race

`useGraphEdgesInfiniteQuery` sets `retry: 2` on the query itself, which
overrides the test wrapper's `retry: false`. React Query's default backoff for
the first retry is ~1000ms — exactly `waitFor`'s default timeout. The
assertion and the retry were racing. Fixed by giving that `waitFor` an explicit
5s budget with the reason recorded inline. Verified stable over 8 consecutive
runs (previously ~1 failure in 5).

### The test name pointed at a real bug

`fetchNextPage()` **resolves** with a result object when a page fails; it only
rejects when `throwOnError` is set. The `try/catch` around it was therefore
dead code for query failures, so:

- `pagesLoadedRef.current += 1` ran even for a page that never landed,
  consuming auto-load budget — literally what the test's name says must not
  happen; and
- the failed cursor stayed in `seenNextCursorsRef`, so a later manual
  `loadMoreEdges()` at the same cursor tripped the duplicate guard and halted
  edge loading permanently with "Detected duplicated edge cursor N".

User-visible consequence: after one transient edge-page failure, the Graph page
could lose the ability to load more edges for the rest of the session.

Fixed in both `attemptLoadNextPage` and `loadMoreEdges` by comparing page count
before and after the call — a resolved promise is not evidence that a page
landed.

Regression guard: `frontend/src/features/graph/__tests__/edgeAutoLoadBudget.test.tsx`.
The first version of this file passed against both the fixed and the buggy
implementation, i.e. it was not a guard at all. Rewritten to drive a manual
retry at the failed cursor, and verified to fail with
`expected [...] to have a length of 2 but got 1` when the fix is reverted.
