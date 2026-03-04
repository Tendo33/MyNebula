# Graph Full Refactor Remaining Gaps Checklist

Date: 2026-03-04  
Branch: `codex/graph-full-refactor`

## Already closed
- [x] v2 settings route requires admin auth and exposes typed contract.
- [x] v2 graph route supports `include_edges=false` by default.
- [x] Frontend graph data path supports staged edge pagination.
- [x] Scheduler triggers unified pipeline service instead of local task chain.

## Merge hardening checklist
- [x] Dashboard/Data page data reads no longer depend on `GraphContext`.
  - Owner: Frontend
  - Verify: `rg -n "useGraph\\(" frontend/src/pages/Dashboard.tsx frontend/src/pages/DataPage.tsx`
  - Exit: no graph data reads via context in these two pages.
- [x] Legacy page-level API usage gate is enabled.
  - Owner: Frontend
  - Verify: `npm --prefix frontend run lint:legacy-api`
  - Exit: gate passes and CI hook is ready.
- [x] Legacy sync adapter coverage is complete.
  - Owner: Backend
  - Verify: `uv run pytest -q tests/api/test_sync_adapter_compat.py`
  - Exit: `/api/sync/schedule|info|full-refresh` compatibility assertions pass.
- [x] E2E flow suite is available.
  - Owner: Frontend
  - Verify: `npm --prefix frontend run test:e2e`
  - Exit: first-load graph / sync refresh / filters-linkage flows are codified (executed when `RUN_E2E=1`).
- [x] Stability drills are reproducible in seeded env (mock mode).
  - Owner: Backend
  - Verify: `uv run python scripts/perf/soak_pipeline.py --runs 50 --mode mock` and `uv run python scripts/perf/rollback_drill.py --runs 10 --mode mock`
  - Exit: success rates reach target thresholds.

## Remaining blockers before merge-to-prod claim
- [x] Alembic smoke (`upgrade/downgrade/upgrade`) on reachable PostgreSQL.
- [x] Browser E2E critical flows executed (`RUN_E2E=1`).
- [x] Live-mode (non-mock) performance and stability evidence.
- [ ] Live `/api/v2/graph/edges` P95 meets `<300ms` target (current: `1385.76ms`).

## Risks to track
- Full-refresh currently uses legacy internals through a v2 adapter path.
- Snapshot fallback path is intentionally permissive (`snapshot_read_fallback_on_error=true`); monitor error budget before tightening.
