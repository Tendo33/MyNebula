# Graph Refactor Performance Report

Date: 2026-03-03  
Branch: `codex/graph-full-refactor`

## Summary
- Introduced versioned graph snapshots and v2 read APIs.
- Graph read path now supports precomputed edges and paged retrieval.
- Frontend data layer now uses React Query cache + Zustand UI state.

## Verified Build/Test Signals
- Backend test suite: `uv run pytest -q` ✅
- Frontend build: `npm --prefix frontend run build` ✅
- Frontend lint: `npm --prefix frontend run lint` ✅ (warnings only)
- Frontend unit tests: `npm --prefix frontend run test` ✅

## Runtime Metrics

### Captured in this run
- Backend test runtime: `~14s`
- Frontend build runtime: `~2s`

### Pending (requires seeded perf environment)
- `/graph` TTI warm/cold
- `/api/v2/graph` P95
- `/api/v2/graph/edges` P95 snapshot-hit path
- Large graph FPS benchmark

## Architecture-level performance improvements
- Removed online edge computation from v2 query path.
- Added snapshot edge pagination (`cursor` + `limit`) to avoid oversized payload bottlenecks.
- Added cache-friendly metadata (`version`, `generated_at`, `request_id`) for future ETag and observability.
- Preserved legacy route compatibility while enabling phased traffic migration to v2.
