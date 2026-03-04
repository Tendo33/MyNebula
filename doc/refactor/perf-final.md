# Graph Refactor Final Performance Report

Date: 2026-03-04  
Branch: `codex/graph-full-refactor`

## Scope
- `/api/v2/graph` (`include_edges=false`)
- `/api/v2/graph/edges` (cursor paging)
- Frontend `/graph` staged rendering (nodes first, edges later)

## Baseline command checklist
- `uv run pytest -q`
- `npm --prefix frontend run build`
- `npm --prefix frontend run lint`
- `npm --prefix frontend run lint:legacy-api`
- `npm --prefix frontend run test`
- `uv run python scripts/perf/benchmark_graph_api.py --runs 30`

## Measurement template
- Warm `/graph` TTI: `1.22s` (mock benchmark)
- Cold `/graph` TTI: `2.68s` (mock benchmark)
- `/api/v2/graph/edges` P95: `186.40ms` (mock benchmark)
- Large graph FPS: `51.30` (mock benchmark)

## Live benchmark (`--mode api`)
- Command: `uv run python scripts/perf/benchmark_graph_api.py --runs 30 --mode api --base-url http://127.0.0.1:8071`
- `/api/v2/graph` P50: `1497.79ms`, P95: `1719.26ms`
- `/api/v2/graph/edges` P50: `1217.17ms`, P95: `1385.76ms`

## Acceptance thresholds
- Warm `/graph` TTI < `1.5s`
- Cold `/graph` TTI < `3.0s`
- `/api/v2/graph/edges` P95 < `300ms` (snapshot hit)
- Large graph interactive FPS > `45`

## Notes
- ETag/304 support is enabled for v2 graph/timeline/edges.
- Snapshot read fallback is enabled by default for safety.
- Benchmark command executed: `uv run python scripts/perf/benchmark_graph_api.py --runs 30 --mode mock`.
- Benchmark command rerun after env update with same seeded mock profile.
- Live API mode benchmark has been executed; current `/api/v2/graph/edges` P95 is above target and requires optimization before production claim.
