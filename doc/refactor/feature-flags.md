# Refactor Feature Flags

Date: 2026-03-04

## Backend
- `SNAPSHOT_READ_FALLBACK_ON_ERROR` (default: `true`)
  - Fallback to live payload build when snapshot read/hydration fails.
- `SLOW_QUERY_LOG_MS` (default: `200`)
  - Slow query warning threshold for graph read services.
- `API_QUERY_TIMEOUT_SECONDS` (default: `15`)
  - Read endpoint timeout guard for heavy graph queries.

## Frontend
- Graph edges are loaded progressively via `/api/v2/graph/edges` paging.
- Page layer API gate is enabled via `npm --prefix frontend run lint:legacy-api`.
- Runtime fallback remains backend adapter + snapshot fallback.

## Rollout recommendation
1. Keep fallback enabled during first production cycle.
2. Observe slow-query and timeout logs for one week.
3. Disable fallback only after stable error budget and rollback drill success.
