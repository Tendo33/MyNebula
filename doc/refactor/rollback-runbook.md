# Snapshot Rollback Runbook

## Preconditions
- v2 graph snapshot tables are migrated.
- At least two snapshots exist for the target user.
- `SNAPSHOT_READ_FALLBACK_ON_ERROR=true` during rollout period.

## Verify active version
1. Call `GET /api/v2/graph?version=active`.
2. Record `version` in response.
3. Call `GET /api/v2/graph/edges?version=active&cursor=0&limit=20` and ensure payload is readable.

## Rollback operation
1. Preferred: service rollback
   - Run a maintenance shell and call `GraphQueryService.rollback_active_snapshot(...)`.
2. Emergency SQL fallback
   - Set `users.active_graph_snapshot_id` to target snapshot id.
   - Mark target snapshot `status='active'`, previous `status='ready'`.
3. Confirm new active id points to the desired snapshot row.

## Post-checks
1. `GET /api/v2/graph?version=active` returns previous version id.
2. `GET /api/v2/graph/edges?version=active&cursor=0&limit=20` succeeds.
3. `GET /api/v2/graph/timeline?version=active` succeeds.
4. Check logs contain no consistency validation errors for active snapshot.

## Emergency fallback
- Keep old `/api/graph*` adapters enabled.
- Keep `SNAPSHOT_READ_FALLBACK_ON_ERROR=true` during first rollout cycle.

## Drill commands
- `uv run python scripts/perf/rollback_drill.py --runs 10 --mode mock`
- `uv run python scripts/perf/rollback_drill.py --runs 10 --mode service` (requires live DB)
