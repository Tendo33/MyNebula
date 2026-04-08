# Remediation Backlog

This file is now primarily historical context plus a verification ledger. The listed tasks have been completed against the current codebase; use it as a release-audit trail, not as an open todo list.

| Priority | Task | Status | Verification | Notes |
| --- | --- | --- | --- | --- |
| P0 | Task 1: trusted proxy allowlist and secure-cookie hardening | Done | `uv run pytest tests/api/test_v2_auth_access.py tests/test_auth_utils.py -q` | Added `TRUSTED_PROXY_IPS`; forwarded headers only trusted from allowlisted proxy IPs |
| P0 | Task 2: full refresh subtask failure propagation | Done | `uv run pytest tests/core/test_full_refresh_task_state_propagation.py tests/api/test_v2_settings_routes.py -q` | Parent task now fails when stars/embedding/clustering subtasks fail |
| P0 | Task 3: unify pipeline and scheduler state semantics | Done | `uv run pytest tests/core/test_pipeline_state_machine.py tests/core/test_scheduler_service.py -q` | Preserves partial failure detail and marks scheduler launch races as failed |
| P0 | Task 4: timezone-aware schedule timestamps | Done | `uv run pytest tests/test_sync_job_and_graph_edges.py tests/api/test_v2_settings_routes.py -q` | Backend returns UTC-aware `next_run_at`; frontend normalizes legacy naive timestamps |
| P1 | Task 5: pgvector ANN indexes for semantic and related search | Done | `uv run pytest tests/core/test_vector_search_query_plan.py -q` | Added ivfflat cosine ANN migration and query contract assertions |
| P1 | Task 6: remove duplicate Data/Dashboard snapshot fetches | Done | `npm --prefix frontend run test -- src/pages/__tests__/data.v2.smoke.test.tsx src/pages/__tests__/dashboard.v2.smoke.test.tsx` and `npm --prefix frontend run build` | `/api/v2/data/repos` now returns cluster metadata; Data page no longer fetches graph snapshot; Dashboard no longer fetches graph payload for summary derivations |
| P1 | Task 7: cancellable Settings polling | Done | `npm --prefix frontend run test -- settings.polling.test.tsx` | Polling now aborts on replacement, auth loss, close/reset, and unmount |
| P1 | Task 8: unify search semantics across Graph/Data/Command Palette | Done | `npm --prefix frontend run test -- src/utils/search.test.ts src/contexts/graphFiltering.test.ts` | Shared frontend matcher added; Data backend `q` now aligns with shared repo fields and `stars:>N` |
| P1 | Task 9: reduce Graph filtering recomputation cost | Done | `npm --prefix frontend run test -- src/contexts/graphFiltering.test.ts src/features/graph/__tests__/useGraphEdgesInfiniteQuery.test.tsx` | Graph filtering split into visible nodes, node id set, edge filter, and cluster derivation with reusable indexes |
| P2 | Task 10: stabilize test and quality gates | Done | `uv run pytest -q`, `uv run ruff check src tests alembic/versions`, `npm --prefix frontend run test`, `npm --prefix frontend run lint -- --quiet`, `npm --prefix frontend run build` | CI/backend/frontend commands now match local verified commands |
| P2 | Task 11: backfill Trellis engineering standards | Done | `.trellis/spec/backend/*.md`, `.trellis/spec/frontend/*.md` | Added frontend query/filtering standards alongside backend conventions |
| P2 | Task 12: publish remediation backlog | Done | `doc/REMEDIATION_BACKLOG.md` | This file is the living backlog snapshot |
