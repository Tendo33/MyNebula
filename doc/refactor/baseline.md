# Graph Full Refactor Baseline

Date: 2026-03-03  
Branch: `codex/graph-full-refactor`

## Environment
- Backend: `uv` + Python 3.13
- Frontend: Node + npm

## Baseline Commands

### Backend tests
Command:
```bash
uv run pytest -q
```
Result:
- ✅ Passed
- `123 passed`
- Runtime: `14.08s`

### Frontend build
Command:
```bash
npm --prefix frontend run build
```
Result:
- ✅ Passed
- Build artifacts generated under `frontend/dist`

### Frontend lint
Original command:
```bash
npm --prefix frontend run lint
```
Initial result:
- ❌ Failed before refactor due missing ESLint v9 flat config (`eslint.config.js`)

After refactor baseline fix:
- ✅ Lint command runs successfully with warnings only

## Performance Baseline Capture

Requested metrics:
- `/graph` first interactive time (x3 samples)
- `/api/graph` P95
- `/api/graph/edges` P95

Current status:
- ⚠️ Not captured in this document because no reproducible seeded runtime dataset + running service benchmark harness was available in this execution environment.
- Follow-up benchmark should be run against a seeded DB snapshot in a dedicated perf environment.
