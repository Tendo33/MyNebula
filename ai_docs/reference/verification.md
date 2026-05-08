# Verification Reference

## 用途

本文件是 MyNebula 验证命令的唯一详细事实源。其他 AI 文档和根入口文件只链接这里，不重复维护完整命令。

## Backend

```bash
uv sync --all-extras
uv run ruff check src tests scripts alembic
uv run ruff format --check src tests scripts alembic
uv run pytest -q
```

## Frontend

```bash
npm --prefix frontend run lint
npx --prefix frontend tsc --noEmit -p frontend/tsconfig.json
npm --prefix frontend run test
npm --prefix frontend run build
```

## Full stack

```bash
uv sync --all-extras
uv run ruff check src tests scripts alembic
uv run ruff format --check src tests scripts alembic
uv run pytest -q
npm --prefix frontend run lint
npx --prefix frontend tsc --noEmit -p frontend/tsconfig.json
npm --prefix frontend run test
npm --prefix frontend run build
```

## CI gate

GitHub Actions 当前以 `.github/workflows/ci.yml` 为准，实际门禁运行：

```bash
uv sync --frozen --extra dev
uv run ruff check src/
uv run ruff format --check src/
uv sync --frozen --all-extras
uv run pytest -q
npx --prefix frontend tsc --noEmit
npm --prefix frontend run lint
npm --prefix frontend run test
npm --prefix frontend run build
```

## Docs and links

```powershell
@'
from pathlib import Path
import re
import sys

docs = [
    *Path("ai_docs").rglob("*.md"),
    *Path("doc").glob("*.md"),
    Path("README.md"),
    Path("README.zh.md"),
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
]
pattern = re.compile(r"\[[^\]]+\]\(([^)#]+)")
missing = []

for doc in docs:
    if not doc.exists():
        continue
    text = doc.read_text(encoding="utf-8")
    for rel in pattern.findall(text):
        if "://" in rel or rel.startswith("#"):
            continue
        target = (doc.parent / rel).resolve()
        if not target.exists():
            missing.append(f"{doc}: {rel}")

if missing:
    print("\n".join(missing))
    sys.exit(1)
'@ | uv run python -
```

## 使用规则

- backend-only 任务跑 `Backend`
- frontend-only 任务跑 `Frontend`
- 同时改前后端、脚本或跨层文档时跑 `Full stack`
- 改 `ai_docs/`、`README*`、`AGENTS.md`、`CLAUDE.md` 或 `doc/*.md` 时，先跑对应任务验证，再额外跑 `Docs and links`
- 判断自动化门禁时只看 `CI gate`，不要把它和 `Full stack` 混为一组

## Hotspot Regression Matrix

以下检查用于修改热点模块时做最小回归，避免每次都只靠“全量跑完再看”：

### Graph / Search hotspots

适用范围：
- `frontend/src/components/graph/Graph2D.tsx`
- `frontend/src/components/ui/CommandPalette.tsx`
- `frontend/src/contexts/GraphContext.tsx`
- `frontend/src/utils/search.ts`
- `src/nebula/api/v2/repos.py` 中的语义搜索

建议最小验证：

```bash
uv run pytest -q tests/api/test_v2_auth_access.py -k "repo_search"
npm --prefix frontend run test -- src/components/ui/__tests__/CommandPalette.test.tsx
npm --prefix frontend run test -- src/pages/__tests__/GraphPage.url-state.test.tsx
npm --prefix frontend run test -- src/features/data/hooks/useDataReposQuery.test.tsx
npx --prefix frontend tsc --noEmit -p frontend/tsconfig.json
```

### Sync / Scheduler hotspots

适用范围：
- `src/nebula/application/services/sync_execution_service.py`
- `src/nebula/application/services/pipeline_service.py`
- `src/nebula/application/services/sync_ops_service.py`
- `src/nebula/core/scheduler.py`
- `frontend/src/pages/Settings.tsx`

建议最小验证：

```bash
uv run pytest -q tests/core/test_scheduler_service.py
uv run pytest -q tests/core/test_pipeline_state_machine.py
uv run pytest -q tests/api/test_v2_sync_pipeline_api.py tests/api/test_v2_settings_routes.py
npm --prefix frontend run test -- src/pages/__tests__/Settings.partial-failed.test.tsx src/pages/__tests__/settings.polling.test.tsx
npx --prefix frontend tsc --noEmit -p frontend/tsconfig.json
```
