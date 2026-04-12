# Contributing to MyNebula

感谢你愿意一起把 MyNebula 做得更稳、更清晰。

## 先了解当前项目

在开始改代码前，建议先读这几份文档：

- `README.md` 或 `README.zh.md`
- `ai_docs/current/operations.md`
- `ai_docs/reference/verification.md`
- `ai_docs/current/release.md`

如果你要改动接口、同步流程或数据库结构，再补读：

- `ai_docs/current/backend.md`
- `ai_docs/current/operations.md`

## 本地开发环境

### 后端

```bash
uv sync --all-extras
cp .env.example .env
docker compose up -d db
uv run alembic upgrade head
uv run uvicorn nebula.main:app --reload --port 8000
```

### 前端

```bash
npm --prefix frontend install
VITE_API_BASE_URL=http://localhost:8000 npm --prefix frontend run dev
```

### Git hooks

```bash
uv run pre-commit install
```

更多说明见 `ai_docs/current/scripts.md` 和 `ai_docs/reference/verification.md`。

## 分支和提交建议

- 从 `main` 拉出新分支
- 分支名建议表达清楚目的，例如 `feat/sync-progress`、`fix/graph-timeout`
- 提交信息建议使用语义化前缀：`feat`、`fix`、`docs`、`refactor`、`test`、`chore`

示例：

```text
feat(sync): add recluster-only pipeline trigger
docs(readme): refresh v2 route reference
fix(frontend): preserve selected ghost node in graph panel
```

## 代码风格

### Python

- 使用 Python 3.10+ 类型语法，例如 `list[str]`
- 新代码尽量补齐类型标注
- 提交前运行 Ruff

```bash
uv run ruff format
uv run ruff check
```

### Frontend

- 代码位于 `frontend/src/`
- 新页面或状态变更优先复用现有 `api/v2`、`contexts`、`stores`、`types`
- 提交前至少运行 lint、单测和 build

```bash
npm --prefix frontend run lint
npx --prefix frontend tsc --noEmit
npm --prefix frontend run test
npm --prefix frontend run build
```

## 测试基线

按改动范围选择最小但足够的验证集合：

### 后端改动

```bash
uv run pytest
```

### 前端改动

```bash
npm --prefix frontend run test
npm --prefix frontend run build
```

### 路由、页面流或交互改动

```bash
npm --prefix frontend run test:e2e
```

### 相关推荐/聚类质量相关改动

```bash
uv run python scripts/evals/run_all_quality_checks.py
```

## 数据库和迁移

- 任何 ORM 结构变更都应配套 Alembic migration
- 不要删除已有 migration 以“修复”链路
- 如果你修改了表结构、初始化逻辑或 pipeline 持久化，请至少验证一次全新数据库启动

常用命令：

```bash
uv run alembic upgrade head
uv run python scripts/reset_db.py
```

## 文档要求

下面这些改动通常必须同步文档：

- 新增或删除 API 路由
- 环境变量变化
- Docker 部署流程变化
- 数据库初始化 / reset / migration 行为变化
- 同步流程、设置页、质量门槛变化

优先更新的文档通常是：

- `README.md`
- `README.zh.md`
- `ai_docs/current/operations.md`
- `ai_docs/current/release.md`
- `ai_docs/reference/verification.md`

## Pull Request 建议

PR 描述尽量包含：

- 这次改了什么
- 为什么要改
- 影响哪些模块
- 如何验证
- 是否需要更新环境变量、迁移、截图或文档

如果改动涉及 UI，附上截图或录屏会更容易 review。

## CI 目前会检查什么

GitHub Actions 当前包含：

- 后端 Ruff lint + format check
- 前端 ESLint
- 前端 TypeScript typecheck
- 前端 Vitest
- 前端生产构建

因此本地最好在提交前把这些命令先跑一遍。

## 提问题和提建议

欢迎直接提 Issue 或 PR。描述越具体，越容易复现和评审：

- 问题现象
- 复现步骤
- 期望行为
- 实际行为
- 本地环境或部署方式

再次感谢你的贡献。
