# Current Architecture

## 什么时候读

- 想快速理解 MyNebula 今天已经落地了什么时先读这里
- 想区分“当前实现”和“路线图/扩展方向”时先读这里

## 当前真实状态

MyNebula 当前是一个已经落地的全栈应用，而不是轻量模板。

它已经包含：

- FastAPI 后端，提供 `/api` 与 `/api/v2` 路由
- PostgreSQL + pgvector 持久化层与 Alembic 迁移
- React + TypeScript + Vite 单页前端
- GitHub Stars 同步、embedding、聚类、图谱快照与定时任务
- 管理登录、只读模式、代理信任和安全 Cookie 边界
- Docker Compose、本地测试、离线质量评估与发布工作流

它当前没有：

- 完整的多用户权限模型
- 面向外部长期承诺的稳定 SDK 面
- 独立于 Web UI 的复杂 CLI 产品面

## 子系统

- 后端真实实现：[backend.md](backend.md)
- 前端真实实现：[frontend.md](frontend.md)
- 运维与配置：[operations.md](operations.md)
- 仓库脚本与维护工具：[scripts.md](scripts.md)
- 发布流程：[release.md](release.md)

## 当前系统骨架

```text
Browser (React SPA)
  -> /api and /api/v2
FastAPI
  -> SQLAlchemy async / PostgreSQL + pgvector
  -> GitHub API / GraphQL
  -> Embedding + LLM providers
  -> APScheduler jobs
```

## 共享参考

- 项目结构：[../reference/project-structure.md](../reference/project-structure.md)
- 路径与命名规则：[../reference/naming-and-paths.md](../reference/naming-and-paths.md)
- 验证命令：[../reference/verification.md](../reference/verification.md)
