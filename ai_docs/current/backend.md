# Current Backend

## 什么时候读

- 要改后端代码前先读这里
- 需要确认 API、服务层、数据层和配置入口的当前事实时读这里

## 当前真实实现

后端主包位于 `src/nebula/`，已经不是模板式基础层，而是带真实业务编排的应用。

已经落地的核心模块：

- `api/`：路由层，包含 v1 兼容接口和 v2 主接口
- `application/services/`：同步、快照、推荐、Dashboard/Data 查询等编排逻辑
- `core/`：配置、认证、日志、embedding、LLM、聚类、调度器
- `db/`：SQLAlchemy 模型、数据库连接和生命周期
- `domain/`：同步与快照生命周期枚举
- `infrastructure/`：快照持久化相关仓储逻辑
- `schemas/`：通用 API schema 与 `schemas/v2/*` 聚合响应

## 当前后端边界

- 路由层保持轻量，负责鉴权、参数校验和 HTTP 翻译
- 业务编排集中在 `application/services/`
- 长任务状态通过数据库持久化，不只存在内存里
- 图谱、Dashboard、Data 等读路径优先走快照或轻量聚合响应
- 当前运行模型仍以单用户优先，很多读写流程默认按首个用户作用域处理
- `api/v2/access.py` 现在提供显式的单用户解析入口，Settings/Sync 等写路由优先在 API 边界拿到 `user` 后再传入 service
- 默认用户自举已带并发保护与冲突恢复，不应再把首次访问视为“天然串行”
- `Graph` 的历史版本查询对不存在的 `version` 返回显式错误，不再静默回退到 active
- 同步链路支持通过 `SYNC_README_FETCH_CONCURRENCY`、`SYNC_LLM_ENHANCEMENT_CONCURRENCY`、`SYNC_PROGRESS_COMMIT_INTERVAL` 做性能调优
- README 抓取、LLM 增强、star list 同步的执行辅助逻辑已从主执行服务中拆到 `application/services/sync_execution_support.py`
- 图快照构建服务支持显式传入 `user`，并记录 `nodes / edges / timeline / total` 分段耗时，便于定位大图性能瓶颈

## 当前稳定入口

- ASGI 应用：`nebula.main:create_app`
- 应用运行命令：`mynebula` 或 `uv run uvicorn nebula.main:app`
- 配置读取：`nebula.core.config`
- 对外更稳定的程序化入口仍以 HTTP API 为主，而不是内部 service 直接导入

## 当前高价值事实源

- 启动与中间件：`src/nebula/main.py`
- 配置模型：`src/nebula/core/config.py`
- 同步流程：`src/nebula/application/services/pipeline_service.py`
- 全量刷新：`src/nebula/application/services/sync_ops_service.py`
- 调度器：`src/nebula/core/scheduler.py`
- 管理认证：`src/nebula/api/v2/auth.py` 与 `src/nebula/core/auth.py`
- ORM 模型：`src/nebula/db/models.py`

## 共享参考

- 后端规范：[../standards/backend.md](../standards/backend.md)
- 项目结构：[../reference/project-structure.md](../reference/project-structure.md)
- 验证命令：[../reference/verification.md](../reference/verification.md)
