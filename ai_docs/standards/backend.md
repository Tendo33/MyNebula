# Backend Standards

## 什么时候读

- 开始任何后端改动前先读这里
- 需要确认路由、服务层、数据库和错误处理默认做法时回到这里

## 分层规则

- 路由层只做鉴权、参数校验、HTTP 翻译
- 编排逻辑放 `application/services/`
- 持久化细节放 `db/` 或 `infrastructure/`
- 共享 schema 放 `schemas/`，v2 聚合响应放 `schemas/v2/`

## 不可打破的默认约束

- 不要把核心业务逻辑塞进 `api/*`
- 不要把长任务状态只保存在内存里，必须可观察、可持久化
- 不要隐藏 `partial_failed` 之类的降级结果
- 不要无条件信任代理头，必须受 `TRUST_PROXY_HEADERS` 和 `TRUSTED_PROXY_IPS` 控制
- 不要把未来规划写成当前已存在的实现

## 代码风格默认值

- 保持函数职责单一，优先写清晰直接的代码
- 错误处理要显式，不走静默失败路径
- 行为、结构、公共导出或脚本入口变了就更新文档
- 新增数据库字段、索引或约束时要同步迁移与测试

## 数据与模型规则

- 需要持久化的新业务事实，必须同时更新 ORM、Alembic、schema 和测试
- 只改变聚合响应时，优先先看是否只需要 schema 层调整
- 尽量让快照读路径与实时重算路径边界清晰

## 共享参考

- 当前后端事实：[../current/backend.md](../current/backend.md)
- 验证命令：[../reference/verification.md](../reference/verification.md)
