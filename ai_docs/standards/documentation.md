# Documentation Standards

## 什么时候读

- 修改 `README*`、`ai_docs/`、根入口文件或兼容页前先读这里

## 分层规则

- `current/` 只写当前真实实现
- `standards/` 只写工程规则和默认值
- `reference/` 只写共享事实

## 写作规则

- 先写能直接执行的信息，再写背景解释
- 把“当前实现”和“推荐扩展”明确分开
- 不用占位词伪装成规范
- 示例只引用当前仓库真实存在的文件、命令和入口
- 主文档系统是 `ai_docs/`，不要再把新主题写回 `doc/`

## 链接规则

- 命令统一链接到 [verification.md](../reference/verification.md)
- 目录结构统一链接到 [project-structure.md](../reference/project-structure.md)
- 路径与导入统一链接到 [naming-and-paths.md](../reference/naming-and-paths.md)

## README 规则

- README 面向人类入口，不承担完整 AI 协作契约
- README 里的文档部分只说明入口和阅读路径
- 详细规则留在 `ai_docs/`

## 根入口规则

- `AGENTS.md` 和 `CLAUDE.md` 都应保持轻量
- 两者都必须优先指向当前存在的 `ai_docs/`
- `AGENTS.md` 中由 Trellis 管理的块必须保留
