# MyNebula AI Docs Start Here

## 什么时候先读这里

- 第一次进入这个仓库时先读这里
- 不确定某项任务该看哪份文档时先回到这里
- 需要区分“当前实现”“工程规范”“共享参考”时先回到这里

## 这套文档是什么

`ai_docs/` 是 MyNebula 当前唯一的主文档系统，面向 AI 助手、自动化工具和工程协作者。

目录分层：

- `current/`：只写当前仓库已经落地的真实实现
- `standards/`：只写默认工程约束、协作约定和 UI/代码规则
- `reference/`：只写共享事实，例如验证命令、目录结构、命名与路径规则

`.trellis/` 仍然保留，但它现在只承担 Trellis 工作流、任务记录和工作区脚本职责，不再是项目主文档入口。

## 按任务快速进入

- 想快速理解仓库：先读 [INDEX.md](INDEX.md)，再读 [current/architecture.md](current/architecture.md)
- 想改后端：先读 [current/backend.md](current/backend.md)，再读 [standards/backend.md](standards/backend.md)
- 想改前端：先读 [current/frontend.md](current/frontend.md)，再读 [standards/frontend.md](standards/frontend.md) 和 [standards/design-system.md](standards/design-system.md)
- 想处理部署、环境变量、重置或运维：先读 [current/operations.md](current/operations.md)
- 想改脚本、版本或发布流程：先读 [current/scripts.md](current/scripts.md) 和 [current/release.md](current/release.md)
- 想确认命令、目录和路径：只引用 `reference/` 下的文档

## 共享入口

- 验证命令唯一详细来源：[reference/verification.md](reference/verification.md)
- 项目结构唯一详细来源：[reference/project-structure.md](reference/project-structure.md)
- 路径与导入规则唯一详细来源：[reference/naming-and-paths.md](reference/naming-and-paths.md)
