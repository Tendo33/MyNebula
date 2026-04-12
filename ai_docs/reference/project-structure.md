# Project Structure Reference

## 用途

本文件集中说明当前仓库目录结构。其他文档要描述目录、位置或扩展点时，只链接这里。

## 当前仓库结构

```text
MyNebula/
├── src/nebula/                 # FastAPI backend package
│   ├── api/                    # v1 / v2 路由层
│   ├── application/services/   # 同步、快照、推荐、聚合查询
│   ├── core/                   # 配置、认证、日志、调度、embedding、llm
│   ├── db/                     # ORM 模型和数据库生命周期
│   ├── domain/                 # 生命周期枚举和共享领域对象
│   ├── infrastructure/         # 快照持久化仓储
│   ├── schemas/                # 通用与 v2 API schema
│   └── utils/                  # 通用后端工具
├── frontend/                   # React + TypeScript SPA
│   └── src/
│       ├── api/                # 前端 API 适配层
│       ├── components/         # UI 与图谱组件
│       ├── contexts/           # 共享上下文与过滤状态
│       ├── features/           # 按业务组织的 query hooks
│       ├── pages/              # 页面容器
│       ├── utils/              # 搜索、格式化等共享工具
│       └── locales/            # i18n 资源
├── alembic/                    # 数据库迁移
├── tests/                      # 后端测试
├── scripts/                    # 维护、评估、性能和检查脚本
├── ai_docs/                    # 当前唯一主文档系统
│   ├── current/                # 当前实现事实
│   ├── standards/              # 工程约束与默认做法
│   └── reference/              # 共享参考事实
├── doc/                        # 兼容入口与 README 图片资源
│   └── images/                 # README 展示图片
├── .trellis/                   # Trellis 工作流、spec、tasks、workspace
├── .github/workflows/          # CI 与 release workflows
├── AGENTS.md                   # 跨工具根入口
├── CLAUDE.md                   # Claude 根入口
├── README.md
└── README.zh.md
```

## 当前文档分工

- `ai_docs/`：项目主文档系统
- `doc/`：兼容跳转页和图片资源，不再承担主文档职责
- `.trellis/`：Trellis 工作流与会话协作资产

## 扩展规则

- 新的项目说明优先加入 `ai_docs/`
- 只有与 Trellis 工作流强绑定的内容才放 `.trellis/`
- 不再向 `doc/` 新增主题型主文档
