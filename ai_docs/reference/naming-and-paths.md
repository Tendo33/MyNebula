# Naming And Paths Reference

## 用途

本文件集中说明命名、导入和文档路径规则。其他文档要引用路径或命名约定时，只链接这里。

## Python 导入规则

- 安装后直接 `import nebula`
- 不要写 `from src.nebula ...`
- 程序化入口优先通过稳定模块导入，例如：
  - `nebula.main`
  - `nebula.core.config`
  - `nebula.schemas`

## 前端路径规则

- 前端源码根是 `frontend/src`
- API 适配层放在 `frontend/src/api`
- 搜索与共享纯函数放在 `frontend/src/utils`
- 页面容器放在 `frontend/src/pages`

## 文档路径规则

- 主文档只放 `ai_docs/`
- `current/` 只写当前实现
- `standards/` 只写规则和默认值
- `reference/` 只写共享事实
- `doc/` 只保留兼容页和图片资源

## 入口文件规则

- `AGENTS.md` 和 `CLAUDE.md` 保持轻量
- 两者都只做入口导航，不重复维护整套文档正文
- 与 Trellis 相关的管理块和工作流引用必须保留，但项目主阅读顺序要先指向 `ai_docs/`

## 路径书写规则

- 说明项目内文件时，优先写仓库相对路径
- 写验证命令时，默认在仓库根目录执行
- 写运行时文件路径时，区分清楚导入路径与文件系统路径，不要混用
