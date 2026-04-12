# Current Release

## 什么时候读

- 准备发版、对齐 CI 门禁或确认 Docker 镜像构建流程时先读这里

## 当前发布事实

- 日常门禁在 `.github/workflows/ci.yml`
- tag 发布在 `.github/workflows/release.yml`
- 版本同步脚本是 `scripts/update_version.py`

## 当前 CI 门禁

`ci.yml` 当前会运行：

- 后端 Ruff check
- 后端 Ruff format check
- 后端单元测试
- 前端 TypeScript typecheck
- 前端 ESLint
- 前端 Vitest
- 前端生产构建

详细命令以 [verification.md](../reference/verification.md) 为准。

## 当前 release 行为

- 推送 `v*` tag 时触发 release workflow
- GitHub Release 使用自动生成的 release notes
- 同时构建并推送 Docker 镜像
- Docker 镜像标签包含 semver、major/minor 和 `latest`

## 当前发版前建议

- 先跑本地 Full stack 验证
- 确认 `CHANGELOG.md`、`README.md` 和 `ai_docs/` 没有与本次变更冲突的过时描述
- 需要更新版本时优先用 `scripts/update_version.py`

## 共享参考

- 验证命令：[../reference/verification.md](../reference/verification.md)
- 文档规范：[../standards/documentation.md](../standards/documentation.md)
