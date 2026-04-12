# Current Scripts

## 什么时候读

- 想了解仓库维护脚本、本地工具和辅助检查时先读这里
- 想确认脚本修改会影响哪些入口和文档时先读这里

## 当前脚本概览

仓库当前提供这些主要脚本：

- `scripts/rename_package.py`：重命名 Python 包和项目名
- `scripts/update_version.py`：同步更新版本号
- `scripts/setup_pre_commit.py`：安装、更新或测试 pre-commit hooks
- `scripts/run_vulture.py`：扫描潜在未使用的 Python 代码
- `scripts/reset_db.py`：重置应用数据库表
- `scripts/evals/run_all_quality_checks.py`：离线质量评估总入口
- `scripts/checks/no_legacy_page_api_usage.sh`：前端 legacy API 使用约束
- `scripts/perf/*`：性能基准、回滚演练和 pipeline soak 工具

## 什么时候用哪个

### 包名或项目名迁移

- 用 `rename_package.py`
- 它会扫描源码、文档和配置里的命名引用
- 改这类脚本时要同步更新 `ai_docs/`，不能只改 README

### 版本更新

- 用 `update_version.py`
- 当前会更新 Python 包版本、前端版本、锁文件和 AI 文档中的版本示例
- 当前版本示例：`1.2.6`

### pre-commit 管理

- 用 `setup_pre_commit.py`
- 负责安装、升级和一次性执行 hooks，不替代完整测试流程

### 死代码扫描

- 用 `run_vulture.py`
- 适合在重构前后做快速扫描，不替代人工判断和测试

### 质量与评估

- 规则验证看 [verification.md](../reference/verification.md)
- 离线推荐/聚类评估由 `scripts/evals/run_all_quality_checks.py` 驱动

## 共享参考

- 发布流程：[release.md](release.md)
- 验证命令：[../reference/verification.md](../reference/verification.md)
