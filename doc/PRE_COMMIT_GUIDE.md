# Pre-commit 使用指南

MyNebula 通过 `pre-commit` 在提交前自动执行一批轻量检查，主要负责基础文件卫生和 Python Ruff 规范，不负责替代完整测试流程。

## 一次性安装

先准备依赖，再安装 Git hooks：

```bash
uv sync --extra dev
uv run pre-commit install
```

安装成功后通常会看到：

```text
pre-commit installed at .git/hooks/pre-commit
```

## 日常工作流

```bash
git add .
git commit -m "docs: refresh docker guide"
```

执行 `git commit` 时，pre-commit 会自动运行。如果某个 hook 修改了文件或发现错误，提交会被拦下，你需要：

1. 查看输出
2. 修复或接受自动修复结果
3. 重新 `git add`
4. 再次提交

## 常用命令

```bash
# 对所有文件执行一次完整检查
uv run pre-commit run --all-files

# 只检查当前暂存区
uv run pre-commit run

# 更新 hook 版本
uv run pre-commit autoupdate

# 卸载 hooks
uv run pre-commit uninstall
```

## 当前仓库启用了哪些 hook

`.pre-commit-config.yaml` 目前包含：

| Hook | 作用 |
| --- | --- |
| `trailing-whitespace` | 删除行尾空格 |
| `end-of-file-fixer` | 确保文件以换行结尾 |
| `check-yaml` | 校验 YAML 语法 |
| `check-toml` | 校验 TOML 语法 |
| `check-json` | 校验 JSON 语法 |
| `check-merge-conflict` | 检测冲突标记 |
| `debug-statements` | 检测遗留调试语句 |
| `ruff` | Python lint，可自动修复一部分问题 |
| `ruff-format` | Python 格式化 |

## 它不会帮你做什么

pre-commit 当前不会自动执行下面这些较重检查：

- `uv run pytest`
- `npm --prefix frontend run lint`
- `npx --prefix frontend tsc --noEmit`
- `npm --prefix frontend run test`
- `npm --prefix frontend run build`
- `uv run python scripts/evals/run_all_quality_checks.py`

这些仍然需要你按改动范围手动执行。

## 常见问题

### 为什么提交时没有自动触发

通常是因为还没安装 hooks：

```bash
uv run pre-commit install
```

### Hook 自动改了文件，怎么办

很正常。重新暂存后再提交即可：

```bash
git add .
git commit -m "..."
```

### 我只想先看结果，不想真的提交

直接运行：

```bash
uv run pre-commit run --all-files
```

### VS Code 里怎么更顺滑

可以让 Python 文件保存时直接走 Ruff：

```json
{
  "editor.formatOnSave": true,
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff"
  }
}
```

这样很多格式问题会在提交前就被消掉。
