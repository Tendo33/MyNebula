# 数据库重置与重新运行指南 (修复版)

由于 `alembic downgrade` 遇到约束命名问题，请按照以下新步骤操作。

## 1. 强制重置数据库

我已为你创建了一个 Python 脚本来强制删除所有表。

在终端中运行：

```bash
uv run python scripts/reset_db.py
```

*注意：这会等待 5 秒钟让你确认。*

## 2. 重建数据库

脚本执行完成后，运行以下命令重新创建表结构：

```bash
uv run alembic upgrade head
```

## 3. 重新运行流程

### 启动应用

如果前后台未运行，请启动它们：

```bash
# 后端
uv run mynebula

# 前端 (新终端)
cd frontend
npm run dev
```

### 触发同步

访问 `http://localhost:3000` 并点击 **Sync Stars**。

或者使用 API：
```bash
curl -X POST "http://localhost:8071/api/sync/stars?sync_mode=full"
```
