# 重置数据库

重置会删掉应用表和 `alembic_version`。步骤和警告在贡献指南的数据库一节，脚本是 [`scripts/reset_db.py`](../scripts/reset_db.py)。

- [数据库和迁移](../CONTRIBUTING.md#数据库和迁移)

```bash
uv run python scripts/reset_db.py
uv run alembic upgrade head
```

脚本会先等 5 秒。这段时间可以按 Ctrl+C 取消。
