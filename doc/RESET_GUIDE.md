# 数据库重置与重新运行指南

适用于本地开发环境；如使用 Docker 部署，请优先按“Docker 方式”操作。

## 1. 强制重置数据库

运行脚本删除所有表：

```bash
uv run python scripts/reset_db.py
```

> 脚本会等待 5 秒用于确认。

---

## 2. 重建数据库结构

### 本地开发

```bash
uv run alembic upgrade head
```

### Docker 部署

```bash
docker compose exec api uv run alembic upgrade head
```

---

## 3. 启动服务

### 本地开发

```bash
# 后端
uv run uvicorn nebula.main:app --reload --port 8000

# 前端（新终端）
npm --prefix frontend install
VITE_API_BASE_URL=http://localhost:8000 npm --prefix frontend run dev
```

然后访问：<http://localhost:5173>

### Docker 部署

```bash
docker compose up -d
```

访问：<http://localhost:8000>

---

## 4. 触发同步

**方式 A：界面操作**

1. 打开 `/settings`
2. 使用 `ADMIN_USERNAME` / `ADMIN_PASSWORD` 登录
3. 触发 **Sync Pipeline**（增量或全量）

**方式 B：API（需管理员会话）**

```bash
curl -X POST "http://localhost:8000/api/v2/sync/start?mode=full"
```

> 注意：该接口需要管理员登录建立会话（Cookie）。
