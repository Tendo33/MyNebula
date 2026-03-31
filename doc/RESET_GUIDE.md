# 数据库重置与重新运行指南

本文档用于两类场景：

- 本地开发时把数据库清空后重新跑一遍
- Docker 环境下彻底清库或从迁移异常中恢复

## 先区分你要哪一种“重置”

### 只想重跑数据，不想删表结构

优先使用 Settings 页面里的 `Full Refresh`，它会重置仓库处理状态并重新跑同步链路，但不会破坏数据库 schema。

### 真正删库重来

使用下面的 reset 流程。

## 本地开发重置

### 1. 停掉正在运行的服务

如果本地跑着 Uvicorn、前端 dev server 或其他脚本，先停掉，避免持有连接。

### 2. 删除应用表

```bash
uv run python scripts/reset_db.py
```

这个脚本会：

- 等待 5 秒确认
- 删除 `Base.metadata` 管理的所有表
- 额外删除 `alembic_version`

### 3. 重新应用迁移

```bash
uv run alembic upgrade head
```

### 4. 启动服务

```bash
uv run uvicorn nebula.main:app --reload --port 8000
VITE_API_BASE_URL=http://localhost:8000 npm --prefix frontend run dev
```

### 5. 重新同步数据

去 `/settings` 登录后，重新触发一次 `incremental` 或 `full` 同步。

## Docker 环境重置

Docker 下常见有两种方式。

### 方式 A：只清空表，保留 volume

```bash
docker compose exec api uv run python scripts/reset_db.py
docker compose exec api uv run alembic upgrade head
```

适合：

- 只是想清空业务数据
- 不想删除整个 PostgreSQL volume

### 方式 B：连 volume 一起删掉

```bash
docker compose down -v
docker compose up -d
```

适合：

- 想得到一个绝对全新的数据库
- 本地测试 migration/bootstrap

注意：这会删除 `postgres_data` volume 中的所有数据。

## 迁移异常恢复

如果启动日志提示 migration version mismatch：

### 本地

```bash
uv run alembic upgrade head
```

### Docker

```bash
docker compose exec api uv run alembic upgrade head
```

代码启动逻辑会在 `alembic_version` 不存在时自动尝试迁移到 head；但如果数据库已经存在且版本落后或分叉，仍然需要你手动执行升级。

## 验证恢复是否完成

至少检查以下几项：

### 健康检查

```bash
curl http://localhost:8000/health
```

### 设置页是否可登录

- 打开 `/settings`
- 确认 `ADMIN_PASSWORD` 和 `ADMIN_SESSION_SECRET` 已配置

### 能否成功触发同步

- `incremental` 或 `full` 能正常启动
- `/api/v2/sync/jobs/{run_id}` 能返回状态

## 常见坑

- 只执行了 `reset_db.py`，但没有重新 `alembic upgrade head`
- 忘了重启后端，仍在使用旧连接
- Docker volume 没删干净，以为自己拿到的是“全新库”
- 管理员鉴权没配，导致以为系统恢复失败，其实只是登录被禁用
