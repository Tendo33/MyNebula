# MyNebula Docker Compose 部署指南

本指南对应仓库根目录当前的 `docker-compose.yml`。

Compose 会启动两个服务：

- `db`: `pgvector/pgvector:pg16`
- `api`: `simonsun3/mynebula:latest`

前端静态资源已经包含在 `api` 镜像里，所以浏览器直接访问后端端口即可。

## 前置要求

```bash
docker --version
docker compose version
```

建议至少：

- 1 核 CPU
- 1 GB 内存
- 能稳定访问 GitHub API
- 能稳定访问 embedding / LLM 服务

## 快速部署

### 1. 获取文件

推荐直接克隆仓库：

```bash
git clone https://github.com/Tendo33/MyNebula.git
cd MyNebula
cp .env.example .env
```

如果你只想最小部署，至少也需要：

- `docker-compose.yml`
- `.env.example`

## 2. 编辑 `.env`

最少配置这些项：

```bash
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
EMBEDDING_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
ADMIN_PASSWORD=
ADMIN_SESSION_SECRET=
READ_ACCESS_MODE=authenticated
FORCE_SECURE_COOKIES=true
TRUST_PROXY_HEADERS=true
TRUSTED_HOSTS=your-domain.com
HTTPS_REDIRECT=true
```

如果你要启用 AI 摘要或聚类命名，再补充：

```bash
LLM_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
LLM_OUTPUT_LANGUAGE=zh
```

生产环境强烈建议额外修改：

- `DATABASE_PASSWORD`
- `ADMIN_USERNAME`
- `API_PORT`

## 3. 启动

```bash
docker compose up -d
```

启动后可访问：

- 主界面：<http://localhost:8000>
- 健康检查：<http://localhost:8000/health>
- OpenAPI：<http://localhost:8000/docs>，仅 `DEBUG=true` 可用

## 首次启动时会发生什么

`api` 服务启动时会：

1. 等待 `db` 健康检查通过
2. 建立数据库连接
3. 执行 `CREATE EXTENSION IF NOT EXISTS vector`
4. 检查 Alembic head
5. 如果 `alembic_version` 缺失，自动尝试迁移到 `head`
6. 启动 FastAPI 与 scheduler

这意味着全新数据库通常不需要你手动跑迁移。

## 什么时候仍然要手动迁移

如果数据库不是全新的，而是旧版本遗留数据，启动时可能出现 migration mismatch。此时执行：

```bash
docker compose exec api uv run alembic upgrade head
```

## Compose 文件里的关键约定

### 数据库容器

- 服务名固定为 `db`
- 容器名默认 `mynebula-db`
- 数据卷为 `postgres_data`
- 宿主机默认暴露 `${DATABASE_PORT:-5432}:5432`

### API 容器

- 服务名固定为 `api`
- 容器名默认 `mynebula-api`
- 容器内部强制覆盖：
  - `API_PORT=8000`
  - `DATABASE_HOST=db`
  - `DATABASE_PORT=5432`
  - `DATABASE_URL=`

所以在 Docker Compose 下：

- `.env` 里的 `DATABASE_HOST` 和 `DATABASE_PORT` 主要影响宿主机访问
- `api` 容器始终通过容器网络访问 `db:5432`

## 常用运维命令

### 查看状态和日志

```bash
docker compose ps
docker compose logs -f api
docker compose logs -f db
```

### 重启

```bash
docker compose restart
```

### 升级镜像

```bash
docker compose pull
docker compose up -d
```

如果升级后涉及数据库结构变化，再执行：

```bash
docker compose exec api uv run alembic upgrade head
```

### 进入容器

```bash
docker compose exec api /bin/sh
docker compose exec db psql -U "${DATABASE_USER:-mynebula}" -d "${DATABASE_NAME:-mynebula}"
```

## 修改端口

如果想把应用暴露到 `9090`：

```bash
API_PORT=9090
```

然后重启：

```bash
docker compose up -d
```

## 数据库端口冲突

如果宿主机已经占用 `5432`，有两种做法。

### 方案 A：不暴露数据库端口

删除 `db` 服务的 `ports` 配置。API 容器仍然能通过 Docker 网络访问数据库。

### 方案 B：修改宿主机映射端口

例如：

```bash
DATABASE_PORT=55432
```

这样宿主机访问数据库就改为 `localhost:55432`，容器内部仍然是 `5432`。

## 常见问题

### 启动成功但 `/settings` 无法登录

通常是没同时设置：

- `ADMIN_PASSWORD`
- `ADMIN_SESSION_SECRET`

管理员鉴权只有这两项都非空时才启用。

如果是通过 Nginx / Caddy / Traefik 等反向代理访问，还要确认：

- `TRUST_PROXY_HEADERS=true`
- `FORCE_SECURE_COOKIES=true`
- `READ_ACCESS_MODE=authenticated`
- 代理正确转发了 `X-Forwarded-Proto=https`

### 应用启动失败并提示 migration mismatch

执行：

```bash
docker compose exec api uv run alembic upgrade head
```

### 更换 embedding 或 LLM 服务需要重建镜像吗

不需要。只要它们通过环境变量配置，改 `.env` 后重启即可：

```bash
docker compose up -d
```

### 想彻底删库重来

```bash
docker compose down -v
docker compose up -d
```

注意这会删除 `postgres_data` volume。

## 相关文档

- `README.md`
- `doc/ENV_VARS.md`
- `doc/RESET_GUIDE.md`
