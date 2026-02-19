# MyNebula Docker Compose 部署指南

本文档介绍如何通过 Docker Compose 一键部署 MyNebula（包含 PostgreSQL 数据库 + 后端 API + 前端界面）。

---

## 目录

- [前置要求](#前置要求)
- [快速部署（3 步完成）](#快速部署3-步完成)
- [配置说明](#配置说明)
- [常见问题](#常见问题)
- [运维操作](#运维操作)
- [架构说明](#架构说明)

---

## 前置要求

| 依赖 | 最低版本 | 说明 |
|------|---------|------|
| Docker | 20.10+ | [安装指南](https://docs.docker.com/get-docker/) |
| Docker Compose | v2.0+ | 通常随 Docker Desktop 自带 |

验证安装：

```bash
docker --version
docker compose version
```

> **服务器要求**：建议至少 1 核 CPU / 1GB 内存。PostgreSQL + pgvector 会占用约 200-500MB 内存。

---

## 快速部署（3 步完成）

### 第 1 步：获取配置文件

你**不需要克隆整个仓库**，只需要两个文件：

```bash
# 创建项目目录
mkdir mynebula && cd mynebula

# 下载配置文件
curl -O https://raw.githubusercontent.com/yourusername/mynebula/main/docker-compose.yml
curl -O https://raw.githubusercontent.com/yourusername/mynebula/main/.env.example

# 创建 .env
cp .env.example .env
```

或者克隆仓库：

```bash
git clone https://github.com/yourusername/mynebula.git
cd mynebula
cp .env.example .env
```

### 第 2 步：编辑 .env（填写必填项）

用编辑器打开 `.env` 文件，**至少**修改以下三项：

```bash
# [必填] GitHub Personal Access Token
# 创建地址: https://github.com/settings/tokens
# 勾选权限: public_repo, read:user
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx

# [必填] Embedding API Key (向量计算)
EMBEDDING_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx

# [推荐] LLM API Key (AI 摘要)
LLM_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx

# [可选] LLM 输出语言（zh=中文，en=英文）
LLM_OUTPUT_LANGUAGE=zh
```

> **安全提示**：生产环境请务必修改 `DATABASE_PASSWORD`，不要使用默认值。

### 第 3 步：启动

```bash
docker compose up -d
```

首次启动会自动完成：
1. 拉取 PostgreSQL (pgvector) 和 MyNebula Docker 镜像
2. 创建数据库并初始化 pgvector 扩展
3. **自动建表**（应用启动时自动执行，无需手动操作）
4. 启动 API 服务并挂载前端界面

等待约 30-60 秒后，访问：

| 服务 | 地址 | 说明 |
|------|------|------|
| Web 界面 | http://localhost:8000 | 主界面（端口由 `API_PORT` 决定） |
| API 文档 | http://localhost:8000/docs | 仅 `DEBUG=true` 时可用 |
| 健康检查 | http://localhost:8000/health | 服务状态 |

---

## 配置说明

### 关于数据库初始化

**你不需要手动初始化数据库表。** MyNebula 应用启动时会自动完成以下操作：

1. 等待 PostgreSQL 容器健康检查通过（`service_healthy`）
2. 连接数据库，自动创建 `vector` 扩展（pgvector）
3. 通过 SQLAlchemy `create_all()` 自动创建所有表（若不存在）

这意味着首次 `docker compose up` 后，数据库就是可用状态，不需要运行任何 SQL 或迁移命令。

### 环境变量完整参考

#### 必填项

| 变量 | 说明 | 示例 |
|------|------|------|
| `GITHUB_TOKEN` | GitHub Personal Access Token | `ghp_xxxx` |
| `EMBEDDING_API_KEY` | Embedding 服务的 API Key | `sk-xxxx` |
| `LLM_API_KEY` | LLM 服务的 API Key（强烈推荐） | `sk-xxxx` |

#### 数据库

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DATABASE_USER` | `mynebula` | 数据库用户名 |
| `DATABASE_PASSWORD` | `mynebula_secret` | 数据库密码（**生产环境请修改**） |
| `DATABASE_NAME` | `mynebula` | 数据库名 |

> 注意：`DATABASE_HOST` 和 `DATABASE_PORT` 由 docker-compose.yml 内部管理（api 容器通过 Docker 网络连接 db 容器），**不需要在 .env 中设置**。

#### Docker 镜像

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DOCKER_IMAGE` | `sjfeng1999/mynebula:latest` | Docker Hub 镜像地址 |

#### 应用

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `API_PORT` | `8000` | 宿主机映射端口 |
| `DEBUG` | `false` | 调试模式（控制 /docs 是否可用） |
| `LOG_LEVEL` | `INFO` | 日志级别 |

#### Embedding 服务

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `EMBEDDING_BASE_URL` | `https://api.siliconflow.cn/v1` | API 地址 |
| `EMBEDDING_MODEL` | `BAAI/bge-large-zh-v1.5` | 模型名称 |
| `EMBEDDING_DIMENSIONS` | `1024` | 向量维度 |

#### LLM 服务

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_BASE_URL` | `https://api.siliconflow.cn/v1` | API 地址 |
| `LLM_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | 模型名称 |
| `LLM_OUTPUT_LANGUAGE` | `zh` | LLM 输出语言（`zh` 或 `en`） |

### 不需要配置的变量（已从 .env.example 移除）

| 变量 | 原因 |
|------|------|
| `DATABASE_HOST` | docker-compose.yml 内部固定为 `db`（容器服务名） |
| `DATABASE_PORT` | docker-compose.yml 内部固定为 `5432` |
| `DATABASE_URL` | docker-compose.yml 中显式设为空，防止覆盖 |
| `VITE_API_BASE_URL` | 前端已内置自动检测逻辑（使用 `window.location.origin`） |
| `FRONTEND_URL` | 代码中未实际使用 |
| `APP_VERSION` | 版本号由 Docker 镜像决定 |
| `LOG_FILE` | 容器内日志建议通过 `docker logs` 查看 |

---

## 常见问题

### Q: 需要手动建表 / 运行 SQL 吗？

**不需要。** 应用启动时会自动创建 pgvector 扩展和所有数据表。你只需要 `docker compose up -d` 即可。

### Q: 升级到新版本怎么做？

```bash
# 拉取最新镜像
docker compose pull

# 重启服务（数据库数据不会丢失）
docker compose up -d
```

数据存储在 Docker Volume `postgres_data` 中，升级应用不影响数据。

如果新版本包含数据库结构变更，可以进入容器运行 Alembic 迁移：

```bash
docker compose exec api uv run alembic upgrade head
```

### Q: 如何查看日志？

```bash
# 查看所有服务日志
docker compose logs -f

# 只看 API 日志
docker compose logs -f api

# 只看数据库日志
docker compose logs -f db

# 查看最近 100 行
docker compose logs --tail=100 api
```

### Q: 如何修改端口？

在 `.env` 中修改 `API_PORT`：

```bash
API_PORT=9090
```

然后重启：

```bash
docker compose up -d
```

访问地址变为 `http://localhost:9090`。

### Q: 数据库端口冲突（本机已有 PostgreSQL）？

默认会将 PostgreSQL 的 5432 端口映射到宿主机。如果本机已有 PostgreSQL 运行，有两种解决方案：

**方案 A**：不暴露数据库端口（推荐，更安全）

编辑 `docker-compose.yml`，删除 db 服务的 `ports` 配置：

```yaml
db:
  image: pgvector/pgvector:pg16
  # ports:                        # 注释或删除这两行
  #   - "${DATABASE_PORT:-5432}:5432"
```

API 容器通过 Docker 内部网络连接数据库，不需要宿主机端口映射。

**方案 B**：修改映射端口

在 `.env` 中添加：

```bash
DATABASE_PORT=15432
```

### Q: 如何使用 Ollama 本地模型？

在 `.env` 中修改 Embedding 配置：

```bash
EMBEDDING_BASE_URL=http://host.docker.internal:11434/v1
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSIONS=768
EMBEDDING_API_KEY=ollama
```

> `host.docker.internal` 是 Docker 提供的特殊域名，指向宿主机。Linux 用户可能需要在 docker-compose.yml 的 api 服务中添加 `extra_hosts: ["host.docker.internal:host-gateway"]`。

### Q: 如何备份数据？

```bash
# 备份数据库
docker compose exec db pg_dump -U mynebula mynebula > backup_$(date +%Y%m%d).sql

# 恢复数据库
cat backup_20260206.sql | docker compose exec -T db psql -U mynebula mynebula
```

### Q: 如何完全重置？

```bash
# 停止并删除容器
docker compose down

# 删除数据卷（会清除所有数据！）
docker volume rm mynebula_postgres_data

# 重新启动
docker compose up -d
```

---

## 运维操作

### 启动 / 停止 / 重启

```bash
# 启动（后台运行）
docker compose up -d

# 停止
docker compose down

# 重启
docker compose restart

# 重启单个服务
docker compose restart api
```

### 查看状态

```bash
# 服务状态
docker compose ps

# 健康检查
curl http://localhost:8000/health
```

### 进入容器调试

```bash
# 进入 API 容器
docker compose exec api bash

# 进入数据库容器
docker compose exec db psql -U mynebula
```

---

## 架构说明

```
┌──── Docker Compose ─────────────────────────────────────┐
│                                                         │
│  ┌─────────────┐       ┌───────────────────────────┐   │
│  │  PostgreSQL  │◄──────│     MyNebula API          │   │
│  │  + pgvector  │ :5432 │  (FastAPI + React SPA)    │   │
│  │  (db)        │       │  (api)                    │   │
│  └──────┬──────┘       └──────────┬────────────────┘   │
│         │                         │                     │
│    postgres_data             :8000 ──► 宿主机           │
│    (持久化卷)                                           │
│                        mynebula-net                      │
└─────────────────────────────────────────────────────────┘

外部依赖 (通过网络调用):
  - GitHub API ← GITHUB_TOKEN
  - Embedding API ← EMBEDDING_API_KEY
  - LLM API ← LLM_API_KEY
```

- **db 容器**：运行 PostgreSQL 16 + pgvector 扩展，数据持久化到 Docker Volume
- **api 容器**：运行 FastAPI 应用，同时托管 React 前端静态文件，单一端口对外服务
- **网络**：两个容器通过 `mynebula-net` Bridge 网络通信，API 通过容器名 `db` 连接数据库
- **前端**：已内置于 Docker 镜像，无需单独构建或部署
