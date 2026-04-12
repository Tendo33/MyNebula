# Current Operations

## 什么时候读

- 要部署、配置、重置或排查运行环境问题时先读这里
- 需要确认 `.env`、Docker Compose、登录边界和恢复流程时先读这里

## 当前运行方式

MyNebula 当前主要有两种运行方式：

- Docker Compose：`db` + `api` 两个服务一起启动，前端构建产物随 `api` 镜像提供
- 本地开发：后端与前端分开运行，数据库通常仍由 Docker 提供

## 当前关键环境变量分组

- GitHub：`GITHUB_TOKEN`
- Embedding：`EMBEDDING_API_KEY`、`EMBEDDING_BASE_URL`、`EMBEDDING_MODEL`
- LLM：`LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL`、`LLM_OUTPUT_LANGUAGE`
- 管理认证：`ADMIN_USERNAME`、`ADMIN_PASSWORD`、`ADMIN_SESSION_SECRET`
- 访问边界：`READ_ACCESS_MODE`
- 代理信任：`TRUST_PROXY_HEADERS`、`TRUSTED_PROXY_IPS`、`TRUSTED_HOSTS`
- 运行时：`DEBUG`、`API_PORT`、`SLOW_QUERY_LOG_MS`、`API_QUERY_TIMEOUT_SECONDS`
- 同步：`SYNC_*`

### 当前环境变量示例片段

```bash
APP_VERSION=1.2.7
READ_ACCESS_MODE=demo
API_PORT=8000
```

## 当前部署与安全事实

- `/docs` 与 `/redoc` 只在 `DEBUG=true` 时可用
- 只有 `ADMIN_PASSWORD` 和 `ADMIN_SESSION_SECRET` 同时非空时，管理写接口和登录能力才有效
- 只有 `TRUST_PROXY_HEADERS=true` 且请求来源命中 `TRUSTED_PROXY_IPS` 时，系统才信任 `X-Forwarded-*`
- `frontend/dist` 存在时，FastAPI 会直接托管前端静态资源

## 当前恢复与重置方式

- 只重跑业务数据而不动 schema：优先使用 `/settings` 里的 `Full Refresh`
- 重置本地数据库：`uv run python scripts/reset_db.py` 后再 `uv run alembic upgrade head`
- Docker 环境清空业务表：`docker compose exec api uv run python scripts/reset_db.py`
- Docker 环境彻底删库：`docker compose down -v`

## 当前参考文件

- 环境变量实现：`src/nebula/core/config.py`
- 应用启动：`src/nebula/main.py`
- Compose：`docker-compose.yml`
- 数据库重置脚本：`scripts/reset_db.py`

## 共享参考

- 验证命令：[../reference/verification.md](../reference/verification.md)
- 项目结构：[../reference/project-structure.md](../reference/project-structure.md)
