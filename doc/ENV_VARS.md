# 环境变量配置指南

本文档说明 MyNebula 支持的环境变量，默认值来源以 `src/nebula/core/config.py` 为准。

## 快速开始

1. 复制示例配置文件：
   ```bash
   cp .env.example .env
   ```

2. 至少配置以下必填项：
   - `GITHUB_TOKEN`
   - `EMBEDDING_API_KEY`

3. 推荐配置（用于后台登录与管理操作）：
   - `ADMIN_PASSWORD`
   - `ADMIN_USERNAME`（可保持默认 `admin`）

4. 可选配置（用于摘要与聚类命名）：
   - `LLM_API_KEY`

---

## 应用与运行时配置

| 变量名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `APP_NAME` | string | `mynebula` | 应用名称 |
| `APP_VERSION` | string | `1.1.0` | 应用版本 |
| `DEBUG` | boolean | `false` | 调试模式（控制 `/docs` 是否开放） |
| `SINGLE_USER_MODE` | boolean | `true` | 单用户模式（读取接口默认使用首个用户） |
| `SNAPSHOT_READ_FALLBACK_ON_ERROR` | boolean | `true` | 快照读取失败时回退到实时构建 |
| `API_PORT` | integer | `8000` | API 监听端口 |
| `SLOW_QUERY_LOG_MS` | integer | `200` | 慢查询日志阈值（毫秒） |
| `API_QUERY_TIMEOUT_SECONDS` | integer | `15` | 查询超时时间（秒） |
| `LOG_LEVEL` | string | `INFO` | 日志级别：`TRACE/DEBUG/INFO/SUCCESS/WARNING/ERROR/CRITICAL` |
| `LOG_FILE` | string | `logs/app.log` | 日志文件路径 |

---

## 管理员鉴权

| 变量名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `ADMIN_USERNAME` | string | `admin` | 管理员用户名（用于 `/settings`） |
| `ADMIN_PASSWORD` | string | *(空)* | 管理员密码（为空则管理员接口不可用） |
| `ADMIN_SESSION_TTL_HOURS` | integer | `24` | 管理员会话有效期（小时） |

---

## GitHub 配置（必填）

| 变量名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `GITHUB_TOKEN` | string | ✅ | GitHub Personal Access Token (PAT) |

> 建议权限：`read:user`、`public_repo`。如需访问私有仓库，请追加相应权限。

---

## 数据库配置（PostgreSQL + pgvector）

支持两种配置方式：

### 方式 1：分离参数（推荐 Docker Compose）

| 变量名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `DATABASE_HOST` | string | `localhost` | 数据库主机 |
| `DATABASE_PORT` | integer | `5432` | 数据库端口 |
| `DATABASE_USER` | string | `mynebula` | 数据库用户 |
| `DATABASE_PASSWORD` | string | `mynebula_secret` | 数据库密码 |
| `DATABASE_NAME` | string | `mynebula` | 数据库名称 |

### 方式 2：完整 URL（优先级更高）

| 变量名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `DATABASE_URL` | string | *(空)* | 完整数据库 URL（覆盖分离式配置） |

支持 `postgresql://...` 或 `postgresql+asyncpg://...`，系统会自动处理 async 驱动前缀。

---

## Embedding 配置（必填）

用于生成仓库描述向量，使用 OpenAI 兼容接口。

| 变量名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `EMBEDDING_API_KEY` | string | *(空)* | API 密钥（必填） |
| `EMBEDDING_BASE_URL` | string | `https://api.siliconflow.cn/v1` | API 基础 URL |
| `EMBEDDING_MODEL` | string | `BAAI/bge-large-zh-v1.5` | 模型名称 |
| `EMBEDDING_DIMENSIONS` | integer | `1024` | 向量维度 |

---

## LLM 配置（可选）

用于摘要与聚类命名，使用 OpenAI 兼容接口。

| 变量名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `LLM_API_KEY` | string | *(空)* | API 密钥 |
| `LLM_BASE_URL` | string | `https://api.siliconflow.cn/v1` | API 基础 URL |
| `LLM_MODEL` | string | `Qwen/Qwen2.5-7B-Instruct` | 模型名称 |
| `LLM_OUTPUT_LANGUAGE` | string | `zh` | 输出语言（`zh` 或 `en`） |

---

## 同步配置

| 变量名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `SYNC_BATCH_SIZE` | integer | `100` | 每批处理的仓库数量（10-500） |
| `SYNC_README_MAX_LENGTH` | integer | `10000` | README 最大抓取长度（字符） |
| `SYNC_DEFAULT_SYNC_MODE` | string | `incremental` | 默认同步模式（`incremental`/`full`） |
| `SYNC_DETECT_UNSTARRED_ON_INCREMENTAL` | boolean | `false` | 增量同步时检测取消 Star 的仓库（增加 API 开销） |

---

## 前端开发（Vite）

| 变量名 | 说明 |
|--------|------|
| `VITE_API_BASE_URL` | 前端开发时 API 目标地址（仅开发环境使用） |

> 在开发模式中，前端默认通过 Vite 代理访问 `/api`，如需直连后端可设置 `VITE_API_BASE_URL`。

---

## 示例配置（开发环境）

```bash
APP_NAME=mynebula
APP_VERSION=1.1.0
DEBUG=true
SINGLE_USER_MODE=true
SNAPSHOT_READ_FALLBACK_ON_ERROR=true
API_PORT=8000
SLOW_QUERY_LOG_MS=200
API_QUERY_TIMEOUT_SECONDS=15

GITHUB_TOKEN=your_pat_xxxxxxxxxxxxxxxx

DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_USER=mynebula
DATABASE_PASSWORD=mynebula_secret
DATABASE_NAME=mynebula

EMBEDDING_API_KEY=your_embedding_api_key
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
EMBEDDING_DIMENSIONS=1024

LLM_API_KEY=your_llm_api_key
LLM_BASE_URL=https://api.siliconflow.cn/v1
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
LLM_OUTPUT_LANGUAGE=zh

ADMIN_USERNAME=admin
ADMIN_PASSWORD=change_me
ADMIN_SESSION_TTL_HOURS=24

SYNC_BATCH_SIZE=100
SYNC_README_MAX_LENGTH=10000
SYNC_DEFAULT_SYNC_MODE=incremental
SYNC_DETECT_UNSTARRED_ON_INCREMENTAL=false
```

---

## 安全建议

1. **不要提交 `.env` 到版本控制**。
2. 生产环境务必设置强密码的 `ADMIN_PASSWORD` 与数据库密码。
3. 定期轮换 API 密钥。
4. 使用 Secrets 管理工具（如 Vault、AWS Secrets Manager）存储敏感信息。
