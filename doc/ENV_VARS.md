# 环境变量配置指南

本文档基于 `src/nebula/core/config.py` 和当前前端开发配置整理，优先级以代码实现为准。

## 最小可运行配置

复制模板：

```bash
cp .env.example .env
```

至少需要设置：

- `GITHUB_TOKEN`
- `EMBEDDING_API_KEY`
- `EMBEDDING_BASE_URL`
- `EMBEDDING_MODEL`

如果你要使用 `/settings` 登录和所有写操作接口，还必须同时设置：

- `ADMIN_PASSWORD`
- `ADMIN_SESSION_SECRET`

如果要启用摘要、标签、聚类命名，再设置：

- `LLM_API_KEY`

## 应用与运行时

| 变量名 | 默认值 | 说明 |
| --- | --- | --- |
| `APP_NAME` | `mynebula` | 应用名 |
| `APP_VERSION` | `1.2.2` | 应用版本 |
| `DEBUG` | `false` | 为 `true` 时开放 `/docs` 和 `/redoc` |
| `SINGLE_USER_MODE` | `true` | 读取接口默认使用首个用户 |
| `SNAPSHOT_READ_FALLBACK_ON_ERROR` | `true` | 快照读取异常时回退到实时构建 |
| `API_PORT` | `8000` | 后端监听端口 |
| `LOG_LEVEL` | `INFO` | `TRACE/DEBUG/INFO/SUCCESS/WARNING/ERROR/CRITICAL` |
| `LOG_FILE` | `logs/app.log` | 日志文件 |
| `SLOW_QUERY_LOG_MS` | `200` | 慢查询阈值（毫秒） |
| `API_QUERY_TIMEOUT_SECONDS` | `15` | 大查询超时（秒） |
| `CORS_ORIGINS` | `""` | 逗号分隔来源列表；留空时仅允许 localhost/127.0.0.1 |

## GitHub

| 变量名 | 默认值 | 说明 |
| --- | --- | --- |
| `GITHUB_TOKEN` | `""` | GitHub PAT，用于同步 stars 和列表 |

建议权限：

- `read:user`
- `public_repo`

如果你需要读取私有仓库，再追加对应私有仓库权限。

## 数据库

### 分离式配置

| 变量名 | 默认值 | 说明 |
| --- | --- | --- |
| `DATABASE_HOST` | `localhost` | 数据库主机 |
| `DATABASE_PORT` | `5432` | 数据库端口 |
| `DATABASE_USER` | `mynebula` | 用户名 |
| `DATABASE_PASSWORD` | `mynebula_secret` | 密码 |
| `DATABASE_NAME` | `mynebula` | 数据库名 |

### URL 配置

| 变量名 | 默认值 | 说明 |
| --- | --- | --- |
| `DATABASE_URL` | `""` | 完整连接串，优先级高于分离式配置 |

支持：

- `postgresql://...`
- `postgresql+asyncpg://...`

代码会自动处理 asyncpg 驱动和部分 SSL 参数。

## Embedding

| 变量名 | 默认值 | 说明 |
| --- | --- | --- |
| `EMBEDDING_API_KEY` | `""` | embedding 服务密钥 |
| `EMBEDDING_BASE_URL` | `https://api.siliconflow.cn/v1` | OpenAI 兼容接口地址 |
| `EMBEDDING_MODEL` | `BAAI/bge-large-zh-v1.5` | 模型名 |
| `EMBEDDING_DIMENSIONS` | `1024` | 向量维度，范围 64-4096 |

## LLM

| 变量名 | 默认值 | 说明 |
| --- | --- | --- |
| `LLM_API_KEY` | `""` | LLM 服务密钥 |
| `LLM_BASE_URL` | `https://api.siliconflow.cn/v1` | OpenAI 兼容接口地址 |
| `LLM_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | 模型名 |
| `LLM_OUTPUT_LANGUAGE` | `zh` | 生成内容语言，`zh` 或 `en` |

## 管理员鉴权

| 变量名 | 默认值 | 说明 |
| --- | --- | --- |
| `ADMIN_USERNAME` | `admin` | 管理员用户名 |
| `ADMIN_PASSWORD` | `""` | 管理员密码 |
| `ADMIN_SESSION_SECRET` | `""` | 签名 session 与 CSRF token 的密钥 |
| `ADMIN_SESSION_TTL_HOURS` | `24` | 管理员会话有效期，范围 1-168 小时 |
| `ADMIN_LOGIN_RATE_LIMIT_WINDOW_SECONDS` | `300` | 管理员登录失败限流时间窗（秒） |
| `ADMIN_LOGIN_RATE_LIMIT_MAX_ATTEMPTS` | `5` | 单个 IP / 用户名在时间窗内允许的失败次数 |
| `FORCE_SECURE_COOKIES` | `false` | 生产环境建议开启，强制写入 Secure Cookie |

重要说明：

- 只有 `ADMIN_PASSWORD` 和 `ADMIN_SESSION_SECRET` 同时非空时，管理员鉴权才会启用
- 登录后后端会写入两个 Cookie：
  - `nebula_admin_session`
  - `nebula_admin_csrf`
- 如果部署在反向代理 / TLS 终止后面，建议同时配置 `TRUST_PROXY_HEADERS=true`

## 访问与反向代理

| 变量名 | 默认值 | 说明 |
| --- | --- | --- |
| `READ_ACCESS_MODE` | `demo` | 读接口访问模式：`demo` 或 `authenticated` |
| `TRUST_PROXY_HEADERS` | `false` | 是否信任 `X-Forwarded-*` 代理头 |
| `TRUSTED_HOSTS` | `""` | `TrustedHostMiddleware` 允许的 Host 列表，逗号分隔 |
| `HTTPS_REDIRECT` | `false` | 是否启用 HTTP -> HTTPS 重定向 |
| `CONTENT_SECURITY_POLICY` | `""` | 可选 CSP header 值 |

模式说明：

- `demo`：允许匿名只读访问，适合本地演示或受控内网环境
- `authenticated`：读接口也要求管理员登录，不再匿名暴露现有数据

## 同步相关

| 变量名 | 默认值 | 说明 |
| --- | --- | --- |
| `SYNC_BATCH_SIZE` | `100` | 每批处理仓库数，范围 10-500 |
| `SYNC_README_MAX_LENGTH` | `10000` | README 存储最大长度 |
| `SYNC_DEFAULT_SYNC_MODE` | `incremental` | 默认同步模式，`incremental` 或 `full` |
| `SYNC_DETECT_UNSTARRED_ON_INCREMENTAL` | `false` | 增量同步时是否主动检测取消 star 的仓库 |

## 前端开发变量

前端开发变量不从根目录 `.env` 读取，而是由 `frontend/` 目录下的 Vite 配置或命令行环境变量提供。

### 常用变量

| 变量名 | 默认值 | 说明 |
| --- | --- | --- |
| `VITE_API_BASE_URL` | `http://localhost:8071` | Vite 代理目标地址 |
| `VITE_API_URL` | 无 | 构建产物备用 API 基地址 |
| `E2E_BASE_URL` | `http://127.0.0.1:4173` | Playwright 目标地址 |

开发时推荐：

```bash
VITE_API_BASE_URL=http://localhost:8000 npm --prefix frontend run dev
```

## 开发环境示例

```bash
APP_NAME=mynebula
APP_VERSION=1.2.2
DEBUG=true
SINGLE_USER_MODE=true
READ_ACCESS_MODE=demo
SNAPSHOT_READ_FALLBACK_ON_ERROR=true
API_PORT=8000
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
SLOW_QUERY_LOG_MS=200
API_QUERY_TIMEOUT_SECONDS=15
CORS_ORIGINS=
TRUST_PROXY_HEADERS=false
TRUSTED_HOSTS=
HTTPS_REDIRECT=false
CONTENT_SECURITY_POLICY=

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
ADMIN_PASSWORD=
ADMIN_SESSION_SECRET=
ADMIN_SESSION_TTL_HOURS=24
ADMIN_LOGIN_RATE_LIMIT_WINDOW_SECONDS=300
ADMIN_LOGIN_RATE_LIMIT_MAX_ATTEMPTS=5
FORCE_SECURE_COOKIES=false

SYNC_BATCH_SIZE=100
SYNC_README_MAX_LENGTH=10000
SYNC_DEFAULT_SYNC_MODE=incremental
SYNC_DETECT_UNSTARRED_ON_INCREMENTAL=false
```

## 安全建议

1. 不要提交 `.env`。
2. 生产环境必须替换默认数据库密码。
3. 管理员密码与 `ADMIN_SESSION_SECRET` 都要用高强度随机值。
4. API key 建议通过 Secret Manager 或 CI/CD Secret 注入。
5. 对公网部署，建议使用 `READ_ACCESS_MODE=authenticated`、`FORCE_SECURE_COOKIES=true`、`TRUST_PROXY_HEADERS=true`。
