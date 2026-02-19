# 环境变量配置指南

本文档详细说明 MyNebula 所有支持的环境变量配置。

## 快速开始

1. 复制示例配置文件：
   ```bash
   cp .env.example .env
   ```

2. 编辑 `.env` 文件，至少配置以下必填项：
   - `GITHUB_TOKEN`
   - `EMBEDDING_API_KEY`

---

## 应用配置

| 变量名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `APP_NAME` | string | `mynebula` | 应用名称 |
| `APP_VERSION` | string | `0.1.0` | 应用版本 |
| `DEBUG` | boolean | `false` | 调试模式（启用 Swagger UI，同时控制开发/生产环境） |
| `SECRET_KEY` | string | - | JWT 签名密钥（生产环境必须修改） |

### 示例

```bash
APP_NAME=mynebula
APP_VERSION=0.1.0
DEBUG=true
SECRET_KEY=your-32-character-secret-key-here
```

---

## 日志配置

| 变量名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `LOG_LEVEL` | string | `INFO` | 日志级别：`DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL` |
| `LOG_FILE` | string | `logs/app.log` | 日志文件路径 |

---

## 数据库配置 (PostgreSQL + pgvector)

有两种配置方式：

### 方式 1: 分离参数（推荐 Docker Compose）

| 变量名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `DATABASE_HOST` | string | `localhost` | 数据库主机 |
| `DATABASE_PORT` | integer | `5432` | 数据库端口 |
| `DATABASE_USER` | string | `mynebula` | 数据库用户 |
| `DATABASE_PASSWORD` | string | `mynebula_secret` | 数据库密码 |
| `DATABASE_NAME` | string | `mynebula` | 数据库名称 |

### 方式 2: 完整 URL（优先级更高）

| 变量名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `DATABASE_URL` | string | - | 完整数据库 URL |

```bash
# 示例
DATABASE_URL=postgresql+asyncpg://mynebula:mynebula_secret@localhost:5432/mynebula
```

### Docker Compose 示例

```bash
DATABASE_HOST=db          # 使用 Docker 服务名
DATABASE_PORT=5432
DATABASE_USER=mynebula
DATABASE_PASSWORD=your_secure_password
DATABASE_NAME=mynebula
```

---

## GitHub 配置 ⚠️ 必填

| 变量名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `GITHUB_TOKEN` | string | ✅ | Personal Access Token (PAT) |

### 获取 GitHub Token

1. 访问 https://github.com/settings/tokens
2. 生成新的 Token (Fine-grained 或 Classic)
3. 确保勾选读取 Star 列表的权限
4. 将 Token 填入 `.env`

```bash
GITHUB_TOKEN=your_pat_xxxxxxxxxxxxxxxxxxxxxxxx
```

---

## Embedding 配置 ⚠️ 必填

用于生成仓库描述的向量嵌入。所有配置使用 OpenAI 兼容接口。

| 变量名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `EMBEDDING_API_KEY` | string | - | API 密钥 ⚠️ **必填** |
| `EMBEDDING_BASE_URL` | string | `https://api.siliconflow.cn/v1` | API 基础 URL (OpenAI 兼容) |
| `EMBEDDING_MODEL` | string | `BAAI/bge-large-zh-v1.5` | 模型名称 |
| `EMBEDDING_DIMENSIONS` | integer | `1024` | 向量维度 |

### 支持的提供商

#### SiliconFlow（推荐国内用户）

```bash
EMBEDDING_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxx
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
EMBEDDING_DIMENSIONS=1024
```

- 注册地址: https://cloud.siliconflow.cn
- 特点: 国内访问快，价格实惠

#### Jina AI

```bash
EMBEDDING_API_KEY=jina_xxxxxxxxxxxxxxxxxxxxx
EMBEDDING_BASE_URL=https://api.jina.ai/v1
EMBEDDING_MODEL=jina-embeddings-v3
EMBEDDING_DIMENSIONS=1024
```

- 注册地址: https://jina.ai

#### OpenAI

```bash
EMBEDDING_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxx
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
```

- 注册地址: https://platform.openai.com
- 特点: 效果好，需要国际访问

#### 智谱 AI

```bash
EMBEDDING_API_KEY=xxxxxxxxxxxxxxxxxxxxx
EMBEDDING_BASE_URL=https://open.bigmodel.cn/api/paas/v4
EMBEDDING_MODEL=embedding-3
EMBEDDING_DIMENSIONS=2048
```

- 注册地址: https://open.bigmodel.cn

#### Ollama（本地部署）

```bash
EMBEDDING_API_KEY=ollama  # Ollama 不需要真实 key，但字段不能为空
EMBEDDING_BASE_URL=http://localhost:11434/v1
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSIONS=768
```

- 安装: https://ollama.com
- 特点: 完全本地，无需网络

---

## LLM 配置（可选）

用于生成 AI 摘要和聚类名称。使用 OpenAI 兼容接口。

| 变量名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `LLM_API_KEY` | string | - | API 密钥 |
| `LLM_BASE_URL` | string | `https://api.siliconflow.cn/v1` | API 基础 URL (OpenAI 兼容) |
| `LLM_MODEL` | string | `Qwen/Qwen2.5-7B-Instruct` | 模型名称 |
| `LLM_OUTPUT_LANGUAGE` | string | `zh` | LLM 输出语言（`zh` 或 `en`） |

### 示例配置

```bash
LLM_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxx
LLM_BASE_URL=https://api.siliconflow.cn/v1
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
LLM_OUTPUT_LANGUAGE=zh
```

---

## 同步配置

控制 Star 同步行为。

| 变量名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| `SYNC_BATCH_SIZE` | integer | `100` | 10-500 | 每批处理的仓库数量 |
| `README_MAX_LENGTH` | integer | `10000` | 1000-100000 | README 内容最大长度（字符） |

---

## Docker Compose 专用配置

这些变量仅在使用 Docker Compose 时生效。

| 变量名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `API_PORT` | integer | `8000` | 后端 API 端口映射 |
| `VITE_API_BASE_URL` | string | `http://localhost:8000` | 前端连接的 API 地址 |

---

## 完整配置示例

### 开发环境

```bash
# ==================== 应用配置 ====================
APP_NAME=mynebula
APP_VERSION=0.1.0
DEBUG=true
SECRET_KEY=dev-secret-key-not-for-production

# ==================== 日志配置 ====================
LOG_LEVEL=DEBUG
LOG_FILE=logs/app.log

# ==================== 数据库配置 ====================
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_USER=mynebula
DATABASE_PASSWORD=mynebula_secret
DATABASE_NAME=mynebula

# ==================== GitHub Token ====================
GITHUB_TOKEN=your_pat_xxxxxxxxxxxxxxxx

# ==================== Embedding (OpenAI 兼容接口) ====================
EMBEDDING_API_KEY=your_embedding_api_key
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
EMBEDDING_DIMENSIONS=1024

# ==================== LLM (OpenAI 兼容接口，可选) ====================
LLM_API_KEY=your_llm_api_key
LLM_BASE_URL=https://api.siliconflow.cn/v1
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
LLM_OUTPUT_LANGUAGE=zh
```

### 生产环境

```bash
# ==================== 应用配置 ====================
APP_NAME=mynebula
APP_VERSION=0.1.0
DEBUG=false
SECRET_KEY=your-very-long-and-secure-secret-key-32chars

# ==================== 日志配置 ====================
LOG_LEVEL=INFO
LOG_FILE=/var/log/mynebula/app.log

# ==================== 数据库配置 ====================
DATABASE_HOST=db
DATABASE_PORT=5432
DATABASE_USER=mynebula
DATABASE_PASSWORD=your_secure_database_password
DATABASE_NAME=mynebula

# ==================== GitHub Token ====================
GITHUB_TOKEN=your_production_pat

# ==================== Embedding (OpenAI 兼容接口) ====================
EMBEDDING_API_KEY=your_embedding_api_key
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
EMBEDDING_DIMENSIONS=1024

# ==================== Docker Compose ====================
API_PORT=8000
VITE_API_BASE_URL=https://your-domain.com
```

---

## 安全建议

1. **永远不要提交 `.env` 文件到版本控制**
2. 生产环境使用强密码和随机生成的 `SECRET_KEY`
3. 定期轮换 API 密钥
4. 使用环境变量管理工具（如 Vault、AWS Secrets Manager）管理敏感信息
5. 限制 GitHub OAuth App 的权限范围
