<div align="center">
  <img src="doc/images/logo2.png" width="120" alt="MyNebula Logo" />
  <h1>MyNebula (我的星云)</h1>
</div>

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16+-blue.svg)](https://www.postgresql.org/)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Transform your GitHub Stars into a semantic knowledge nebula.**

将你的 GitHub Star 列表转化为三维知识星云。通过语义分析，让相似的项目自动聚集，通过时间轴展示你的技术兴趣演变。

![MyNebula Banner](doc/images/banner.png)

## ✨ Features

- 🌐 **星云图谱 (Nebula Graph)**: 3D 可视化你的 Star 列表，相似项目自动聚类
- 🔍 **语义搜索 (Semantic Search)**: 自然语言查询，如"找一个轻量级的 Python 依赖管理工具"
- 🤖 **AI 摘要 (AI Summary)**: 自动生成仓库的一句话总结
- ⏰ **时间旅行 (Time Travel)**: 时间轴展示你的技术兴趣演变
- 🔌 **多 Embedding 提供商**: 支持 OpenAI、SiliconFlow、Jina、Ollama 等
- 🐳 **自托管 (Self-hosted)**: Docker 一键部署，数据完全自主

---

## 🚀 Quick Start

### Option A: Docker Compose (推荐)

一键部署完整应用栈：

```bash
# 1. 克隆仓库
git clone https://github.com/yourusername/mynebula.git
cd mynebula

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填写必要配置（详见下方说明）

# 3. 启动所有服务
docker-compose up -d

# 4. 查看日志
docker-compose logs -f
```

服务启动后：
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs (开发模式)

### Option B: 本地开发模式

#### Prerequisites

- Python 3.10+
- Node.js 20+
- Docker (仅用于 PostgreSQL)
- GitHub OAuth App

#### 1. 安装依赖

```bash
# 安装 uv (Python 包管理器)
pip install uv

# 安装 Python 依赖
uv sync

# 安装前端依赖
cd frontend && npm install && cd ..
```

#### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件（详见 doc/ENV_VARS.md）
```

#### 3. 启动数据库

```bash
docker-compose up -d db
```

#### 4. 初始化数据库

```bash
uv run alembic upgrade head
```

#### 5. 启动服务

**后端** (终端 1):
```bash
uv run uvicorn nebula.main:app --reload
```

**前端** (终端 2):
```bash
cd frontend && npm run dev
```

访问：
- Frontend: http://localhost:5173
- API Docs: http://localhost:8000/docs

---

## 📦 Configuration

### 环境变量概览

详细配置说明请参考 [doc/ENV_VARS.md](doc/ENV_VARS.md)。

| 变量组 | 必填 | 说明 |
|--------|------|------|
| `GITHUB_*` | ✅ | GitHub OAuth 认证 |
| `EMBEDDING_*` | ✅ | Embedding 服务配置 |
| `DATABASE_*` | ❌ | 数据库配置（有默认值） |
| `LLM_*` | ❌ | LLM 服务（用于 AI 摘要） |

### GitHub OAuth 配置

1. 访问 https://github.com/settings/developers
2. 创建新的 OAuth App
3. 设置 Callback URL:
   - 开发环境: `http://localhost:8000/api/auth/callback`
   - 生产环境: `https://your-domain.com/api/auth/callback`
4. 将 Client ID 和 Client Secret 填入 `.env`

```bash
GITHUB_CLIENT_ID=your_client_id
GITHUB_CLIENT_SECRET=your_client_secret
GITHUB_REDIRECT_URI=http://localhost:8000/api/auth/callback
```

### Embedding 提供商

支持多种 OpenAI 兼容的 Embedding API：

| 提供商 | Base URL | 推荐模型 | 维度 |
|--------|----------|----------|------|
| **SiliconFlow** (推荐国内) | `https://api.siliconflow.cn/v1` | `BAAI/bge-large-zh-v1.5` | 1024 |
| **Jina AI** | `https://api.jina.ai/v1` | `jina-embeddings-v3` | 1024 |
| **OpenAI** | `https://api.openai.com/v1` | `text-embedding-3-small` | 1536 |
| **智谱 AI** | `https://open.bigmodel.cn/api/paas/v4` | `embedding-3` | 2048 |
| **Ollama** (本地) | `http://localhost:11434/v1` | `nomic-embed-text` | 768 |

SiliconFlow 配置示例：

```bash
EMBEDDING_API_KEY=your_api_key
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
EMBEDDING_DIMENSIONS=1024
```

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     MyNebula Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐         ┌──────────────────┐              │
│  │  React Frontend  │  HTTP   │  FastAPI Backend │              │
│  │  (3D Force Graph)│◄───────►│                  │              │
│  └──────────────────┘         └────────┬─────────┘              │
│                                        │                         │
│         ┌──────────────────────────────┼──────────────┐         │
│         ▼                              ▼              ▼         │
│  ┌─────────────────┐          ┌──────────────┐  ┌──────────┐   │
│  │  PostgreSQL     │          │ GitHub API   │  │ Embedding│   │
│  │  + pgvector     │          │              │  │ Provider │   │
│  └─────────────────┘          └──────────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
mynebula/
├── src/nebula/
│   ├── api/                # FastAPI routes
│   │   ├── auth.py         # GitHub OAuth
│   │   ├── repos.py        # Repository CRUD & search
│   │   ├── graph.py        # Graph visualization data
│   │   └── sync.py         # Star synchronization
│   ├── core/               # Business logic
│   │   ├── config.py       # Configuration management
│   │   ├── embedding.py    # Embedding service
│   │   ├── github_client.py# GitHub API wrapper
│   │   └── clustering.py   # UMAP + HDBSCAN
│   ├── db/                 # Database layer
│   │   ├── database.py     # Connection management
│   │   └── models.py       # SQLAlchemy models
│   ├── schemas/            # Pydantic schemas
│   ├── utils/              # Utility functions
│   └── main.py             # Application entry
├── frontend/               # React frontend (coming soon)
├── alembic/                # Database migrations
├── docker-compose.yml      # Docker configuration
└── pyproject.toml          # Project dependencies
```

## 🔧 Development

### Running Tests

```bash
uv run pytest
```

### Code Quality

```bash
# Format code
uv run ruff format

# Lint code
uv run ruff check

# Fix linting issues
uv run ruff check --fix
```

### Database Migrations

```bash
# Create a new migration
uv run alembic revision --autogenerate -m "description"

# Apply migrations
uv run alembic upgrade head

# Rollback
uv run alembic downgrade -1
```

## 🛣 Roadmap

### Phase 1: 基础架构 ✅
- [x] PostgreSQL + pgvector 向量数据库
- [x] GitHub OAuth 认证流程
- [x] 多提供商 Embedding 服务
- [x] Star 列表同步 API

### Phase 2: 核心数据管道 ✅
- [x] README 内容获取与处理
- [x] 批量 Embedding 计算
- [x] 向量入库流程
- [x] 语义相似度搜索

### Phase 3: 语义能力 ✅
- [x] 自然语言查询 API
- [x] UMAP 降维算法
- [x] 聚类名称生成 (LLM)
- [x] AI 摘要生成

### Phase 4: 前端可视化 ✅
- [x] React + Three.js 3D 力导图
- [x] 节点交互 (悬停/点击)
- [x] 语义搜索 UI
- [x] 时间轴滑块
- [x] 配置面板

### Phase 5: 部署与运维 ✅
- [x] Docker Compose 配置
- [x] 部署文档
- [x] 环境变量说明

### Future Enhancements
- [ ] 多用户支持
- [ ] 趋势发现
- [ ] 技术栈 DNA 生成
- [ ] 导出/分享功能

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [pgvector](https://github.com/pgvector/pgvector) - Vector similarity for PostgreSQL
- [UMAP](https://github.com/lmcinnes/umap) - Dimensionality reduction
- [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan) - Clustering algorithm
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [react-force-graph](https://github.com/vasturiano/react-force-graph) - 3D force graph
