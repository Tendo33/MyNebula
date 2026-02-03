# 🌌 MyNebula (我的星云)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16+-blue.svg)](https://www.postgresql.org/)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Transform your GitHub Stars into a semantic knowledge nebula.**

将你的 GitHub Star 列表转化为三维知识星云。通过语义分析，让相似的项目自动聚集，通过时间轴展示你的技术兴趣演变。

![MyNebula Preview](https://via.placeholder.com/800x400?text=MyNebula+Preview)

## ✨ Features

- 🌐 **星云图谱 (Nebula Graph)**: 3D 可视化你的 Star 列表，相似项目自动聚类
- 🔍 **语义搜索 (Semantic Search)**: 自然语言查询，如"找一个轻量级的 Python 依赖管理工具"
- 🤖 **AI 摘要 (AI Summary)**: 自动生成仓库的一句话总结
- ⏰ **时间旅行 (Time Travel)**: 时间轴展示你的技术兴趣演变
- 🔌 **多 Embedding 提供商**: 支持 OpenAI、SiliconFlow、Jina、Ollama 等
- 🐳 **自托管 (Self-hosted)**: Docker 一键部署，数据完全自主

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- GitHub OAuth App (用于认证)

### 1. Clone and Setup

```bash
git clone https://github.com/yourusername/mynebula.git
cd mynebula

# Install uv if not already installed
pip install uv

# Install dependencies
uv sync
```

### 2. Configure Environment

```bash
# Copy example config
cp .env.example .env

# Edit .env with your settings:
# - GitHub OAuth credentials
# - Embedding provider (SiliconFlow recommended for CN users)
# - Database credentials (or use defaults)
```

### 3. Start PostgreSQL

```bash
# Start PostgreSQL with pgvector
docker-compose up -d db

# Wait for database to be ready
docker-compose logs -f db
```

### 4. Initialize Database

```bash
# Run database migrations
uv run alembic upgrade head
```

### 5. Start the Server

```bash
# Development mode
uv run uvicorn nebula.main:app --reload

# Or use the CLI
uv run mynebula
```

Visit http://localhost:8000/docs for the API documentation.

## 📦 Configuration

### GitHub OAuth Setup

1. Go to https://github.com/settings/developers
2. Create a new OAuth App
3. Set the callback URL to `http://localhost:8000/api/auth/callback`
4. Copy Client ID and Client Secret to `.env`

### Embedding Providers

MyNebula supports multiple embedding providers through OpenAI-compatible APIs:

| Provider | Base URL | Recommended Model |
|----------|----------|-------------------|
| **SiliconFlow** (推荐国内) | `https://api.siliconflow.cn/v1` | `BAAI/bge-large-zh-v1.5` |
| **Jina AI** | `https://api.jina.ai/v1` | `jina-embeddings-v3` |
| **OpenAI** | `https://api.openai.com/v1` | `text-embedding-3-small` |
| **Ollama** (本地) | `http://localhost:11434/v1` | `nomic-embed-text` |

Example `.env` configuration for SiliconFlow:

```bash
EMBEDDING_PROVIDER=siliconflow
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

- [x] Phase 1: Core Backend
  - [x] PostgreSQL + pgvector setup
  - [x] GitHub OAuth & Star sync
  - [x] Embedding service (multi-provider)
  - [x] Semantic search API
- [ ] Phase 2: Advanced Features
  - [ ] UMAP clustering & visualization data
  - [ ] AI summary generation
  - [ ] README fetching & processing
- [ ] Phase 3: Frontend
  - [ ] React + Three.js 3D visualization
  - [ ] Semantic search UI
  - [ ] Timeline component
- [ ] Phase 4: Enhancements
  - [ ] Multi-user support
  - [ ] Trend discovery
  - [ ] Tech stack DNA generation

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
