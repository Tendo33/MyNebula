# MyNebula Backend Dockerfile (包含前端静态文件)
# ==================== Stage 1: Build Frontend ====================
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

# 复制前端依赖文件
COPY frontend/package.json frontend/pnpm-lock.yaml ./

# 安装前端依赖
RUN corepack enable && pnpm install --frozen-lockfile

# 复制前端源代码
COPY frontend/ ./

# 构建前端（生成静态文件）
ARG VITE_API_BASE_URL
RUN VITE_API_BASE_URL=${VITE_API_BASE_URL} pnpm run build

# ==================== Stage 2: Build Python environment ====================
# 编译器只存在于这一层。numpy / scikit-learn / numba / psycopg2 需要它们来构建
# wheel，但运行时不需要——把它们留在最终镜像里只会增加体积和 CVE 面。
FROM python:3.12-slim AS python-builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv==0.10.0

WORKDIR /app

# 先装依赖（不装项目本身），让依赖层可以被缓存
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project

# 再复制源码并安装项目本身，使 `nebula.main:app` 可导入
COPY src/ ./src/
COPY alembic/ ./alembic/
COPY alembic.ini ./
RUN uv sync --frozen --no-dev

# ==================== Stage 3: Runtime ====================
# 与 builder 使用同一基础镜像和同一 Python 次版本，复制过来的 virtualenv 才有效。
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/app/.venv/bin:$PATH"

# curl 是运行时依赖：Dockerfile 的 HEALTHCHECK 和 docker-compose 的 api
# healthcheck 都调用它。这里不再安装任何编译器。
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 从 builder 复制已解析好的虚拟环境与后端源码
COPY --from=python-builder /app/.venv ./.venv
COPY --from=python-builder /app/src ./src
COPY --from=python-builder /app/alembic ./alembic
COPY --from=python-builder /app/alembic.ini ./alembic.ini

# 复制前端构建产物
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# 直接调用 venv 里的 uvicorn；运行时镜像不再包含 uv。
CMD ["uvicorn", "nebula.main:app", "--host", "0.0.0.0", "--port", "8000"]
