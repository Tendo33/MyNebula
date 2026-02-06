# MyNebula Backend Dockerfile (包含前端静态文件)
# ==================== Stage 1: Build Frontend ====================
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

# 复制前端依赖文件
COPY frontend/package*.json ./

# 安装前端依赖
RUN npm ci

# 复制前端源代码
COPY frontend/ ./

# 构建前端（生成静态文件）
ARG VITE_API_BASE_URL
RUN VITE_API_BASE_URL=${VITE_API_BASE_URL} npm run build

# ==================== Stage 2: Backend with Frontend ====================
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy dependency files and project metadata
COPY pyproject.toml uv.lock README.md LICENSE ./

# Install dependencies (without project) to improve caching
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code
COPY src/ ./src/
COPY alembic/ ./alembic/
COPY alembic.ini ./

# Install the project itself
RUN uv sync --frozen --no-dev

# Copy frontend build from frontend-builder stage
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uv", "run", "uvicorn", "nebula.main:app", "--host", "0.0.0.0", "--port", "8000"]
