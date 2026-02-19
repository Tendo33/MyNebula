"""FastAPI application entry point.

This module creates and configures the FastAPI application for MyNebula.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from nebula.api import api_router
from nebula.core.config import get_app_settings
from nebula.core.embedding import close_embedding_service
from nebula.core.llm import close_llm_service
from nebula.core.scheduler import close_scheduler_service, get_scheduler_service
from nebula.db import close_db, init_db
from nebula.utils import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    Handles startup and shutdown events.
    """
    settings = get_app_settings()

    # Setup logging
    setup_logging(level=settings.log_level, log_file=settings.log_file)
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Debug mode: {settings.debug}")

    # Initialize database
    try:
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

    # Start scheduler service
    try:
        scheduler = get_scheduler_service()
        await scheduler.start()
        logger.info("Scheduler service started")
    except Exception as e:
        logger.warning(f"Failed to start scheduler service: {e}")
        # Non-critical, continue without scheduler

    yield

    # Cleanup
    logger.info("Shutting down...")
    await close_scheduler_service()
    await close_embedding_service()
    await close_llm_service()
    await close_db()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Create and configure FastAPI application.

    Returns:
        Configured FastAPI application
    """
    settings = get_app_settings()

    app = FastAPI(
        title="MyNebula API",
        description="Transform your GitHub Stars into a semantic knowledge nebula",
        version=settings.app_version,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan,
    )

    # Configure CORS - 确保所有请求都能正确返回 CORS 头
    # 注意：allow_origins=["*"] 与 allow_credentials=True 不能同时使用
    # 在开发模式下我们使用具体的来源地址
    # Configure CORS - 允许所有本地开发端口
    # 使用 regex 匹配 localhost 和 127.0.0.1 的任意端口
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    # Include API routes
    app.include_router(api_router, prefix="/api")

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        from nebula.db.database import check_db_connection

        db_healthy = await check_db_connection()
        return {
            "status": "healthy" if db_healthy else "degraded",
            "database": "connected" if db_healthy else "disconnected",
            "version": settings.app_version,
            "single_user_mode": settings.single_user_mode,
        }

    # 静态文件服务配置
    # 前端构建产物路径（相对于项目根目录）
    frontend_dist = Path(__file__).parent.parent.parent / "frontend" / "dist"

    if frontend_dist.exists():
        # 挂载静态资源目录（CSS, JS, images等）
        app.mount(
            "/assets",
            StaticFiles(directory=str(frontend_dist / "assets")),
            name="assets",
        )

        # SPA 路由处理：所有非 API 请求都返回 index.html
        @app.get("/{full_path:path}")
        async def serve_spa(request: Request, full_path: str):
            """Serve the SPA for all routes not matched by API or static files."""
            # 如果请求的是具体文件（如 favicon.ico, manifest.json等）
            requested_file = frontend_dist / full_path
            if requested_file.is_file():
                return FileResponse(requested_file)

            # 否则返回 index.html（SPA 路由）
            return FileResponse(frontend_dist / "index.html")

        logger.info(f"Serving frontend static files from: {frontend_dist}")
    else:
        logger.warning(
            f"Frontend dist directory not found at {frontend_dist}. Static file serving disabled."
        )

        # 如果没有前端文件，提供一个简单的 root endpoint
        @app.get("/")
        async def root():
            """Root endpoint with API information."""
            return {
                "name": settings.app_name,
                "version": settings.app_version,
                "docs": "/docs" if settings.debug else "Disabled in production",
                "health": "/health",
                "api": "/api",
            }

    return app


# Create application instance
app = create_app()


def run():
    """Run the application using uvicorn.

    This function is called when running `mynebula` command.
    """
    import uvicorn

    settings = get_app_settings()
    uvicorn.run(
        "nebula.main:app",
        host="0.0.0.0",
        port=settings.api_port,
        reload=settings.debug,
        reload_dirs=["src"],  # Explicitly watch src directory
        reload_excludes=[".history"],  # Ignore .history directory
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
