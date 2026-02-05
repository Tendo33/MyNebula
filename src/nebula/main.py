"""FastAPI application entry point.

This module creates and configures the FastAPI application for MyNebula.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    cors_origins = [
        "http://localhost:5173",  # Vite 开发服务器
        "http://localhost:3000",  # 生产前端
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
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
        }

    # Root endpoint
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
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
