"""V2 API routes."""

from fastapi import APIRouter

from .dashboard import router as dashboard_router
from .data import router as data_router
from .graph import router as graph_router
from .settings import router as settings_router
from .sync import router as sync_router

v2_router = APIRouter()
v2_router.include_router(graph_router, prefix="/graph", tags=["v2-graph"])
v2_router.include_router(dashboard_router, prefix="/dashboard", tags=["v2-dashboard"])
v2_router.include_router(data_router, prefix="/data", tags=["v2-data"])
v2_router.include_router(settings_router, prefix="/settings", tags=["v2-settings"])
v2_router.include_router(sync_router, prefix="/sync", tags=["v2-sync"])

__all__ = ["v2_router"]
