"""API routes module.

This module contains FastAPI routers for MyNebula:
- repos: Repository CRUD and search
- graph: Graph data for visualization
- sync: Star synchronization endpoints
"""

from fastapi import APIRouter

from .graph import router as graph_router
from .repos import router as repos_router
from .sync import router as sync_router

api_router = APIRouter()

api_router.include_router(repos_router, prefix="/repos", tags=["repos"])
api_router.include_router(graph_router, prefix="/graph", tags=["graph"])
api_router.include_router(sync_router, prefix="/sync", tags=["sync"])

__all__ = ["api_router"]
