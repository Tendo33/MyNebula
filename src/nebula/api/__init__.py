"""API routes module.

This module contains FastAPI routers for MyNebula:
- v2: Version 2 API surface (graph, sync, settings, dashboard, data)
"""

from fastapi import APIRouter

from .v2 import v2_router

api_router = APIRouter()

api_router.include_router(v2_router, prefix="/v2")

__all__ = ["api_router"]
