"""Database module.

This module provides database connectivity and models for MyNebula:
- database: Async PostgreSQL connection management
- models: SQLAlchemy ORM models with pgvector support
"""

from .database import (
    AsyncSessionLocal,
    close_db,
    get_db,
    init_db,
)
from .models import (
    Base,
    Cluster,
    StarList,
    StarredRepo,
    SyncTask,
    User,
)

__all__ = [
    # Database
    "get_db",
    "init_db",
    "close_db",
    "AsyncSessionLocal",
    # Models
    "Base",
    "User",
    "StarredRepo",
    "StarList",
    "Cluster",
    "SyncTask",
]
