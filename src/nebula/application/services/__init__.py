"""Application services."""

from typing import Any

__all__ = ["GraphSnapshotBuilderService", "GraphQueryService", "SyncPipelineService"]


def __getattr__(name: str) -> Any:
    if name == "GraphQueryService":
        from .graph_query_service import GraphQueryService

        return GraphQueryService
    if name == "GraphSnapshotBuilderService":
        from .graph_snapshot_service import GraphSnapshotBuilderService

        return GraphSnapshotBuilderService
    if name == "SyncPipelineService":
        from .pipeline_service import SyncPipelineService

        return SyncPipelineService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
