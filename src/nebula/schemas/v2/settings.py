"""V2 settings API schemas."""

from pydantic import BaseModel, Field

from nebula.schemas.schedule import (
    FullRefreshRequest,
    FullRefreshResponse,
    JobStatusResponse,
    ScheduleConfig,
    ScheduleResponse,
    SyncInfoResponse,
)


class GraphDefaults(BaseModel):
    """Frontend graph defaults delivered by settings endpoint."""

    max_clusters: int = 8
    min_clusters: int = 3
    related_min_semantic: float = 0.65
    hq_rendering: bool = True
    show_trajectories: bool = True


class SettingsResponse(BaseModel):
    """Consolidated v2 settings payload."""

    schedule: ScheduleResponse
    sync_info: SyncInfoResponse
    graph_defaults: GraphDefaults
    version: str | None = None
    generated_at: str | None = None
    request_id: str | None = None


class ScheduleUpdateResponse(BaseModel):
    """Schedule update response with metadata."""

    schedule: ScheduleResponse
    version: str | None = None
    generated_at: str | None = None
    request_id: str | None = None


class FullRefreshStartResponse(BaseModel):
    """Full refresh trigger response with metadata."""

    task: FullRefreshResponse
    version: str | None = None
    generated_at: str | None = None
    request_id: str | None = None


class FullRefreshJobResponse(BaseModel):
    """Full refresh job status with metadata."""

    job: JobStatusResponse
    version: str | None = None
    generated_at: str | None = None
    request_id: str | None = None


__all__ = [
    "FullRefreshJobResponse",
    "FullRefreshRequest",
    "FullRefreshStartResponse",
    "GraphDefaults",
    "JobStatusResponse",
    "ScheduleConfig",
    "ScheduleResponse",
    "ScheduleUpdateResponse",
    "SettingsResponse",
    "SyncInfoResponse",
]
