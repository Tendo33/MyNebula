"""Schedule-related Pydantic schemas for automatic sync configuration."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ScheduleConfig(BaseModel):
    """Schema for creating/updating sync schedule configuration."""

    is_enabled: bool = Field(
        default=False, description="Whether scheduled sync is enabled"
    )
    schedule_hour: int = Field(
        default=9, ge=0, le=23, description="Hour to run sync (0-23)"
    )
    schedule_minute: int = Field(
        default=0, ge=0, le=59, description="Minute to run sync (0-59)"
    )
    timezone: str = Field(
        default="Asia/Shanghai", description="Timezone for schedule (IANA format)"
    )


class ScheduleResponse(BaseModel):
    """Schema for sync schedule response."""

    is_enabled: bool = Field(..., description="Whether scheduled sync is enabled")
    schedule_hour: int = Field(..., description="Hour to run sync (0-23)")
    schedule_minute: int = Field(..., description="Minute to run sync (0-59)")
    timezone: str = Field(..., description="Timezone for schedule")
    last_run_at: datetime | None = Field(None, description="Last execution time")
    last_run_status: str | None = Field(
        None, description="Status of last run: 'success', 'failed', or 'running'"
    )
    last_run_error: str | None = Field(
        None, description="Error message if last run failed"
    )
    next_run_at: datetime | None = Field(
        None, description="Calculated next execution time"
    )

    model_config = ConfigDict(from_attributes=True)


class SyncInfoResponse(BaseModel):
    """Schema for comprehensive sync status information."""

    last_sync_at: datetime | None = Field(None, description="Last sync completion time")
    github_token_configured: bool = Field(
        ..., description="Whether GitHub token is configured for sync operations"
    )
    total_repos: int = Field(..., description="Total number of starred repositories")
    synced_repos: int = Field(..., description="Number of synced repositories")
    embedded_repos: int = Field(
        ..., description="Number of repositories with embeddings"
    )
    summarized_repos: int = Field(
        ..., description="Number of repositories with AI summaries"
    )
    schedule: ScheduleResponse | None = Field(
        None, description="Schedule configuration"
    )


class FullRefreshRequest(BaseModel):
    """Schema for full refresh request."""

    confirm: bool = Field(
        default=False,
        description="Confirmation flag - must be True to proceed with full refresh",
    )


class FullRefreshResponse(BaseModel):
    """Schema for full refresh response."""

    task_id: int = Field(..., description="Background task ID for tracking progress")
    message: str = Field(..., description="Status message")
    reset_count: int = Field(default=0, description="Number of repositories reset")
