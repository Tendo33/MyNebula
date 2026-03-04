"""V2 sync pipeline schemas."""

from datetime import datetime

from pydantic import BaseModel, Field


class PipelineStartResponse(BaseModel):
    """Response when starting the v2 sync pipeline."""

    pipeline_run_id: int = Field(..., description="Pipeline run ID")
    status: str = Field(..., description="Initial pipeline status")
    phase: str = Field(..., description="Initial pipeline phase")
    message: str = Field(..., description="Human readable status")
    version: str | None = Field(None, description="Pipeline version token")
    generated_at: str | None = Field(None, description="Response generation timestamp")
    request_id: str | None = Field(None, description="Request correlation ID")


class PipelineStatusResponse(BaseModel):
    """Response for pipeline status query."""

    pipeline_run_id: int = Field(..., description="Pipeline run ID")
    user_id: int = Field(..., description="User ID")
    status: str = Field(..., description="Pipeline status")
    phase: str = Field(..., description="Current phase")
    last_error: str | None = Field(None, description="Latest error message")
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    version: str | None = Field(None, description="Pipeline version token")
    generated_at: str | None = Field(None, description="Response generation timestamp")
    request_id: str | None = Field(None, description="Request correlation ID")
