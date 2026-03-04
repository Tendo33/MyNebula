"""V2 dashboard API schemas."""

from pydantic import BaseModel, Field


class DashboardSummary(BaseModel):
    """Top-level dashboard counters."""

    total_repos: int = 0
    embedded_repos: int = 0
    total_clusters: int = 0
    total_edges: int = 0


class DashboardCluster(BaseModel):
    """Dashboard cluster card payload."""

    id: int
    name: str | None = None
    repo_count: int = 0
    color: str | None = None


class DashboardResponse(BaseModel):
    """Consolidated dashboard payload."""

    summary: DashboardSummary
    top_clusters: list[DashboardCluster] = Field(default_factory=list)
    version: str | None = None
    generated_at: str | None = None
    request_id: str | None = None
