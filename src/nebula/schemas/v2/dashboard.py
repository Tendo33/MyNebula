"""V2 dashboard API schemas."""

from pydantic import BaseModel, Field


class DashboardSummary(BaseModel):
    """Top-level dashboard counters."""

    total_repos: int = 0
    embedded_repos: int = 0
    total_topics: int = 0
    total_clusters: int = 0
    total_edges: int = 0


class DashboardLanguageStat(BaseModel):
    """Dashboard language ranking payload."""

    language: str
    count: int = 0


class DashboardTopicStat(BaseModel):
    """Dashboard topic ranking payload."""

    topic: str
    count: int = 0


class DashboardCluster(BaseModel):
    """Dashboard cluster card payload."""

    id: int
    name: str | None = None
    repo_count: int = 0
    color: str | None = None
    keywords: list[str] = Field(default_factory=list)


class DashboardResponse(BaseModel):
    """Consolidated dashboard payload."""

    summary: DashboardSummary
    top_languages: list[DashboardLanguageStat] = Field(default_factory=list)
    top_topics: list[DashboardTopicStat] = Field(default_factory=list)
    top_clusters: list[DashboardCluster] = Field(default_factory=list)
    version: str | None = None
    generated_at: str | None = None
    request_id: str | None = None
