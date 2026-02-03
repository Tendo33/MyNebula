"""Repository-related Pydantic schemas."""

from datetime import datetime

from pydantic import BaseModel, Field


class RepoBase(BaseModel):
    """Base repository schema with common fields."""

    full_name: str = Field(..., description="Full repository name (owner/repo)")
    owner: str = Field(..., description="Repository owner")
    name: str = Field(..., description="Repository name")
    description: str | None = Field(None, description="Repository description")
    language: str | None = Field(None, description="Primary programming language")
    topics: list[str] = Field(
        default_factory=list, description="Repository topics/tags"
    )
    html_url: str = Field(..., description="GitHub URL")


class RepoCreate(RepoBase):
    """Schema for creating a new starred repo record."""

    github_repo_id: int = Field(..., description="GitHub repository ID")
    homepage_url: str | None = None
    stargazers_count: int = 0
    forks_count: int = 0
    watchers_count: int = 0
    open_issues_count: int = 0
    starred_at: datetime | None = None
    repo_created_at: datetime | None = None
    repo_updated_at: datetime | None = None
    repo_pushed_at: datetime | None = None


class RepoResponse(RepoBase):
    """Schema for repository API response."""

    id: int = Field(..., description="Internal database ID")
    github_repo_id: int = Field(..., description="GitHub repository ID")
    homepage_url: str | None = None

    # Statistics
    stargazers_count: int = 0
    forks_count: int = 0
    watchers_count: int = 0
    open_issues_count: int = 0

    # AI-generated content
    ai_summary: str | None = Field(None, description="AI-generated summary")

    # Clustering
    cluster_id: int | None = Field(None, description="Assigned cluster ID")

    # 3D coordinates
    coord_x: float | None = None
    coord_y: float | None = None
    coord_z: float | None = None

    # Timestamps
    starred_at: datetime | None = None
    repo_updated_at: datetime | None = None

    # Processing status
    is_embedded: bool = False
    is_summarized: bool = False

    class Config:
        from_attributes = True


class RepoSearchRequest(BaseModel):
    """Schema for semantic search request."""

    query: str = Field(
        ..., description="Natural language search query", min_length=1, max_length=500
    )
    limit: int = Field(
        default=20, description="Maximum results to return", ge=1, le=100
    )
    language: str | None = Field(None, description="Filter by programming language")
    cluster_id: int | None = Field(None, description="Filter by cluster")
    min_stars: int | None = Field(None, description="Minimum stargazers count", ge=0)


class RepoSearchResponse(BaseModel):
    """Schema for search result item."""

    repo: RepoResponse
    score: float = Field(..., description="Similarity score (0-1)")
    highlight: str | None = Field(None, description="Highlighted match snippet")


class RepoListResponse(BaseModel):
    """Schema for paginated repository list."""

    items: list[RepoResponse]
    total: int
    page: int
    per_page: int
    has_more: bool
