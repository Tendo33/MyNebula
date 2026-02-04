"""Graph visualization Pydantic schemas."""

from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    """Schema for a graph node (repository)."""

    id: int = Field(..., description="Node ID (repo database ID)")
    github_id: int = Field(..., description="GitHub repository ID")
    full_name: str = Field(..., description="Repository full name")
    name: str = Field(..., description="Repository name")
    description: str | None = Field(None, description="Repository description")
    language: str | None = Field(None, description="Primary language")
    html_url: str = Field(..., description="GitHub URL")

    # Owner info
    owner: str = Field(..., description="Repository owner")
    owner_avatar_url: str | None = Field(None, description="Owner avatar URL")

    # 3D position
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")
    z: float = Field(..., description="Z coordinate")

    # Visual properties
    cluster_id: int | None = Field(None, description="Cluster ID")
    color: str | None = Field(None, description="Node color (hex)")
    size: float = Field(default=1.0, description="Node size based on stars")

    # User's star list (GitHub user-defined category)
    star_list_id: int | None = Field(None, description="User's star list ID")
    star_list_name: str | None = Field(None, description="User's star list name")

    # Stats for tooltip
    stargazers_count: int = 0
    ai_summary: str | None = None
    ai_tags: list[str] | None = Field(None, description="AI-generated tags")
    topics: list[str] | None = Field(None, description="GitHub topics")
    starred_at: str | None = None


class StarListInfo(BaseModel):
    """Schema for user's star list."""

    id: int = Field(..., description="Star list ID")
    name: str = Field(..., description="List name")
    description: str | None = Field(None, description="List description")
    repo_count: int = Field(default=0, description="Number of repos in list")


class GraphEdge(BaseModel):
    """Schema for a graph edge (connection between repos)."""

    source: int = Field(..., description="Source node ID")
    target: int = Field(..., description="Target node ID")
    weight: float = Field(..., description="Edge weight (similarity)", ge=0, le=1)


class ClusterInfo(BaseModel):
    """Schema for cluster information."""

    id: int = Field(..., description="Cluster ID")
    name: str | None = Field(None, description="Cluster name")
    description: str | None = Field(None, description="Cluster description")
    keywords: list[str] = Field(default_factory=list, description="Cluster keywords")
    color: str = Field(..., description="Cluster color (hex)")
    repo_count: int = Field(..., description="Number of repos in cluster")

    # 3D center position
    center_x: float | None = None
    center_y: float | None = None
    center_z: float | None = None


class GraphData(BaseModel):
    """Schema for complete graph data."""

    nodes: list[GraphNode] = Field(..., description="Graph nodes")
    edges: list[GraphEdge] = Field(default_factory=list, description="Graph edges")
    clusters: list[ClusterInfo] = Field(
        default_factory=list, description="Cluster information"
    )
    star_lists: list[StarListInfo] = Field(
        default_factory=list, description="User's star lists"
    )

    # Metadata
    total_nodes: int = Field(..., description="Total number of nodes")
    total_edges: int = Field(..., description="Total number of edges")
    total_clusters: int = Field(..., description="Total number of clusters")
    total_star_lists: int = Field(default=0, description="Total number of star lists")


class TimelinePoint(BaseModel):
    """Schema for timeline data point."""

    date: str = Field(..., description="Date (YYYY-MM)")
    count: int = Field(..., description="Number of stars in this period")
    repos: list[str] = Field(default_factory=list, description="Repo names starred")
    top_languages: list[str] = Field(default_factory=list, description="Top languages")
    top_topics: list[str] = Field(default_factory=list, description="Top topics")


class TimelineData(BaseModel):
    """Schema for timeline visualization data."""

    points: list[TimelinePoint] = Field(..., description="Timeline data points")
    total_stars: int = Field(..., description="Total starred repos")
    date_range: tuple[str, str] = Field(..., description="Date range (start, end)")
