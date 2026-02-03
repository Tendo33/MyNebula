"""User-related Pydantic schemas."""

from datetime import datetime

from pydantic import BaseModel, Field


class UserBase(BaseModel):
    """Base user schema."""

    username: str = Field(..., description="GitHub username")
    email: str | None = Field(None, description="User email")
    avatar_url: str | None = Field(None, description="Avatar URL")


class UserResponse(UserBase):
    """Schema for user API response."""

    id: int = Field(..., description="Internal user ID")
    github_id: int = Field(..., description="GitHub user ID")
    total_stars: int = Field(default=0, description="Total starred repos")
    synced_stars: int = Field(default=0, description="Synced starred repos")
    last_sync_at: datetime | None = Field(None, description="Last sync timestamp")
    created_at: datetime = Field(..., description="Account creation time")

    class Config:
        from_attributes = True


class UserStats(BaseModel):
    """Schema for user statistics."""

    total_stars: int = Field(..., description="Total starred repositories")
    synced_stars: int = Field(..., description="Synced repositories")
    embedded_repos: int = Field(..., description="Repos with embeddings")
    total_clusters: int = Field(..., description="Number of clusters")

    # Language breakdown
    top_languages: list[dict] = Field(
        default_factory=list,
        description="Top languages with counts",
    )

    # Topic breakdown
    top_topics: list[dict] = Field(
        default_factory=list,
        description="Top topics with counts",
    )

    # Time stats
    first_star_date: str | None = Field(None, description="First star date")
    last_star_date: str | None = Field(None, description="Most recent star date")
    most_active_month: str | None = Field(None, description="Month with most stars")
