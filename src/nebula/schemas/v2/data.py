"""V2 data API schemas."""

from pydantic import BaseModel, Field


class DataRepoItem(BaseModel):
    """Repository item payload for Data page."""

    id: int
    full_name: str
    name: str
    owner: str
    owner_avatar_url: str | None = None
    description: str | None = None
    ai_summary: str | None = None
    topics: list[str] = Field(default_factory=list)
    language: str | None = None
    stargazers_count: int
    html_url: str
    cluster_id: int | None = None
    star_list_id: int | None = None
    starred_at: str | None = None
    last_commit_time: str | None = None


class DataReposResponse(BaseModel):
    """Paged repositories response."""

    items: list[DataRepoItem] = Field(default_factory=list)
    count: int
    limit: int
    offset: int
    version: str | None = None
    generated_at: str | None = None
    request_id: str | None = None
