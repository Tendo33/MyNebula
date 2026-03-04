"""V2 graph API schemas."""

from pydantic import BaseModel, Field

from nebula.schemas.graph import GraphEdge


class GraphEdgesPage(BaseModel):
    """Paged edge response for large graph datasets."""

    edges: list[GraphEdge] = Field(default_factory=list)
    next_cursor: int | None = Field(None, description="Cursor for next page")
    version: str = Field(..., description="Snapshot version")
    generated_at: str | None = Field(None, description="Snapshot generation time")
    request_id: str | None = Field(None, description="Request correlation ID")
