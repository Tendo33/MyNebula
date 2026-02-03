"""SQLAlchemy ORM models with pgvector support.

This module defines the database models for MyNebula:
- User: GitHub user information
- StarredRepo: Starred repositories with embeddings
- Cluster: Semantic clusters
- SyncTask: Synchronization task tracking
"""

from datetime import datetime
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from nebula.core.config import get_embedding_settings


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


class User(Base):
    """GitHub user model."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    github_id: Mapped[int] = mapped_column(
        Integer, unique=True, nullable=False, index=True
    )
    username: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False, index=True
    )
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    avatar_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    access_token: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )  # Encrypted in production

    # Statistics
    total_stars: Mapped[int] = mapped_column(Integer, default=0)
    synced_stars: Mapped[int] = mapped_column(Integer, default=0)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    last_sync_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    starred_repos: Mapped[list["StarredRepo"]] = relationship(
        "StarredRepo", back_populates="user", cascade="all, delete-orphan"
    )
    sync_tasks: Mapped[list["SyncTask"]] = relationship(
        "SyncTask", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, username={self.username})>"


class StarredRepo(Base):
    """Starred repository model with vector embedding."""

    __tablename__ = "starred_repos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False, index=True
    )
    github_repo_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)

    # Repository info
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)  # owner/repo
    owner: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    language: Mapped[str | None] = mapped_column(String(100), nullable=True)
    topics: Mapped[list[str] | None] = mapped_column(ARRAY(String), nullable=True)
    homepage_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    html_url: Mapped[str] = mapped_column(String(500), nullable=False)

    # Statistics
    stargazers_count: Mapped[int] = mapped_column(Integer, default=0)
    forks_count: Mapped[int] = mapped_column(Integer, default=0)
    watchers_count: Mapped[int] = mapped_column(Integer, default=0)
    open_issues_count: Mapped[int] = mapped_column(Integer, default=0)

    # Content for embedding
    readme_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding_text: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # Combined text used for embedding

    # Vector embedding (dimension set from settings)
    # Note: Vector dimension is set dynamically based on embedding model
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(get_embedding_settings().dimensions), nullable=True
    )

    # AI-generated content
    ai_summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Clustering
    cluster_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("clusters.id"), nullable=True, index=True
    )

    # 3D coordinates for visualization (after UMAP)
    coord_x: Mapped[float | None] = mapped_column(nullable=True)
    coord_y: Mapped[float | None] = mapped_column(nullable=True)
    coord_z: Mapped[float | None] = mapped_column(nullable=True)

    # Timestamps
    starred_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    repo_created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    repo_updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    repo_pushed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Processing flags
    is_readme_fetched: Mapped[bool] = mapped_column(Boolean, default=False)
    is_embedded: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    is_summarized: Mapped[bool] = mapped_column(Boolean, default=False)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="starred_repos")
    cluster: Mapped[Optional["Cluster"]] = relationship(
        "Cluster", back_populates="repos"
    )

    # Indexes
    __table_args__ = (
        Index("ix_starred_repos_user_github", "user_id", "github_repo_id", unique=True),
        Index("ix_starred_repos_starred_at", "user_id", "starred_at"),
        Index("ix_starred_repos_language", "user_id", "language"),
    )

    def __repr__(self) -> str:
        return f"<StarredRepo(id={self.id}, full_name={self.full_name})>"


class Cluster(Base):
    """Semantic cluster model."""

    __tablename__ = "clusters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False, index=True
    )

    # Cluster info
    name: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )  # LLM-generated name
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    keywords: Mapped[list[str] | None] = mapped_column(ARRAY(String), nullable=True)
    color: Mapped[str | None] = mapped_column(
        String(20), nullable=True
    )  # Hex color for visualization

    # Statistics
    repo_count: Mapped[int] = mapped_column(Integer, default=0)

    # Cluster center embedding
    center_embedding: Mapped[list[float] | None] = mapped_column(
        Vector(get_embedding_settings().dimensions), nullable=True
    )

    # 3D center coordinates
    center_x: Mapped[float | None] = mapped_column(nullable=True)
    center_y: Mapped[float | None] = mapped_column(nullable=True)
    center_z: Mapped[float | None] = mapped_column(nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    repos: Mapped[list["StarredRepo"]] = relationship(
        "StarredRepo", back_populates="cluster"
    )

    def __repr__(self) -> str:
        return f"<Cluster(id={self.id}, name={self.name}, count={self.repo_count})>"


class SyncTask(Base):
    """Synchronization task tracking model."""

    __tablename__ = "sync_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False, index=True
    )

    # Task info
    task_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # stars, readme, embedding, cluster
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending", index=True
    )  # pending, running, completed, failed

    # Progress
    total_items: Mapped[int] = mapped_column(Integer, default=0)
    processed_items: Mapped[int] = mapped_column(Integer, default=0)
    failed_items: Mapped[int] = mapped_column(Integer, default=0)

    # Error tracking
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_details: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="sync_tasks")

    def __repr__(self) -> str:
        return f"<SyncTask(id={self.id}, type={self.task_type}, status={self.status})>"
