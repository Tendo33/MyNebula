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
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from nebula.core.config import get_embedding_settings

# Naming convention for constraints to avoid "unnamed constraint" errors in Alembic
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Base class for all models."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


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
    taxonomy_versions: Mapped[list["TaxonomyVersion"]] = relationship(
        "TaxonomyVersion", back_populates="user", cascade="all, delete-orphan"
    )
    taxonomy_overrides: Mapped[list["UserTaxonomyOverride"]] = relationship(
        "UserTaxonomyOverride", back_populates="user", cascade="all, delete-orphan"
    )
    sync_schedule: Mapped[Optional["SyncSchedule"]] = relationship(
        "SyncSchedule",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
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

    # Owner avatar (for display)
    owner_avatar_url: Mapped[str | None] = mapped_column(String(500), nullable=True)

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
    ai_tags: Mapped[list[str] | None] = mapped_column(
        ARRAY(String), nullable=True
    )  # AI-generated tags based on README

    # User's GitHub Star List (user-defined category)
    star_list_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("star_lists.id"), nullable=True, index=True
    )

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

    # Content hashes for smart change detection
    description_hash: Mapped[str | None] = mapped_column(
        String(32), nullable=True
    )  # MD5 hash of description
    topics_hash: Mapped[str | None] = mapped_column(
        String(32), nullable=True
    )  # MD5 hash of sorted topics

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="starred_repos")
    cluster: Mapped[Optional["Cluster"]] = relationship(
        "Cluster", back_populates="repos"
    )
    star_list: Mapped[Optional["StarList"]] = relationship(
        "StarList", back_populates="repos"
    )

    # Indexes
    __table_args__ = (
        Index("ix_starred_repos_user_github", "user_id", "github_repo_id", unique=True),
        Index("ix_starred_repos_starred_at", "user_id", "starred_at"),
        Index("ix_starred_repos_language", "user_id", "language"),
    )

    def __repr__(self) -> str:
        return f"<StarredRepo(id={self.id}, full_name={self.full_name})>"


class StarList(Base):
    """User's GitHub Star List (user-defined category).

    Represents a user's custom list for organizing starred repos on GitHub.
    """

    __tablename__ = "star_lists"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False, index=True
    )
    github_list_id: Mapped[str | None] = mapped_column(
        String(100), nullable=True, index=True
    )  # GitHub's list ID (from GraphQL)

    # List info
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_public: Mapped[bool] = mapped_column(Boolean, default=True)

    # Statistics
    repo_count: Mapped[int] = mapped_column(Integer, default=0)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    repos: Mapped[list["StarredRepo"]] = relationship(
        "StarredRepo", back_populates="star_list"
    )

    # Indexes
    __table_args__ = (
        Index("ix_star_lists_user_github", "user_id", "github_list_id", unique=True),
    )

    def __repr__(self) -> str:
        return f"<StarList(id={self.id}, name={self.name}, count={self.repo_count})>"


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


class TaxonomyVersion(Base):
    """Taxonomy version snapshot.

    A version can be global (`user_id` is null) or user-specific.
    """

    __tablename__ = "taxonomy_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=True, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(50), default="auto")
    status: Mapped[str] = mapped_column(String(20), default="draft", index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    stats: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    user: Mapped["User"] = relationship("User", back_populates="taxonomy_versions")
    terms: Mapped[list["TaxonomyTerm"]] = relationship(
        "TaxonomyTerm", back_populates="version", cascade="all, delete-orphan"
    )
    mappings: Mapped[list["TaxonomyMapping"]] = relationship(
        "TaxonomyMapping", back_populates="version", cascade="all, delete-orphan"
    )
    candidates: Mapped[list["TaxonomyCandidate"]] = relationship(
        "TaxonomyCandidate", back_populates="version", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_taxonomy_versions_user_status", "user_id", "status"),
    )


class TaxonomyTerm(Base):
    """Canonical taxonomy term within a version."""

    __tablename__ = "taxonomy_terms"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("taxonomy_versions.id"), nullable=False, index=True
    )
    term: Mapped[str] = mapped_column(String(255), nullable=False)
    normalized_term: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    aliases: Mapped[list[str] | None] = mapped_column(ARRAY(String), nullable=True)
    term_metadata: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    version: Mapped["TaxonomyVersion"] = relationship(
        "TaxonomyVersion", back_populates="terms"
    )

    __table_args__ = (
        Index("ix_taxonomy_terms_version_normalized", "version_id", "normalized_term"),
    )


class TaxonomyMapping(Base):
    """Resolved source term to canonical term mapping."""

    __tablename__ = "taxonomy_mappings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("taxonomy_versions.id"), nullable=False, index=True
    )
    source_term: Mapped[str] = mapped_column(String(255), nullable=False)
    source_normalized: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    canonical_term: Mapped[str] = mapped_column(String(255), nullable=False)
    canonical_normalized: Mapped[str] = mapped_column(
        String(255), nullable=False, index=True
    )
    confidence_score: Mapped[float] = mapped_column(Float, default=0.0)
    confidence_level: Mapped[str] = mapped_column(String(20), default="low", index=True)
    evidence: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    version: Mapped["TaxonomyVersion"] = relationship(
        "TaxonomyVersion", back_populates="mappings"
    )

    __table_args__ = (
        Index("ix_taxonomy_mappings_version_source", "version_id", "source_normalized"),
    )


class TaxonomyCandidate(Base):
    """Candidate term relationship generated by offline analysis."""

    __tablename__ = "taxonomy_candidates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("taxonomy_versions.id"), nullable=False, index=True
    )
    left_term: Mapped[str] = mapped_column(String(255), nullable=False)
    left_normalized: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    right_term: Mapped[str] = mapped_column(String(255), nullable=False)
    right_normalized: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    confidence_level: Mapped[str] = mapped_column(String(20), default="low", index=True)
    decision: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    evidence: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    version: Mapped["TaxonomyVersion"] = relationship(
        "TaxonomyVersion", back_populates="candidates"
    )

    __table_args__ = (
        Index(
            "ix_taxonomy_candidates_version_pair",
            "version_id",
            "left_normalized",
            "right_normalized",
        ),
    )


class UserTaxonomyOverride(Base):
    """User-specific override for source-term canonical mapping."""

    __tablename__ = "user_taxonomy_overrides"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False, index=True
    )
    source_term: Mapped[str] = mapped_column(String(255), nullable=False)
    source_normalized: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    canonical_term: Mapped[str] = mapped_column(String(255), nullable=False)
    canonical_normalized: Mapped[str] = mapped_column(
        String(255), nullable=False, index=True
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    user: Mapped["User"] = relationship("User", back_populates="taxonomy_overrides")

    __table_args__ = (
        Index(
            "ix_user_taxonomy_overrides_user_source",
            "user_id",
            "source_normalized",
            unique=True,
        ),
    )


class SyncSchedule(Base):
    """Scheduled sync configuration model.

    Stores user's automatic sync schedule settings for periodic synchronization.
    """

    __tablename__ = "sync_schedules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False, unique=True, index=True
    )

    # Schedule configuration
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    schedule_hour: Mapped[int] = mapped_column(Integer, default=9)  # 0-23, default 9 AM
    schedule_minute: Mapped[int] = mapped_column(Integer, default=0)  # 0-59
    timezone: Mapped[str] = mapped_column(String(50), default="Asia/Shanghai")

    # Execution status
    last_run_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_run_status: Mapped[str | None] = mapped_column(
        String(20), nullable=True
    )  # 'success', 'failed', 'running'
    last_run_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="sync_schedule")

    def __repr__(self) -> str:
        return f"<SyncSchedule(id={self.id}, user_id={self.user_id}, enabled={self.is_enabled}, time={self.schedule_hour}:{self.schedule_minute:02d})>"


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
