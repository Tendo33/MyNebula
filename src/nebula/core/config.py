"""Extended application configuration for MyNebula.

This module extends the base Settings class with database, embedding,
and GitHub configurations needed for the MyNebula application.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""

    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    user: str = Field(default="mynebula", description="Database user")
    password: str = Field(default="mynebula_secret", description="Database password")
    name: str = Field(default="mynebula", description="Database name")
    url: str | None = Field(
        default=None, description="Full database URL (overrides other settings)"
    )

    model_config = SettingsConfigDict(
        env_prefix="DATABASE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @computed_field
    @property
    def async_url(self) -> str:
        """Get async database URL for SQLAlchemy."""
        if self.url:
            # Ensure it uses asyncpg driver
            url = self.url
            if url.startswith("postgresql://"):
                url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
            elif not url.startswith("postgresql+asyncpg://"):
                url = f"postgresql+asyncpg://{url}"

            # Fix URL parameters for asyncpg
            if "?" in url:
                base, query = url.split("?", 1)
                params = query.split("&")
                new_params = []
                for p in params:
                    if p.startswith("sslmode="):
                        new_params.append("ssl=require")
                    elif p.startswith("channel_binding="):
                        continue
                    else:
                        new_params.append(p)
                url = base + "?" + "&".join(new_params) if new_params else base

            return url
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    @computed_field
    @property
    def sync_url(self) -> str:
        """Get sync database URL for Alembic migrations."""
        if self.url:
            url = self.url
            if "+asyncpg" in url:
                url = url.replace("+asyncpg", "")
            return url
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class EmbeddingSettings(BaseSettings):
    """Embedding service configuration (OpenAI compatible API)."""

    api_key: str = Field(default="", description="API key for embedding service")
    base_url: str = Field(
        default="https://api.siliconflow.cn/v1",
        description="Base URL for embedding API (OpenAI compatible)",
    )
    model: str = Field(
        default="BAAI/bge-large-zh-v1.5",
        description="Embedding model name",
    )
    dimensions: int = Field(
        default=1024,
        description="Embedding vector dimensions",
        ge=64,
        le=4096,
    )

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Warn if API key is empty."""
        if not v:
            import warnings

            warnings.warn(
                "EMBEDDING_API_KEY is not set. Embedding features will not work.",
                stacklevel=2,
            )
        return v


class LLMSettings(BaseSettings):
    """LLM service configuration for summarization (OpenAI compatible API)."""

    api_key: str = Field(default="", description="API key for LLM service")
    base_url: str = Field(
        default="https://api.siliconflow.cn/v1",
        description="Base URL for LLM API (OpenAI compatible)",
    )
    model: str = Field(
        default="Qwen/Qwen2.5-7B-Instruct",
        description="LLM model name",
    )
    output_language: Literal["zh", "en"] = Field(
        default="zh",
        description="Language used in generated summaries and cluster labels",
    )

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class SyncSettings(BaseSettings):
    """Synchronization configuration."""

    batch_size: int = Field(
        default=100,
        description="Number of repos to process per batch",
        ge=10,
        le=500,
    )
    readme_max_length: int = Field(
        default=10000,
        description="Maximum README content length to store",
        ge=1000,
        le=100000,
    )
    default_sync_mode: Literal["incremental", "full"] = Field(
        default="incremental",
        description="Default sync mode: incremental (fast) or full (complete)",
    )
    detect_unstarred_on_incremental: bool = Field(
        default=False,
        description=(
            "Whether incremental sync should fetch the full starred list to detect unstarred repos"
        ),
    )

    model_config = SettingsConfigDict(
        env_prefix="SYNC_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class AppSettings(BaseSettings):
    """Main application settings combining all configurations."""

    # Basic settings
    app_name: str = Field(default="mynebula", description="Application name")
    app_version: str = Field(default="0.2.6", description="Application version")
    debug: bool = Field(
        default=False, description="Debug mode (also controls environment)"
    )
    frontend_url: str = Field(
        default="http://localhost:5173",
        description="Frontend URL for redirects",
    )
    single_user_mode: bool = Field(
        default=True,
        description=(
            "Whether API runs in single-user mode (reads use the first user by default)"
        ),
    )

    # GitHub Personal Access Token
    github_token: str = Field(
        default="",
        description="GitHub Personal Access Token for API access",
    )

    # Admin auth (for protecting mutating operations and settings page)
    admin_username: str = Field(
        default="admin",
        description="Admin username for settings access",
    )
    admin_password: str = Field(
        default="",
        description="Admin password for settings access",
    )
    admin_session_ttl_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Admin session TTL in hours",
    )

    # Logging
    log_level: str = Field(default="INFO", description="Log level")
    log_file: str = Field(default="logs/app.log", description="Log file path")

    # Server
    api_port: int = Field(default=8000, description="API server port")

    # Sub-configurations (loaded separately for clean env prefix handling)
    # These are accessed via get_*_settings() functions

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        allowed = ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"Log level must be one of {allowed}")
        return v_upper


@lru_cache
def get_app_settings() -> AppSettings:
    """Get cached application settings instance."""
    return AppSettings()


@lru_cache
def get_database_settings() -> DatabaseSettings:
    """Get cached database settings instance."""
    return DatabaseSettings()


@lru_cache
def get_embedding_settings() -> EmbeddingSettings:
    """Get cached embedding settings instance."""
    return EmbeddingSettings()


@lru_cache
def get_llm_settings() -> LLMSettings:
    """Get cached LLM settings instance."""
    return LLMSettings()


@lru_cache
def get_sync_settings() -> SyncSettings:
    """Get cached sync settings instance."""
    return SyncSettings()


def reload_all_settings() -> None:
    """Clear all settings caches to reload from environment."""
    get_app_settings.cache_clear()
    get_database_settings.cache_clear()
    get_embedding_settings.cache_clear()
    get_llm_settings.cache_clear()
    get_sync_settings.cache_clear()
