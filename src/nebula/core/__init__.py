"""Core business logic module.

This module contains the core services and logic for MyNebula:
- config: Extended configuration management
- embedding: Unified embedding service for multiple providers
- github_client: GitHub API wrapper
- clustering: UMAP + HDBSCAN clustering
- summarizer: AI-powered summarization
"""

from .config import (
    AppSettings,
    DatabaseSettings,
    EmbeddingSettings,
    GitHubSettings,
    LLMSettings,
    SyncSettings,
    get_app_settings,
    get_database_settings,
    get_embedding_settings,
    get_github_settings,
    get_llm_settings,
    get_sync_settings,
    reload_all_settings,
)
from .embedding import EmbeddingService, close_embedding_service, get_embedding_service
from .github_client import GitHubClient, GitHubRepo, GitHubUser

__all__ = [
    # Config
    "AppSettings",
    "DatabaseSettings",
    "EmbeddingSettings",
    "GitHubSettings",
    "LLMSettings",
    "SyncSettings",
    "get_app_settings",
    "get_database_settings",
    "get_embedding_settings",
    "get_github_settings",
    "get_llm_settings",
    "get_sync_settings",
    "reload_all_settings",
    # Embedding
    "EmbeddingService",
    "get_embedding_service",
    "close_embedding_service",
    # GitHub
    "GitHubClient",
    "GitHubUser",
    "GitHubRepo",
]
