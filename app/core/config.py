"""
Application configuration using pydantic-settings.
All settings are loaded dynamically from environment variables or a .env file.
Supports multiple LLM providers (Groq, Google Gemini, OpenAI) with zero vendor lock-in.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the Multi-Agent Research System."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------------------------------------------------------------------------
    # Application Info & Modes
    # ---------------------------------------------------------------------------
    APP_NAME: str = Field(default="LangChain Multi-Agent Research System")
    APP_VERSION: str = Field(default="2.0.0")
    DEBUG: bool = Field(default=False)
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO"
    )

    # ---------------------------------------------------------------------------
    # Multi-LLM Provider Configuration
    # ---------------------------------------------------------------------------
    # Provider selection: 'auto', 'groq', 'gemini', or 'openai'
    # 'auto' intelligently selects the provider based on available API keys!
    LLM_PROVIDER: str = Field(
        default="auto",
        description="Active LLM provider: auto | groq | gemini | openai",
    )

    # Groq (Lightning-fast open models - LLaMA 3.3, 3.1, Mixtral)
    GROQ_API_KEY: str = Field(default="", description="Groq Cloud API Key")
    GROQ_MODEL: str = Field(default="llama-3.3-70b-versatile", description="Groq model name")

    # Google Gemini (Gemini 1.5 Flash, 1.5 Pro)
    GEMINI_API_KEY: str = Field(default="", description="Google Gemini API key")
    GOOGLE_API_KEY: str = Field(default="", description="Alias for Gemini API key")
    GEMINI_MODEL: str = Field(default="gemini-1.5-flash", description="Gemini model name")

    # OpenAI (GPT-4o-mini, GPT-4o)
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key")
    OPENAI_MODEL: str = Field(default="gpt-4o-mini", description="OpenAI model name")

    # Global LLM settings
    LLM_MODEL: str = Field(default="gpt-4o-mini", description="Default/fallback model name")
    LLM_TEMPERATURE: float = Field(default=0.1, ge=0.0, le=2.0)

    # ---------------------------------------------------------------------------
    # Search APIs
    # ---------------------------------------------------------------------------
    TAVILY_API_KEY: str = Field(default="", description="Tavily search API key")

    # ---------------------------------------------------------------------------
    # Optional External Services
    # ---------------------------------------------------------------------------
    COHERE_API_KEY: str = Field(default="", description="Cohere reranking API key (optional)")
    LANGSMITH_API_KEY: str = Field(default="", description="LangSmith tracing key (optional)")
    LANGSMITH_PROJECT: str = Field(default="multi-agent-research")
    LANGSMITH_TRACING: bool = Field(default=False)

    # ---------------------------------------------------------------------------
    # Vector Store – Qdrant (supports cloud, local, or automatic in-memory fallback)
    # ---------------------------------------------------------------------------
    QDRANT_URL: str = Field(default="http://localhost:6333", description="Qdrant base URL")
    QDRANT_API_KEY: str = Field(default="", description="Qdrant API key (cloud only)")
    QDRANT_COLLECTION_NAME: str = Field(
        default="research_documents", description="Default Qdrant collection"
    )

    # ---------------------------------------------------------------------------
    # Cache & Task Queue – Redis / Celery (optional in direct mode)
    # ---------------------------------------------------------------------------
    REDIS_URL: str = Field(default="redis://localhost:6379", description="Redis connection URL")
    CELERY_BROKER_URL: str = Field(default="redis://localhost:6379/1")
    CELERY_RESULT_BACKEND: str = Field(default="redis://localhost:6379/2")

    # ---------------------------------------------------------------------------
    # Relational Database – PostgreSQL or SQLite fallback
    # ---------------------------------------------------------------------------
    DATABASE_URL: str = Field(
        default="sqlite+aiosqlite:///./research.db",
        description="Async database connection string",
    )

    # ---------------------------------------------------------------------------
    # Auth / JWT
    # ---------------------------------------------------------------------------
    JWT_SECRET_KEY: str = Field(
        default="dev-insecure-secret-key-change-in-production-1234567890",
        description="Secret key used to sign JWT tokens",
    )
    JWT_ALGORITHM: str = Field(default="HS256")
    JWT_EXPIRE_MINUTES: int = Field(default=60, ge=1)

    # ---------------------------------------------------------------------------
    # Rate Limiting
    # ---------------------------------------------------------------------------
    RATE_LIMIT_REQUESTS: int = Field(default=100, ge=1)
    RATE_LIMIT_WINDOW: int = Field(default=60, ge=1)
    MAX_CONCURRENT_REQUESTS: int = Field(default=50, ge=1)

    # ---------------------------------------------------------------------------
    # Embeddings & Document Chunking
    # ---------------------------------------------------------------------------
    EMBEDDING_PROVIDER: str = Field(default="auto", description="auto | openai | gemini | local")
    EMBEDDING_MODEL: str = Field(default="text-embedding-3-large")
    EMBEDDING_DIM: int = Field(default=3072, ge=1)
    CHUNK_SIZE: int = Field(default=512, ge=64)
    CHUNK_OVERLAP: int = Field(default=50, ge=0)
    MAX_SEARCH_RESULTS: int = Field(default=10, ge=1)
    MAX_SCRAPE_CHARS: int = Field(default=8000, ge=100)

    @field_validator("CHUNK_OVERLAP")
    @classmethod
    def overlap_less_than_chunk(cls, v: int, info: object) -> int:
        data = getattr(info, "data", {})
        chunk_size = data.get("CHUNK_SIZE", 512)
        if v >= chunk_size:
            raise ValueError("CHUNK_OVERLAP must be less than CHUNK_SIZE")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance (singleton)."""
    return Settings()


# Module-level convenience alias
settings: Settings = get_settings()
