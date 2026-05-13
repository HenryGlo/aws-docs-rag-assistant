"""Centralized configuration using Pydantic Settings.

Loads environment variables from .env file with type validation.
All settings have sensible defaults except secrets (API keys).
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Claude (Anthropic API)
    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-6"
    max_tokens: int = 1024
    temperature: float = 0.0

    # Embeddings (sentence-transformers, runs locally)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Vector store (ChromaDB)
    chroma_persist_dir: Path = Path("./data/vectorstore")
    collection_name: str = "aws_docs"

    # Retrieval
    top_k: int = 5
    similarity_threshold: float = 0.7

    # Evaluation
    # Cap parallel calls to the Anthropic judge during RAGAs scoring.
    # Default 4 stays well under the per-minute rate limit on Anthropic
    # build-tier 1; raise on higher tiers.
    ragas_max_workers: int = 4

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Logging
    log_level: str = "INFO"

    # Data paths
    data_dir: Path = Path("./data")
    raw_dir: Path = Path("./data/raw")
    processed_dir: Path = Path("./data/processed")


settings = Settings()
