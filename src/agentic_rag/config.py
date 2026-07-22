"""
Application configuration loaded from environment variables and .env.

Centralizes all runtime settings (API keys, model names, chunk sizes,
index paths) so every module reads from one source of truth.
"""

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Typed settings object backed by process env vars and optional .env file.

    Pydantic validates types on load and ignores unknown env keys so extra
    variables in .env do not crash the app.
    """

    # Read .env from the project root; ignore keys we do not declare below.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI credentials and model selection. Empty key is allowed at import
    # time so CLI help/stats work before .env is configured.
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # Where the on-disk FAISS index is stored, plus ingestion/retrieval tuning.
    vector_index_dir: Path = Field(default=Path("./data/index"))
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_top_k: int = 4

    @field_validator("openai_api_key")
    @classmethod
    def strip_key(cls, value: str) -> str:
        """Remove accidental whitespace from copied API keys."""
        return value.strip()

    def ensure_dirs(self) -> None:
        """Create the vector index directory if it does not exist yet."""
        self.vector_index_dir.mkdir(parents=True, exist_ok=True)

    def require_api_key(self) -> str:
        """
        Return a non-empty API key or raise with a setup hint.

        Called only when making OpenAI requests (embed/query), not at startup.
        """
        if not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        return self.openai_api_key


# Singleton used across the codebase; instantiated once on first import.
settings = Settings()
