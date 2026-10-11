from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Secrets and configuration are supplied by the process environment.
        # In particular, OPENAI_API_KEY is never read from a dotenv file.
        extra="ignore"
    )

    app_name: str = "Compliance Checker"
    log_level: str = "INFO"
    database_path: Path = PROJECT_ROOT / "data/compliance.sqlite3"
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    openai_timeout_seconds: float = Field(default=60, gt=0)
    embedding_model_id: str = "Alibaba-NLP/gte-large-en-v1.5"
    embedding_device: str = "cpu"
    embedding_cache_dir: Path = PROJECT_ROOT / "data/models"
    embedding_local_files_only: bool = False
    embedding_batch_size: int = Field(default=8, gt=0)
    embedding_revision: str | None = None
    qdrant_path: Path = PROJECT_ROOT / "data/qdrant"
    qdrant_collection: str = "policy_chunks"
    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None
    qdrant_timeout_seconds: int = Field(default=10, gt=0)

    @field_validator(
        "openai_api_key",
        "qdrant_api_key",
        "qdrant_url",
        "embedding_revision",
        mode="before",
    )
    @classmethod
    def blank_as_none(cls, value):
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("database_path", "embedding_cache_dir", "qdrant_path")
    @classmethod
    def resolve_path(cls, value: Path) -> Path:
        return value if value.is_absolute() else PROJECT_ROOT / value

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        value = value.upper()
        if value not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("Unsupported log level")
        return value
