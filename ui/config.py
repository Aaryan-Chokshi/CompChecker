from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class UISettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="UI_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_base_url: str = "http://127.0.0.1:8000/api/v1"
    api_timeout_seconds: float = Field(default=5, gt=0)
    host: str = "127.0.0.1"
    port: int = Field(default=8050, ge=1, le=65535)
    debug: bool = False

    @field_validator("api_base_url")
    @classmethod
    def validate_api_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("UI_API_BASE_URL must be an HTTP(S) URL.")
        return value.rstrip("/")
