"""API runtime settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    """Environment-backed settings for the FastAPI service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    CHURN_API_KEY: str = ""
    CHURN_CORS_ORIGINS: str = ""
    CHURN_DEBUG: bool = False

    @property
    def cors_origins(self) -> list[str]:
        """Parse comma-separated CORS origins."""
        if not self.CHURN_CORS_ORIGINS.strip():
            return []
        return [origin.strip() for origin in self.CHURN_CORS_ORIGINS.split(",") if origin.strip()]


@lru_cache(maxsize=1)
def get_api_settings() -> ApiSettings:
    """Return cached API settings."""
    return ApiSettings()
