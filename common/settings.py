"""Central, environment-driven configuration.

Values come from environment variables prefixed with ``SENTINEL_`` or from a
local ``.env`` file. Secrets are never hardcoded; see ``.env.example``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Community parquet mirror of the official CICIDS2017 "TrafficLabelling" CSVs,
# pinned to a commit so every download is byte-for-byte reproducible.
DEFAULT_CICIDS_BASE_URL = (
    "https://huggingface.co/datasets/bvsam/cic-ids-2017/resolve/70bac6246d99cf046186a02e1cce6883e2ffe7ea/traffic_labels"
)


class Settings(BaseSettings):
    """Runtime settings for SentinelAI."""

    model_config = SettingsConfigDict(env_prefix="SENTINEL_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    cicids_base_url: str = DEFAULT_CICIDS_BASE_URL
    models_dir: Path = Path("models")
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"

    @property
    def raw_dir(self) -> Path:
        """Untouched downloads."""
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        """Normalized events and feature tables."""
        return self.data_dir / "processed"

    @property
    def samples_dir(self) -> Path:
        """Small sample logs committed to the repository."""
        return self.data_dir / "samples"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
