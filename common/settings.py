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
    # Optional artifacts the API and dashboard use when they exist. Unset means
    # the default location under data_dir / models_dir (see the properties below).
    kb_dir: Path | None = None
    detector_dir: Path | None = None
    featurizer_path: Path | None = None
    demo_flows_path: Path | None = None
    injection_model_dir: Path | None = None
    max_detections: int = 500

    @property
    def knowledge_base_dir(self) -> Path:
        """Built ATT&CK / CVE index (``make kb``)."""
        return self.kb_dir or self.processed_dir / "kb"

    @property
    def flow_model_dir(self) -> Path:
        """Best flow detector saved by ``make train``."""
        return self.detector_dir or self.models_dir / "cicids2017" / "best"

    @property
    def flow_featurizer_path(self) -> Path:
        """Featurizer the flow detector was trained with (``make data``)."""
        return self.featurizer_path or self.processed_dir / "features" / "cicids2017" / "featurizer.json"

    @property
    def injection_classifier_dir(self) -> Path:
        """Fine-tuned prompt-injection classifier (``sentinel-guard train``)."""
        return self.injection_model_dir or self.models_dir / "injection" / "distilbert"

    @property
    def flow_demo_path(self) -> Path:
        """Normalized flows the detection feed is built from (default: the port-scan afternoon)."""
        default = self.processed_dir / "events" / "cicids2017" / "Friday-WorkingHours-Afternoon-PortScan.parquet"
        return self.demo_flows_path or default

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
