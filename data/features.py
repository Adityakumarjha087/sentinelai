"""Feature engineering for network-flow anomaly detection.

Design choices that matter for honest evaluation:

* **No identifiers as features.** IPs, flow ids and raw ports are excluded. In
  CICIDS2017 the attacker always uses the same addresses, so a model fed IPs
  "detects" attacks by memorizing them. IPs are used only as *grouping keys* for
  behavioural aggregates (how many flows / distinct ports per source per minute).
* **No time-of-day by default.** Attacks in the dataset were launched on a fixed
  schedule, so the clock leaks the label. Enable ``include_time`` only for
  deployments where working-hours context is genuinely meaningful.
* **Fit on training data only.** Imputation medians and the choice of
  log-scaled columns are learned in :meth:`FlowFeaturizer.fit` and stored as
  plain JSON (never pickle) so a saved featurizer is safe to load.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from data.schema import COMMON_COLUMNS

logger = logging.getLogger(__name__)

META_COLUMNS: tuple[str, ...] = ("event_id", "timestamp", "label", "is_attack")
_WELL_KNOWN_PORT_MAX = 1023
_EPHEMERAL_PORT_MIN = 49152


def behavioral_features(frame: pd.DataFrame, window: str = "1min") -> pd.DataFrame:
    """Per-flow counts of what its source and destination did in the same time window.

    Windows are tumbling (``timestamp`` floored to ``window``), which matches the
    one-minute resolution of CICIDS2017 timestamps. Port scans show up as a high
    ``src_unique_dst_ports``; floods as a high ``dst_flow_count``.
    """
    keys = pd.DataFrame(
        {
            "src": frame["src_ip"],
            "dst": frame["dst_ip"],
            "port": frame["dst_port"],
            "bucket": frame["timestamp"].dt.floor(window),
            "one": 1,
        },
        index=frame.index,
    )
    by_src = keys.groupby(["src", "bucket"], sort=False, dropna=False)
    by_dst = keys.groupby(["dst", "bucket"], sort=False, dropna=False)
    return pd.DataFrame(
        {
            "src_flow_count": by_src["one"].transform("size"),
            "src_unique_dst_ports": by_src["port"].transform("nunique"),
            "src_unique_dst_ips": by_src["dst"].transform("nunique"),
            "dst_flow_count": by_dst["one"].transform("size"),
            "dst_unique_src_ips": by_dst["src"].transform("nunique"),
        },
        index=frame.index,
    ).astype("float64")


def port_protocol_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Coarse port classes and protocol indicators (not the raw port numbers)."""
    src_port, dst_port = frame["src_port"], frame["dst_port"]
    return pd.DataFrame(
        {
            "dst_port_well_known": (dst_port <= _WELL_KNOWN_PORT_MAX).fillna(False),
            "dst_port_ephemeral": (dst_port >= _EPHEMERAL_PORT_MIN).fillna(False),
            "src_port_well_known": (src_port <= _WELL_KNOWN_PORT_MAX).fillna(False),
            "proto_tcp": (frame["protocol"] == "tcp").fillna(False),
            "proto_udp": (frame["protocol"] == "udp").fillna(False),
        },
        index=frame.index,
    ).astype("float64")


def time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Cyclical hour-of-day encoding (UTC). See the module docstring before using."""
    hour = frame["timestamp"].dt.hour + frame["timestamp"].dt.minute / 60.0
    angle = 2 * np.pi * hour / 24.0
    return pd.DataFrame({"hour_sin": np.sin(angle), "hour_cos": np.cos(angle)}, index=frame.index)


class FlowFeaturizer:
    """Turn normalized flow events into a clean numeric feature matrix.

    Steps: take the numeric flow statistics, add behavioural / port / protocol
    features, replace ``inf`` with missing, impute missing values with training
    medians, and compress heavy-tailed columns with a signed ``log1p``.
    """

    def __init__(
        self, window: str = "1min", include_time: bool = False, log_threshold: float = 100.0
    ) -> None:
        self.window = window
        self.include_time = include_time
        self.log_threshold = log_threshold
        self.feature_names_: list[str] = []
        self.medians_: dict[str, float] = {}
        self.log_columns_: list[str] = []

    def _engineer(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in ("timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "protocol")
                   if c not in frame.columns]  # fmt: skip
        if missing:
            raise ValueError(f"frame is missing required columns: {missing}")
        stats = frame.drop(columns=[c for c in COMMON_COLUMNS if c in frame.columns])
        parts = [
            stats.select_dtypes(include="number").astype("float64"),
            behavioral_features(frame, self.window),
            port_protocol_features(frame),
        ]
        if self.include_time:
            parts.append(time_features(frame))
        return pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan)

    def fit(self, frame: pd.DataFrame) -> Self:
        """Learn the feature list, imputation medians and log-scaled columns."""
        engineered = self._engineer(frame)
        usable = engineered.loc[:, engineered.notna().any(axis=0)]
        if usable.empty:
            raise ValueError("no usable numeric features to fit on")
        self.feature_names_ = [str(column) for column in usable.columns]
        self.medians_ = {str(k): float(v) for k, v in usable.median().items()}
        peak = usable.abs().max()
        self.log_columns_ = [str(column) for column in peak[peak > self.log_threshold].index]
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a ``float32`` feature matrix with no missing or infinite values."""
        if not self.feature_names_:
            raise RuntimeError("FlowFeaturizer is not fitted; call fit() or load() first")
        features = self._engineer(frame).reindex(columns=self.feature_names_)
        features = features.fillna(self.medians_)
        logged = features[self.log_columns_]
        features[self.log_columns_] = np.sign(logged) * np.log1p(logged.abs())
        return features.astype("float32")

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Fit on ``frame`` and transform it."""
        return self.fit(frame).transform(frame)

    def to_dict(self) -> dict[str, Any]:
        """Serializable state."""
        return {
            "window": self.window,
            "include_time": self.include_time,
            "log_threshold": self.log_threshold,
            "feature_names": self.feature_names_,
            "medians": self.medians_,
            "log_columns": self.log_columns_,
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> Self:
        """Rebuild a fitted featurizer from :meth:`to_dict` output."""
        featurizer = cls(
            window=str(state["window"]),
            include_time=bool(state["include_time"]),
            log_threshold=float(state["log_threshold"]),
        )
        featurizer.feature_names_ = [str(name) for name in state["feature_names"]]
        featurizer.medians_ = {str(k): float(v) for k, v in state["medians"].items()}
        featurizer.log_columns_ = [str(name) for name in state["log_columns"]]
        return featurizer

    def save(self, path: Path) -> None:
        """Write the fitted state as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load a featurizer saved with :meth:`save`."""
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _load_without_raw(path: Path) -> pd.DataFrame:
    columns = [name for name in pq.ParquetFile(path).schema_arrow.names if name != "raw"]
    return pd.read_parquet(path, columns=columns)


def build_feature_tables(
    normalized_dir: Path, out_dir: Path, *, fit_on: str = "monday"
) -> FlowFeaturizer:
    """Featurize every normalized flow file, fitting on the benign baseline only.

    The featurizer is fitted on files whose name contains ``fit_on`` (Monday is
    the attack-free day in CICIDS2017), or on the first file if none match. Each
    output table holds :data:`META_COLUMNS` plus the feature columns. The fitted
    state is saved next to them as ``featurizer.json``.
    """
    files = sorted(normalized_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no normalized parquet files in {normalized_dir}")
    fit_files = [file for file in files if fit_on.lower() in file.name.lower()]
    if not fit_files:
        logger.warning("no file matches %r; fitting on %s", fit_on, files[0].name)
        fit_files = files[:1]

    fit_frame = pd.concat([_load_without_raw(file) for file in fit_files], ignore_index=True)
    featurizer = FlowFeaturizer().fit(fit_frame)
    del fit_frame

    out_dir.mkdir(parents=True, exist_ok=True)
    for file in files:
        frame = _load_without_raw(file)
        features = featurizer.transform(frame)
        table = pd.concat([frame[list(META_COLUMNS)], features], axis=1)
        table.to_parquet(out_dir / file.name, compression="zstd", index=False)
        logger.info("features %s: %d rows x %d features", file.name, *features.shape)
    featurizer.save(out_dir / "featurizer.json")
    return featurizer
