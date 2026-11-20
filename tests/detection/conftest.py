"""Fixtures for the detection tests: small synthetic feature tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data.features import build_feature_tables
from data.normalize import write_normalized
from detection.dataset import DetectionData, load_detection_data


@pytest.fixture(scope="session")
def features_dir(flows: pd.DataFrame, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Feature tables with a benign-only "Monday" and a mixed "Friday"."""
    root = tmp_path_factory.mktemp("detection")
    benign = flows[flows["label"] == "BENIGN"]
    write_normalized([benign.iloc[:200]], root / "events" / "Monday-WorkingHours.parquet")
    mixed = pd.concat([benign.iloc[200:], flows[flows["label"] != "BENIGN"]])
    write_normalized([mixed], root / "events" / "Friday-WorkingHours.parquet")
    build_feature_tables(root / "events", root / "features")
    return root / "features"


@pytest.fixture(scope="session")
def detection_data(features_dir: Path) -> DetectionData:
    return load_detection_data(features_dir)
