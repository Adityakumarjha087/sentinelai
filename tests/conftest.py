"""Shared fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from common.settings import get_settings
from data.features import FlowFeaturizer
from data.normalize import write_normalized
from data.parsers.cicids import parse_cicids_file
from data.synthetic import SYNTHETIC_FILE_NAME, generate_cicids_like_csv
from detection.models import IsolationForestDetector
from detection.train import RunResult, save_model
from guardrails.injection import use_classifier

REPO_ROOT = Path(__file__).resolve().parents[1]
OPTIONAL_ARTIFACTS = (
    "SENTINEL_KB_DIR",
    "SENTINEL_DETECTOR_DIR",
    "SENTINEL_FEATURIZER_PATH",
    "SENTINEL_DEMO_FLOWS_PATH",
    "SENTINEL_INJECTION_MODEL_DIR",
)


@pytest.fixture(scope="session", autouse=True)
def _no_local_artifacts(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Keep tests hermetic: never load the knowledge base or model a developer built locally."""
    absent = tmp_path_factory.mktemp("absent")
    with pytest.MonkeyPatch.context() as patch:
        for name in OPTIONAL_ARTIFACTS:
            patch.setenv(name, str(absent / name.lower()))
        get_settings.cache_clear()
        yield
    get_settings.cache_clear()


@pytest.fixture(scope="session")
def sysmon_sample() -> Path:
    """The synthetic Sysmon export committed with the repository."""
    return REPO_ROOT / "data" / "samples" / "sysmon_sample.xml"


@pytest.fixture(scope="session")
def synthetic_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A synthetic flow file in the original (quirky) CICIDS2017 CSV format."""
    path = tmp_path_factory.mktemp("raw") / SYNTHETIC_FILE_NAME
    return generate_cicids_like_csv(path, n_benign=400, n_scan=120, n_ddos=100)


@pytest.fixture(scope="session")
def flows(synthetic_csv: Path) -> pd.DataFrame:
    """The synthetic flows, normalized."""
    return pd.concat(parse_cicids_file(synthetic_csv, chunk_rows=250), ignore_index=True)


@pytest.fixture(autouse=True)
def _no_registered_classifier() -> Iterator[None]:
    """Each test starts and ends with pattern-only injection detection."""
    use_classifier(None)
    yield
    use_classifier(None)


@dataclass(frozen=True)
class FlowArtifacts:
    model_dir: Path
    featurizer_path: Path
    flows_path: Path


@pytest.fixture(scope="session")
def flow_artifacts(flows: pd.DataFrame, tmp_path_factory: pytest.TempPathFactory) -> FlowArtifacts:
    """A small Isolation Forest trained on the synthetic benign flows, saved like ``make train`` does."""
    root = tmp_path_factory.mktemp("flow-artifacts")
    benign = flows[flows["label"] == "BENIGN"]
    featurizer = FlowFeaturizer().fit(benign)
    detector = IsolationForestDetector(n_estimators=50, max_samples=256).fit(featurizer.transform(benign).to_numpy())
    threshold = float(np.quantile(detector.score(featurizer.transform(benign).to_numpy()), 0.95))
    result = RunResult(
        name=IsolationForestDetector.name, params=detector.params(), threshold=threshold, validation={}, test={},
        test_recall_by_label={}, fit_seconds=0.0, score_seconds_per_1k=0.0,
    )  # fmt: skip
    save_model(detector, result, featurizer.feature_names_, root / "model")
    featurizer.save(root / "featurizer.json")
    write_normalized([flows], root / "flows.parquet")
    return FlowArtifacts(root / "model", root / "featurizer.json", root / "flows.parquet")
