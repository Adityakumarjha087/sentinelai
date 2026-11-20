from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from detection.dataset import DetectionData, load_detection_data


def test_training_split_is_benign_baseline_only(detection_data: DetectionData) -> None:
    assert len(detection_data.train) == 200
    assert not detection_data.train.is_attack.any()
    assert set(detection_data.train.labels) == {"BENIGN"}


def test_evaluation_is_split_in_stratified_halves(detection_data: DetectionData) -> None:
    validation = Counter(detection_data.validation.labels)
    test = Counter(detection_data.test.labels)
    # Friday holds 200 benign, 120 scan and 100 DDoS flows.
    assert validation == {"BENIGN": 100, "PortScan": 60, "DDoS": 50}
    assert test == validation


def test_matrices_are_model_ready(detection_data: DetectionData) -> None:
    for split in (detection_data.train, detection_data.validation, detection_data.test):
        assert split.features.dtype == np.float32
        assert split.features.shape == (len(split), len(detection_data.feature_names))
        assert np.isfinite(split.features).all()
        assert split.is_attack.tolist() == [label != "BENIGN" for label in split.labels]


def test_same_seed_gives_same_split_and_other_seed_differs(features_dir: Path) -> None:
    first = load_detection_data(features_dir, seed=1)
    again = load_detection_data(features_dir, seed=1)
    other = load_detection_data(features_dir, seed=2)
    assert np.array_equal(first.test.features, again.test.features)
    assert not np.array_equal(first.test.features, other.test.features)


def test_row_caps_keep_every_label(features_dir: Path) -> None:
    data = load_detection_data(features_dir, max_train_rows=50, max_eval_rows=42)
    assert len(data.train) == 50
    assert len(data.validation) + len(data.test) == 42
    assert set(data.validation.labels) == {"BENIGN", "PortScan", "DDoS"}


def test_without_a_baseline_file_half_of_benign_trains(features_dir: Path) -> None:
    data = load_detection_data(features_dir, baseline="no-such-day")
    assert len(data.train) == 200  # half of the 400 benign flows
    assert not data.train.is_attack.any()
    assert len(data.validation) + len(data.test) == 420


def test_missing_tables_raise(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_detection_data(tmp_path)
    (tmp_path / "featurizer.json").write_text('{"feature_names": ["a"]}')
    with pytest.raises(FileNotFoundError, match="no feature tables"):
        load_detection_data(tmp_path)
