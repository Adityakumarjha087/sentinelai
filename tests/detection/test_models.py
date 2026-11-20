from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from detection.dataset import DetectionData
from detection.models import (
    DETECTOR_TYPES,
    AutoencoderDetector,
    Detector,
    IsolationForestDetector,
    _Autoencoder,
)


def make(name: str) -> Detector:
    if name == "autoencoder":
        return AutoencoderDetector(epochs=40, batch_size=32, patience=40)
    return IsolationForestDetector(n_estimators=50)


@pytest.fixture(scope="module", params=sorted(DETECTOR_TYPES))
def fitted(request: pytest.FixtureRequest, detection_data: DetectionData) -> Detector:
    return make(request.param).fit(detection_data.train.features)


def test_attacks_score_higher_than_benign(fitted: Detector, detection_data: DetectionData) -> None:
    test = detection_data.test
    scores = fitted.score(test.features)
    assert scores.shape == (len(test),)
    assert scores.dtype == np.float32
    assert np.isfinite(scores).all()
    assert roc_auc_score(test.is_attack, scores) > 0.9


def test_save_and_load_reproduce_scores(fitted: Detector, detection_data: DetectionData, tmp_path: Path) -> None:
    fitted.save(tmp_path / fitted.name)
    restored = DETECTOR_TYPES[fitted.name].load(tmp_path / fitted.name)
    assert restored.params() == fitted.params()
    np.testing.assert_allclose(
        restored.score(detection_data.test.features),
        fitted.score(detection_data.test.features),
        rtol=1e-5,
    )


@pytest.mark.parametrize("name", sorted(DETECTOR_TYPES))
def test_unfitted_models_refuse_to_score_or_save(name: str, tmp_path: Path) -> None:
    detector = make(name)
    with pytest.raises(RuntimeError, match="not fitted"):
        detector.score(np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(RuntimeError, match="not fitted"):
        detector.save(tmp_path)


def test_same_seed_trains_the_same_autoencoder(detection_data: DetectionData) -> None:
    train, test = detection_data.train.features, detection_data.test.features
    first = AutoencoderDetector(epochs=3, batch_size=32).fit(train).score(test)
    second = AutoencoderDetector(epochs=3, batch_size=32).fit(train).score(test)
    np.testing.assert_allclose(first, second, rtol=1e-5)


def test_autoencoder_learns_and_stops_early(detection_data: DetectionData) -> None:
    detector = AutoencoderDetector(epochs=40, batch_size=32, patience=40)
    detector.fit(detection_data.train.features)
    history = detector.history_
    assert history[-1]["holdout_loss"] < history[0]["holdout_loss"]

    impatient = AutoencoderDetector(epochs=200, batch_size=32, patience=1, learning_rate=0.5)
    impatient.fit(detection_data.train.features)
    assert len(impatient.history_) < 200


def test_autoencoder_is_a_symmetric_bottleneck() -> None:
    network = _Autoencoder(n_features=20, hidden=(16, 8), latent=4)
    widths = [m.out_features for m in [*network.encoder, *network.decoder] if isinstance(m, torch.nn.Linear)]
    assert widths == [16, 8, 4, 8, 16, 20]
    assert network(torch.zeros(5, 20)).shape == (5, 20)


def test_constant_features_and_extreme_values_stay_finite() -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(size=(200, 4)).astype(np.float32)
    train[:, 0] = 7.0  # zero variance
    detector = AutoencoderDetector(epochs=2, batch_size=64).fit(train)
    extreme = np.full((3, 4), 1e12, dtype=np.float32)
    assert np.isfinite(detector.score(extreme)).all()
