from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import pytest

from common.settings import get_settings
from detection.dataset import DetectionData
from detection.models import AutoencoderDetector, IsolationForestDetector
from detection.train import (
    METADATA_FILE,
    RESULTS_FILE,
    RunResult,
    load_model,
    main,
    results_table,
    run_experiment,
    train_and_evaluate,
)


@pytest.fixture(scope="module")
def experiment(
    detection_data: DetectionData, tmp_path_factory: pytest.TempPathFactory
) -> tuple[list[RunResult], Path, str]:
    root = tmp_path_factory.mktemp("experiment")
    tracking_uri = f"sqlite:///{root / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.create_experiment("test", artifact_location=(root / "artifacts").as_uri())
    results = run_experiment(
        detection_data,
        [IsolationForestDetector(n_estimators=50), AutoencoderDetector(epochs=40, batch_size=32)],
        out_dir=root / "models",
        tracking_uri=tracking_uri,
        experiment="test",
    )
    return results, root / "models", tracking_uri


def test_threshold_comes_from_validation_and_metrics_from_test(
    detection_data: DetectionData,
) -> None:
    result = train_and_evaluate(IsolationForestDetector(n_estimators=50), detection_data)
    assert result.name == "isolation_forest"
    assert result.validation["f1"] > 0.8
    assert result.test["f1"] > 0.8
    assert set(result.test_recall_by_label) == {"DDoS", "PortScan"}
    assert result.fit_seconds > 0
    assert result.score_seconds_per_1k > 0


def test_experiment_evaluates_every_detector(experiment: tuple[list[RunResult], Path, str]) -> None:
    results, _, _ = experiment
    assert [result.name for result in results] == ["isolation_forest", "autoencoder"]
    for result in results:
        assert set(result.test) == {
            "precision",
            "recall",
            "f1",
            "false_positive_rate",
            "roc_auc",
            "pr_auc",
        }
        assert all(0.0 <= value <= 1.0 for value in result.test.values())


def test_best_model_is_chosen_on_validation_and_saved(
    experiment: tuple[list[RunResult], Path, str], detection_data: DetectionData
) -> None:
    results, out_dir, _ = experiment
    winner = max(results, key=lambda result: result.validation["f1"])
    loaded = load_model(out_dir / "best")
    assert loaded.metadata["model"] == winner.name
    assert loaded.threshold == pytest.approx(winner.threshold)
    assert loaded.feature_names == detection_data.feature_names
    alerts = loaded.is_anomalous(detection_data.test.features)
    recall = float(alerts[detection_data.test.is_attack].mean())
    assert recall == pytest.approx(winner.test["recall"], abs=1e-6)


def test_results_table_is_written(experiment: tuple[list[RunResult], Path, str]) -> None:
    results, out_dir, _ = experiment
    table = (out_dir / RESULTS_FILE).read_text()
    assert table == results_table(results)
    assert "| isolation_forest |" in table
    assert "| autoencoder |" in table
    assert "| PortScan |" in table


def test_runs_are_tracked_in_mlflow(experiment: tuple[list[RunResult], Path, str]) -> None:
    results, _, tracking_uri = experiment
    mlflow.set_tracking_uri(tracking_uri)
    runs = mlflow.search_runs(experiment_names=["test"], output_format="list")
    by_name = {run.info.run_name: run for run in runs}
    assert set(by_name) == {"isolation_forest", "autoencoder"}
    for result in results:
        run = by_name[result.name]
        assert run.data.metrics["test_f1"] == pytest.approx(result.test["f1"])
        assert run.data.metrics["recall_portscan"] == pytest.approx(
            result.test_recall_by_label["PortScan"]
        )
        assert run.data.params["model"] == result.name
        artifacts = mlflow.artifacts.list_artifacts(
            run_id=run.info.run_id, artifact_path=result.name
        )
        assert METADATA_FILE in {Path(item.path).name for item in artifacts}
    history = mlflow.MlflowClient().get_metric_history(
        by_name["autoencoder"].info.run_id, "holdout_loss"
    )
    assert len(history) > 1


def test_tampered_model_files_are_refused(
    experiment: tuple[list[RunResult], Path, str], tmp_path: Path
) -> None:
    _, out_dir, _ = experiment
    copy = tmp_path / "best"
    copy.mkdir()
    for path in (out_dir / "best").iterdir():
        (copy / path.name).write_bytes(path.read_bytes())
    metadata = json.loads((copy / METADATA_FILE).read_text())
    victim = next(name for name in metadata["files_sha256"] if name.startswith("model."))
    (copy / victim).write_bytes((copy / victim).read_bytes() + b"\x00")
    with pytest.raises(ValueError, match="does not match its recorded checksum"):
        load_model(copy)


def test_cli_trains_and_saves(
    features_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_dir = tmp_path / "data"
    target = data_dir / "processed" / "features" / "synthetic"
    target.mkdir(parents=True)
    for path in features_dir.iterdir():
        (target / path.name).write_bytes(path.read_bytes())
    monkeypatch.setenv("SENTINEL_DATA_DIR", str(data_dir))
    monkeypatch.setenv("SENTINEL_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("SENTINEL_MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    try:
        assert main(["--dataset", "synthetic", "--epochs", "3"]) == 0
    finally:
        get_settings.cache_clear()
    assert "| isolation_forest |" in capsys.readouterr().out
    loaded = load_model(tmp_path / "models" / "synthetic" / "best")
    assert np.isfinite(loaded.threshold)
