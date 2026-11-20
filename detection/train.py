"""Train the detectors, compare them, track runs in MLflow and save the best one.

Usage: ``sentinel-detect --dataset cicids2017``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow

from common.settings import get_settings
from detection.dataset import DetectionData, FloatArray, load_detection_data
from detection.metrics import best_f1_threshold, evaluate, per_label_recall
from detection.models import (
    DETECTOR_TYPES,
    AutoencoderDetector,
    Detector,
    IsolationForestDetector,
)

logger = logging.getLogger("detection.train")

METADATA_FILE = "metadata.json"
RESULTS_FILE = "results.md"
_METRIC_ORDER = ("precision", "recall", "f1", "false_positive_rate", "roc_auc", "pr_auc")


@dataclass(frozen=True)
class RunResult:
    """Outcome of training and evaluating one detector."""

    name: str
    params: dict[str, Any]
    threshold: float
    validation: dict[str, float]
    test: dict[str, float]
    test_recall_by_label: dict[str, float]
    fit_seconds: float
    score_seconds_per_1k: float


def _metric_key(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]+", "_", text).strip("_").lower()


def train_and_evaluate(detector: Detector, data: DetectionData) -> RunResult:
    """Fit on benign training rows, pick the threshold on validation, report on test."""
    started = time.perf_counter()
    detector.fit(data.train.features)
    fit_seconds = time.perf_counter() - started

    validation_scores = detector.score(data.validation.features)
    threshold = best_f1_threshold(data.validation.is_attack, validation_scores)
    started = time.perf_counter()
    test_scores = detector.score(data.test.features)
    score_seconds = time.perf_counter() - started
    return RunResult(
        name=detector.name,
        params=detector.params(),
        threshold=threshold,
        validation=evaluate(data.validation.is_attack, validation_scores, threshold),
        test=evaluate(data.test.is_attack, test_scores, threshold),
        test_recall_by_label=per_label_recall(
            data.test.labels, data.test.is_attack, test_scores, threshold
        ),
        fit_seconds=fit_seconds,
        score_seconds_per_1k=1000 * score_seconds / max(len(data.test), 1),
    )


def results_table(results: Sequence[RunResult]) -> str:
    """Markdown comparison of test metrics."""
    header = "| model | " + " | ".join(_METRIC_ORDER) + " | fit (s) |"
    lines = [header, "|---|" + "---|" * (len(_METRIC_ORDER) + 1)]
    for result in results:
        cells = " | ".join(f"{result.test[name]:.4f}" for name in _METRIC_ORDER)
        lines.append(f"| {result.name} | {cells} | {result.fit_seconds:.1f} |")
    labels = sorted({label for result in results for label in result.test_recall_by_label})
    lines += ["", "| attack label | " + " | ".join(r.name for r in results) + " |"]
    lines.append("|---|" + "---|" * len(results))
    for label in labels:
        cells = " | ".join(
            f"{r.test_recall_by_label.get(label, float('nan')):.3f}" for r in results
        )
        lines.append(f"| {label} | {cells} |")
    return "\n".join(lines) + "\n"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_model(
    detector: Detector, result: RunResult, feature_names: list[str], directory: Path
) -> None:
    """Write the model plus everything needed to use it: threshold, features, metrics."""
    if directory.exists():
        shutil.rmtree(directory)
    detector.save(directory)
    files = {path.name: _sha256(path) for path in sorted(directory.iterdir())}
    metadata = {
        "model": result.name,
        "threshold": result.threshold,
        "feature_names": feature_names,
        "params": result.params,
        "validation_metrics": result.validation,
        "test_metrics": result.test,
        "test_recall_by_label": result.test_recall_by_label,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "files_sha256": files,
    }
    (directory / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


@dataclass(frozen=True)
class LoadedModel:
    """A saved detector ready to score feature rows."""

    detector: Detector
    threshold: float
    feature_names: list[str]
    metadata: dict[str, Any]

    def is_anomalous(self, features: FloatArray) -> Any:
        """Boolean alert decision per row."""
        return self.detector.score(features) >= self.threshold


def load_model(directory: Path) -> LoadedModel:
    """Load a model saved by :func:`save_model`, refusing files that changed on disk.

    The digest check catches corruption and accidental replacement. It is not
    authentication: someone who can rewrite the model can rewrite the metadata.
    """
    metadata = json.loads((directory / METADATA_FILE).read_text(encoding="utf-8"))
    for name, expected in metadata["files_sha256"].items():
        if _sha256(directory / name) != expected:
            raise ValueError(f"model file {name} does not match its recorded checksum")
    detector = DETECTOR_TYPES[metadata["model"]].load(directory)
    return LoadedModel(
        detector=detector,
        threshold=float(metadata["threshold"]),
        feature_names=[str(name) for name in metadata["feature_names"]],
        metadata=metadata,
    )


def run_experiment(
    data: DetectionData,
    detectors: Sequence[Detector],
    *,
    out_dir: Path,
    tracking_uri: str,
    experiment: str = "sentinelai-detection",
) -> list[RunResult]:
    """Train every detector, log each run to MLflow, and keep the best in ``out_dir/best``.

    "Best" is the highest validation F1; test metrics are never used for selection.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    results: list[RunResult] = []
    best: tuple[Detector, RunResult] | None = None
    for detector in detectors:
        with mlflow.start_run(run_name=detector.name):
            result = train_and_evaluate(detector, data)
            mlflow.log_params({"model": detector.name, **result.params})
            mlflow.log_params(
                {
                    "train_rows": len(data.train),
                    "validation_rows": len(data.validation),
                    "test_rows": len(data.test),
                    "n_features": len(data.feature_names),
                }
            )
            mlflow.log_metric("threshold", result.threshold)
            mlflow.log_metric("fit_seconds", result.fit_seconds)
            mlflow.log_metric("score_seconds_per_1k", result.score_seconds_per_1k)
            mlflow.log_metrics({f"val_{k}": v for k, v in result.validation.items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in result.test.items()})
            mlflow.log_metrics(
                {f"recall_{_metric_key(k)}": v for k, v in result.test_recall_by_label.items()}
            )
            for step in getattr(detector, "history_", []):
                mlflow.log_metrics(
                    {"train_loss": step["train_loss"], "holdout_loss": step["holdout_loss"]},
                    step=int(step["epoch"]),
                )
            with tempfile.TemporaryDirectory() as scratch:
                save_model(detector, result, data.feature_names, Path(scratch) / detector.name)
                mlflow.log_artifacts(scratch)
        logger.info(
            "%s: test F1 %.4f, ROC-AUC %.4f", result.name, result.test["f1"], result.test["roc_auc"]
        )
        results.append(result)
        if best is None or result.validation["f1"] > best[1].validation["f1"]:
            best = (detector, result)

    if best is None:
        raise ValueError("no detectors to train")
    save_model(best[0], best[1], data.feature_names, out_dir / "best")
    (out_dir / RESULTS_FILE).write_text(results_table(results), encoding="utf-8")
    logger.info("best model: %s -> %s", best[1].name, out_dir / "best")
    return results


def main(argv: Sequence[str] | None = None) -> int:
    """Run the training CLI; returns a process exit code."""
    parser = argparse.ArgumentParser(prog="sentinel-detect", description=__doc__)
    parser.add_argument("--dataset", choices=("cicids2017", "synthetic"), default="cicids2017")
    parser.add_argument("--max-train-rows", type=int, default=300_000)
    parser.add_argument("--max-eval-rows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    settings = get_settings()
    data = load_detection_data(
        settings.processed_dir / "features" / args.dataset,
        max_train_rows=args.max_train_rows,
        max_eval_rows=args.max_eval_rows,
        seed=args.seed,
    )
    logger.info(
        "rows: train %d, validation %d, test %d",
        len(data.train),
        len(data.validation),
        len(data.test),
    )
    results = run_experiment(
        data,
        [
            IsolationForestDetector(seed=args.seed),
            AutoencoderDetector(epochs=args.epochs, seed=args.seed),
        ],
        out_dir=settings.models_dir / args.dataset,
        tracking_uri=settings.mlflow_tracking_uri,
    )
    print(results_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
