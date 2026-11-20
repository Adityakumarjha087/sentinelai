"""Assemble train / validation / test splits from the Phase 1 feature tables.

Protocol (semi-supervised anomaly detection):

* **train** - benign flows from the baseline capture (Monday, attack-free).
  Models never see an attack or a label during fitting.
* **validation / test** - all remaining captures, split 50/50 and stratified by
  attack label. Validation is used only to choose the alert threshold and to
  pick the better model; every reported number comes from the test half.

Caveat: the validation/test split is random, so flows from the same attack burst
land in both halves. That is acceptable here because nothing is *trained* on
validation data, but it would be leakage for a supervised classifier.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class Split:
    """Feature matrix with ground truth, row-aligned."""

    features: FloatArray
    is_attack: BoolArray
    labels: npt.NDArray[np.object_]

    def __len__(self) -> int:
        return int(self.features.shape[0])


@dataclass(frozen=True)
class DetectionData:
    """The three splits plus the feature order the models were trained with."""

    train: Split
    validation: Split
    test: Split
    feature_names: list[str]


def _to_split(frame: pd.DataFrame, feature_names: list[str]) -> Split:
    return Split(
        features=np.ascontiguousarray(frame[feature_names].to_numpy(dtype=np.float32)),
        is_attack=frame["is_attack"].to_numpy(dtype=bool),
        labels=frame["label"].to_numpy(dtype=object),
    )


def _sample(frame: pd.DataFrame, max_rows: int | None, rng: np.random.Generator) -> pd.DataFrame:
    """Subsample to at most ``max_rows`` keeping each label's share (at least one row each)."""
    if max_rows is None or len(frame) <= max_rows:
        return frame
    fraction = max_rows / len(frame)
    picked: list[npt.NDArray[np.int64]] = []
    for positions in frame.groupby("label", sort=True).indices.values():
        keep = max(1, round(len(positions) * fraction))
        picked.append(rng.choice(positions, size=keep, replace=False))
    return frame.iloc[np.sort(np.concatenate(picked))]


def _stratified_halves(frame: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    first: list[npt.NDArray[np.int64]] = []
    second: list[npt.NDArray[np.int64]] = []
    for positions in frame.groupby("label", sort=True).indices.values():
        shuffled = rng.permutation(positions)
        middle = (len(shuffled) + 1) // 2
        first.append(shuffled[:middle])
        second.append(shuffled[middle:])
    return (
        frame.iloc[np.sort(np.concatenate(first))],
        frame.iloc[np.sort(np.concatenate(second))],
    )


def load_detection_data(
    features_dir: Path,
    *,
    baseline: str = "monday",
    max_train_rows: int | None = 300_000,
    max_eval_rows: int | None = None,
    seed: int = 42,
) -> DetectionData:
    """Load feature tables written by ``sentinel-data features`` and split them.

    Args:
        features_dir: Directory with ``*.parquet`` tables and ``featurizer.json``.
        baseline: File-name fragment identifying the attack-free capture. If no
            file matches (e.g. the single-file synthetic set), half of all benign
            flows are used for training instead.
        max_train_rows: Cap on training rows (random subsample).
        max_eval_rows: Cap on validation+test rows (label-stratified subsample).
        seed: Seed for every random choice, so splits are reproducible.
    """
    state = json.loads((features_dir / "featurizer.json").read_text(encoding="utf-8"))
    feature_names = [str(name) for name in state["feature_names"]]
    files = sorted(features_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no feature tables in {features_dir}")
    rng = np.random.default_rng(seed)
    columns = ["label", "is_attack", *feature_names]

    baseline_files = [file for file in files if baseline.lower() in file.name.lower()]
    if baseline_files:
        train = pd.concat([pd.read_parquet(f, columns=columns) for f in baseline_files])
        train = train[~train["is_attack"].astype(bool)]
        other = [f for f in files if f not in baseline_files]
        if not other:
            raise ValueError("only baseline files found; nothing to evaluate on")
        evaluation = pd.concat([pd.read_parquet(f, columns=columns) for f in other])
    else:
        everything = pd.concat([pd.read_parquet(f, columns=columns) for f in files])
        benign = np.flatnonzero(~everything["is_attack"].to_numpy(dtype=bool))
        train_positions = rng.choice(benign, size=len(benign) // 2, replace=False)
        mask = np.zeros(len(everything), dtype=bool)
        mask[train_positions] = True
        train, evaluation = everything[mask], everything[~mask]

    train = train.reset_index(drop=True)
    evaluation = evaluation.reset_index(drop=True)
    if max_train_rows is not None and len(train) > max_train_rows:
        train = train.iloc[np.sort(rng.choice(len(train), size=max_train_rows, replace=False))]
    evaluation = _sample(evaluation, max_eval_rows, rng).reset_index(drop=True)
    validation, test = _stratified_halves(evaluation, rng)
    return DetectionData(
        train=_to_split(train, feature_names),
        validation=_to_split(validation, feature_names),
        test=_to_split(test, feature_names),
        feature_names=feature_names,
    )
