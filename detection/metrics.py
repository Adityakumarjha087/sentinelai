"""Detection metrics. Scores are always "higher = more anomalous"."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)

Scores = npt.NDArray[np.floating]
Truth = npt.NDArray[np.bool_]


def best_f1_threshold(is_attack: Truth, scores: Scores) -> float:
    """The score threshold that maximizes F1 (alert when ``score >= threshold``)."""
    precision, recall, thresholds = precision_recall_curve(is_attack, scores)
    # The final precision/recall point has no threshold; drop it.
    precision, recall = precision[:-1], recall[:-1]
    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall, denominator, out=np.zeros_like(denominator), where=denominator > 0
    )
    return float(thresholds[int(np.argmax(f1))])


def evaluate(is_attack: Truth, scores: Scores, threshold: float) -> dict[str, float]:
    """Precision, recall, F1 and false-positive rate at ``threshold``, plus ROC/PR AUC."""
    predicted = scores >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        is_attack, predicted, average="binary", zero_division=0
    )
    benign = ~is_attack
    false_positive_rate = float(predicted[benign].mean()) if benign.any() else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": false_positive_rate,
        "roc_auc": float(roc_auc_score(is_attack, scores)),
        "pr_auc": float(average_precision_score(is_attack, scores)),
    }


def per_label_recall(
    labels: npt.NDArray[np.object_], is_attack: Truth, scores: Scores, threshold: float
) -> dict[str, float]:
    """Share of each attack label's flows that raise an alert at ``threshold``."""
    predicted = scores >= threshold
    return {
        str(label): float(predicted[labels == label].mean())
        for label in sorted({str(value) for value in labels[is_attack]})
    }
