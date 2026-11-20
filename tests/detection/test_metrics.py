from __future__ import annotations

import numpy as np
import pytest

from detection.metrics import best_f1_threshold, evaluate, per_label_recall

TRUTH = np.array([False, False, False, False, True, True, True, True])
SCORES = np.array([0.1, 0.2, 0.3, 0.8, 0.4, 0.7, 0.9, 1.0])
LABELS = np.array(["BENIGN"] * 4 + ["PortScan", "PortScan", "DDoS", "DDoS"], dtype=object)


def test_best_threshold_maximizes_f1() -> None:
    # Alerting from 0.4 catches all 4 attacks with 1 false positive: F1 = 8/9.
    assert best_f1_threshold(TRUTH, SCORES) == pytest.approx(0.4)


def test_evaluate_at_a_threshold() -> None:
    metrics = evaluate(TRUTH, SCORES, threshold=0.4)
    assert metrics["precision"] == pytest.approx(0.8)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(8 / 9)
    assert metrics["false_positive_rate"] == pytest.approx(0.25)
    assert metrics["roc_auc"] == pytest.approx(14 / 16)  # 14 of 16 attack/benign pairs ranked right
    assert 0.0 < metrics["pr_auc"] <= 1.0


def test_threshold_is_inclusive_and_degenerate_cases_do_not_crash() -> None:
    assert evaluate(TRUTH, SCORES, threshold=1.0)["recall"] == pytest.approx(0.25)
    nothing = evaluate(TRUTH, SCORES, threshold=5.0)
    assert (nothing["precision"], nothing["recall"], nothing["f1"]) == (0.0, 0.0, 0.0)


def test_per_label_recall_covers_attack_labels_only() -> None:
    recall = per_label_recall(LABELS, TRUTH, SCORES, threshold=0.7)
    assert recall == {"DDoS": 1.0, "PortScan": 0.5}
