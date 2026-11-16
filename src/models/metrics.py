"""Small, dependency-free classification metrics for model reports.

Kept independent of scikit-learn so evaluation works even when only the
inference extra is installed and so metric definitions are easy to test.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and sorted_values[end] == sorted_values[index]:
            end += 1
        ranks[order[index:end]] = (index + 1 + end) / 2.0
        index = end
    return ranks


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """Rank-based ROC AUC with tie correction; ``None`` for a single class."""
    truth = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    positives = int(truth.sum())
    negatives = len(truth) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = _average_ranks(scores)
    auc = (ranks[truth == 1].sum() - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )
    return float(auc)


def classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Direction metrics with the majority-class baseline reported alongside."""
    truth = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    if truth.shape != scores.shape or truth.ndim != 1 or len(truth) == 0:
        raise ValueError("y_true and y_score must be non-empty 1-D arrays of equal length")
    if not np.isfinite(scores).all():
        raise ValueError("y_score contains non-finite values")

    predictions = (scores >= threshold).astype(int)
    tp = int(((predictions == 1) & (truth == 1)).sum())
    tn = int(((predictions == 0) & (truth == 0)).sum())
    fp = int(((predictions == 1) & (truth == 0)).sum())
    fn = int(((predictions == 0) & (truth == 1)).sum())

    accuracy = float((tp + tn) / len(truth))
    positive_rate = float(truth.mean())
    majority = max(positive_rate, 1.0 - positive_rate)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "n": int(len(truth)),
        "accuracy": accuracy,
        "majority_baseline_accuracy": float(majority),
        "roc_auc": roc_auc(truth, scores),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "positive_rate": positive_rate,
        "threshold": float(threshold),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }
