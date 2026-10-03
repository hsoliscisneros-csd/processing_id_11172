"""Threshold tuning on validation scores."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def optimal_threshold_f1(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Grid search threshold maximizing F1 on validation."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(y_true)) < 2:
        return 0.5, float(f1_score(y_true, (scores >= 0.5).astype(int), zero_division=0))
    qs = np.linspace(0.05, 0.95, 37)
    candidates = np.unique(np.quantile(scores, qs))
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        f1 = f1_score(y_true, (scores >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1
