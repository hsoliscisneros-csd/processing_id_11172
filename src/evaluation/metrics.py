"""Classification metrics for EXP-04+."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> dict:
    out: dict = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
        out["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None
        out["mcc"] = None
    return out
