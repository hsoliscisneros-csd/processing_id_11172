"""Statistical significance tests for model comparison."""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest, chi2


def mcnemar_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    *,
    correction: bool = True,
) -> dict:
    """McNemar test for paired classifiers on the same instances."""
    y_true = np.asarray(y_true, dtype=int)
    pred_a = np.asarray(pred_a, dtype=int)
    pred_b = np.asarray(pred_b, dtype=int)
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    b = int((~correct_a & correct_b).sum())
    c = int((correct_a & ~correct_b).sum())
    n = b + c
    if n == 0:
        return {
            "b": b,
            "c": c,
            "n_discordant": 0,
            "statistic": 0.0,
            "p_value": 1.0,
            "method": "none",
        }
    if n < 25:
        p_value = float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)
        method = "exact_binomial"
        statistic = float(abs(b - c))
    else:
        if correction:
            statistic = float((abs(b - c) - 1) ** 2 / n)
        else:
            statistic = float((b - c) ** 2 / n)
        p_value = float(1.0 - chi2.cdf(statistic, df=1))
        method = "chi2_cc" if correction else "chi2"
    return {
        "b": b,
        "c": c,
        "n_discordant": n,
        "statistic": statistic,
        "p_value": p_value,
        "method": method,
    }
