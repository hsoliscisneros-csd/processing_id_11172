"""EXP-04: Baseline ML with temporal CV and tuned thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import binary_metrics
from src.evaluation.splits import temporal_cv_splits, temporal_split
from src.evaluation.thresholds import optimal_threshold_f1
from src.features.build import FEATURE_COLS

SEED = 42
LABEL = "is_anomaly_heuristic"


def get_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[cols].fillna(0).values
    y = df[LABEL].astype(int).values
    return X, y


def fit_rf(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    ).fit(X_train, y_train)


def fit_xgb(X_train: np.ndarray, y_train: np.ndarray):
    import xgboost as xgb

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        random_state=SEED,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    return model


def supervised_scores(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    raw = model.decision_function(X)
    return (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)


def eval_supervised(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
) -> dict:
    scores_val = supervised_scores(model, X_val)
    threshold, val_f1 = optimal_threshold_f1(y_val, scores_val)
    scores_test = supervised_scores(model, X_test)
    preds = (scores_test >= threshold).astype(int)
    out = binary_metrics(y_test, scores_test, preds)
    out["threshold"] = threshold
    out["val_f1_at_threshold"] = val_f1
    return out


def eval_unsupervised(model, X_train, X_val, y_val, X_test, y_test) -> dict:
    model.fit(X_train)
    if hasattr(model, "decision_function"):
        raw_val = -model.decision_function(X_val)
        raw_test = -model.decision_function(X_test)
    else:
        raw_val = -model.score_samples(X_val)
        raw_test = -model.score_samples(X_test)
    scores_val = (raw_val - raw_val.min()) / (raw_val.max() - raw_val.min() + 1e-9)
    threshold, _ = optimal_threshold_f1(y_val, scores_val)
    scores_test = (raw_test - raw_test.min()) / (raw_test.max() - raw_test.min() + 1e-9)
    preds = (scores_test >= threshold).astype(int)
    out = binary_metrics(y_test, scores_test, preds)
    out["threshold"] = threshold
    return out


def run_temporal_cv(df: pd.DataFrame, n_folds: int = 5) -> dict:
    fold_metrics: dict[str, list[dict]] = {
        "random_forest": [],
        "xgboost": [],
        "isolation_forest": [],
        "lof": [],
    }
    for fold in temporal_cv_splits(df, n_folds=n_folds):
        train, val = fold.train, fold.val
        X_train, y_train = get_xy(train)
        X_val, y_val = get_xy(val)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        rf = fit_rf(X_train_s, y_train)
        m = eval_supervised(rf, X_train_s, y_train, X_val_s, y_val, X_val_s, y_val)
        m["fold"] = fold.fold_id
        fold_metrics["random_forest"].append(m)

        try:
            xgb_m = fit_xgb(X_train_s, y_train)
            m = eval_supervised(xgb_m, X_train_s, y_train, X_val_s, y_val, X_val_s, y_val)
            m["fold"] = fold.fold_id
            fold_metrics["xgboost"].append(m)
        except ImportError:
            pass

        iso = IsolationForest(
            n_estimators=200, contamination=float(y_train.mean()), random_state=SEED
        )
        m = eval_unsupervised(iso, X_train_s, X_val_s, y_val, X_val_s, y_val)
        m["fold"] = fold.fold_id
        fold_metrics["isolation_forest"].append(m)

        lof = LocalOutlierFactor(
            n_neighbors=20, novelty=True, contamination=float(y_train.mean())
        )
        m = eval_unsupervised(lof, X_train_s, X_val_s, y_val, X_val_s, y_val)
        m["fold"] = fold.fold_id
        fold_metrics["lof"].append(m)

    summary = {}
    for name, runs in fold_metrics.items():
        if not runs:
            continue
        summary[name] = {
            "f1_mean": float(np.mean([r["f1"] for r in runs])),
            "f1_std": float(np.std([r["f1"] for r in runs])),
            "pr_auc_mean": float(np.mean([r["pr_auc"] for r in runs if r["pr_auc"] is not None])),
            "mcc_mean": float(np.mean([r["mcc"] for r in runs if r["mcc"] is not None])),
            "threshold_mean": float(np.mean([r["threshold"] for r in runs])),
            "n_folds": len(runs),
        }
    return {"folds": fold_metrics, "summary": summary}


def save_pr_curve(y_test, scores, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_test, scores, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Train baselines (EXP-04)")
    parser.add_argument("--input", type=Path, default=Path("data/features/features.parquet"))
    parser.add_argument("--output", type=Path, default=Path("experiments/outputs/baselines"))
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    train, val, test = temporal_split(df)
    X_train, y_train = get_xy(train)
    X_val, y_val = get_xy(val)
    X_test, y_test = get_xy(test)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    cv_results = run_temporal_cv(df, n_folds=args.cv_folds)

    results: dict = {
        "split": {"train": len(train), "val": len(val), "test": len(test)},
        "label": LABEL,
        "temporal_cv": cv_results["summary"],
        "models": {},
    }

    args.output.mkdir(parents=True, exist_ok=True)
    fig_dir = Path("reports/figures/exp04")
    fig_dir.mkdir(parents=True, exist_ok=True)

    rf = fit_rf(X_train_s, y_train)
    results["models"]["random_forest"] = eval_supervised(
        rf, X_train_s, y_train, X_val_s, y_val, X_test_s, y_test
    )
    rf_scores = supervised_scores(rf, X_test_s)
    rf_t = results["models"]["random_forest"]["threshold"]
    save_pr_curve(
        y_test,
        rf_scores,
        fig_dir / "pr_random_forest.png",
        f"Random Forest (τ={rf_t:.3f})",
    )

    try:
        xgb_model = fit_xgb(X_train_s, y_train)
        results["models"]["xgboost"] = eval_supervised(
            xgb_model, X_train_s, y_train, X_val_s, y_val, X_test_s, y_test
        )
        xgb_scores = supervised_scores(xgb_model, X_test_s)
        xgb_t = results["models"]["xgboost"]["threshold"]
        save_pr_curve(
            y_test,
            xgb_scores,
            fig_dir / "pr_xgboost.png",
            f"XGBoost (τ={xgb_t:.3f})",
        )
    except ImportError:
        results["models"]["xgboost"] = {"error": "xgboost not installed"}

    iso = IsolationForest(n_estimators=200, contamination=float(y_train.mean()), random_state=SEED)
    results["models"]["isolation_forest"] = eval_unsupervised(
        iso, X_train_s, X_val_s, y_val, X_test_s, y_test
    )

    lof = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=float(y_train.mean()))
    results["models"]["lof"] = eval_unsupervised(lof, X_train_s, X_val_s, y_val, X_test_s, y_test)

    best_name = max(
        (k for k in results["models"] if "f1" in results["models"][k]),
        key=lambda k: results["models"][k]["f1"],
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    if best_name == "xgboost" and "error" not in results["models"]["xgboost"]:
        t = results["models"]["xgboost"]["threshold"]
        preds = (xgb_scores >= t).astype(int)
    else:
        t = results["models"]["random_forest"]["threshold"]
        preds = (rf_scores >= t).astype(int)
    ConfusionMatrixDisplay.from_predictions(y_test, preds, ax=ax)
    ax.set_title(f"Confusion — {best_name} (τ={t:.3f})")
    plt.tight_layout()
    plt.savefig(fig_dir / "confusion_matrix.png", bbox_inches="tight")
    plt.close()

    out_file = args.output / "metrics.json"
    out_file.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (args.output / "cv_folds.json").write_text(
        json.dumps(cv_results["folds"], indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2))
    print(f"Saved {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
