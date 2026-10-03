"""Train/val/test splits without leakage."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalFold:
    fold_id: int
    train: pd.DataFrame
    val: pd.DataFrame


def temporal_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values("timestamp")
    n = len(ordered)
    i_train = int(n * train_ratio)
    i_val = int(n * (train_ratio + val_ratio))
    train = ordered.iloc[:i_train]
    val = ordered.iloc[i_train:i_val]
    test = ordered.iloc[i_val:]
    return train, val, test


def spatial_split(
    df: pd.DataFrame,
    *,
    train_municipalities: list[str],
    val_municipalities: list[str] | None = None,
    state: str | None = "Chiapas",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Hold out entire municipalities for test (no BSSID leakage across regions)."""
    work = df.copy()
    if state and "state" in work.columns:
        work = work[work["state"] == state]
    val_municipalities = val_municipalities or []
    train_set = set(train_municipalities)
    val_set = set(val_municipalities)
    train = work[work["municipality"].isin(train_set)]
    val = work[work["municipality"].isin(val_set)]
    held_out = train_set | val_set
    test = work[~work["municipality"].isin(held_out)]
    return train, val, test


def spatial_loo_folds(
    df: pd.DataFrame,
    *,
    train_municipalities: list[str],
    state: str | None = "Chiapas",
    min_val_obs: int = 200,
) -> list[tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Leave-one-municipality-out: each train municipality becomes val once."""
    work = df.copy()
    if state and "state" in work.columns:
        work = work[work["state"] == state]
    folds: list[tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    held_out = set(train_municipalities)
    test = work[~work["municipality"].isin(held_out)]

    for val_muni in train_municipalities:
        val_df = work[work["municipality"] == val_muni]
        if len(val_df) < min_val_obs:
            continue
        train_munis = [m for m in train_municipalities if m != val_muni]
        train_df = work[work["municipality"].isin(train_munis)]
        folds.append((val_muni, train_df, val_df, test))
    return folds


def geo_grid_split(
    df: pd.DataFrame,
    *,
    train_grids: list[str],
    val_grids: list[str],
    test_grids: list[str],
    state: str | None = "Chiapas",
    lat_median: float | None = None,
    lon_median: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """OOD por cuadrante: cada BSSID va a un solo grid (mediana lat/lon)."""
    work = df.copy()
    if state and "state" in work.columns:
        work = work[work["state"] == state]
    lat_m = float(work["latitude"].median()) if lat_median is None else lat_median
    lon_m = float(work["longitude"].median()) if lon_median is None else lon_median
    loc = (
        work.groupby("bssid", as_index=False)
        .agg(latitude=("latitude", "median"), longitude=("longitude", "median"))
    )
    loc["grid_id"] = (
        (loc["latitude"] >= lat_m).astype(int).astype(str)
        + "_"
        + (loc["longitude"] >= lon_m).astype(int).astype(str)
    )
    work = work.drop(columns=[c for c in work.columns if c == "grid_id"], errors="ignore")
    work = work.merge(loc[["bssid", "grid_id"]], on="bssid", how="left")
    train = work[work["grid_id"].isin(train_grids)]
    val = work[work["grid_id"].isin(val_grids)]
    test = work[work["grid_id"].isin(test_grids)]
    return train, val, test


def temporal_cv_splits(
    df: pd.DataFrame,
    n_folds: int = 5,
    min_train_ratio: float = 0.50,
) -> list[TemporalFold]:
    """Walk-forward CV: train crece, val es ventana siguiente (sin mezclar tiempos)."""
    ordered = df.sort_values("timestamp").reset_index(drop=True)
    n = len(ordered)
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")

    min_train = max(int(n * min_train_ratio), 1)
    val_window = max((n - min_train) // n_folds, 1)
    folds: list[TemporalFold] = []

    for k in range(n_folds):
        train_end = min_train + k * val_window
        val_end = min(train_end + val_window, n)
        if train_end >= n or val_end <= train_end:
            break
        folds.append(
            TemporalFold(
                fold_id=k,
                train=ordered.iloc[:train_end],
                val=ordered.iloc[train_end:val_end],
            )
        )
    return folds
