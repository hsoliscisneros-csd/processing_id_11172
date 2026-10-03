"""EXP-02: Feature engineering (stage_1)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h3
import networkx as nx
import numpy as np
import pandas as pd

from src.data.wifi_utils import estimate_rssi, path_loss_params

H3_RES = 9
GRID_DEG = 0.0005  # ~55 m en latitud media México
SECURITY_COLS = ["sec_wpa3", "sec_wpa2", "sec_wpa", "sec_wep", "sec_open"]
SCAN_KEY = ["device_id", "timestamp"]


def band_label(frequency: int) -> str:
    if frequency > 5900:
        return "6ghz"
    if frequency > 5000:
        return "5ghz"
    return "24ghz"


def add_temporal(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"], unit="ms", utc=True).dt.tz_convert(
        "America/Mexico_City"
    )
    out["hour"] = ts.dt.hour
    out["day_of_week"] = ts.dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    return out


def add_radio_security(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["band"] = out["frequency"].astype(int).apply(band_label)
    out["is_wep"] = (out["security_type"].str.upper() == "WEP").astype(int)
    out["is_open"] = (out["security_type"].str.upper() == "OPEN").astype(int)

    sec = out["security_type"].fillna("OPEN").str.upper()
    for col in SECURITY_COLS:
        out[col] = 0
    out.loc[sec == "WPA3", "sec_wpa3"] = 1
    out.loc[sec == "WPA2", "sec_wpa2"] = 1
    out.loc[sec == "WPA", "sec_wpa"] = 1
    out.loc[sec == "WEP", "sec_wep"] = 1
    out.loc[sec == "OPEN", "sec_open"] = 1
    return out


def add_physics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rssi_residual"] = out.apply(
        lambda r: r["level"] - estimate_rssi(float(r["estimated_distance"]), int(r["frequency"]))
        if pd.notna(r["estimated_distance"]) and pd.notna(r["level"])
        else np.nan,
        axis=1,
    )
    return out


def add_spatial(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    has_geo = out["latitude"].notna() & out["longitude"].notna()
    out["h3_index"] = None
    out.loc[has_geo, "h3_index"] = out.loc[has_geo].apply(
        lambda r: h3.latlng_to_cell(r["latitude"], r["longitude"], H3_RES), axis=1
    )
    out["lat_grid"] = (out["latitude"] / GRID_DEG).round() * GRID_DEG
    out["lon_grid"] = (out["longitude"] / GRID_DEG).round() * GRID_DEG

    muni_counts = out["municipality"].fillna("UNKNOWN").value_counts()
    out["municipality_freq"] = out["municipality"].fillna("UNKNOWN").map(muni_counts)
    return out


def add_scan_context(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    scan_id = out.groupby(SCAN_KEY, sort=False).ngroup()
    out["_scan_id"] = scan_id

    scan_stats = out.groupby("_scan_id").agg(
        n_aps_scan=("bssid", "nunique"),
        pct_open_scan=("is_open", "mean"),
        n_vendors_scan=("vendor", "nunique"),
    )
    out = out.join(scan_stats, on="_scan_id")

    grid = out.dropna(subset=["lat_grid", "lon_grid"]).copy()
    if len(grid):
        grid_stats = (
            grid.groupby(["lat_grid", "lon_grid"])
            .agg(
                n_aps_50m=("bssid", "nunique"),
                n_vendors_50m=("vendor", "nunique"),
                pct_open_50m=("is_open", "mean"),
            )
            .reset_index()
        )
        out = out.merge(grid_stats, on=["lat_grid", "lon_grid"], how="left")
    else:
        out["n_aps_50m"] = np.nan
        out["n_vendors_50m"] = np.nan
        out["pct_open_50m"] = np.nan

    # excluir el AP actual del conteo local
    out["n_aps_50m"] = (out["n_aps_50m"].fillna(1) - 1).clip(lower=0)
    out.drop(columns=["_scan_id"], inplace=True)
    return out


def add_graph_features(df: pd.DataFrame, sample_scans: int = 8000) -> pd.DataFrame:
    """AP degree y PageRank por co-observación en el mismo escaneo."""
    out = df.copy()
    scans = out[SCAN_KEY].drop_duplicates()
    if len(scans) > sample_scans:
        scans = scans.sample(n=sample_scans, random_state=42)
    sub = out.merge(scans, on=SCAN_KEY, how="inner")

    g = nx.Graph()
    for _, grp in sub.groupby(SCAN_KEY, sort=False):
        bssids = grp["bssid"].unique().tolist()
        for i, a in enumerate(bssids):
            g.add_node(a)
            for b in bssids[i + 1 :]:
                g.add_edge(a, b)

    if g.number_of_nodes() == 0:
        out["ap_degree"] = 0
        out["ap_pagerank"] = 0.0
        return out

    degree = dict(g.degree())
    try:
        pr = nx.pagerank(g, max_iter=50)
    except nx.PowerIterationFailedConvergence:
        pr = {n: 0.0 for n in g.nodes()}

    out["ap_degree"] = out["bssid"].map(degree).fillna(0).astype(int)
    out["ap_pagerank"] = out["bssid"].map(pr).fillna(0.0)
    return out


FEATURE_COLS = [
    "level",
    "frequency",
    "channel",
    "bandwidth",
    "estimated_distance",
    "distance_residual",
    "rssi_residual",
    "hour",
    "day_of_week",
    "is_weekend",
    "is_wep",
    "is_open",
    "is_randomized_mac",
    "n_aps_scan",
    "pct_open_scan",
    "n_vendors_scan",
    "n_aps_50m",
    "n_vendors_50m",
    "pct_open_50m",
    "municipality_freq",
    "ap_degree",
    "ap_pagerank",
    *SECURITY_COLS,
]

ID_COLS = ["id", "bssid", "device_id", "timestamp", "municipality", "h3_index"]
LABEL_COLS = ["is_anomaly_heuristic", "threat_type_heuristic"]


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    work = df.copy()
    work = add_temporal(work)
    work = add_radio_security(work)
    work = add_physics(work)
    work = add_spatial(work)
    work = add_scan_context(work)
    work = add_graph_features(work)

    # estándar 802.11 como categoría ordinal simple
    std_map = {s: i for i, s in enumerate(work["standard"].fillna("unknown").unique())}
    work["standard_code"] = work["standard"].fillna("unknown").map(std_map)
    band_map = {"24ghz": 0, "5ghz": 1, "6ghz": 2}
    work["band_code"] = work["band"].map(band_map).fillna(0).astype(int)

    feature_cols = FEATURE_COLS + ["standard_code", "band_code"]
    for col in feature_cols:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    meta = {
        "rows": len(work),
        "feature_columns": feature_cols,
        "nan_rates": {c: float(work[c].isna().mean()) for c in feature_cols if c in work.columns},
        "features_version": "features_v1.0",
    }
    out_cols = [c for c in ID_COLS if c in work.columns] + feature_cols + LABEL_COLS
    out_cols = [c for c in out_cols if c in work.columns]
    return work[out_cols], meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Build WiFi features (EXP-02)")
    parser.add_argument(
        "--input", type=Path, default=Path("data/datos_wifi_curated_anon.parquet")
    )
    parser.add_argument("--output", type=Path, default=Path("data/features/features.parquet"))
    parser.add_argument("--manifest", type=Path, default=Path("data/features/manifest.json"))
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    features, meta = build_features(df)
    meta["input"] = str(args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.output, index=False)
    args.manifest.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Wrote {args.output} ({len(features)} rows, {len(meta['feature_columns'])} features)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
