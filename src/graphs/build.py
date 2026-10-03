"""EXP-03: Knowledge graph construction — physical, spatial, temporal."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h3
import networkx as nx
import numpy as np
import pandas as pd

SCAN_KEY = ["device_id", "timestamp"]
DEFAULT_NEARBY_M = 30.0
DEFAULT_H3_RES = 9
DEFAULT_MIN_CO_OBS = 2


def _safe_mode(series: pd.Series, default: str = "") -> str:
    s = series.dropna()
    if s.empty:
        return default
    m = s.mode()
    return str(m.iloc[0]) if len(m) else default


def _ap_node_attrs(grp: pd.DataFrame) -> dict:
    return {
        "ssid_mode": _safe_mode(grp["ssid"]),
        "vendor_mode": _safe_mode(grp["vendor"]),
        "security_mode": _safe_mode(grp["security_type"]),
        "level_mean": float(grp["level"].mean()),
        "frequency_mode": int(grp["frequency"].mode().iloc[0]) if grp["frequency"].notna().any() else 0,
        "obs_count": int(len(grp)),
        "is_anomaly_rate": float(grp["is_anomaly_heuristic"].mean())
        if "is_anomaly_heuristic" in grp.columns
        else 0.0,
    }


def build_physical_graph(
    df: pd.DataFrame,
    nearby_threshold_m: float = DEFAULT_NEARBY_M,
    min_co_observations: int = DEFAULT_MIN_CO_OBS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """AP nodes + NEARBY edges from co-scans within distance threshold."""
    edge_acc: dict[tuple[str, str], list[float]] = defaultdict(list)

    for _, scan in df.groupby(SCAN_KEY, sort=False):
        sub = scan.dropna(subset=["bssid", "estimated_distance"])
        if len(sub) < 2:
            continue
        records = sub[["bssid", "estimated_distance", "level"]].to_dict("records")
        for i, a in enumerate(records):
            for b in records[i + 1 :]:
                d_avg = (a["estimated_distance"] + b["estimated_distance"]) / 2
                if d_avg > nearby_threshold_m:
                    continue
                key = tuple(sorted((a["bssid"], b["bssid"])))
                weight = 1.0 / (1.0 + d_avg)
                edge_acc[key].append(weight)

    ap_nodes = []
    for bssid, grp in df.groupby("bssid"):
        attrs = _ap_node_attrs(grp)
        ap_nodes.append({"node_id": bssid, "node_type": "AccessPoint", **attrs})

    edges = []
    for (src, dst), weights in edge_acc.items():
        if len(weights) < min_co_observations:
            continue
        edges.append(
            {
                "src": src,
                "dst": dst,
                "relation": "NEARBY",
                "weight": float(np.mean(weights)),
                "co_obs_count": len(weights),
            }
        )

    return pd.DataFrame(ap_nodes), pd.DataFrame(edges)


def build_spatial_graph(
    df: pd.DataFrame,
    h3_resolution: int = DEFAULT_H3_RES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """H3 location nodes + ADJACENT edges between grid neighbors."""
    geo = df.dropna(subset=["latitude", "longitude"]).copy()
    geo["h3_index"] = geo.apply(
        lambda r: h3.latlng_to_cell(r["latitude"], r["longitude"], h3_resolution),
        axis=1,
    )

    loc_nodes = []
    for h3_idx, grp in geo.groupby("h3_index"):
        lat, lon = h3.cell_to_latlng(h3_idx)
        loc_nodes.append(
            {
                "node_id": h3_idx,
                "node_type": "Location",
                "lat": lat,
                "lon": lon,
                "ap_count": int(grp["bssid"].nunique()),
                "obs_count": int(len(grp)),
                "municipality_mode": _safe_mode(grp["municipality"]),
            }
        )

    nodes_df = pd.DataFrame(loc_nodes)
    h3_set = set(nodes_df["node_id"])
    edge_set: set[tuple[str, str]] = set()

    for h3_idx in h3_set:
        for neighbor in h3.grid_disk(h3_idx, 1):
            if neighbor == h3_idx or neighbor not in h3_set:
                continue
            key = tuple(sorted((h3_idx, neighbor)))
            if key in edge_set:
                continue
            edge_set.add(key)

    edges = []
    for a, b in edge_set:
        edges.append({"src": a, "dst": b, "relation": "ADJACENT", "weight": 1.0})

    return nodes_df, pd.DataFrame(edges)


def build_temporal_graph(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ScanEvent nodes + TEMPORAL_NEXT edges per auditor."""
    scan_nodes = []
    edges = []

    for (device_id, ts), grp in df.groupby(SCAN_KEY, sort=False):
        scan_id = f"{device_id}_{ts}"
        lat = grp["latitude"].mean() if grp["latitude"].notna().any() else None
        lon = grp["longitude"].mean() if grp["longitude"].notna().any() else None
        scan_nodes.append(
            {
                "node_id": scan_id,
                "node_type": "ScanEvent",
                "device_id": device_id,
                "timestamp": int(ts),
                "n_aps": int(grp["bssid"].nunique()),
                "lat": lat,
                "lon": lon,
            }
        )

    nodes_df = pd.DataFrame(scan_nodes)

    for device_id, dev_grp in nodes_df.groupby("device_id", sort=False):
        ordered = dev_grp.sort_values("timestamp")
        ids = ordered["node_id"].tolist()
        times = ordered["timestamp"].tolist()
        for i in range(len(ids) - 1):
            dt_ms = times[i + 1] - times[i]
            edges.append(
                {
                    "src": ids[i],
                    "dst": ids[i + 1],
                    "relation": "TEMPORAL_NEXT",
                    "weight": float(dt_ms / 1000.0),
                    "delta_ms": int(dt_ms),
                }
            )

    return nodes_df, pd.DataFrame(edges)


def graph_stats(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict:
    g = nx.Graph()
    for nid in nodes["node_id"]:
        g.add_node(nid)
    for _, e in edges.iterrows():
        g.add_edge(e["src"], e["dst"], weight=e.get("weight", 1.0))
    degrees = [d for _, d in g.degree()] if g.number_of_nodes() else [0]
    return {
        "n_nodes": int(g.number_of_nodes()),
        "n_edges": int(g.number_of_edges()),
        "n_components": int(nx.number_connected_components(g)) if g.number_of_nodes() else 0,
        "avg_degree": float(np.mean(degrees)),
        "max_degree": int(max(degrees)) if degrees else 0,
    }


def save_graph(
    out_dir: Path,
    name: str,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
) -> dict:
    path = out_dir / name
    path.mkdir(parents=True, exist_ok=True)
    nodes.to_parquet(path / "nodes.parquet", index=False)
    edges.to_parquet(path / "edges.parquet", index=False)
    stats = graph_stats(nodes, edges)
    stats["name"] = name
    return stats


def build_all(
    df: pd.DataFrame,
    nearby_threshold_m: float = DEFAULT_NEARBY_M,
    h3_resolution: int = DEFAULT_H3_RES,
    min_co_observations: int = DEFAULT_MIN_CO_OBS,
) -> dict:
    phys_n, phys_e = build_physical_graph(df, nearby_threshold_m, min_co_observations)
    spat_n, spat_e = build_spatial_graph(df, h3_resolution)
    temp_n, temp_e = build_temporal_graph(df)
    return {
        "physical": (phys_n, phys_e),
        "spatial": (spat_n, spat_e),
        "temporal": (temp_n, temp_e),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build WiFi knowledge graphs (EXP-03)")
    parser.add_argument(
        "--input", type=Path, default=Path("data/datos_wifi_curated_anon.parquet")
    )
    parser.add_argument("--output", type=Path, default=Path("data/graphs"))
    parser.add_argument("--nearby-m", type=float, default=DEFAULT_NEARBY_M)
    parser.add_argument("--h3-res", type=int, default=DEFAULT_H3_RES)
    parser.add_argument("--min-co-obs", type=int, default=DEFAULT_MIN_CO_OBS)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    graphs = build_all(df, args.nearby_m, args.h3_res, args.min_co_obs)

    manifest = {
        "graphs_version": "graphs_v1.0",
        "input": str(args.input),
        "params": {
            "nearby_threshold_m": args.nearby_m,
            "h3_resolution": args.h3_res,
            "min_co_observations": args.min_co_obs,
        },
        "graphs": {},
    }

    for name, (nodes, edges) in graphs.items():
        manifest["graphs"][name] = save_graph(args.output, name, nodes, edges)

    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
