"""Sensitivity of physical-graph fragmentation to NEARBY construction parameters."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from src.graphs.build import SCAN_KEY, graph_stats

DEFAULT_INPUT = Path("data/datos_wifi_curated_anon.parquet")
DEFAULT_OUT = Path("reports/sensitivity")


def precompute_coobservations(df: pd.DataFrame) -> tuple[set[str], dict[tuple[str, str], list[float]]]:
    """Collect all co-scan AP pairs with mean distance (one pass)."""
    edge_acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    for _, scan in df.groupby(SCAN_KEY, sort=False):
        sub = scan.dropna(subset=["bssid", "estimated_distance"])
        if len(sub) < 2:
            continue
        records = sub[["bssid", "estimated_distance"]].to_dict("records")
        for i, a in enumerate(records):
            for b in records[i + 1 :]:
                d_avg = (a["estimated_distance"] + b["estimated_distance"]) / 2
                key = tuple(sorted((a["bssid"], b["bssid"])))
                edge_acc[key].append(d_avg)

    all_bssids = set(df["bssid"].dropna().unique())
    ever_coobs = {b for pair in edge_acc for b in pair}
    return all_bssids, edge_acc


def stats_from_edges(
    all_bssids: set[str],
    edge_acc: dict[tuple[str, str], list[float]],
    nearby_m: float,
    min_co: int,
) -> dict:
    edges = []
    for (src, dst), dists in edge_acc.items():
        kept = [d for d in dists if d <= nearby_m]
        if len(kept) < min_co:
            continue
        weight = 1.0 / (1.0 + float(np.mean(kept)))
        edges.append({"src": src, "dst": dst, "weight": weight})

    nodes = pd.DataFrame({"node_id": sorted(all_bssids)})
    edges_df = pd.DataFrame(edges)
    stats = graph_stats(nodes, edges_df)

    g = nx.Graph()
    for nid in all_bssids:
        g.add_node(nid)
    for e in edges:
        g.add_edge(e["src"], e["dst"])
    degrees = [d for _, d in g.degree()]
    iso_frac = sum(1 for d in degrees if d == 0) / max(len(degrees), 1)
    comp_sizes = [len(c) for c in nx.connected_components(g)] if g.number_of_nodes() else [0]
    stats["isolated_frac"] = float(iso_frac)
    stats["largest_component"] = int(max(comp_sizes))
    return stats


def sweep(
    all_bssids: set[str],
    edge_acc: dict[tuple[str, str], list[float]],
    nearby_values: list[float],
    min_co_values: list[int],
) -> pd.DataFrame:
    rows = []
    for nearby_m, min_co in product(nearby_values, min_co_values):
        s = stats_from_edges(all_bssids, edge_acc, nearby_m, min_co)
        rows.append(
            {
                "nearby_m": nearby_m,
                "min_co_obs": min_co,
                **s,
            }
        )
    return pd.DataFrame(rows)


def _plot_metric(df: pd.DataFrame, metric: str, ylabel: str, stem: str, out: Path) -> None:
    """One standalone panel, sized for ~7.5 cm side-by-side inclusion."""
    palette = plt.cm.viridis(np.linspace(0.15, 0.85, len(df["min_co_obs"].unique())))
    fig, ax = plt.subplots(figsize=(5.0, 3.9))
    for i, min_co in enumerate(sorted(df["min_co_obs"].unique())):
        sub = df[df["min_co_obs"] == min_co].sort_values("nearby_m")
        ax.plot(
            sub["nearby_m"],
            sub[metric],
            marker="o",
            ms=4,
            label=f"min co-obs = {min_co}",
            color=palette[i],
        )
    ax.set_xlabel("Distance threshold (m)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.legend(fontsize=9, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_sensitivity(df: pd.DataFrame, out: Path) -> None:
    _plot_metric(df, "isolated_frac", "Isolated node fraction",
                 "graph_threshold_sensitivity_a", out)
    _plot_metric(df, "n_edges", "Edge count",
                 "graph_threshold_sensitivity_b", out)


def main() -> int:
    p = argparse.ArgumentParser(description="Physical graph threshold sensitivity")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--nearby",
        type=float,
        nargs="+",
        default=[15, 20, 30, 50, 75, 100, 150],
    )
    p.add_argument("--min-co", type=int, nargs="+", default=[1, 2, 3, 5])
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(args.input)
    all_bssids, edge_acc = precompute_coobservations(raw)
    never_coobs_frac = 1.0 - len({b for pair in edge_acc for b in pair}) / len(all_bssids)

    grid = sweep(all_bssids, edge_acc, args.nearby, args.min_co)
    grid["never_coobs_frac"] = never_coobs_frac
    grid.to_csv(args.output / "threshold_sensitivity.csv", index=False)
    plot_sensitivity(grid, args.output)

    baseline = grid[(grid["nearby_m"] == 30) & (grid["min_co_obs"] == 2)].iloc[0]
    summary = {
        "never_coobserved_fraction": never_coobs_frac,
        "baseline_isolated_frac": float(baseline["isolated_frac"]),
        "isolated_frac_min": float(grid["isolated_frac"].min()),
        "isolated_frac_max": float(grid["isolated_frac"].max()),
        "n_configurations": len(grid),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(grid.pivot(index="nearby_m", columns="min_co_obs", values="isolated_frac").round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
