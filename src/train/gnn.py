"""EXP-05.1: GCN baseline (shared pipeline, no leakage)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.train.gnn_common import evaluate_model, prepare_graph_dataset, train_node_model
from src.train.gnn_models import build_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Train GCN (EXP-05.1)")
    parser.add_argument("--curated", type=Path, default=Path("data/datos_wifi_curated_anon.parquet"))
    parser.add_argument("--graph-dir", type=Path, default=Path("data/graphs/physical"))
    parser.add_argument("--output", type=Path, default=Path("experiments/outputs/gnn"))
    parser.add_argument("--edges", choices=("train", "full"), default="train")
    parser.add_argument("--features", choices=("safe", "rich"), default="rich")
    args = parser.parse_args()

    curated = pd.read_parquet(args.curated)
    nodes = pd.read_parquet(args.graph_dir / "nodes.parquet")
    edges = pd.read_parquet(args.graph_dir / "edges.parquet")

    dataset = prepare_graph_dataset(
        curated, nodes, edges, edge_mode=args.edges, feature_mode=args.features
    )
    model = build_model("gcn", len(dataset.feature_cols))
    model, threshold = train_node_model(model, dataset, use_edges=True)
    metrics = evaluate_model(model, dataset, threshold, use_edges=True)

    args.output.mkdir(parents=True, exist_ok=True)
    out = {
        "model": "gcn_physical",
        "edges": args.edges,
        "features": args.features,
        "metrics": metrics,
    }
    (args.output / "metrics.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    import torch

    torch.save(model.state_dict(), args.output / "gcn_physical.pt")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
