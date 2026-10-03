"""EXP-05: refinement ladder (simple → more expressive)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.evaluation.metrics import binary_metrics
from src.evaluation.thresholds import optimal_threshold_f1
from src.train.gnn_common import evaluate_model, prepare_graph_dataset, train_node_model
from src.train.gnn_models import build_model

# Orden: de control simple a arquitectura más expresiva
LADDER = [
    {
        "id": "00_rf_node",
        "model": "sklearn_rf",
        "edges": "train",
        "features": "rich",
        "subgraph": "all",
        "note": "Random Forest nodo (misma tarea que GNN, sin mensajes)",
    },
    {
        "id": "00_mlp_safe",
        "model": "mlp",
        "edges": "full",
        "features": "safe",
        "subgraph": "all",
        "note": "Control sin grafo; features sin leakage",
    },
    {
        "id": "01_gcn_safe_full",
        "model": "gcn",
        "edges": "full",
        "features": "safe",
        "subgraph": "all",
        "note": "GCN + grafo completo (estructura con leakage)",
    },
    {
        "id": "02_gcn_safe_train",
        "model": "gcn",
        "edges": "train",
        "features": "safe",
        "subgraph": "all",
        "note": "GCN + aristas solo train (anti-leakage)",
    },
    {
        "id": "03_gcn_rich_train",
        "model": "gcn",
        "edges": "train",
        "features": "rich",
        "subgraph": "all",
        "note": "GCN + grado + features radio ampliadas",
    },
    {
        "id": "04_sage_rich_train",
        "model": "sage",
        "edges": "train",
        "features": "rich",
        "subgraph": "all",
        "note": "GraphSAGE mean aggregator (EXP-05.2)",
    },
    {
        "id": "05_gat_rich_train",
        "model": "gat",
        "edges": "train",
        "features": "rich",
        "subgraph": "all",
        "note": "GAT 4 heads (EXP-05.3)",
    },
    {
        "id": "06_rf_graph_train",
        "model": "sklearn_rf",
        "edges": "train",
        "features": "graph",
        "subgraph": "all",
        "note": "B: RF + pagerank, SSID/vendor, WPA",
    },
    {
        "id": "07_sage_graph_train",
        "model": "sage",
        "edges": "train",
        "features": "graph",
        "subgraph": "all",
        "note": "B: GraphSAGE + features de grafo enriquecidas",
    },
    {
        "id": "08_rf_graph_lcc",
        "model": "sklearn_rf",
        "edges": "train",
        "features": "graph",
        "subgraph": "lcc",
        "note": "B+C: RF en componente conexa mayor",
    },
    {
        "id": "09_sage_graph_lcc",
        "model": "sage",
        "edges": "train",
        "features": "graph",
        "subgraph": "lcc",
        "note": "B+C (LCC): cluster denso 455 nodos — sesgado",
    },
    {
        "id": "10_rf_graph_noniso",
        "model": "sklearn_rf",
        "edges": "train",
        "features": "graph",
        "subgraph": "non_isolate",
        "note": "B+C: RF sin nodos aislados (grado>=1)",
    },
    {
        "id": "11_sage_graph_noniso",
        "model": "sage",
        "edges": "train",
        "features": "graph",
        "subgraph": "non_isolate",
        "edge_weight": "none",
        "note": "B+C: GraphSAGE sin nodos aislados",
    },
    {
        "id": "12_sage_proximity_noniso",
        "model": "sage",
        "edges": "train",
        "features": "graph",
        "subgraph": "non_isolate",
        "edge_weight": "proximity",
        "note": "D: peso 1/(1+d) en convoluciones",
    },
    {
        "id": "13_sage_coobs_noniso",
        "model": "sage",
        "edges": "train",
        "features": "graph",
        "subgraph": "non_isolate",
        "edge_weight": "co_obs",
        "note": "D: peso log(1+co_obs_count)",
    },
    {
        "id": "14_sage_combined_noniso",
        "model": "sage",
        "edges": "train",
        "features": "graph",
        "subgraph": "non_isolate",
        "edge_weight": "combined",
        "note": "D: proximidad x co-observacion",
    },
    {
        "id": "15_gcn_combined_noniso",
        "model": "gcn",
        "edges": "train",
        "features": "graph",
        "subgraph": "non_isolate",
        "edge_weight": "combined",
        "note": "D: GCN con peso combinado",
    },
]

BASELINE_REF = {
    "random_forest": {"f1": 0.656, "pr_auc": 0.746},
    "xgboost": {"f1": 0.647, "pr_auc": 0.747},
}


def _dataset_kwargs(spec: dict) -> dict:
    return {
        "edge_mode": spec["edges"],
        "feature_mode": spec["features"],
        "subgraph_mode": spec.get("subgraph", "all"),
        "edge_weight_mode": spec.get("edge_weight", "none"),
    }


def run_sklearn_node(
    curated: pd.DataFrame,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    spec: dict,
) -> dict:
    dataset = prepare_graph_dataset(curated, nodes, edges, **_dataset_kwargs(spec))
    df = dataset.node_df
    cols = dataset.feature_cols
    tr, va, te = dataset.train_mask, dataset.val_mask, dataset.test_mask
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(df.loc[tr, cols], df.loc[tr, "label"].astype(int))
    scores_val = clf.predict_proba(df.loc[va, cols])[:, 1]
    threshold, _ = optimal_threshold_f1(df.loc[va, "label"].astype(int).values, scores_val)
    scores_test = clf.predict_proba(df.loc[te, cols])[:, 1]
    preds = (scores_test >= threshold).astype(int)
    metrics = binary_metrics(df.loc[te, "label"].astype(int).values, scores_test, preds)
    metrics["threshold"] = threshold
    metrics["n_nodes_test"] = int(te.sum())
    metrics["n_edges"] = int(dataset.data.edge_index.shape[1] // 2)
    metrics["edge_mode"] = dataset.edge_mode
    metrics["feature_cols"] = cols
    metrics["subgraph_mode"] = dataset.subgraph_mode
    metrics["graph_stats"] = dataset.graph_stats
    return {
        "variant_id": spec["id"],
        "model": spec["model"],
        "edges": spec["edges"],
        "features": spec["features"],
        "subgraph": spec.get("subgraph", "all"),
        "edge_weight": spec.get("edge_weight", "none"),
        "note": spec["note"],
        "metrics": metrics,
    }


def run_variant(
    curated: pd.DataFrame,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    spec: dict,
) -> dict:
    dataset = prepare_graph_dataset(curated, nodes, edges, **_dataset_kwargs(spec))
    use_edges = spec["model"] != "mlp"
    model = build_model(spec["model"], len(dataset.feature_cols))
    model, threshold = train_node_model(model, dataset, use_edges=use_edges)
    metrics = evaluate_model(model, dataset, threshold, use_edges=use_edges)
    return {
        "variant_id": spec["id"],
        "model": spec["model"],
        "edges": spec["edges"],
        "features": spec["features"],
        "subgraph": spec.get("subgraph", "all"),
        "edge_weight": spec.get("edge_weight", "none"),
        "note": spec["note"],
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="EXP-05 ladder (simple → expressive)")
    parser.add_argument("--curated", type=Path, default=Path("data/datos_wifi_curated_anon.parquet"))
    parser.add_argument("--graph-dir", type=Path, default=Path("data/graphs/physical"))
    parser.add_argument("--output", type=Path, default=Path("experiments/outputs/gnn/ladder"))
    parser.add_argument("--only", type=str, default="", help="IDs separados por coma")
    args = parser.parse_args()

    curated = pd.read_parquet(args.curated)
    nodes = pd.read_parquet(args.graph_dir / "nodes.parquet")
    edges = pd.read_parquet(args.graph_dir / "edges.parquet")

    selected = LADDER
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        selected = [s for s in LADDER if s["id"] in wanted]

    args.output.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for spec in selected:
        print(f"\n=== {spec['id']} — {spec['note']} ===")
        if spec["model"] == "sklearn_rf":
            out = run_sklearn_node(curated, nodes, edges, spec)
        else:
            out = run_variant(curated, nodes, edges, spec)
        results.append(out)
        m = out["metrics"]
        print(
            f"F1={m['f1']:.3f} PR-AUC={m['pr_auc']:.3f} "
            f"MCC={m['mcc']:.3f} edges={m['n_edges']} "
            f"nodes={m.get('graph_stats', {}).get('n_nodes', '?')} "
            f"comp={m.get('graph_stats', {}).get('n_components', '?')}"
        )
        (args.output / f"{spec['id']}.json").write_text(
            json.dumps(out, indent=2) + "\n", encoding="utf-8"
        )

    summary = {
        "baseline_observation_level": BASELINE_REF,
        "variants": [
            {
                "id": r["variant_id"],
                "model": r["model"],
                "edges": r["edges"],
                "features": r["features"],
                "subgraph": r.get("subgraph", "all"),
                "edge_weight": r.get("edge_weight", "none"),
                "f1": r["metrics"]["f1"],
                "pr_auc": r["metrics"]["pr_auc"],
                "mcc": r["metrics"]["mcc"],
                "n_nodes_test": r["metrics"]["n_nodes_test"],
                "n_edges": r["metrics"]["n_edges"],
            }
            for r in results
        ],
        "best_by_f1": max(results, key=lambda r: r["metrics"]["f1"])["variant_id"],
        "best_by_pr_auc": max(results, key=lambda r: r["metrics"]["pr_auc"])["variant_id"],
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("\n=== Resumen escalera ===")
    print(json.dumps(summary, indent=2))
    print(f"Saved {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
