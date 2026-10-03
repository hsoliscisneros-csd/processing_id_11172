"""EXP-05b / Track B: Dual-head + dual_rf (GNN conectados + RF aislados)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier

from src.evaluation.metrics import binary_metrics
from src.evaluation.thresholds import optimal_threshold_f1
from src.train.baselines import SEED as RF_SEED
from src.train.gnn_common import (
    SEED,
    densify_knn_edges,
    evaluate_model,
    forward_model,
    prepare_graph_dataset,
    to_pyg_data,
    train_node_model,
)
from src.train.gnn_models import DualHeadDetector, build_model

try:
    from src.train.baselines import fit_rf
except ImportError:
    fit_rf = None  # type: ignore


def _connected_mask(data) -> torch.Tensor:
    n = data.x.size(0)
    deg = torch.zeros(n, dtype=torch.long)
    if data.edge_index.numel():
        ones = torch.ones(data.edge_index.size(1), dtype=torch.long)
        deg = deg.scatter_add(0, data.edge_index[0].cpu(), ones)
    return deg > 0


def _fit_rf(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    if fit_rf is not None:
        return fit_rf(X, y)
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RF_SEED if "RF_SEED" in dir() else SEED,
    ).fit(X, y)


def _partition_metrics(
    y: np.ndarray, scores: np.ndarray, mask: np.ndarray, threshold: float
) -> dict | None:
    if mask.sum() == 0:
        return None
    yt = y[mask].astype(int)
    s = scores[mask]
    pred = (s >= threshold).astype(int)
    return binary_metrics(yt, s, pred)


def _eval_scores(
    dataset,
    scores: np.ndarray,
    threshold: float,
    connected: np.ndarray,
) -> dict:
    test = dataset.test_mask
    y = dataset.data.y.cpu().numpy().astype(int)
    yt = y[test]
    st = scores[test]
    pred = (st >= threshold).astype(int)
    metrics = binary_metrics(yt, st, pred)
    metrics["threshold"] = float(threshold)
    metrics["n_nodes_test"] = int(test.sum())
    conn_t = connected[test]
    metrics["pr_auc_connected"] = None
    metrics["pr_auc_isolated"] = None
    m_c = _partition_metrics(yt, st, conn_t, threshold)
    m_i = _partition_metrics(yt, st, ~conn_t, threshold)
    if m_c:
        metrics["pr_auc_connected"] = m_c.get("pr_auc")
        metrics["f1_connected"] = m_c.get("f1")
        metrics["n_test_connected"] = int(conn_t.sum())
    if m_i:
        metrics["pr_auc_isolated"] = m_i.get("pr_auc")
        metrics["f1_isolated"] = m_i.get("f1")
        metrics["n_test_isolated"] = int((~conn_t).sum())
    return metrics


def _physical_edge_df(dataset) -> pd.DataFrame:
    """Reconstruye aristas físicas desde edge_index del dataset."""
    data = dataset.data
    ids = dataset.node_df["node_id"].astype(str).values
    ei = data.edge_index.cpu().numpy()
    # undirected duplicadas → quedarse con u<v
    rows = []
    seen = set()
    ew = getattr(data, "edge_weight", None)
    for i in range(ei.shape[1]):
        u, v = int(ei[0, i]), int(ei[1, i])
        if u >= v:
            continue
        key = (u, v)
        if key in seen:
            continue
        seen.add(key)
        w = float(ew[i].item()) if ew is not None else 1.0
        rows.append(
            {"src": ids[u], "dst": ids[v], "weight": w, "co_obs_count": 1.0, "edge_type": "physical"}
        )
    return pd.DataFrame(rows)


def _train_sage_knn(dataset, *, k: int = 8) -> tuple[dict, object, float]:
    """SAGE sobre grafo físico + kNN semántico desde aislados."""
    from dataclasses import replace

    from sklearn.neighbors import NearestNeighbors

    torch.manual_seed(SEED)
    if hasattr(dataset.data, "connected_mask"):
        delattr(dataset.data, "connected_mask")

    data0 = dataset.data
    connected = _connected_mask(data0).numpy()
    X = data0.x.cpu().numpy()
    n = X.shape[0]
    iso_idx = np.where(~connected)[0]

    # edge_index físico existente
    ei_phys = data0.edge_index.cpu()
    ew_phys = getattr(data0, "edge_weight", None)
    if ew_phys is not None:
        ew_phys = ew_phys.cpu()
    else:
        ew_phys = torch.ones(ei_phys.size(1))

    nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric="euclidean", n_jobs=-1)
    nn.fit(X)
    dists, inds = nn.kneighbors(X[iso_idx], return_distance=True)
    knn_src: list[int] = []
    knn_dst: list[int] = []
    knn_w: list[float] = []
    for local, src_i in enumerate(iso_idx):
        for dist, dst_i in zip(dists[local], inds[local]):
            dst_i = int(dst_i)
            if dst_i == int(src_i):
                continue
            w = float(1.0 / (1.0 + dist))
            knn_src.extend([int(src_i), dst_i])
            knn_dst.extend([dst_i, int(src_i)])
            knn_w.extend([w, w])

    if knn_src:
        ei_knn = torch.tensor([knn_src, knn_dst], dtype=torch.long)
        ew_knn = torch.tensor(knn_w, dtype=torch.float32)
        ew_knn = ew_knn / ew_knn.max().clamp(min=1e-9)
        edge_index = torch.cat([ei_phys, ei_knn], dim=1)
        edge_weight = torch.cat([ew_phys, ew_knn], dim=0)
    else:
        edge_index, edge_weight = ei_phys, ew_phys

    data = data0.clone()
    data.edge_index = edge_index
    data.edge_weight = edge_weight
    ds = replace(dataset, data=data)

    model = build_model("sage", len(dataset.feature_cols))
    model, threshold = train_node_model(model, ds, use_edges=True)
    metrics = evaluate_model(model, ds, threshold, use_edges=True)
    with torch.no_grad():
        scores = torch.sigmoid(forward_model(model, ds.data, use_edges=True)).cpu().numpy()
    part = _eval_scores(ds, scores, threshold, connected)
    metrics["pr_auc_connected"] = part.get("pr_auc_connected")
    metrics["pr_auc_isolated"] = part.get("pr_auc_isolated")
    metrics["f1_connected"] = part.get("f1_connected")
    metrics["f1_isolated"] = part.get("f1_isolated")
    metrics["n_edges_physical"] = int(ei_phys.size(1) // 2)
    metrics["n_edges_densified"] = int(edge_index.size(1) // 2)
    metrics["knn_k"] = k
    return metrics, model, float(threshold)


def _train_dual_rf(
    dataset_full,
    curated,
    nodes,
    edges,
    *,
    edge_mode: str,
    feature_mode: str,
    edge_weight: str,
) -> tuple[dict, object, object, float]:
    """SAGE en non_isolate (protocolo freeze) + RF en grafo completo; fusión."""
    torch.manual_seed(SEED)
    if hasattr(dataset_full.data, "connected_mask"):
        delattr(dataset_full.data, "connected_mask")

    connected = _connected_mask(dataset_full.data).numpy()
    node_ids = dataset_full.node_df["node_id"].astype(str).values

    ds_conn = prepare_graph_dataset(
        curated,
        nodes,
        edges,
        edge_mode=edge_mode,
        feature_mode=feature_mode,
        subgraph_mode="non_isolate",
        edge_weight_mode=edge_weight,
    )
    sage = build_model("sage", len(ds_conn.feature_cols))
    sage, _ = train_node_model(sage, ds_conn, use_edges=True)
    sage.eval()
    with torch.no_grad():
        sage_conn = (
            torch.sigmoid(forward_model(sage, ds_conn.data, use_edges=True)).cpu().numpy()
        )
    conn_ids = ds_conn.node_df["node_id"].astype(str).values
    id_to_sage = {nid: float(s) for nid, s in zip(conn_ids, sage_conn)}
    sage_scores = np.array([id_to_sage.get(nid, 0.0) for nid in node_ids], dtype=np.float64)

    X = dataset_full.data.x.cpu().numpy()
    y = dataset_full.data.y.cpu().numpy().astype(int)
    train = dataset_full.train_mask
    rf = _fit_rf(X[train], y[train])
    rf_scores = rf.predict_proba(X)[:, 1]

    fused = np.where(connected, sage_scores, rf_scores)
    # Donde no hay score SAGE (no debería), ya usamos 0; preferir RF si faltara
    missing = connected & ~np.isin(node_ids, conn_ids)
    fused = np.where(missing, rf_scores, fused)

    val = dataset_full.val_mask
    t, _ = optimal_threshold_f1(y[val], fused[val])
    metrics = _eval_scores(dataset_full, fused, t, connected)
    metrics["rf_train_nodes"] = int(train.sum())
    metrics["sage_protocol"] = "non_isolate"
    metrics["n_sage_nodes"] = int(len(conn_ids))
    return metrics, sage, rf, float(t)


def _fit_xgb(X: np.ndarray, y: np.ndarray):
    from src.train.baselines import fit_xgb

    return fit_xgb(X, y)


def _train_tabular(dataset, kind: str = "rf") -> tuple[dict, object, float]:
    connected = _connected_mask(dataset.data).numpy()
    X = dataset.data.x.cpu().numpy()
    y = dataset.data.y.cpu().numpy().astype(int)
    if kind == "xgb":
        model = _fit_xgb(X[dataset.train_mask], y[dataset.train_mask])
    else:
        model = _fit_rf(X[dataset.train_mask], y[dataset.train_mask])
    scores = model.predict_proba(X)[:, 1]
    t, _ = optimal_threshold_f1(y[dataset.val_mask], scores[dataset.val_mask])
    metrics = _eval_scores(dataset, scores, t, connected)
    return metrics, model, float(t)


def _train_dual_tab(
    dataset_full,
    curated,
    nodes,
    edges,
    *,
    edge_mode: str,
    feature_mode: str,
    edge_weight: str,
    tab: str = "xgb",
) -> tuple[dict, object, object, float]:
    """SAGE non_isolate + tabular (RF/XGB) en aislados; en conectados max(sage, tab)."""
    torch.manual_seed(SEED)
    if hasattr(dataset_full.data, "connected_mask"):
        delattr(dataset_full.data, "connected_mask")

    connected = _connected_mask(dataset_full.data).numpy()
    node_ids = dataset_full.node_df["node_id"].astype(str).values

    ds_conn = prepare_graph_dataset(
        curated,
        nodes,
        edges,
        edge_mode=edge_mode,
        feature_mode=feature_mode,
        subgraph_mode="non_isolate",
        edge_weight_mode=edge_weight,
    )
    sage = build_model("sage", len(ds_conn.feature_cols))
    sage, _ = train_node_model(sage, ds_conn, use_edges=True)
    sage.eval()
    with torch.no_grad():
        sage_conn = (
            torch.sigmoid(forward_model(sage, ds_conn.data, use_edges=True)).cpu().numpy()
        )
    conn_ids = ds_conn.node_df["node_id"].astype(str).values
    id_to_sage = {nid: float(s) for nid, s in zip(conn_ids, sage_conn)}
    sage_scores = np.array([id_to_sage.get(nid, 0.0) for nid in node_ids], dtype=np.float64)

    X = dataset_full.data.x.cpu().numpy()
    y = dataset_full.data.y.cpu().numpy().astype(int)
    train = dataset_full.train_mask
    tab_m = _fit_xgb(X[train], y[train]) if tab == "xgb" else _fit_rf(X[train], y[train])
    tab_scores = tab_m.predict_proba(X)[:, 1]

    # Conectados: tomar el mejor score por nodo (evita degradar RF/XGB saturado).
    # Aislados: tabular.
    fused = np.where(connected, np.maximum(sage_scores, tab_scores), tab_scores)

    val = dataset_full.val_mask
    t, _ = optimal_threshold_f1(y[val], fused[val])
    metrics = _eval_scores(dataset_full, fused, t, connected)
    metrics["tab_model"] = tab
    metrics["fusion"] = "max_on_connected"
    metrics["sage_protocol"] = "non_isolate"
    return metrics, sage, tab_m, float(t)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train DualHead / dual_rf (Track B)")
    parser.add_argument("--curated", type=Path, default=Path("data/datos_wifi_curated_anon.parquet"))
    parser.add_argument("--graph-dir", type=Path, default=Path("data/graphs/physical"))
    parser.add_argument("--output", type=Path, default=Path("experiments/outputs/dual_head"))
    parser.add_argument("--edges", choices=("train", "full"), default="train")
    parser.add_argument("--features", choices=("safe", "rich", "graph"), default="graph")
    parser.add_argument("--edge-weight", default="combined")
    parser.add_argument(
        "--ablation",
        choices=("dual", "sage_full", "mlp_full", "dual_rf", "rf_all", "sage_knn", "xgb_all", "dual_xgb"),
        default="dual_xgb",
        help="dual_xgb=SAGE noniso+XGB; xgb_all=control",
    )
    parser.add_argument("--knn-k", type=int, default=8)
    args = parser.parse_args()

    curated = pd.read_parquet(args.curated)
    nodes = pd.read_parquet(args.graph_dir / "nodes.parquet")
    edges = pd.read_parquet(args.graph_dir / "edges.parquet")

    use_edges = args.ablation not in ("mlp_full", "rf_all", "xgb_all")
    dataset = prepare_graph_dataset(
        curated,
        nodes,
        edges,
        edge_mode=args.edges,
        feature_mode=args.features,
        subgraph_mode="all",
        edge_weight_mode=args.edge_weight if use_edges else "none",
    )

    n = int(dataset.data.x.size(0))
    n_conn = int(_connected_mask(dataset.data).sum().item())
    stem = args.ablation
    args.output.mkdir(parents=True, exist_ok=True)

    if args.ablation == "sage_knn":
        metrics, model, threshold = _train_sage_knn(dataset, k=args.knn_k)
        torch.save(model.state_dict(), args.output / f"{stem}.pt")
        model_name = "sage_knn"
    elif args.ablation in ("rf_all", "xgb_all"):
        kind = "xgb" if args.ablation == "xgb_all" else "rf"
        metrics, tab_m, threshold = _train_tabular(dataset, kind=kind)
        joblib.dump(tab_m, args.output / f"{stem}.joblib")
        model_name = args.ablation
    elif args.ablation in ("dual_rf", "dual_xgb"):
        tab = "xgb" if args.ablation == "dual_xgb" else "rf"
        metrics, sage, tab_m, threshold = _train_dual_tab(
            dataset,
            curated,
            nodes,
            edges,
            edge_mode=args.edges,
            feature_mode=args.features,
            edge_weight=args.edge_weight,
            tab=tab,
        )
        torch.save(sage.state_dict(), args.output / f"{stem}_sage.pt")
        joblib.dump(tab_m, args.output / f"{stem}_tab.joblib")
        model_name = args.ablation
    elif args.ablation == "dual_rf_legacy":
        metrics, sage, rf, threshold = _train_dual_rf(
            dataset,
            curated,
            nodes,
            edges,
            edge_mode=args.edges,
            feature_mode=args.features,
            edge_weight=args.edge_weight,
        )
        torch.save(sage.state_dict(), args.output / f"{stem}_sage.pt")
        joblib.dump(rf, args.output / f"{stem}_rf.joblib")
        model_name = "dual_rf"
    else:
        in_dim = len(dataset.feature_cols)
        if args.ablation == "dual":
            model: torch.nn.Module = DualHeadDetector(in_dim)
            dataset.data.connected_mask = _connected_mask(dataset.data)  # type: ignore[attr-defined]
        elif args.ablation == "sage_full":
            model = build_model("sage", in_dim)
        else:
            model = build_model("mlp", in_dim)
        model, threshold = train_node_model(model, dataset, use_edges=use_edges)
        metrics = evaluate_model(model, dataset, threshold, use_edges=use_edges)
        conn = _connected_mask(dataset.data).numpy()
        with torch.no_grad():
            scores = (
                torch.sigmoid(forward_model(model, dataset.data, use_edges=use_edges))
                .cpu()
                .numpy()
            )
        part = _eval_scores(dataset, scores, threshold, conn)
        metrics["pr_auc_connected"] = part.get("pr_auc_connected")
        metrics["pr_auc_isolated"] = part.get("pr_auc_isolated")
        metrics["f1_connected"] = part.get("f1_connected")
        metrics["f1_isolated"] = part.get("f1_isolated")
        torch.save(model.state_dict(), args.output / f"{stem}.pt")
        model_name = f"dual_head_{args.ablation}"

    out = {
        "model": model_name,
        "track": "B",
        "version": "v1" if args.ablation in ("dual_rf", "rf_all", "sage_knn") else "v0",
        "edges": args.edges,
        "features": args.features,
        "edge_weight": args.edge_weight,
        "subgraph": "all",
        "n_nodes": n,
        "n_connected": n_conn,
        "n_isolated": n - n_conn,
        "threshold": threshold,
        "metrics": metrics,
        "graph_stats": dataset.graph_stats,
        "gate_note": "G1: dual_rf PR-AUC >= rf_all y > sage_full en test completo",
    }
    (args.output / f"metrics_{stem}.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
