"""Utilidades compartidas para EXP-05 (nodos AP, grafos, entrenamiento)."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

from src.evaluation.metrics import binary_metrics
from src.evaluation.splits import spatial_split, temporal_split
from src.evaluation.thresholds import optimal_threshold_f1
from src.graphs.build import build_physical_graph

SEED = 42
LABEL = "is_anomaly_heuristic"

SAFE_FEATURES = [
    "level_mean",
    "level_std",
    "frequency_mode",
    "obs_count",
    "dist_mean",
    "dist_residual_mean",
    "open_rate",
    "wep_rate",
    "randomized_mac_rate",
]

RICH_EXTRA = ["graph_degree"]

GRAPH_EXTRA = [
    "graph_degree",
    "ap_pagerank",
    "ssid_nunique",
    "wpa2_rate",
    "wpa3_rate",
    "vendor_code",
]


@dataclass(frozen=True)
class GraphDataset:
    data: Data
    node_df: pd.DataFrame
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    feature_cols: list[str]
    edge_mode: str
    subgraph_mode: str
    edge_weight_mode: str
    graph_stats: dict


def build_node_table(
    curated: pd.DataFrame,
    graph_nodes: pd.DataFrame | None = None,
    *,
    include_label: bool = True,
) -> pd.DataFrame:
    if curated.empty:
        cols = ["node_id", *SAFE_FEATURES]
        if include_label:
            cols.append("label")
        return pd.DataFrame(columns=cols)

    work = curated.copy()
    work["is_open"] = (work["security_type"] == "OPEN").astype(float)
    work["is_wep"] = (work["security_type"] == "WEP").astype(float)
    work["is_wpa2"] = work["security_type"].isin(["WPA2", "WPA2/WPA3"]).astype(float)
    work["is_wpa3"] = (work["security_type"] == "WPA3").astype(float)
    if "is_randomized_mac" not in work.columns:
        from src.data.wifi_utils import is_randomized_mac

        work["is_randomized_mac"] = work["bssid"].map(is_randomized_mac).astype(float)
    else:
        work["is_randomized_mac"] = work["is_randomized_mac"].astype(float)

    if "distance_residual" not in work.columns and "estimated_distance" in work.columns:
        from src.data.wifi_utils import estimate_distance

        expected = work.apply(
            lambda r: estimate_distance(int(r["level"]), int(r["frequency"]))
            if pd.notna(r["level"]) and pd.notna(r["frequency"])
            else np.nan,
            axis=1,
        )
        work["distance_residual"] = work["estimated_distance"] - expected

    agg_spec: dict = {
        "level_mean": ("level", "mean"),
        "level_std": ("level", "std"),
        "frequency_mode": ("frequency", lambda s: int(s.mode().iloc[0]) if len(s.mode()) else 0),
        "obs_count": (LABEL, "count"),
        "dist_mean": ("estimated_distance", "mean"),
        "dist_residual_mean": ("distance_residual", "mean"),
        "open_rate": ("is_open", "mean"),
        "wep_rate": ("is_wep", "mean"),
        "wpa2_rate": ("is_wpa2", "mean"),
        "wpa3_rate": ("is_wpa3", "mean"),
        "ssid_nunique": ("ssid", "nunique"),
        "vendor_code": ("vendor", lambda s: float(hash(_safe_mode(s)) % 997) / 997.0),
        "randomized_mac_rate": ("is_randomized_mac", "mean"),
    }
    if include_label:
        agg_spec["label"] = (LABEL, lambda s: int(s.mean() >= 0.5))

    agg = work.groupby("bssid").agg(**agg_spec).reset_index()
    agg = agg.rename(columns={"bssid": "node_id"})
    agg["level_std"] = agg["level_std"].fillna(0)
    for col in (
        "dist_mean",
        "dist_residual_mean",
        "open_rate",
        "wep_rate",
        "wpa2_rate",
        "wpa3_rate",
        "ssid_nunique",
        "vendor_code",
        "randomized_mac_rate",
    ):
        agg[col] = agg[col].fillna(0)

    if graph_nodes is not None:
        agg = agg[agg["node_id"].isin(graph_nodes["node_id"])]
    return agg


def _safe_mode(series: pd.Series, default: str = "") -> str:
    s = series.dropna()
    if s.empty:
        return default
    m = s.mode()
    return str(m.iloc[0]) if len(m) else default


def _edges_for_mode(
    train_df: pd.DataFrame,
    full_edges: pd.DataFrame,
    all_bssids: set[str],
    edge_mode: str,
) -> pd.DataFrame:
    if edge_mode == "full":
        edges = full_edges
    elif edge_mode == "train":
        _, edges = build_physical_graph(train_df)
        edges = edges if len(edges) else pd.DataFrame(columns=["src", "dst"])
    else:
        raise ValueError(f"edge_mode desconocido: {edge_mode}")
    return edges[edges["src"].isin(all_bssids) & edges["dst"].isin(all_bssids)]


def _add_graph_centrality(node_df: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    out = node_df.copy()
    degree: dict[str, int] = {nid: 0 for nid in out["node_id"]}
    g = nx.Graph()
    for nid in out["node_id"]:
        g.add_node(nid)
    for _, e in edges.iterrows():
        g.add_edge(e["src"], e["dst"])
        degree[e["src"]] = degree.get(e["src"], 0) + 1
        degree[e["dst"]] = degree.get(e["dst"], 0) + 1
    out["graph_degree"] = out["node_id"].map(degree).fillna(0).astype(float)
    if g.number_of_edges() == 0:
        out["ap_pagerank"] = 0.0
        return out
    try:
        pr = nx.pagerank(g, max_iter=50)
    except nx.PowerIterationFailedConvergence:
        pr = {n: 0.0 for n in g.nodes()}
    out["ap_pagerank"] = out["node_id"].map(pr).fillna(0.0).astype(float)
    return out


def _largest_component_nodes(node_ids: set[str], edges: pd.DataFrame) -> set[str]:
    g = nx.Graph()
    for nid in node_ids:
        g.add_node(nid)
    for _, e in edges.iterrows():
        if e["src"] in node_ids and e["dst"] in node_ids:
            g.add_edge(e["src"], e["dst"])
    if g.number_of_nodes() == 0:
        return set()
    components = list(nx.connected_components(g))
    if not components:
        return set()
    return set(max(components, key=len))


def _graph_stats(node_df: pd.DataFrame, edges: pd.DataFrame) -> dict:
    g = nx.Graph()
    for nid in node_df["node_id"]:
        g.add_node(nid)
    for _, e in edges.iterrows():
        g.add_edge(e["src"], e["dst"])
    degrees = [d for _, d in g.degree()] if g.number_of_nodes() else [0]
    return {
        "n_nodes": int(g.number_of_nodes()),
        "n_edges": int(g.number_of_edges()),
        "n_components": int(nx.number_connected_components(g)) if g.number_of_nodes() else 0,
        "avg_degree": float(np.mean(degrees)),
        "max_degree": int(max(degrees)) if degrees else 0,
    }


def prepare_graph_dataset(
    curated: pd.DataFrame,
    nodes: pd.DataFrame,
    full_edges: pd.DataFrame,
    *,
    edge_mode: str = "train",
    feature_mode: str = "safe",
    subgraph_mode: str = "all",
    edge_weight_mode: str = "none",
    split_mode: str = "temporal",
    spatial_cfg: dict | None = None,
    train_df: pd.DataFrame | None = None,
    val_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
) -> GraphDataset:
    if train_df is not None and val_df is not None and test_df is not None:
        pass  # use provided frames
    elif split_mode == "spatial":
        sc = spatial_cfg or {}
        train_df, val_df, test_df = spatial_split(
            curated,
            train_municipalities=sc["train_municipalities"],
            val_municipalities=sc.get("val_municipalities"),
            state=sc.get("state", "Chiapas"),
        )
    elif split_mode == "grid":
        from src.evaluation.splits import geo_grid_split

        sc = spatial_cfg or {}
        train_df, val_df, test_df = geo_grid_split(
            curated,
            train_grids=sc["train_grids"],
            val_grids=sc["val_grids"],
            test_grids=sc["test_grids"],
            state=sc.get("state", "Chiapas"),
            lat_median=sc.get("lat_median"),
            lon_median=sc.get("lon_median"),
        )
    elif split_mode == "temporal":
        train_df, val_df, test_df = temporal_split(curated)
    else:
        raise ValueError(f"split_mode desconocido: {split_mode}")
    graph_nodes = nodes.copy()

    train_nodes = build_node_table(train_df, graph_nodes)
    val_nodes = build_node_table(val_df, graph_nodes)
    test_nodes = build_node_table(test_df, graph_nodes)

    all_bssids = (
        set(train_nodes["node_id"])
        | set(val_nodes["node_id"])
        | set(test_nodes["node_id"])
    )
    graph_nodes = graph_nodes[graph_nodes["node_id"].isin(all_bssids)]
    sub_edges = _edges_for_mode(train_df, full_edges, all_bssids, edge_mode)

    feat_table = build_node_table(train_df, graph_nodes, include_label=False)
    label_table = build_node_table(
        curated[curated["bssid"].isin(all_bssids)], graph_nodes
    )[["node_id", "label"]]
    full = graph_nodes[["node_id"]].merge(feat_table, on="node_id", how="left")
    full = full.merge(label_table, on="node_id", how="left")

    if subgraph_mode == "lcc":
        lcc = _largest_component_nodes(set(full["node_id"]), sub_edges)
        full = full[full["node_id"].isin(lcc)].reset_index(drop=True)
        sub_edges = sub_edges[
            sub_edges["src"].isin(lcc) & sub_edges["dst"].isin(lcc)
        ].reset_index(drop=True)
        train_nodes = train_nodes[train_nodes["node_id"].isin(lcc)]
        val_nodes = val_nodes[val_nodes["node_id"].isin(lcc)]
        test_nodes = test_nodes[test_nodes["node_id"].isin(lcc)]
    elif subgraph_mode == "non_isolate":
        full = _add_graph_centrality(full, sub_edges)
        active = set(full.loc[full["graph_degree"] > 0, "node_id"])
        full = full[full["node_id"].isin(active)].reset_index(drop=True)
        sub_edges = sub_edges[
            sub_edges["src"].isin(active) & sub_edges["dst"].isin(active)
        ].reset_index(drop=True)
        train_nodes = train_nodes[train_nodes["node_id"].isin(active)]
        val_nodes = val_nodes[val_nodes["node_id"].isin(active)]
        test_nodes = test_nodes[test_nodes["node_id"].isin(active)]
    elif subgraph_mode != "all":
        raise ValueError(f"subgraph_mode desconocido: {subgraph_mode}")

    if subgraph_mode != "non_isolate":
        full = _add_graph_centrality(full, sub_edges)
    stats = _graph_stats(full, sub_edges)

    feature_cols = list(SAFE_FEATURES)
    if feature_mode == "rich":
        feature_cols = feature_cols + list(RICH_EXTRA)
    elif feature_mode == "graph":
        feature_cols = list(SAFE_FEATURES) + list(GRAPH_EXTRA)
    elif feature_mode != "safe":
        raise ValueError(f"feature_mode desconocido: {feature_mode}")

    full[feature_cols] = full[feature_cols].fillna(0)
    level_raw = full["level_mean"].fillna(0).values.astype(np.float32).copy()
    scaler = StandardScaler()
    full[feature_cols] = scaler.fit_transform(full[feature_cols])

    data = to_pyg_data(full, sub_edges, feature_cols, edge_weight_mode=edge_weight_mode)
    data.level_raw = torch.tensor(level_raw)
    if getattr(data, "edge_index", None) is not None and data.edge_index.numel():
        row, col = data.edge_index
        data.edge_rssi = 0.5 * (data.level_raw[row] + data.level_raw[col])
    train_mask = full["node_id"].isin(set(train_nodes["node_id"])).values
    val_mask = full["node_id"].isin(set(val_nodes["node_id"])).values
    test_mask = full["node_id"].isin(set(test_nodes["node_id"])).values

    return GraphDataset(
        data=data,
        node_df=full,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        feature_cols=feature_cols,
        edge_mode=edge_mode,
        subgraph_mode=subgraph_mode,
        edge_weight_mode=edge_weight_mode,
        graph_stats=stats,
    )


def densify_knn_edges(
    node_ids: np.ndarray,
    X: np.ndarray,
    physical_edges: pd.DataFrame,
    connected: np.ndarray,
    *,
    k: int = 8,
    max_isolates: int | None = None,
) -> pd.DataFrame:
    """Añade aristas kNN en espacio de features desde aislados → vecinos.

    Novedad Track B: message passing sobre grafo físico + densificación semántica
    para nodos sin vecindad RF.
    """
    from sklearn.neighbors import NearestNeighbors

    iso_idx = np.where(~connected)[0]
    if max_isolates is not None and len(iso_idx) > max_isolates:
        rng = np.random.default_rng(SEED)
        iso_idx = rng.choice(iso_idx, size=max_isolates, replace=False)

    if len(iso_idx) == 0:
        return physical_edges.copy()

    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X)), metric="euclidean", n_jobs=-1)
    nn.fit(X)
    # Vecinos entre todos; descartar self
    dists, inds = nn.kneighbors(X[iso_idx], return_distance=True)
    rows: list[dict] = []
    id_list = list(node_ids)
    for local, src_i in enumerate(iso_idx):
        for dist, dst_i in zip(dists[local], inds[local]):
            if int(dst_i) == int(src_i):
                continue
            rows.append(
                {
                    "src": id_list[int(src_i)],
                    "dst": id_list[int(dst_i)],
                    "weight": float(1.0 / (1.0 + dist)),
                    "co_obs_count": 1.0,
                    "edge_type": "knn",
                }
            )
    knn_df = pd.DataFrame(rows)
    phys = physical_edges.copy()
    if "edge_type" not in phys.columns:
        phys["edge_type"] = "physical"
    if knn_df.empty:
        return phys
    return pd.concat([phys, knn_df], ignore_index=True)


def _scalar_edge_weight(edge: pd.Series, mode: str) -> float:
    w = float(edge.get("weight", 1.0))
    co = float(edge.get("co_obs_count", 1.0))
    if mode == "none":
        return 1.0
    if mode == "proximity":
        return w
    if mode == "co_obs":
        return float(np.log1p(co))
    if mode == "combined":
        return w * float(np.log1p(co))
    raise ValueError(f"edge_weight_mode desconocido: {mode}")


def to_pyg_data(
    node_df: pd.DataFrame,
    edges: pd.DataFrame,
    feature_cols: list[str],
    *,
    edge_weight_mode: str = "none",
) -> Data:
    id_map = {nid: i for i, nid in enumerate(node_df["node_id"])}
    x = node_df[feature_cols].fillna(0).values.astype(np.float32)
    y = node_df["label"].fillna(0).values.astype(np.float32)
    edge_list: list[list[int]] = []
    edge_weights: list[float] = []
    edge_dists: list[float] = []
    for _, e in edges.iterrows():
        if e["src"] in id_map and e["dst"] in id_map:
            u, v = id_map[e["src"]], id_map[e["dst"]]
            raw_w = float(e.get("weight", 1.0))
            # weight en grafo físico = 1/(1+d) → recuperar d
            d = max(0.0, 1.0 / max(raw_w, 1e-6) - 1.0)
            w = _scalar_edge_weight(e, edge_weight_mode)
            edge_list.extend([[u, v], [v, u]])
            edge_weights.extend([w, w])
            edge_dists.extend([d, d])
    edge_index = (
        torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        if edge_list
        else torch.empty((2, 0), dtype=torch.long)
    )
    data = Data(
        x=torch.tensor(x.copy()),
        edge_index=edge_index,
        y=torch.tensor(y.copy()),
    )
    if edge_dists:
        data.edge_dist = torch.tensor(edge_dists, dtype=torch.float32)
    if edge_weight_mode != "none" and edge_weights:
        ew = torch.tensor(edge_weights, dtype=torch.float32)
        ew = ew / ew.max().clamp(min=1e-9)
        data.edge_weight = ew
    return data


def train_node_model(
    model: torch.nn.Module,
    dataset: GraphDataset,
    *,
    epochs: int = 200,
    patience: int = 20,
    lr: float = 1e-3,
    use_edges: bool = True,
    seed: int = SEED,
    early_metric: str = "f1",
) -> tuple[torch.nn.Module, float]:
    torch.manual_seed(seed)
    data = dataset.data
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    train_idx = torch.tensor(dataset.train_mask)
    val_idx = torch.tensor(dataset.val_mask)

    pos = float(data.y[train_idx].sum().item())
    neg = float(len(train_idx) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)])

    best_state = None
    best_val = -1.0
    best_t = 0.5
    wait = 0

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits = forward_model(model, data, use_edges=use_edges)
        loss = F.binary_cross_entropy_with_logits(
            logits[train_idx], data.y[train_idx], pos_weight=pos_weight
        )
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(forward_model(model, data, use_edges=use_edges)[val_idx])
            scores_np = scores.cpu().numpy()
            y_val = data.y[val_idx].cpu().numpy().astype(int)
            t, vf1 = optimal_threshold_f1(y_val, scores_np)
            if early_metric == "pr_auc" and len(np.unique(y_val)) > 1:
                from sklearn.metrics import average_precision_score

                vmetric = float(average_precision_score(y_val, scores_np))
            else:
                vmetric = float(vf1)
        if vmetric > best_val:
            best_val = vmetric
            best_t = t
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_t


def evaluate_layered(
    model: torch.nn.Module,
    dataset: GraphDataset,
    threshold: float,
    *,
    use_edges: bool = True,
) -> dict:
    """PR-AUC / F1 en full test, connected-only y isolate-only."""
    model.eval()
    data = dataset.data
    with torch.no_grad():
        scores = torch.sigmoid(forward_model(model, data, use_edges=use_edges)).cpu().numpy()
    y = data.y.cpu().numpy().astype(int)
    test = dataset.test_mask
    deg = np.zeros(data.x.size(0), dtype=np.int64)
    if data.edge_index.numel():
        ei = data.edge_index[0].cpu().numpy()
        np.add.at(deg, ei, 1)
    connected = deg > 0

    def _pack(mask: np.ndarray) -> dict | None:
        m = test & mask
        if m.sum() == 0:
            return None
        yt, st = y[m], scores[m]
        pred = (st >= threshold).astype(int)
        out = binary_metrics(yt, st, pred)
        out["n"] = int(m.sum())
        return out

    full = _pack(np.ones_like(test, dtype=bool))
    conn = _pack(connected)
    iso = _pack(~connected)
    return {
        "threshold": float(threshold),
        "full": full,
        "connected": conn,
        "isolated": iso,
        "n_test": int(test.sum()),
        "n_test_connected": int((test & connected).sum()),
        "n_test_isolated": int((test & ~connected).sum()),
    }


def forward_model(model: torch.nn.Module, data: Data, *, use_edges: bool) -> torch.Tensor:
    if use_edges:
        edge_weight = getattr(data, "edge_weight", None)
        edge_dist = getattr(data, "edge_dist", None)
        connected_mask = getattr(data, "connected_mask", None)
        kwargs = {}
        if connected_mask is not None:
            kwargs["connected_mask"] = connected_mask
        if edge_dist is not None:
            kwargs["edge_dist"] = edge_dist
        if kwargs:
            return model(data.x, data.edge_index, edge_weight=edge_weight, **kwargs)
        return model(data.x, data.edge_index, edge_weight=edge_weight)
    return model(data.x)


def evaluate_model(
    model: torch.nn.Module,
    dataset: GraphDataset,
    threshold: float,
    *,
    use_edges: bool = True,
    return_predictions: bool = False,
) -> dict:
    model.eval()
    data = dataset.data
    with torch.no_grad():
        scores = torch.sigmoid(forward_model(model, data, use_edges=use_edges)).cpu().numpy()
    test_mask = dataset.test_mask
    y_test = data.y[test_mask].cpu().numpy().copy().astype(int)
    s_test = scores[test_mask]
    preds = (s_test >= threshold).astype(int)
    metrics = binary_metrics(y_test, s_test, preds)
    metrics["threshold"] = threshold
    metrics["n_nodes_test"] = int(test_mask.sum())
    metrics["n_edges"] = int(data.edge_index.shape[1] // 2)
    metrics["edge_mode"] = dataset.edge_mode
    metrics["subgraph_mode"] = dataset.subgraph_mode
    metrics["edge_weight_mode"] = dataset.edge_weight_mode
    metrics["graph_stats"] = dataset.graph_stats
    metrics["feature_cols"] = dataset.feature_cols
    if return_predictions:
        node_ids = dataset.node_df.loc[test_mask, "node_id"].tolist()
        metrics["predictions"] = {
            "node_id": node_ids,
            "y_true": y_test.tolist(),
            "y_score": s_test.tolist(),
            "y_pred": preds.tolist(),
        }
    return metrics
