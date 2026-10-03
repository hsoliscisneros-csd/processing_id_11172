"""EXP-06: Physics-Informed GNN (PI-GNN) with ablation A0-A3, lambda grid, spatial split."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from src.evaluation.significance import mcnemar_test
from src.evaluation.splits import spatial_loo_folds
from src.evaluation.thresholds import optimal_threshold_f1
from src.train.gnn_common import SEED, evaluate_model, forward_model, prepare_graph_dataset
from src.train.gnn_models import NodeSAGE


class PiSAGE(NodeSAGE):
    """GraphSAGE with accessible embeddings for physics losses."""

    def encode(self, x, edge_index, edge_weight=None, **_kwargs):
        h = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index, edge_weight=edge_weight))
        return h

    def forward(self, x, edge_index, edge_weight=None, **_kwargs):
        h = self.encode(x, edge_index, edge_weight=edge_weight)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.head(h).squeeze(-1)


def _implicit_distance(emb: torch.Tensor) -> torch.Tensor:
    return torch.norm(emb, dim=1).clamp(min=1e-6)


def edge_pathloss_loss(
    edge_dist: torch.Tensor,
    edge_rssi: torch.Tensor,
    edge_index: torch.Tensor,
    node_train_mask: torch.Tensor,
    n: torch.Tensor,
    *,
    p0: float = -45.0,
    dist_min: float = 0.5,
) -> torch.Tensor:
    """Residual RF en aristas: MSE(RSSI_obs, P0 - 10 n log10(d)).

    Solo aristas con ambos extremos en train y d > dist_min.
    """
    if edge_index.numel() == 0 or edge_dist is None or edge_rssi is None:
        return n.new_tensor(0.0)
    row, col = edge_index
    keep = node_train_mask[row] & node_train_mask[col] & (edge_dist > dist_min)
    if not keep.any():
        return n.new_tensor(0.0)
    d = edge_dist[keep].clamp(min=0.1)
    rssi_hat = p0 - 10.0 * n * torch.log10(d)
    return F.mse_loss(edge_rssi[keep], rssi_hat)


def physics_loss(
    emb: torch.Tensor,
    dist_target: torch.Tensor,
    mask: torch.Tensor,
    *,
    loss_type: str = "mse",
    dist_min: float = 0.5,
) -> torch.Tensor:
    if not mask.any():
        return emb.new_tensor(0.0)
    reliable = mask & (dist_target > dist_min)
    if not reliable.any():
        return emb.new_tensor(0.0)
    pred = _implicit_distance(emb[reliable])
    target = dist_target[reliable].clamp(min=0.1, max=200.0)
    lp = torch.log1p(pred)
    lt = torch.log1p(target)
    if loss_type == "huber":
        return F.huber_loss(lp, lt, delta=1.0)
    if loss_type == "log_cosh":
        return torch.mean(torch.log(torch.cosh(lp - lt)))
    return F.mse_loss(lp, lt)


def _effective_lambda(base: float, epoch: int, warmup_epochs: int, ramp_epochs: int) -> float:
    if warmup_epochs <= 0:
        return base
    if epoch < warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return base
    progress = min(1.0, (epoch - warmup_epochs) / ramp_epochs)
    return base * progress


def smoothness_loss(
    emb: torch.Tensor,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    *,
    max_pairs: int = 4096,
) -> torch.Tensor:
    if edge_index.numel() == 0:
        return emb.new_tensor(0.0)
    row, col = edge_index
    benign = (labels[row] < 0.5) & (labels[col] < 0.5)
    if not benign.any():
        return emb.new_tensor(0.0)
    idx = torch.where(benign)[0]
    if idx.numel() > max_pairs:
        idx = idx[torch.randperm(idx.numel(), device=idx.device)[:max_pairs]]
    return F.mse_loss(emb[row[idx]], emb[col[idx]])


def _dist_targets(node_df: pd.DataFrame) -> np.ndarray:
    if "dist_mean" in node_df.columns:
        return node_df["dist_mean"].fillna(0).values.astype(np.float32)
    return np.zeros(len(node_df), dtype=np.float32)


def train_pignn(
    model: PiSAGE,
    dataset,
    *,
    lambda_physics: float,
    lambda_smooth: float,
    epochs: int = 200,
    patience: int = 20,
    lr: float = 1e-3,
    loss_variant: dict | None = None,
) -> tuple[PiSAGE, float, float]:
    lv = loss_variant or {}
    loss_type = lv.get("physics_loss", "mse")
    physics_warmup = int(lv.get("physics_warmup_epochs", 0))
    physics_ramp = int(lv.get("physics_ramp_epochs", 0))
    smooth_warmup = int(lv.get("smooth_warmup_epochs", 0))
    dist_min = float(lv.get("dist_min", 0.5))

    torch.manual_seed(SEED)
    data = dataset.data
    dist_target = torch.tensor(_dist_targets(dataset.node_df), dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    train_idx = torch.tensor(dataset.train_mask)
    val_idx = torch.tensor(dataset.val_mask)

    pos = float(data.y[train_idx].sum().item())
    neg = float(len(train_idx) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)])

    best_state = None
    best_val_f1 = -1.0
    best_t = 0.5
    wait = 0

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        emb = model.encode(
            data.x,
            data.edge_index,
            edge_weight=getattr(data, "edge_weight", None),
        )
        logits = model.head(emb).squeeze(-1)
        loss_cls = F.binary_cross_entropy_with_logits(
            logits[train_idx], data.y[train_idx], pos_weight=pos_weight
        )
        l1 = _effective_lambda(lambda_physics, epoch, physics_warmup, physics_ramp)
        l2 = _effective_lambda(lambda_smooth, epoch, smooth_warmup, physics_ramp)
        loss_phys = physics_loss(
            emb, dist_target, train_idx, loss_type=loss_type, dist_min=dist_min
        )
        loss_smooth = smoothness_loss(emb, data.edge_index, data.y)
        loss = loss_cls + l1 * loss_phys + l2 * loss_smooth
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(forward_model(model, data, use_edges=True)[val_idx])
            t, vf1 = optimal_threshold_f1(
                data.y[val_idx].cpu().numpy().astype(int), scores.cpu().numpy()
            )
        if vf1 > best_val_f1:
            best_val_f1 = vf1
            best_t = t
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_t, best_val_f1


def _physics_violation_rate(model: PiSAGE, dataset, *, tolerance: float = 0.5) -> float:
    model.eval()
    data = dataset.data
    dist_target = _dist_targets(dataset.node_df)
    with torch.no_grad():
        emb = model.encode(
            data.x,
            data.edge_index,
            edge_weight=getattr(data, "edge_weight", None),
        )
        pred = _implicit_distance(emb).cpu().numpy()
    test_mask = dataset.test_mask
    d = dist_target[test_mask]
    p = pred[test_mask]
    valid = d > 0.5
    if not valid.any():
        return 0.0
    rel_err = np.abs(p[valid] - d[valid]) / d[valid]
    return float((rel_err > tolerance).mean())


def _dataset_from_cfg(
    cfg: dict,
    curated: pd.DataFrame,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    train_df: pd.DataFrame | None = None,
    val_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
):
    spatial_cfg = cfg.get("spatial") if cfg.get("split", "temporal") == "spatial" else None
    return prepare_graph_dataset(
        curated,
        nodes,
        edges,
        feature_mode=cfg.get("features", "graph"),
        edge_mode=cfg.get("edge_mode", "train"),
        subgraph_mode=cfg.get("subgraph", "non_isolate"),
        edge_weight_mode=cfg.get("edge_weight", "combined"),
        split_mode=cfg.get("split", "temporal"),
        spatial_cfg=spatial_cfg,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )


def run_variant(
    cfg: dict,
    variant_id: str,
    lambdas: dict,
    *,
    save_predictions: bool = False,
    train_df: pd.DataFrame | None = None,
    val_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
) -> dict:
    curated = pd.read_parquet(Path(cfg["input"]))
    graph_dir = Path(cfg.get("graph_dir", "data/graphs/physical"))
    nodes = pd.read_parquet(graph_dir / "nodes.parquet")
    edges = pd.read_parquet(graph_dir / "edges.parquet")
    dataset = _dataset_from_cfg(
        cfg, curated, nodes, edges, train_df=train_df, val_df=val_df, test_df=test_df
    )
    loss_variant = {**cfg.get("loss_variant", {}), **lambdas.get("loss_variant", {})}
    model = PiSAGE(len(dataset.feature_cols), hidden=int(cfg.get("hidden", 64)))
    model, threshold, val_f1 = train_pignn(
        model,
        dataset,
        lambda_physics=float(lambdas.get("lambda_physics", 0.0)),
        lambda_smooth=float(lambdas.get("lambda_smooth", 0.0)),
        epochs=int(cfg.get("epochs", 200)),
        patience=int(cfg.get("patience", 20)),
        lr=float(cfg.get("lr", 1e-3)),
        loss_variant=loss_variant,
    )
    metrics = evaluate_model(
        model, dataset, threshold, use_edges=True, return_predictions=save_predictions
    )
    metrics["variant"] = variant_id
    metrics["split"] = cfg.get("split", "temporal")
    metrics["lambda_physics"] = lambdas.get("lambda_physics", 0.0)
    metrics["lambda_smooth"] = lambdas.get("lambda_smooth", 0.0)
    metrics["loss_variant"] = loss_variant.get("physics_loss", "mse")
    metrics["val_f1"] = float(val_f1)
    metrics["physics_violation_rate"] = _physics_violation_rate(model, dataset)
    return metrics


def _expand_variants(cfg: dict, variant_filter: str | None) -> dict[str, dict]:
    items: dict[str, dict] = {}
    if cfg.get("grid"):
        g = cfg["grid"]
        l1 = g["lambda_physics"]
        l2 = g["lambda_smooth"]
        for a, b in itertools.product(l1, l2):
            vid = f"G_l1_{a}_l2_{b}".replace(".", "p")
            items[vid] = {"lambda_physics": float(a), "lambda_smooth": float(b)}
    baseline = cfg.get("variants", {})
    for vid, spec in baseline.items():
        if vid not in items:
            items[vid] = spec
    if not items:
        vid = variant_filter or cfg.get("variant", "A3")
        items = {
            vid: {
                "lambda_physics": cfg.get("lambda_physics", 0.1),
                "lambda_smooth": cfg.get("lambda_smooth", 0.1),
            }
        }
    if variant_filter:
        items = {variant_filter: items[variant_filter]}
    return items


def _summary_row(metrics: dict) -> dict:
    return {
        "variant": metrics["variant"],
        "split": metrics.get("split", "temporal"),
        "lambda_physics": metrics["lambda_physics"],
        "lambda_smooth": metrics["lambda_smooth"],
        "val_f1": round(metrics["val_f1"], 4),
        "f1": round(metrics["f1"], 4),
        "pr_auc": round(metrics["pr_auc"], 4),
        "physics_violation_rate": round(metrics["physics_violation_rate"], 4),
        "n_nodes_test": metrics["n_nodes_test"],
    }


def run_mcnemar(out_root: Path, variant_a: str, variant_b: str) -> dict:
    pa = json.loads((out_root / variant_a / "metrics.json").read_text(encoding="utf-8"))
    pb = json.loads((out_root / variant_b / "metrics.json").read_text(encoding="utf-8"))
    pred_a = {n: p for n, p in zip(pa["predictions"]["node_id"], pa["predictions"]["y_pred"])}
    pred_b = {n: p for n, p in zip(pb["predictions"]["node_id"], pb["predictions"]["y_pred"])}
    y_true_a = {n: y for n, y in zip(pa["predictions"]["node_id"], pa["predictions"]["y_true"])}
    common = sorted(set(pred_a) & set(pred_b))
    if not common:
        raise ValueError("No overlapping test nodes for McNemar")
    y = np.array([y_true_a[n] for n in common], dtype=int)
    result = mcnemar_test(y, np.array([pred_a[n] for n in common]), np.array([pred_b[n] for n in common]))
    out = {
        "variant_a": variant_a,
        "variant_b": variant_b,
        "f1_a": pa["f1"],
        "f1_b": pb["f1"],
        "n_paired": len(common),
        **result,
    }
    return out


def run_spatial_loo(cfg: dict, items: dict[str, dict], out_root: Path) -> list[dict]:
    """Leave-one-municipality-out: average val F1 per variant for lambda selection."""
    curated = pd.read_parquet(Path(cfg["input"]))
    spatial = cfg.get("spatial", {})
    folds = spatial_loo_folds(
        curated,
        train_municipalities=spatial["train_municipalities"],
        state=spatial.get("state", "Chiapas"),
        min_val_obs=int(cfg.get("loo_min_val_obs", 200)),
    )
    if not folds:
        raise ValueError("No LOO folds generated; check train_municipalities")

    summary = []
    for vid, lambdas in items.items():
        fold_f1s = []
        for val_muni, train_df, val_df, test_df in folds:
            metrics = run_variant(
                cfg,
                f"{vid}__{val_muni[:12]}",
                lambdas,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
            )
            fold_f1s.append(metrics["val_f1"])
        row = {
            "variant": vid,
            "lambda_physics": lambdas.get("lambda_physics", 0.0),
            "lambda_smooth": lambdas.get("lambda_smooth", 0.0),
            "loss_variant": lambdas.get("loss_variant", cfg.get("loss_variant", {})),
            "mean_val_f1": round(float(np.mean(fold_f1s)), 4),
            "std_val_f1": round(float(np.std(fold_f1s)), 4),
            "n_folds": len(fold_f1s),
            "fold_val_f1": [round(f, 4) for f in fold_f1s],
        }
        summary.append(row)
        print(f"LOO {vid}: mean_val_f1={row['mean_val_f1']} (+/- {row['std_val_f1']})")

    summary.sort(key=lambda r: -r["mean_val_f1"])
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "loo_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    best = summary[0]
    (out_root / "loo_best.json").write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
    print(f"LOO best: {best['variant']} mean_val_f1={best['mean_val_f1']}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="EXP-06 PI-GNN ablation")
    parser.add_argument("--config", type=Path, default=Path("experiments/configs/pignn_ablation.yaml"))
    parser.add_argument("--variant", type=str, default=None, help="Single variant id (A0-A3, G_l1_...)")
    parser.add_argument(
        "--mcnemar",
        nargs=2,
        metavar=("A", "B"),
        default=None,
        help="Run McNemar on saved predictions (e.g. --mcnemar A0 A1)",
    )
    parser.add_argument(
        "--loo",
        action="store_true",
        help="Run leave-one-municipality-out validation for variant selection",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    out_root = Path(cfg.get("output_dir", "experiments/outputs/gnn/pignn"))

    if args.mcnemar:
        result = run_mcnemar(out_root, args.mcnemar[0], args.mcnemar[1])
        out_path = out_root / f"mcnemar_{args.mcnemar[0]}_vs_{args.mcnemar[1]}.json"
        out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0

    save_preds = bool(cfg.get("save_predictions", False))
    items = _expand_variants(cfg, args.variant)

    if args.loo or cfg.get("run_loo"):
        run_spatial_loo(cfg, items, out_root)
        if not cfg.get("loo_only"):
            pass  # continue to final runs below
        else:
            return 0

    summary = []

    for vid, lambdas in items.items():
        print(f"=== {vid} split={cfg.get('split','temporal')} l1={lambdas['lambda_physics']} l2={lambdas['lambda_smooth']} ===")
        metrics = run_variant(cfg, vid, lambdas, save_predictions=save_preds)
        out_dir = out_root / vid
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        row = _summary_row(metrics)
        summary.append(row)
        print(json.dumps(row, ensure_ascii=False))

    summary_name = "grid_summary.json" if cfg.get("grid") else "summary.json"
    (out_root / summary_name).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if cfg.get("grid"):
        csv_path = out_root / "grid_summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)
        best = max(summary, key=lambda r: r["val_f1"])
        (out_root / "grid_best.json").write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
        print(f"Best by val_f1: {best['variant']} (val_f1={best['val_f1']}, test_f1={best['f1']})")

    if cfg.get("mcnemar_pair") and save_preds:
        a, b = cfg["mcnemar_pair"]
        result = run_mcnemar(out_root, a, b)
        out_path = out_root / f"mcnemar_{a}_vs_{b}.json"
        out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("McNemar:", json.dumps(result, indent=2))

    if cfg.get("evaluate_gate") and save_preds:
        from src.experiments.spatial_diagnostics import evaluate_gate

        gate = evaluate_gate(out_root, baseline=cfg.get("gate_baseline", "A0"))
        (out_root / "gate_decision.json").write_text(
            json.dumps(gate, indent=2) + "\n", encoding="utf-8"
        )
        print("Gate:", json.dumps(gate, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
