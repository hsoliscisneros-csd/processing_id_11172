"""PhysAgg / PhysAgg-R — ablations A0, A1, A2, A2e, A3 plus fragmentation layers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from src.evaluation.significance import mcnemar_test
from src.evaluation.thresholds import optimal_threshold_f1
from src.train.gnn_common import (
    SEED,
    evaluate_layered,
    evaluate_model,
    forward_model,
    prepare_graph_dataset,
    train_node_model,
)
from src.train.gnn_models import PhysSAGE, build_model
from src.train.pignn import (
    PiSAGE,
    edge_pathloss_loss,
    physics_loss,
    smoothness_loss,
    train_pignn,
    _dist_targets,
)


SPATIAL_ROBUST = {
    "state": "Chiapas",
    # Val = municipio grande del metro (no Las Flechas)
    "train_municipalities": [
        "Tuxtla Gutiérrez",
        "Chiapa de Corzo",
        "Jardines del Grijalva",
    ],
    "val_municipalities": ["Suchiapa"],
}

SPATIAL_LEGACY = {
    "state": "Chiapas",
    "train_municipalities": [
        "Tuxtla Gutiérrez",
        "Suchiapa",
        "Chiapa de Corzo",
        "Jardines del Grijalva",
    ],
    "val_municipalities": ["Las Flechas"],
}


def _train_physagg_combo(
    model: PhysSAGE,
    dataset,
    *,
    lambda_nodal: float = 0.0,
    lambda_edge: float = 0.0,
    lambda_smooth: float = 0.0,
    epochs: int = 200,
    patience: int = 20,
    lr: float = 1e-3,
    seed: int = SEED,
    p0: float = -45.0,
) -> tuple[PhysSAGE, float]:
    """A2e (edge) / A3 (nodal) / combinaciones."""
    torch.manual_seed(seed)
    data = dataset.data
    dist_target = torch.tensor(_dist_targets(dataset.node_df), dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    train_idx = torch.tensor(dataset.train_mask)
    val_idx = torch.tensor(dataset.val_mask)
    train_mask_t = torch.tensor(dataset.train_mask, dtype=torch.bool)
    pos = float(data.y[train_idx].sum().item())
    neg = float(len(train_idx) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)])

    best_state = None
    best_val = -1.0
    best_t = 0.5
    wait = 0
    edge_dist = getattr(data, "edge_dist", None)
    edge_weight = getattr(data, "edge_weight", None)
    edge_rssi = getattr(data, "edge_rssi", None)

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        emb = model.encode(
            data.x, data.edge_index, edge_weight=edge_weight, edge_dist=edge_dist
        )
        logits = model.head(emb).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(
            logits[train_idx], data.y[train_idx], pos_weight=pos_weight
        )
        n = model.conv1._n()
        if lambda_nodal > 0:
            loss = loss + lambda_nodal * physics_loss(emb, dist_target, train_idx)
        if lambda_edge > 0 and edge_dist is not None and edge_rssi is not None:
            loss = loss + lambda_edge * edge_pathloss_loss(
                edge_dist,
                edge_rssi,
                data.edge_index,
                train_mask_t,
                n,
                p0=p0,
            )
        if lambda_smooth > 0:
            loss = loss + lambda_smooth * smoothness_loss(emb, data.edge_index, data.y)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(forward_model(model, data, use_edges=True)[val_idx])
            scores_np = scores.cpu().numpy()
            y_val = data.y[val_idx].cpu().numpy().astype(int)
            t, _ = optimal_threshold_f1(y_val, scores_np)
            vmetric = (
                float(average_precision_score(y_val, scores_np))
                if len(np.unique(y_val)) > 1
                else 0.0
            )
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


def _build_dataset(args, curated, nodes, edges):
    spatial_cfg = None
    split_mode = args.split
    if args.split == "spatial":
        spatial_cfg = SPATIAL_ROBUST if args.spatial_val == "robust" else SPATIAL_LEGACY
    elif args.split == "grid":
        spatial_cfg = {
            "state": "Chiapas",
            "train_grids": ["1_0", "1_1"],
            "val_grids": ["0_1"],
            "test_grids": ["0_0"],
        }
    return prepare_graph_dataset(
        curated,
        nodes,
        edges,
        edge_mode="train",
        feature_mode=args.features,
        subgraph_mode=args.subgraph,
        edge_weight_mode=args.edge_weight,
        split_mode=split_mode,
        spatial_cfg=spatial_cfg,
    )


def run_one(args, dataset, run: str, seed: int) -> dict:
    in_dim = len(dataset.feature_cols)
    if run == "A0":
        model = build_model("sage", in_dim)
        model, thr = train_node_model(
            model, dataset, use_edges=True, seed=seed, early_metric="pr_auc"
        )
        extra = {"n_pathloss": None}
    elif run == "A1":
        model = PiSAGE(in_dim)
        # train_pignn uses SEED global; set torch seed
        torch.manual_seed(seed)
        model, thr, _ = train_pignn(
            model,
            dataset,
            lambda_physics=args.lambda1,
            lambda_smooth=args.lambda2,
            lr=args.lr,
        )
        extra = {"lambda1": args.lambda1, "lambda2": args.lambda2}
    elif run == "A2":
        model = PhysSAGE(in_dim, learn_n=True)
        if args.n_fixed:
            with torch.no_grad():
                ln = float(np.log(args.n_fixed))
                model.conv1.log_n.fill_(ln)
                model.conv2.log_n.fill_(ln)
            model.conv1.log_n.requires_grad_(False)
            model.conv2.log_n.requires_grad_(False)
        model, thr = train_node_model(
            model, dataset, use_edges=True, seed=seed, early_metric="pr_auc", lr=args.lr
        )
        extra = {
            "n_pathloss": model.path_loss_n(),
            "alpha": float(model.conv1.log_alpha.exp().detach()),
            "n_fixed": bool(args.n_fixed),
        }
    elif run == "A2e":
        model = PhysSAGE(in_dim, learn_n=True)
        model, thr = _train_physagg_combo(
            model,
            dataset,
            lambda_edge=args.lambda_edge,
            lambda_smooth=args.lambda2,
            seed=seed,
            lr=args.lr,
            p0=args.p0,
        )
        extra = {
            "n_pathloss": model.path_loss_n(),
            "lambda_edge": args.lambda_edge,
            "alpha": float(model.conv1.log_alpha.exp().detach()),
        }
    else:  # A3
        model = PhysSAGE(in_dim)
        model, thr = _train_physagg_combo(
            model,
            dataset,
            lambda_nodal=args.lambda1,
            lambda_smooth=args.lambda2,
            seed=seed,
            lr=args.lr,
        )
        extra = {
            "n_pathloss": model.path_loss_n(),
            "lambda1": args.lambda1,
            "lambda2": args.lambda2,
        }

    metrics = evaluate_model(model, dataset, thr, use_edges=True, return_predictions=True)
    layered = evaluate_layered(model, dataset, thr, use_edges=True)
    return {
        "run": run,
        "seed": seed,
        "threshold": thr,
        "metrics": {k: v for k, v in metrics.items() if k != "predictions"},
        "layered": layered,
        "predictions": metrics.get("predictions"),
        **extra,
    }


def train_rf_fair(dataset) -> dict:
    """RF en el mismo noniso / features que el GNN."""
    from src.train.baselines import fit_rf

    X = dataset.data.x.cpu().numpy()
    y = dataset.data.y.cpu().numpy().astype(int)
    rf = fit_rf(X[dataset.train_mask], y[dataset.train_mask])
    scores = rf.predict_proba(X)[:, 1]
    t, _ = optimal_threshold_f1(y[dataset.val_mask], scores[dataset.val_mask])
    test = dataset.test_mask
    pred = (scores[test] >= t).astype(int)
    from src.evaluation.metrics import binary_metrics

    m = binary_metrics(y[test], scores[test], pred)
    # layered
    deg = np.zeros(X.shape[0], dtype=np.int64)
    if dataset.data.edge_index.numel():
        np.add.at(deg, dataset.data.edge_index[0].cpu().numpy(), 1)
    connected = deg > 0

    def _layer(mask):
        msk = test & mask
        if msk.sum() == 0:
            return None
        return binary_metrics(
            y[msk], scores[msk], (scores[msk] >= t).astype(int)
        ) | {"n": int(msk.sum())}

    return {
        "run": "RF_fair",
        "threshold": float(t),
        "metrics": m,
        "layered": {
            "full": _layer(np.ones_like(test, dtype=bool)),
            "connected": _layer(connected),
            "isolated": _layer(~connected),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="PhysAgg-R power ablation")
    parser.add_argument("--curated", type=Path, default=Path("data/datos_wifi_curated_anon.parquet"))
    parser.add_argument("--graph-dir", type=Path, default=Path("data/graphs/physical"))
    parser.add_argument("--output", type=Path, default=Path("experiments/outputs/physagg_power"))
    parser.add_argument("--features", default="graph")
    parser.add_argument("--edge-weight", default="combined")
    parser.add_argument("--subgraph", default="non_isolate")
    parser.add_argument(
        "--run",
        choices=("A0", "A1", "A2", "A2e", "A3", "all", "rf"),
        default="all",
    )
    parser.add_argument("--lambda1", type=float, default=0.3)
    parser.add_argument("--lambda2", type=float, default=0.01)
    parser.add_argument("--lambda-edge", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--p0", type=float, default=-45.0)
    parser.add_argument("--n-fixed", type=float, default=0.0, help="si >0 fija n")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--seeds", type=str, default="", help="comma list e.g. 42,0,1,2,3")
    parser.add_argument(
        "--split",
        choices=("temporal", "spatial", "grid"),
        default="temporal",
    )
    parser.add_argument(
        "--spatial-val",
        choices=("robust", "legacy"),
        default="robust",
    )
    parser.add_argument("--with-rf", action="store_true")
    parser.add_argument("--mcnemar", type=str, default="", help="e.g. A2e,A0")
    args = parser.parse_args()
    if args.n_fixed <= 0:
        args.n_fixed = 0.0

    curated = pd.read_parquet(args.curated)
    nodes = pd.read_parquet(args.graph_dir / "nodes.parquet")
    edges = pd.read_parquet(args.graph_dir / "edges.parquet")
    dataset = _build_dataset(args, curated, nodes, edges)
    args.output.mkdir(parents=True, exist_ok=True)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()] if args.seeds else [args.seed]
    runs = ["A0", "A1", "A2", "A2e", "A3"] if args.run == "all" else [args.run]
    if args.run == "rf":
        runs = []

    all_results = []
    pred_store: dict[str, dict] = {}

    for seed in seeds:
        for run in runs:
            out = run_one(args, dataset, run, seed)
            pred = out.pop("predictions", None)
            tag = f"{run}_s{seed}"
            (args.output / f"metrics_{tag}.json").write_text(
                json.dumps(out, indent=2) + "\n", encoding="utf-8"
            )
            if pred:
                pred_store[tag] = pred
                (args.output / f"pred_{tag}.json").write_text(
                    json.dumps(pred) + "\n", encoding="utf-8"
                )
            all_results.append(out)
            print(json.dumps({k: out[k] for k in out if k != "layered"}, indent=2))
            print("layered", json.dumps(out.get("layered"), indent=2))

    if args.with_rf or args.run == "rf":
        rf_out = train_rf_fair(dataset)
        (args.output / "metrics_RF_fair.json").write_text(
            json.dumps(rf_out, indent=2) + "\n", encoding="utf-8"
        )
        all_results.append(rf_out)
        print(json.dumps(rf_out, indent=2))

    # Aggregate mean±std for multi-seed
    summary_rows = []
    for run in runs:
        subset = [r for r in all_results if r.get("run") == run]
        if not subset:
            continue
        prs = [r["metrics"].get("pr_auc") for r in subset if r["metrics"].get("pr_auc") is not None]
        f1s = [r["metrics"].get("f1") for r in subset if r["metrics"].get("f1") is not None]
        summary_rows.append(
            {
                "run": run,
                "n_seeds": len(subset),
                "pr_auc_mean": float(np.mean(prs)) if prs else None,
                "pr_auc_std": float(np.std(prs)) if prs else None,
                "f1_mean": float(np.mean(f1s)) if f1s else None,
                "f1_std": float(np.std(f1s)) if f1s else None,
            }
        )

    mcnemar_out = None
    if args.mcnemar and "," in args.mcnemar:
        a, b = [x.strip() for x in args.mcnemar.split(",", 1)]
        # use first seed present
        ka = next((k for k in pred_store if k.startswith(f"{a}_")), None)
        kb = next((k for k in pred_store if k.startswith(f"{b}_")), None)
        if ka and kb:
            pa, pb = pred_store[ka], pred_store[kb]
            mcnemar_out = mcnemar_test(
                np.array(pa["y_true"]),
                np.array(pa["y_pred"]),
                np.array(pb["y_pred"]),
            )
            mcnemar_out["pair"] = [a, b]

    payload = {
        "split": args.split,
        "spatial_val": args.spatial_val if args.split == "spatial" else None,
        "subgraph": args.subgraph,
        "edge_weight": args.edge_weight,
        "seeds": seeds,
        "summary": summary_rows,
        "mcnemar": mcnemar_out,
        "runs": all_results,
    }
    (args.output / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print("SUMMARY", json.dumps(summary_rows, indent=2))
    if mcnemar_out:
        print("MCNEMAR", json.dumps(mcnemar_out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
