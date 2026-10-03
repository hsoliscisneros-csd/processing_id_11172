# Experiment log (paper companion)

Manuscript ID **11172** — Fragmentation-Conditioned WiFi Threat Detection under Crowdsensing.

Public data path used by this package:

`data/datos_wifi_curated_anon.parquet`

## PhysAgg-R ablations (connected / non-isolate)

Entry point:

```bash
python -m src.train.physagg --run A2e --seeds 42,0,1,2,3 --split temporal --subgraph non_isolate
```

| Setting | Best GNN (PR-AUC) | Note |
|:---|:---|:---|
| Temporal, non-isolate, 5 seeds | A2e ≈ 0.886 ± 0.005 | vs SAGE baseline A0 ≈ 0.880 ± 0.002 |
| Full-graph / isolate regimes | RF stronger on isolates | Message passing is connectivity-limited |
| Municipality hold-out (Suchiapa) | SAGE A0 ≈ 0.628 | PhysAgg does not win OOD |

### Temporal non-isolate (illustrative single-seed snapshot)

| Run | F1 | PR-AUC |
|:---|:---|:---|
| A0 SAGE | 0.666 | 0.764 |
| A1 physics-informed loss | 0.670 | 0.764 |
| A2 PhysAgg operator | 0.685 | 0.768 |
| A3 PhysAgg + loss | 0.734 | 0.839 |

A3 improves in-domain connected refinement; OOD municipality hold-out does not favour A3. The paper claim emphasises connectivity-conditioned evaluation and bounded PhysAgg refinement on connected regions.

## Spatial OOD manifest

- Config: `experiments/manifests/spatial_ood_v0.yaml`
- Protocol used in the manuscript: municipality hold-out (not denser spatial grids).

## Reproduction notes

1. Install deps from `requirements.txt` (PyTorch Geometric optional for GNN runs).
2. Point loaders at `data/datos_wifi_curated_anon.parquet` (defaults already updated in this package).
3. Heuristic labels are threat-risk proxies, not adjudicated attacks.
