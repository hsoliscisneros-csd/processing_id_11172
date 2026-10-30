# Fragmentation-Conditioned WiFi Threat Detection under Crowdsensing

**Manuscript ID:** 11172, IEEE Latin America Transactions.

**Authors:**

- José A. Macías-Montoya
- Horacio I. Solís-Cisneros
- Josué R. Velázquez-González

**Affiliation:** Data Intelligence, Generative AI and Resilient Cybersecurity Research Group, Universidad Politécnica de Chiapas, Tuxtla Gutiérrez, Chiapas, Mexico.

**Repository:** https://github.com/hsoliscisneros-csd/processing_id_11172

This package lets a reader rebuild the anonymised corpus checks, the physical-graph isolation rate, and the PhysAgg-R tables reported in the manuscript. Released metric files are included, so the tables can be read without retraining. Retraining the graph models needs PyTorch Geometric.

## How to run

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/01_eda_wifi.ipynb
```

To rebuild the graph and retrain PhysAgg-R (A0, A1, A2, A2e, A3) on the temporal split and the non-isolated subgraph:

```bash
python -m src.graphs.build --input data/datos_wifi_curated_anon.parquet --output data/graphs
python -m src.train.physagg --run all --seeds 42,0,1,2,3 --split temporal --subgraph non_isolate --mcnemar A2e,A0 --with-rf
```

`notebooks/03_reproduce_physagg.ipynb` reads the released tables under `experiments/outputs/physagg_power/`. Set `RUN_TRAINING = True` in that notebook to retrain instead.

Run codes used below: **A0** GraphSAGE, **A1** nodal physics, **A2** PhysAgg operator, **A2e** PhysAgg plus the edge RSSI residual, **A3** PhysAgg plus nodal physics, **RF fair** random forest on the same connected mask.

## File descriptions

| File | What it is for |
|:---|:---|
| `.gitignore` | Keeps caches, virtual environments, checkpoints, and bulky prediction dumps out of the repository. The compact metric JSON files stay tracked. |
| `LICENSE` | MIT licence for the code. |
| `README.md` | This guide. |
| `requirements.txt` | Python packages for the notebooks and the training commands (pandas, scikit-learn, PyTorch, PyTorch Geometric, H3, and friends). |
| `data/DATA_DICTIONARY.md` | Field-by-field description of the released parquet: units, privacy notes, and the heuristic label. |
| `data/datos_wifi_curated_anon.parquet` | Anonymised curated scans, 53,371 rows. Input to the notebooks and to `src.graphs.build`. BSSID and device IDs are salted hashes. Coordinates are rounded to 3 decimals. Raw SSIDs are omitted. |
| `data/qc_report.json` | Quality-control counts for that parquet (rows, missing values, and the checks recorded at release). |
| `experiments/EXPERIMENT_LOG.md` | Short record of the PhysAgg-R settings that match the manuscript: temporal non-isolate means, the Suchiapa hold-out, and how to launch a run. |
| `experiments/configs/physagg_power.yaml` | Settings of the reported PhysAgg-R comparison: non-isolated subgraph, seeds 42, 0, 1, 2, 3, runs A0–A3, and Suchiapa as the spatial validation municipality. |
| `experiments/configs/physagg_v0.yaml` | Earlier PhysAgg ablation on the temporal split and the non-isolated subgraph. |
| `experiments/configs/dual_head_v0.yaml` | Dual-head setup: a graph model on connected nodes and a tabular head on isolates. |
| `experiments/configs/pignn_ablation.yaml` | Ablation A0–A3 of the earlier physics-informed GNN. |
| `experiments/configs/pignn_v1.yaml` | Default physics-informed GNN variant A3. |
| `experiments/configs/pignn_lambda_grid.yaml` | Grid over the physics-loss weights on the temporal split. |
| `experiments/configs/pignn_spatial.yaml` | Spatial split comparing A0 and A1, with a paired significance test. |
| `experiments/configs/pignn_spatial_grid.yaml` | Spatial grid over the physics-loss weights. |
| `experiments/configs/pignn_spatial_loss.yaml` | Spatial variants of the physics loss. |
| `experiments/configs/pignn_spatial_loo.yaml` | Leave-one-municipality-out used to choose the loss weights. |
| `experiments/configs/pignn_spatial_loo_best.yaml` | Loss weights kept after that leave-one-municipality-out selection. |
| `experiments/manifests/spatial_ood_v0.yaml` | Spatial protocols (municipality hold-out, leave-one-municipality-out, and a geographic grid). The manuscript reports the municipality hold-out. |
| `experiments/manifests/spatial_ood_v0.json` | The same spatial manifest in JSON. |
| `notebooks/01_eda_wifi.ipynb` | Exploratory pass over the released parquet and `qc_report.json`: coverage, security mix, and scan timing. |
| `notebooks/02_paper_results_view.ipynb` | Recomputes the physical-graph isolation rate at 30 m and the random-forest ablation from columns in the released parquet. |
| `notebooks/03_reproduce_physagg.ipynb` | Loads the released PhysAgg-R metric tables. Optional retraining from the same notebook. |
| `src/__init__.py` | Marks `src` as a package so `python -m src...` resolves. |
| `src/evaluation/__init__.py` | Marks the evaluation package. |
| `src/evaluation/metrics.py` | Precision, recall, F1, and PR-AUC used in the tables. |
| `src/evaluation/significance.py` | Paired comparison between two runs (McNemar) on the same test nodes. |
| `src/evaluation/splits.py` | Temporal and spatial train, validation, and test splits without leakage. |
| `src/evaluation/thresholds.py` | Chooses the decision threshold on validation scores. |
| `src/features/__init__.py` | Marks the feature package. |
| `src/features/build.py` | Builds the tabular features from the scans. |
| `src/graphs/__init__.py` | Marks the graph package. |
| `src/graphs/build.py` | Builds the physical, spatial, and temporal graphs from the parquet. |
| `src/graphs/sensitivity.py` | How the isolated-node fraction changes with the distance cutoff and the minimum number of co-observations. |
| `src/train/__init__.py` | Marks the training package. |
| `src/train/baselines.py` | Tabular baselines, including random forest, with temporal cross-validation. |
| `src/train/dual_head.py` | Graph model on connected nodes and random forest on isolates. |
| `src/train/gnn.py` | GCN baseline on the same leakage-safe pipeline. |
| `src/train/gnn_common.py` | Shared node, graph, and training utilities. |
| `src/train/gnn_ladder.py` | Refinement ladder from a simpler model to a more expressive one. |
| `src/train/gnn_models.py` | MLP, GCN, GraphSAGE, and GAT used in that ladder. |
| `src/train/ood_eval.py` | Temporal-versus-spatial gap for PhysAgg runs A0–A3. |
| `src/train/physagg.py` | Manuscript entry point for ablations A0, A1, A2, A2e, and A3. |
| `src/train/pignn.py` | Earlier physics-informed GNN: ablation, lambda grid, and spatial split. |

### Released scores, temporal split, non-isolated subgraph

Each `metrics_*.json` file stores the scores of one trained run. `summary.json` stores the five-seed mean and standard deviation of PR-AUC and F1. `03_reproduce_physagg.ipynb` reads this folder.

| File | What it is for |
|:---|:---|
| `experiments/outputs/physagg_power/temporal_noniso/metrics_A0_s42.json` | A0, seed 42. |
| `experiments/outputs/physagg_power/temporal_noniso/metrics_A0_s0.json` | A0, seed 0. |
| `experiments/outputs/physagg_power/temporal_noniso/metrics_A0_s1.json` | A0, seed 1. |
| `experiments/outputs/physagg_power/temporal_noniso/metrics_A0_s2.json` | A0, seed 2. |
| `experiments/outputs/physagg_power/temporal_noniso/metrics_A0_s3.json` | A0, seed 3. |
| `experiments/outputs/physagg_power/temporal_noniso/metrics_A1_s42.json` | A1, seed 42. |
| `experiments/outputs/physagg_power/temporal_noniso/metrics_A1_s0.json` | A1, seed 0. |
| `experiments/outputs/physagg_power/temporal_noniso/metrics_A1_s1.json` | A1, seed 1. |
| `experiments/outputs/physagg_power/temporal_noniso/metrics_A1_s2.json` | A1, seed 2. |
| `experiments/outputs/physagg_power/temporal_noniso/metrics_A1_s3.json` | A1, seed 3. |
| `experiments/outputs/physagg_power/temporal_noniso/metrics_A2_s42.json` | A2, seed 42. |
| `experiments/outputs/physagg_power/temporal_noniso/metrics_A2_s0.json` | A2, seed 0. |
| `experiments/outputs/physagg_power/temporal_noniso/metrics_A2_s1.json` | A2, seed 1. |
| `experiments/outputs/physagg_power/temporal_noniso/metrics_A2_s2.json` | A2, seed 2. |
| `experiments/outputs/physagg_power/temporal_noniso/metrics_A2_s3.json` | A2, seed 3. |
| `experiments/outputs/physagg_power/temporal_noniso/metrics_A2e_s42.json` | A2e, seed 42. |
| `experiments/outputs/physagg_power/temporal_noniso/metrics_A2e_s0.json` | A2e, seed 0. |
| `experiments/outputs/physagg_power/temporal_noniso/metrics_A2e_s1.json` | A2e, seed 1. |
| `experiments/outputs/physagg_power/temporal_noniso/metrics_A2e_s2.json` | A2e, seed 2. |
| `experiments/outputs/physagg_power/temporal_noniso/metrics_A2e_s3.json` | A2e, seed 3. |
| `experiments/outputs/physagg_power/temporal_noniso/metrics_A3_s42.json` | A3, seed 42. |
| `experiments/outputs/physagg_power/temporal_noniso/metrics_A3_s0.json` | A3, seed 0. |
| `experiments/outputs/physagg_power/temporal_noniso/metrics_A3_s1.json` | A3, seed 1. |
| `experiments/outputs/physagg_power/temporal_noniso/metrics_A3_s2.json` | A3, seed 2. |
| `experiments/outputs/physagg_power/temporal_noniso/metrics_A3_s3.json` | A3, seed 3. |
| `experiments/outputs/physagg_power/temporal_noniso/metrics_RF_fair.json` | Random forest on the same connected mask. |
| `experiments/outputs/physagg_power/temporal_noniso/summary.json` | Five-seed mean and standard deviation for A0, A1, A2, A2e, and A3. |

### Released scores, municipality hold-out (Suchiapa)

Seed 42. Same run codes. This is the spatial check in the manuscript.

| File | What it is for |
|:---|:---|
| `experiments/outputs/physagg_power/spatial_suchiapa/metrics_A0_s42.json` | A0 on the Suchiapa hold-out. |
| `experiments/outputs/physagg_power/spatial_suchiapa/metrics_A1_s42.json` | A1 on the Suchiapa hold-out. |
| `experiments/outputs/physagg_power/spatial_suchiapa/metrics_A2_s42.json` | A2 on the Suchiapa hold-out. |
| `experiments/outputs/physagg_power/spatial_suchiapa/metrics_A2e_s42.json` | A2e on the Suchiapa hold-out. |
| `experiments/outputs/physagg_power/spatial_suchiapa/metrics_A3_s42.json` | A3 on the Suchiapa hold-out. |
| `experiments/outputs/physagg_power/spatial_suchiapa/metrics_RF_fair.json` | Random forest on the same hold-out mask. |
| `experiments/outputs/physagg_power/spatial_suchiapa/summary.json` | Scores gathered for that hold-out. |

## Privacy

BSSID and device IDs are salted SHA-256 hashes. The salt is not distributed. Coordinates are rounded to 3 decimals. Raw SSIDs are omitted. Heuristic labels are threat-risk proxies, not adjudicated attacks.

## Licence

Code: MIT. Dataset: CC BY 4.0.
