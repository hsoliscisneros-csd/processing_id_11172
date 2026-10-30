# Fragmentation-Conditioned WiFi Threat Detection under Crowdsensing

Companion code and anonymised data for the IEEE Latin America Transactions manuscript **ID 11172**.

**Repository:** https://github.com/hsoliscisneros-csd/processing_id_11172

## Authors

José A. Macías-Montoya, Horacio I. Solís-Cisneros, and Josué R. Velázquez-González.

Data Intelligence, Generative AI and Resilient Cybersecurity Research Group, Universidad Politécnica de Chiapas, Tuxtla Gutiérrez, Chiapas, Mexico.

## File list

- `.gitignore`
- `LICENSE`
- `README.md`
- `requirements.txt`
- `data/DATA_DICTIONARY.md`
- `data/datos_wifi_curated_anon.parquet`
- `data/qc_report.json`
- `experiments/EXPERIMENT_LOG.md`
- `experiments/configs/dual_head_v0.yaml`
- `experiments/configs/physagg_power.yaml`
- `experiments/configs/physagg_v0.yaml`
- `experiments/configs/pignn_ablation.yaml`
- `experiments/configs/pignn_lambda_grid.yaml`
- `experiments/configs/pignn_spatial.yaml`
- `experiments/configs/pignn_spatial_grid.yaml`
- `experiments/configs/pignn_spatial_loo.yaml`
- `experiments/configs/pignn_spatial_loo_best.yaml`
- `experiments/configs/pignn_spatial_loss.yaml`
- `experiments/configs/pignn_v1.yaml`
- `experiments/manifests/spatial_ood_v0.json`
- `experiments/manifests/spatial_ood_v0.yaml`
- `experiments/outputs/physagg_power/spatial_suchiapa/metrics_A0_s42.json`
- `experiments/outputs/physagg_power/spatial_suchiapa/metrics_A1_s42.json`
- `experiments/outputs/physagg_power/spatial_suchiapa/metrics_A2_s42.json`
- `experiments/outputs/physagg_power/spatial_suchiapa/metrics_A2e_s42.json`
- `experiments/outputs/physagg_power/spatial_suchiapa/metrics_A3_s42.json`
- `experiments/outputs/physagg_power/spatial_suchiapa/metrics_RF_fair.json`
- `experiments/outputs/physagg_power/spatial_suchiapa/summary.json`
- `experiments/outputs/physagg_power/temporal_noniso/metrics_A0_s0.json`
- `experiments/outputs/physagg_power/temporal_noniso/metrics_A0_s1.json`
- `experiments/outputs/physagg_power/temporal_noniso/metrics_A0_s2.json`
- `experiments/outputs/physagg_power/temporal_noniso/metrics_A0_s3.json`
- `experiments/outputs/physagg_power/temporal_noniso/metrics_A0_s42.json`
- `experiments/outputs/physagg_power/temporal_noniso/metrics_A1_s0.json`
- `experiments/outputs/physagg_power/temporal_noniso/metrics_A1_s1.json`
- `experiments/outputs/physagg_power/temporal_noniso/metrics_A1_s2.json`
- `experiments/outputs/physagg_power/temporal_noniso/metrics_A1_s3.json`
- `experiments/outputs/physagg_power/temporal_noniso/metrics_A1_s42.json`
- `experiments/outputs/physagg_power/temporal_noniso/metrics_A2_s0.json`
- `experiments/outputs/physagg_power/temporal_noniso/metrics_A2_s1.json`
- `experiments/outputs/physagg_power/temporal_noniso/metrics_A2_s2.json`
- `experiments/outputs/physagg_power/temporal_noniso/metrics_A2_s3.json`
- `experiments/outputs/physagg_power/temporal_noniso/metrics_A2_s42.json`
- `experiments/outputs/physagg_power/temporal_noniso/metrics_A2e_s0.json`
- `experiments/outputs/physagg_power/temporal_noniso/metrics_A2e_s1.json`
- `experiments/outputs/physagg_power/temporal_noniso/metrics_A2e_s2.json`
- `experiments/outputs/physagg_power/temporal_noniso/metrics_A2e_s3.json`
- `experiments/outputs/physagg_power/temporal_noniso/metrics_A2e_s42.json`
- `experiments/outputs/physagg_power/temporal_noniso/metrics_A3_s0.json`
- `experiments/outputs/physagg_power/temporal_noniso/metrics_A3_s1.json`
- `experiments/outputs/physagg_power/temporal_noniso/metrics_A3_s2.json`
- `experiments/outputs/physagg_power/temporal_noniso/metrics_A3_s3.json`
- `experiments/outputs/physagg_power/temporal_noniso/metrics_A3_s42.json`
- `experiments/outputs/physagg_power/temporal_noniso/metrics_RF_fair.json`
- `experiments/outputs/physagg_power/temporal_noniso/summary.json`
- `notebooks/01_eda_wifi.ipynb`
- `notebooks/02_paper_results_view.ipynb`
- `notebooks/03_reproduce_physagg.ipynb`
- `src/__init__.py`
- `src/evaluation/__init__.py`
- `src/evaluation/metrics.py`
- `src/evaluation/significance.py`
- `src/evaluation/splits.py`
- `src/evaluation/thresholds.py`
- `src/features/__init__.py`
- `src/features/build.py`
- `src/graphs/__init__.py`
- `src/graphs/build.py`
- `src/graphs/sensitivity.py`
- `src/train/__init__.py`
- `src/train/baselines.py`
- `src/train/dual_head.py`
- `src/train/gnn.py`
- `src/train/gnn_common.py`
- `src/train/gnn_ladder.py`
- `src/train/gnn_models.py`
- `src/train/ood_eval.py`
- `src/train/physagg.py`
- `src/train/pignn.py`

## Quick start

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/01_eda_wifi.ipynb
```

### Reproduce PhysAgg-R

`notebooks/03_reproduce_physagg.ipynb` loads the released metric tables under
`experiments/outputs/physagg_power/`. To retrain from scratch, set
`RUN_TRAINING = True` in that notebook, or use the CLI:

```bash
python -m src.graphs.build --input data/datos_wifi_curated_anon.parquet --output data/graphs
python -m src.train.physagg --run all --seeds 42,0,1,2,3 --split temporal --subgraph non_isolate --mcnemar A2e,A0 --with-rf
```

Needs PyTorch Geometric for GNN runs. Outputs land in `experiments/outputs/` (gitignored).

## Privacy

BSSID/device IDs are salted SHA-256 hashes (salt not distributed). Coordinates are rounded to 3 decimals. Raw SSIDs are omitted.

## Licence

Code: MIT (unless otherwise noted). Dataset: CC BY 4.0.
