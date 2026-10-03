# Reproducibility package — Fragmentation-Conditioned WiFi Threat Detection

Companion code and anonymised data for IEEE Latin America Transactions manuscript **ID 11172**.

**Repository:** https://github.com/hsoliscisneros-csd/processing_id_11172

## Contents

| Path | Description |
|:---|:---|
| `data/datos_wifi_curated_anon.parquet` | Anonymised curated scans (53,371 rows) |
| `data/DATA_DICTIONARY.md` | Field descriptions |
| `data/qc_report.json` | QC summary for the released parquet |
| `notebooks/01_eda_wifi.ipynb` | EDA on the released parquet (executed outputs) |
| `notebooks/02_paper_results_view.ipynb` | Recomputes isolation + public-column RF ablation from the package |
| `notebooks/03_reproduce_physagg.ipynb` | Conductor notebook (B) that calls the PhysAgg CLI (A) |
| `src/` | Minimal training/evaluation source used in the paper |
| `experiments/configs`, `experiments/manifests` | Experiment configuration |
| `experiments/EXPERIMENT_LOG.md` | Short English experiment notes for the paper |

## Quick start

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/01_eda_wifi.ipynb
```

### Reproduce PhysAgg-R (B + A)

1. Open `notebooks/03_reproduce_physagg.ipynb`.
2. Set `RUN_TRAINING = True`.
3. Run all cells (builds `data/graphs/physical`, then launches the paper CLI recipes).

Equivalent CLI:

```bash
python -m src.graphs.build --input data/datos_wifi_curated_anon.parquet --output data/graphs
python -m src.train.physagg --run all --seeds 42,0,1,2,3 --split temporal --subgraph non_isolate --mcnemar A2e,A0 --with-rf
```

Needs PyTorch Geometric for GNN runs. Outputs land in `experiments/outputs/` (gitignored).

## Privacy

BSSID/device IDs are salted SHA-256 hashes (salt not distributed). Coordinates are rounded to 3 decimals. Raw SSIDs are omitted.

## Licence

Code: MIT (unless otherwise noted). Dataset: CC BY 4.0.
