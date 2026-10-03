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

PhysAgg-R ablations (optional; needs PyTorch Geometric):

```bash
python -m src.train.physagg --run A2e --seeds 42,0,1,2,3 --split temporal --subgraph non_isolate
```

## Privacy

BSSID/device IDs are salted SHA-256 hashes (salt not distributed). Coordinates are rounded to 3 decimals. Raw SSIDs are omitted.

## Licence

Code: MIT (unless otherwise noted). Dataset: CC BY 4.0.
