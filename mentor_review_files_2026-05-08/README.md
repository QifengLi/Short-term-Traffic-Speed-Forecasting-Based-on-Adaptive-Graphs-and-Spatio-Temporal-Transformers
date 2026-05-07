# Short-term Traffic Flow Forecasting Based on Adaptive Graphs and Spatio-Temporal Transformers

This repository contains a complete graduation-project pipeline for short-term traffic forecasting.

## Features

- AG-STT main model (adaptive graph + spatio-temporal transformer)
- Baselines: `dcrnn`, `stgcn`, `lstm`, `linear`
- Ablation configs for AG-STT (`no_dynamic`, `no_adaptive`, `static_only`)
- End-to-end scripts for data preparation, training, evaluation, significance testing, and plotting
- Auto-export of `json/csv/md` benchmark reports and thesis-ready summary files

## Install

```bash
pip install -r requirements.txt
```

## Recommended Environment (for reproducibility)

- Python `3.11`
- PyTorch `2.11.0+cu128` (CUDA optional; `device=auto` will fall back to CPU)
- `requirements.txt` currently uses lower-bound constraints (`>=`). For strict reproducibility, use the versions above.

## Real-Data Pipeline

### 1) Prepare datasets

```bash
python scripts/prepare_pemsd7_dataset.py
python scripts/prepare_metr_la_dataset.py
```

Data prerequisites (manual placement under `data/raw/`):

- `data/raw/PeMSD7_Full/PeMSD7_V_228.csv`
- `data/raw/PeMSD7_Full/PeMSD7_W_228.csv`
- `data/raw/metr-la.h5`
- `data/raw/adj_mx.pkl`

Reference sources:

- METR-LA / `adj_mx.pkl`: DCRNN public data links
- PEMSD7 files: common STGCN/traffic-forecasting public releases

### 2) Run benchmarks

```bash
python scripts/run_experiments.py --configs configs/pemsd7_agstt.yaml configs/pemsd7_lstm_baseline.yaml configs/pemsd7_linear_baseline.yaml configs/pemsd7_stgcn_baseline.yaml configs/pemsd7_dcrnn_baseline.yaml --output-dir outputs/pemsd7
python scripts/run_experiments.py --configs configs/metr_la_agstt.yaml configs/metr_la_lstm_baseline.yaml configs/metr_la_linear_baseline.yaml configs/metr_la_stgcn_baseline.yaml configs/metr_la_dcrnn_baseline.yaml --output-dir outputs/metr_la
```

### 3) Significance tests

```bash
python scripts/significance_test.py --reference outputs/pemsd7_agstt_predictions.npz --challengers outputs/pemsd7_dcrnn_predictions.npz outputs/pemsd7_lstm_predictions.npz outputs/pemsd7_stgcn_predictions.npz outputs/pemsd7_linear_predictions.npz --out-json outputs/pemsd7/significance_test.json --out-md outputs/pemsd7/significance_test.md
python scripts/significance_test.py --reference outputs/metr_la_agstt_predictions.npz --challengers outputs/metr_la_dcrnn_predictions.npz outputs/metr_la_lstm_predictions.npz outputs/metr_la_stgcn_predictions.npz outputs/metr_la_linear_predictions.npz --out-json outputs/metr_la/significance_test.json --out-md outputs/metr_la/significance_test.md
```

### 4) Generate figures

```bash
python scripts/plot_benchmark_results.py --benchmark-json outputs/pemsd7/benchmark_summary.json --metrics-jsons outputs/pemsd7_agstt_metrics.json outputs/pemsd7_dcrnn_metrics.json outputs/pemsd7_lstm_metrics.json outputs/pemsd7_stgcn_metrics.json outputs/pemsd7_linear_metrics.json --out-dir outputs/pemsd7/figures
python scripts/plot_benchmark_results.py --benchmark-json outputs/metr_la/benchmark_summary.json --metrics-jsons outputs/metr_la_agstt_metrics.json outputs/metr_la_dcrnn_metrics.json outputs/metr_la_lstm_metrics.json outputs/metr_la_stgcn_metrics.json outputs/metr_la_linear_metrics.json --out-dir outputs/metr_la/figures
```

### 5) Generate final summary

```bash
python scripts/generate_final_results_summary.py
```

## Core Outputs

- `outputs/pemsd7/benchmark_summary.{json,csv,md}`
- `outputs/metr_la/benchmark_summary.{json,csv,md}`
- `outputs/pemsd7/significance_test.{json,md}`
- `outputs/metr_la/significance_test.{json,md}`
- `outputs/current_results_summary.{json,md}`

## Minimal Run (quick sanity check)

If pretrained checkpoints already exist:

```bash
python evaluate.py --config configs/metr_la_agstt.yaml --checkpoint checkpoints/metr_la_metr_la_agstt_full_best.pt
```

If checkpoints are missing, run training first:

```bash
python train.py --config configs/metr_la_agstt.yaml
python evaluate.py --config configs/metr_la_agstt.yaml
```

## Data Format

Traffic file (`.npz`) must contain an array with shape `[T, N, F]` (default key: `data`).

Optional adjacency file (`.npy`/`.csv`) must be shape `[N, N]`.

## Security Notes

- `evaluate.py` uses safe checkpoint loading by default.
- Use `--allow-unsafe-checkpoint` only for trusted files.
