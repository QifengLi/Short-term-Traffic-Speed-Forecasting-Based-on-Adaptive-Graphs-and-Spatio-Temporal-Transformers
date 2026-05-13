# Short-term Traffic Speed Forecasting Based on Adaptive Graphs and Spatio-Temporal Transformers

This repository contains the code and final evidence materials for a graduation project on short-term traffic-state forecasting with adaptive graph learning and spatio-temporal Transformer blocks.

The implemented model is AG-STT, a compact Adaptive Graph Spatio-Temporal Transformer. It combines a static road graph, a learnable adaptive graph, an input-conditioned dynamic graph, temporal self-attention, graph diffusion convolution and a direct multi-step prediction head.

## Repository Structure

```text
src/                 Core data, model and utility modules
configs/             Experiment configurations
scripts/             Dataset preparation, experiment and analysis utilities
paper/               Final report document and supporting evidence
train.py             Training entry point
evaluate.py          Evaluation entry point
requirements.txt     Python dependencies
```

Large raw datasets, checkpoints and prediction arrays are not included in the repository. The report explains the data sources and the evidence files included under `paper/`.

## Environment

The project was developed with Python 3.11 and PyTorch. Install dependencies with:

```bash
pip install -r requirements.txt
```

## Datasets

The experiments use PEMSD7-228 and METR-LA. Raw dataset files should be placed locally under `data/raw/` before running the preparation utilities.

Expected raw files:

```text
data/raw/PeMSD7_Full/PeMSD7_V_228.csv
data/raw/PeMSD7_Full/PeMSD7_W_228.csv
data/raw/metr-la.h5
data/raw/adj_mx.pkl
```

Prepared `.npz` and adjacency `.npy` files are intentionally ignored by Git because they are derived data files.

## Basic Usage

Prepare datasets:

```bash
python scripts/prepare_pemsd7_dataset.py
python scripts/prepare_metr_la_dataset.py
```

Train one model:

```bash
python train.py --config configs/pemsd7_agstt.yaml
```

Evaluate one model:

```bash
python evaluate.py --config configs/pemsd7_agstt.yaml
```

Run a grouped benchmark:

```bash
python scripts/run_experiments.py --configs configs/pemsd7_agstt.yaml configs/pemsd7_dcrnn_baseline.yaml configs/pemsd7_stgcn_baseline.yaml configs/pemsd7_lstm_baseline.yaml configs/pemsd7_linear_baseline.yaml
```

## Reported Results

The final report presents controlled experiments on PEMSD7-228 and METR-LA, including baseline comparisons, ablations, fair-budget experiments, paired significance tests, seed-level checks and rolling split evidence.

The paper avoids claiming that AG-STT uniformly outperforms all baselines. Under the standard budget, AG-STT is competitive, while the DCRNN-style baseline remains slightly stronger in MAE. Supplementary fair-budget and constrained-fusion experiments provide additional evidence about when AG-STT improves.

## Notes

- Raw datasets and model checkpoints are excluded to keep the repository lightweight.
- Saved evidence summaries and final report assets are provided under `paper/`.
- The code is organised for reproducibility rather than large-scale production deployment.

