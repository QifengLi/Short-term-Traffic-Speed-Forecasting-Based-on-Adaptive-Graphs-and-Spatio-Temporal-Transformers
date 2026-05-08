# 06 No-dynamic Real-data Evidence 2026-05-08

## Purpose

This note completes the previously missing real-data no-dynamic ablation. Earlier project materials only had demo-level no-dynamic evidence, so the dynamic graph could not be isolated on the real PEMSD7-228 and METR-LA benchmark slices.

## Configs

- `configs/pemsd7_agstt_no_dynamic.yaml`
- `configs/metr_la_agstt_no_dynamic.yaml`

Both configs keep the standard-budget setting:

- `max_timesteps: 4032`
- `seq_len: 12`
- `pred_len: 12`
- `epochs: 6`
- `use_static_graph: true`
- `use_adaptive_graph: true`
- `use_dynamic_graph: false`

## Results

| Dataset | Variant | MAE | RMSE | MAPE (%) | Metrics file |
|---|---|---:|---:|---:|---|
| PEMSD7-228 | full standard AG-STT | 3.6007 | 6.8320 | 9.34 | `outputs/pemsd7_agstt_metrics.json` |
| PEMSD7-228 | no-dynamic | 3.5444 | 6.8014 | 9.14 | `outputs/improvements/pemsd7_agstt_no_dynamic_metrics.json` |
| METR-LA | full standard AG-STT | 4.3751 | 8.8248 | 12.24 | `outputs/metr_la_agstt_metrics.json` |
| METR-LA | no-dynamic | 4.3529 | 8.7726 | 12.62 | `outputs/improvements/metr_la_agstt_no_dynamic_metrics.json` |

## Interpretation

The no-dynamic ablation is now backed by real standard-budget runs. Removing the dynamic graph slightly improves MAE and RMSE on both datasets in these runs, while METR-LA MAPE becomes worse. Therefore, the dynamic graph should be described conservatively as an implemented window-conditioned mechanism with mixed evidence, not as a uniformly positive component.

The stronger component claim remains the adaptive graph result, especially the long-budget PEMSD7-228 comparison where full AG-STT reaches MAE 3.1828 and no-adaptive reaches MAE 3.3765.

## Reproduction

```bash
python scripts/run_experiments.py --configs configs/pemsd7_agstt_no_dynamic.yaml configs/metr_la_agstt_no_dynamic.yaml --output-dir outputs/improvements
python scripts/run_experiments.py --skip-train --configs configs/pemsd7_agstt_long_budget.yaml configs/pemsd7_agstt_no_adaptive.yaml configs/pemsd7_agstt_no_dynamic.yaml configs/metr_la_agstt_long_budget.yaml configs/metr_la_agstt_no_adaptive.yaml configs/metr_la_agstt_no_dynamic.yaml --output-dir outputs/improvements
```

Generated checkpoints:

- `checkpoints/pemsd7_228_pemsd7_agstt_no_dynamic_best.pt`
- `checkpoints/metr_la_metr_la_agstt_no_dynamic_best.pt`

