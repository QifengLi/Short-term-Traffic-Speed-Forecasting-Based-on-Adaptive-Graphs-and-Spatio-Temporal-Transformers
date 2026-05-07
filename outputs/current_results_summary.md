# 当前毕业设计完成情况（最终闭环版）

- 报告日期：2026-04-27
- 课题：Short-term Traffic Flow Forecasting Based on Adaptive Graphs and Spatio-Temporal Transformers

## 完成状态
- real_datasets_connected: True
- strong_baselines_completed: True
- statistical_significance_completed: True
- visualizations_completed: True
- report_and_defense_materials_updated: True
- fair_budget_completed: True
- fusion_optimization_completed: True
- robustness_n5_completed: True
- robustness_v3_n5_completed: True
- cross_split_v3_completed: True
- cross_split_rolling_completed: True
- seed_level_significance_v3_completed: True
- all_pending_items_closed: True

## 数据集：pemsd7_228
- 数据形状=[12672, 228, 1]，范围=[3.0000, 82.6000]，均值=58.8892，标准差=13.4833
- 邻接矩阵=[228, 228]，非零=18620，稀疏度=0.6418

| Run Name | Model | MAE | RMSE | MAPE(%) |
|---|---|---|---|---|
| pemsd7_dcrnn_baseline | dcrnn | 3.5898 | 7.0305 | 8.99 |
| pemsd7_agstt_full | agstt | 3.6007 | 6.8320 | 9.34 |
| pemsd7_lstm_baseline | lstm | 3.7110 | 7.4007 | 9.25 |
| pemsd7_stgcn_baseline | stgcn | 4.0883 | 7.3853 | 11.53 |
| pemsd7_linear_baseline | linear | 4.3167 | 8.1672 | 11.26 |

## 数据集：metr_la
- 数据形状=[34272, 207, 1]，范围=[0.0000, 70.0000]，均值=53.7190，标准差=20.2614
- 邻接矩阵=[207, 207]，非零=1722，稀疏度=0.9598

| Run Name | Model | MAE | RMSE | MAPE(%) |
|---|---|---|---|---|
| metr_la_dcrnn_baseline | dcrnn | 4.3335 | 8.9771 | 11.90 |
| metr_la_agstt_full | agstt | 4.3751 | 8.8248 | 12.24 |
| metr_la_lstm_baseline | lstm | 4.3854 | 8.9903 | 12.16 |
| metr_la_stgcn_baseline | stgcn | 4.5435 | 8.8454 | 12.62 |
| metr_la_linear_baseline | linear | 5.1071 | 9.9206 | 13.67 |

## 补强实验与闭环证据

| Run Name | Model | MAE |
|---|---|---|
| pemsd7_agstt_long_budget_fair | agstt | 3.1828 |
| pemsd7_dcrnn_long_budget_fair | dcrnn | 3.2543 |
| pemsd7_lstm_long_budget_fair | lstm | 3.3995 |
| pemsd7_stgcn_long_budget_fair | stgcn | 3.4658 |
| pemsd7_linear_long_budget_fair | linear | 3.6610 |
| metr_la_agstt_long_budget_fair | agstt | 3.7914 |
| metr_la_stgcn_long_budget_fair | stgcn | 3.8832 |
| metr_la_dcrnn_long_budget_fair | dcrnn | 3.8921 |
| metr_la_lstm_long_budget_fair | lstm | 3.9287 |
| metr_la_linear_long_budget_fair | linear | 4.2371 |

| Run Name | MAE |
|---|---|
| pemsd7_agstt_fusion_constrained_v3 | 3.5503 |
| pemsd7_agstt_fusion_constrained_v2 | 3.5571 |
| pemsd7_agstt_fusion_constrained_v1 | 3.5573 |
| metr_la_agstt_fusion_constrained_v3 | 4.2797 |
| metr_la_agstt_fusion_constrained_v1 | 4.2835 |
| metr_la_agstt_fusion_constrained_v2 | 4.2857 |

| Dataset | Run Name | N | MAE(mean±std) |
|---|---|---|---|
| metr_la | metr_la_agstt_full | 5 | 4.3081±0.0455 |
| metr_la | metr_la_dcrnn_baseline | 5 | 4.3332±0.0155 |
| pemsd7_228 | pemsd7_agstt_full | 5 | 3.5999±0.0182 |
| pemsd7_228 | pemsd7_dcrnn_baseline | 5 | 3.6077±0.0120 |

| Dataset | Run Name | N | MAE(mean±std) |
|---|---|---|---|
| metr_la | metr_la_agstt_fusion_constrained_v3 | 5 | 4.2851±0.0308 |
| pemsd7_228 | pemsd7_agstt_fusion_constrained_v3 | 5 | 3.6077±0.0462 |

| Dataset | N | Mean(ref-ch) | 95% CI | p-value | Significant |
|---|---|---|---|---|---|
| metr_la | 5 | -0.048122 | [-0.065103, -0.031140] | 0.090909 | False |
| pemsd7_228 | 5 | -0.000049 | [-0.029722, 0.033398] | 1.000000 | False |

| Dataset | N Splits | AG-STT Wins | DCRNN Wins | Mean Delta(AGSTT-DCRNN) |
|---|---|---|---|---|
| metr_la | 3 | 1 | 2 | 0.0365 |
| pemsd7_228 | 3 | 2 | 1 | 0.4253 |

## 备注
- METR-LA evaluation uses null_val=0.0 to avoid invalid MAPE inflation caused by zero-speed ground truth.
- GPU environment is enabled (CUDA PyTorch) and used in training/evaluation when device=auto.
- Cross-split rolling evaluation now includes s1/s2/s3 windows with paired model comparison.
- Seed-level significance for AG-STT(v3) vs DCRNN is included in outputs/improvements/seed_level_significance_v3_vs_dcrnn.*
