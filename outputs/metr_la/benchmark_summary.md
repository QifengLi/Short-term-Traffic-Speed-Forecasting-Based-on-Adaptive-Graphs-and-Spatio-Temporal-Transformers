# Benchmark Summary

| Config | Run Name | Model | MAE | RMSE | MAPE(%) | Metrics File |
|---|---|---|---:|---:|---:|---|
| configs\metr_la_dcrnn_baseline.yaml | metr_la_dcrnn_baseline | dcrnn | 4.3335 | 8.9771 | 11.90 | outputs\metr_la_dcrnn_metrics.json |
| configs\metr_la_agstt.yaml | metr_la_agstt_full | agstt | 4.3751 | 8.8248 | 12.24 | outputs\metr_la_agstt_metrics.json |
| configs\metr_la_lstm_baseline.yaml | metr_la_lstm_baseline | lstm | 4.3854 | 8.9903 | 12.16 | outputs\metr_la_lstm_metrics.json |
| configs\metr_la_stgcn_baseline.yaml | metr_la_stgcn_baseline | stgcn | 4.5435 | 8.8454 | 12.62 | outputs\metr_la_stgcn_metrics.json |
| configs\metr_la_linear_baseline.yaml | metr_la_linear_baseline | linear | 5.1071 | 9.9206 | 13.67 | outputs\metr_la_linear_metrics.json |