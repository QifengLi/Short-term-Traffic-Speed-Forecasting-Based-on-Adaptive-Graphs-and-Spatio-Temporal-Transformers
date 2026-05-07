# Benchmark Summary

| Config | Run Name | Model | MAE | RMSE | MAPE(%) | Metrics File |
|---|---|---|---:|---:|---:|---|
| configs\pemsd7_dcrnn_baseline.yaml | pemsd7_dcrnn_baseline | dcrnn | 3.5898 | 7.0305 | 8.99 | outputs\pemsd7_dcrnn_metrics.json |
| configs\pemsd7_agstt.yaml | pemsd7_agstt_full | agstt | 3.6007 | 6.8320 | 9.34 | outputs\pemsd7_agstt_metrics.json |
| configs\pemsd7_lstm_baseline.yaml | pemsd7_lstm_baseline | lstm | 3.7110 | 7.4007 | 9.25 | outputs\pemsd7_lstm_metrics.json |
| configs\pemsd7_stgcn_baseline.yaml | pemsd7_stgcn_baseline | stgcn | 4.0883 | 7.3853 | 11.53 | outputs\pemsd7_stgcn_metrics.json |
| configs\pemsd7_linear_baseline.yaml | pemsd7_linear_baseline | linear | 4.3167 | 8.1672 | 11.26 | outputs\pemsd7_linear_metrics.json |