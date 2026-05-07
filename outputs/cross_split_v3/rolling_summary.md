# Cross-Split Rolling Summary

| Dataset | Split | AG-STT MAE | DCRNN MAE | Delta(AGSTT-DCRNN) | Better |
|---|---|---:|---:|---:|---|
| metr_la | s1 | 5.1402 | 5.0804 | 0.0598 | dcrnn |
| metr_la | s2 | 4.5890 | 4.6302 | -0.0412 | agstt |
| metr_la | s3 | 4.5798 | 4.4890 | 0.0908 | dcrnn |
| pemsd7_228 | s1 | 3.6417 | 3.6765 | -0.0348 | agstt |
| pemsd7_228 | s2 | 3.3196 | 3.3645 | -0.0449 | agstt |
| pemsd7_228 | s3 | 6.6260 | 5.2705 | 1.3555 | dcrnn |

| Dataset | Model | N Splits | Mean MAE | Median MAE | Std MAE | Best MAE | Worst MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| metr_la | dcrnn | 3 | 4.7332 | 4.6302 | 0.2522 | 4.4890 | 5.0804 |
| metr_la | agstt | 3 | 4.7696 | 4.5890 | 0.2620 | 4.5798 | 5.1402 |
| pemsd7_228 | dcrnn | 3 | 4.1039 | 3.6765 | 0.8347 | 3.3645 | 5.2705 |
| pemsd7_228 | agstt | 3 | 4.5291 | 3.6417 | 1.4886 | 3.3196 | 6.6260 |

| Dataset | N Splits | AG-STT Wins | DCRNN Wins | Ties | Mean Delta(AGSTT-DCRNN) |
|---|---:|---:|---:|---:|---:|
| metr_la | 3 | 1 | 2 | 0 | 0.0365 |
| pemsd7_228 | 3 | 2 | 1 | 0 | 0.4253 |
