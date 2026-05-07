# Seed-Level Significance Summary

| Dataset | N Seeds | Mean(ref-ch) | Std(diff) | 95% CI | p-value | Ref Wins | Ch Wins | Significant(0.05) | Interpretation |
|---|---:|---:|---:|---|---:|---:|---:|---|---|
| metr_la | 5 | -0.048122 | 0.019154 | [-0.065103, -0.031140] | 0.090909 | 5 | 0 | False | agstt_fusion_constrained_v3 better |
| pemsd7_228 | 5 | -0.000049 | 0.037903 | [-0.029722, 0.033398] | 1.000000 | 2 | 3 | False | agstt_fusion_constrained_v3 better |

| Dataset | Seed | Ref MAE | Ch MAE | Ref-Ch | Better |
|---|---:|---:|---:|---:|---|
| metr_la | 7 | 4.244815 | 4.317163 | -0.072349 | agstt_fusion_constrained_v3 |
| metr_la | 42 | 4.279741 | 4.323475 | -0.043735 | agstt_fusion_constrained_v3 |
| metr_la | 123 | 4.264539 | 4.333080 | -0.068541 | agstt_fusion_constrained_v3 |
| metr_la | 2026 | 4.333537 | 4.362163 | -0.028626 | agstt_fusion_constrained_v3 |
| metr_la | 3407 | 4.302780 | 4.330138 | -0.027358 | agstt_fusion_constrained_v3 |
| pemsd7_228 | 7 | 3.579956 | 3.616032 | -0.036076 | agstt_fusion_constrained_v3 |
| pemsd7_228 | 42 | 3.550269 | 3.589872 | -0.039603 | agstt_fusion_constrained_v3 |
| pemsd7_228 | 123 | 3.619420 | 3.611922 | 0.007498 | dcrnn_baseline |
| pemsd7_228 | 2026 | 3.687745 | 3.622555 | 0.065189 | dcrnn_baseline |
| pemsd7_228 | 3407 | 3.600916 | 3.598168 | 0.002748 | dcrnn_baseline |
