from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_model_from_prediction_path(pred_path: str) -> str:
    lower = pred_path.lower()
    for m in ("dcrnn", "lstm", "stgcn", "linear"):
        if m in lower:
            return m
    return Path(pred_path).stem


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def copy_if_exists(src: Path, dst: Path) -> bool:
    if src.exists():
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)
        return True
    return False


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", type=Path, default=Path("outputs/current_results_summary.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/experiment_summary_2026-04-28"))
    parser.add_argument("--zip", action="store_true")
    args = parser.parse_args()

    summary = load_json(args.summary_json)

    out_dir = args.out_dir
    figs_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    refs_dir = out_dir / "refs"
    ensure_dir(out_dir)
    ensure_dir(figs_dir)
    ensure_dir(tables_dir)
    ensure_dir(refs_dir)

    # ---------- dataset overview ----------
    ds_rows = []
    for ds_name, ds in summary["datasets"].items():
        null_val = ""
        if ds.get("significance_vs_agstt"):
            null_val = ds["significance_vs_agstt"][0].get("null_val", "")
        ds_rows.append(
            {
                "dataset": ds_name,
                "timesteps": ds["data_shape"][0],
                "nodes": ds["data_shape"][1],
                "features": ds["data_shape"][2],
                "data_min": ds["data_min"],
                "data_max": ds["data_max"],
                "data_mean": ds["data_mean"],
                "data_std": ds["data_std"],
                "adj_rows": ds["adj_shape"][0],
                "adj_cols": ds["adj_shape"][1],
                "adj_nonzero": ds["adj_nonzero"],
                "adj_density": ds["adj_density"],
                "null_val_eval": null_val,
            }
        )
    write_csv(
        tables_dir / "dataset_overview.csv",
        ds_rows,
        [
            "dataset",
            "timesteps",
            "nodes",
            "features",
            "data_min",
            "data_max",
            "data_mean",
            "data_std",
            "adj_rows",
            "adj_cols",
            "adj_nonzero",
            "adj_density",
            "null_val_eval",
        ],
    )

    # ---------- benchmark ----------
    bench_rows = []
    for ds_name, ds in summary["datasets"].items():
        for row in ds["benchmark"]:
            bench_rows.append(
                {
                    "dataset": ds_name,
                    "model": row["model"],
                    "run_name": row["run_name"],
                    "mae": row["mae"],
                    "rmse": row["rmse"],
                    "mape_percent": row["mape_percent"],
                    "config": row["config"],
                    "metrics_path": row["metrics_path"],
                }
            )
    write_csv(
        tables_dir / "main_benchmark_combined.csv",
        bench_rows,
        ["dataset", "model", "run_name", "mae", "rmse", "mape_percent", "config", "metrics_path"],
    )

    # ---------- significance ----------
    sig_rows = []
    for ds_name, ds in summary["datasets"].items():
        for row in ds["significance_vs_agstt"]:
            challenger_model = parse_model_from_prediction_path(row["challenger"])
            sig_rows.append(
                {
                    "dataset": ds_name,
                    "reference_model": "agstt",
                    "challenger_model": challenger_model,
                    "null_val": row.get("null_val", ""),
                    "n_valid_points": row.get("n_valid_points", ""),
                    "mean_diff_ref_minus_challenger": row["mean_error_diff_ref_minus_challenger"],
                    "p_value_two_sided": row["permutation_p_value_two_sided"],
                    "ci_low": row["bootstrap_95ci"][0],
                    "ci_high": row["bootstrap_95ci"][1],
                    "significant_at_0_05": row["significant_at_0_05"],
                    "interpretation": row["interpretation"],
                }
            )
    write_csv(
        tables_dir / "significance_combined.csv",
        sig_rows,
        [
            "dataset",
            "reference_model",
            "challenger_model",
            "null_val",
            "n_valid_points",
            "mean_diff_ref_minus_challenger",
            "p_value_two_sided",
            "ci_low",
            "ci_high",
            "significant_at_0_05",
            "interpretation",
        ],
    )

    # ---------- supplementary ----------
    write_csv(
        tables_dir / "improvements_long_budget_and_no_adaptive.csv",
        summary["improvements"]["long_budget"],
        ["config", "run_name", "model", "mae", "rmse", "mape_percent", "metrics_path"],
    )
    write_csv(
        tables_dir / "fair_budget_summary.csv",
        summary["improvements"]["fair_budget"],
        ["config", "run_name", "model", "mae", "rmse", "mape_percent", "metrics_path"],
    )
    write_csv(
        tables_dir / "fusion_sweeps_summary.csv",
        summary["improvements"]["fusion_sweeps"],
        ["config", "run_name", "model", "mae", "rmse", "mape_percent", "metrics_path"],
    )
    write_csv(
        tables_dir / "robustness_n5_summary.csv",
        summary["improvements"]["robustness_n5"],
        ["dataset", "run_name", "model", "n_runs", "mae_mean", "mae_std", "rmse_mean", "rmse_std", "mape_mean", "mape_std"],
    )
    write_csv(
        tables_dir / "robustness_v3_n5_summary.csv",
        summary["improvements"]["robustness_v3_n5"],
        ["dataset", "run_name", "model", "n_runs", "mae_mean", "mae_std", "rmse_mean", "rmse_std", "mape_mean", "mape_std"],
    )
    write_csv(
        tables_dir / "cross_split_v3_rolling_per_split.csv",
        summary["improvements"]["cross_split_v3_rolling"]["per_split"],
        ["dataset", "split", "agstt_mae", "dcrnn_mae", "delta_agstt_minus_dcrnn", "better_model"],
    )
    write_csv(
        tables_dir / "cross_split_v3_rolling_head_to_head.csv",
        summary["improvements"]["cross_split_v3_rolling"]["head_to_head"],
        ["dataset", "n_splits", "agstt_win_count", "dcrnn_win_count", "tie_count", "mean_delta_agstt_minus_dcrnn"],
    )
    write_csv(
        tables_dir / "seed_level_significance_v3_vs_dcrnn_summary.csv",
        summary["improvements"]["seed_level_significance_v3_vs_dcrnn"]["summary"],
        [
            "dataset",
            "reference_label",
            "challenger_label",
            "n_seeds",
            "mean_diff_ref_minus_challenger",
            "std_diff_ref_minus_challenger",
            "p_value_two_sided",
            "significant_at_0_05",
            "reference_win_count",
            "challenger_win_count",
            "tie_count",
            "interpretation",
        ],
    )

    # ---------- copy key figures ----------
    fig_map = {
        "pemsd7_overall_metrics_bar.png": Path("outputs/pemsd7/figures/overall_metrics_bar.png"),
        "pemsd7_horizon_mae_curve.png": Path("outputs/pemsd7/figures/horizon_mae_curve.png"),
        "metr_la_overall_metrics_bar.png": Path("outputs/metr_la/figures/overall_metrics_bar.png"),
        "metr_la_horizon_mae_curve.png": Path("outputs/metr_la/figures/horizon_mae_curve.png"),
        "pemsd7_single_node_curve_node211.png": Path("outputs/analysis/pemsd7_single_node_curve_node211.png"),
        "metr_la_single_node_curve_node163.png": Path("outputs/analysis/metr_la_single_node_curve_node163.png"),
    }
    copied = {}
    for dst_name, src in fig_map.items():
        copied[dst_name] = copy_if_exists(src, figs_dir / dst_name)

    # ---------- copy key raw/json/md evidence ----------
    ref_map = {
        "outputs/pemsd7/benchmark_summary.json": "pemsd7_benchmark_summary.json",
        "outputs/metr_la/benchmark_summary.json": "metr_la_benchmark_summary.json",
        "outputs/pemsd7/significance_test.json": "pemsd7_significance_test.json",
        "outputs/metr_la/significance_test.json": "metr_la_significance_test.json",
        "outputs/fair_budget/benchmark_summary.json": "fair_budget_benchmark_summary.json",
        "outputs/fusion_sweeps/benchmark_summary.json": "fusion_sweeps_benchmark_summary.json",
        "outputs/improvements/robustness_summary_n5.json": "robustness_summary_n5.json",
        "outputs/improvements/robustness_summary_v3_n5.json": "robustness_summary_v3_n5.json",
        "outputs/cross_split_v3/rolling_summary.json": "cross_split_v3_rolling_summary.json",
        "outputs/improvements/seed_level_significance_v3_vs_dcrnn.json": "seed_level_significance_v3_vs_dcrnn.json",
        "outputs/ablation/benchmark_summary.json": "ablation_demo_level_benchmark_summary.json",
    }
    for rel, dst_name in ref_map.items():
        src = Path(rel)
        if src.exists():
            copy_if_exists(src, refs_dir / dst_name)

    # ---------- write markdown docs ----------
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    readme = [
        "# 实验数据与结果汇总包",
        "",
        f"- 生成时间：{now}",
        f"- 数据来源：`{args.summary_json}`",
        "- 内容范围：主实验、显著性、补充实验、稳健性、跨时段滚动、节点误差分析",
        "",
        "## 文件结构",
        "- `01_数据集与预处理概览.md`",
        "- `02_主实验结果与显著性.md`",
        "- `03_补充实验与稳健性.md`",
        "- `04_可视化与误差分析.md`",
        "- `05_复现与证据路径.md`",
        "- `tables/*.csv`",
        "- `figures/*.png`",
        "- `refs/*.json`",
        "",
        "## 使用建议",
        "1. 论文正文直接引用 `tables/main_benchmark_combined.csv` 与 `tables/significance_combined.csv`。",
        "2. 答辩PPT优先使用 `figures/` 下 6 张图。",
        "3. 口径说明统一引用 `05_复现与证据路径.md`。",
    ]
    (out_dir / "00_目录说明.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    # 01
    ds_table_rows = []
    for r in ds_rows:
        ds_table_rows.append(
            [
                str(r["dataset"]),
                str(r["nodes"]),
                str(r["timesteps"]),
                str(r["features"]),
                f'{r["data_min"]:.4f}',
                f'{r["data_max"]:.4f}',
                f'{r["data_mean"]:.4f}',
                f'{r["data_std"]:.4f}',
                str(r["adj_nonzero"]),
                f'{r["adj_density"]:.6f}',
                str(r["null_val_eval"]),
            ]
        )
    md01 = [
        "# 01 数据集与预处理概览",
        "",
        markdown_table(
            ["dataset", "nodes", "timesteps", "features", "min", "max", "mean", "std", "adj_nonzero", "adj_density", "null_val"],
            ds_table_rows,
        ),
        "",
        "## 关键说明",
        "- 主实验统一使用 `Tin=12`, `Tout=12`。",
        "- 主实验窗口使用 `max_timesteps=4032`；长预算/公平预算使用 `8064`。",
        "- METR-LA 主评估与显著性检验均使用 `null_val=0.0`。",
        "",
        "原始表格文件：`tables/dataset_overview.csv`",
    ]
    (out_dir / "01_数据集与预处理概览.md").write_text("\n".join(md01) + "\n", encoding="utf-8")

    # 02 main + sig
    # build benchmark row sets
    pems_main = [r for r in bench_rows if r["dataset"] == "pemsd7_228"]
    metr_main = [r for r in bench_rows if r["dataset"] == "metr_la"]
    model_order = {"dcrnn": 0, "agstt": 1, "lstm": 2, "stgcn": 3, "linear": 4}
    pems_main.sort(key=lambda x: model_order.get(x["model"], 99))
    metr_main.sort(key=lambda x: model_order.get(x["model"], 99))

    def rows_main(rows):
        return [[r["model"], f'{r["mae"]:.4f}', f'{r["rmse"]:.4f}', f'{r["mape_percent"]:.2f}'] for r in rows]

    sig_metr = [r for r in sig_rows if r["dataset"] == "metr_la"]
    sig_metr.sort(key=lambda x: model_order.get(x["challenger_model"], 99))
    sig_rows_md = [
        [
            f'AG-STT vs {r["challenger_model"]}',
            f'{r["mean_diff_ref_minus_challenger"]:+.6f}',
            f'{r["p_value_two_sided"]:.6f}',
            f'[{r["ci_low"]:.6f}, {r["ci_high"]:.6f}]',
            str(r["interpretation"]),
        ]
        for r in sig_metr
    ]

    md02 = [
        "# 02 主实验结果与显著性",
        "",
        "## PEMSD7-228 主结果（标准预算）",
        markdown_table(["model", "MAE", "RMSE", "MAPE(%)"], rows_main(pems_main)),
        "",
        "## METR-LA 主结果（标准预算）",
        markdown_table(["model", "MAE", "RMSE", "MAPE(%)"], rows_main(metr_main)),
        "",
        "## METR-LA 显著性（与主评估同 mask 口径）",
        markdown_table(["comparison", "mean(ref-ch)", "p_value", "95%CI", "interpretation"], sig_rows_md),
        "",
        "## 结论口径",
        "- AG-STT 与 DCRNN 在当前标准预算下为 competitive 关系。",
        "- AG-STT 对 LSTM/STGCN/Linear 在多数指标或统计意义上更优（具体见表）。",
        "",
        "原始表格文件：`tables/main_benchmark_combined.csv`, `tables/significance_combined.csv`",
    ]
    (out_dir / "02_主实验结果与显著性.md").write_text("\n".join(md02) + "\n", encoding="utf-8")

    # 03 supplementary
    md03 = [
        "# 03 补充实验与稳健性",
        "",
        "## 文件清单",
        "- `tables/improvements_long_budget_and_no_adaptive.csv`",
        "- `tables/fair_budget_summary.csv`",
        "- `tables/fusion_sweeps_summary.csv`",
        "- `tables/robustness_n5_summary.csv`",
        "- `tables/robustness_v3_n5_summary.csv`",
        "- `tables/cross_split_v3_rolling_per_split.csv`",
        "- `tables/cross_split_v3_rolling_head_to_head.csv`",
        "- `tables/seed_level_significance_v3_vs_dcrnn_summary.csv`",
        "",
        "## 证据分层建议写法",
        "1. 自适应图有效性：真实数据对比证据较强（no_adaptive 对比）。",
        "2. 融合约束优化：有支持证据（fusion sweep）。",
        "3. 动态图：当前主要为探索性证据（demo-level），答辩建议如实描述。",
        "",
        "## 稳健性与跨时段边界",
        "- 已提供 n=5 多seed汇总与 seed 级配对检验。",
        "- 已提供 s1/s2/s3 滚动切分结果与 head-to-head 统计。",
    ]
    (out_dir / "03_补充实验与稳健性.md").write_text("\n".join(md03) + "\n", encoding="utf-8")

    # 04 visuals
    md04 = [
        "# 04 可视化与误差分析",
        "",
        "## 总体指标柱状图",
        "![pems overall](figures/pemsd7_overall_metrics_bar.png)",
        "",
        "![metr overall](figures/metr_la_overall_metrics_bar.png)",
        "",
        "## Per-horizon MAE 曲线",
        "![pems horizon](figures/pemsd7_horizon_mae_curve.png)",
        "",
        "![metr horizon](figures/metr_la_horizon_mae_curve.png)",
        "",
        "## 单节点预测曲线",
        "![pems node](figures/pemsd7_single_node_curve_node211.png)",
        "",
        "![metr node](figures/metr_la_single_node_curve_node163.png)",
        "",
        "## 节点误差相关原始文件",
        "- `outputs/analysis/pemsd7_top10_hardest_nodes_agstt.csv`",
        "- `outputs/analysis/metr_la_top10_hardest_nodes_agstt.csv`",
        "- `outputs/analysis/pemsd7_top10_largest_agstt_dcrnn_gap_nodes.csv`",
        "- `outputs/analysis/metr_la_top10_largest_agstt_dcrnn_gap_nodes.csv`",
    ]
    (out_dir / "04_可视化与误差分析.md").write_text("\n".join(md04) + "\n", encoding="utf-8")

    # 05 refs/repro
    md05 = [
        "# 05 复现与证据路径",
        "",
        "## 核心结果路径",
        "- 主结果 JSON：`refs/pemsd7_benchmark_summary.json`、`refs/metr_la_benchmark_summary.json`",
        "- 显著性 JSON：`refs/pemsd7_significance_test.json`、`refs/metr_la_significance_test.json`",
        "- 当前总汇总：`outputs/current_results_summary.json`",
        "",
        "## 关键命令（与项目脚本一致）",
        "```bash",
        "python scripts/run_experiments.py --configs ... --output-dir outputs/pemsd7",
        "python scripts/run_experiments.py --configs ... --output-dir outputs/metr_la",
        "python scripts/significance_test.py --reference ... --challengers ... --config configs/metr_la_agstt.yaml",
        "python scripts/generate_final_results_summary.py",
        "```",
        "",
        "## 口径声明（论文/答辩推荐原文）",
        "METR-LA 的主评估指标与统计显著性检验均使用 `null_val=0.0` 掩码规则，确保统计页与主结果页口径一致。",
    ]
    (out_dir / "05_复现与证据路径.md").write_text("\n".join(md05) + "\n", encoding="utf-8")

    # manifest
    all_files = sorted([str(p.relative_to(out_dir)).replace("\\", "/") for p in out_dir.rglob("*") if p.is_file()])
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_summary_json": str(args.summary_json),
        "out_dir": str(out_dir),
        "file_count": len(all_files),
        "files": all_files,
        "copied_figures": copied,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # optional zip
    if args.zip:
        zip_path = out_dir.with_suffix(".zip")
        if zip_path.exists():
            zip_path.unlink()
        shutil.make_archive(str(out_dir), "zip", root_dir=out_dir.parent, base_dir=out_dir.name)

    print(f"Summary bundle generated: {out_dir}")
    print(f"Total files: {len(all_files)}")


if __name__ == "__main__":
    main()
