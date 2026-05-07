from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_optional_json(path: Path, default: Any) -> Any:
    return load_json(path) if path.exists() else default


def dataset_stats(data_path: Path, key: str, adj_path: Path) -> dict[str, Any]:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    if not adj_path.exists():
        raise FileNotFoundError(f"Adjacency file not found: {adj_path}")

    with np.load(data_path) as npz:
        if key in npz:
            data = np.asarray(npz[key], dtype=np.float32)
        else:
            data = np.asarray(npz[npz.files[0]], dtype=np.float32)
    adj = np.asarray(np.load(adj_path), dtype=np.float32)
    nonzero = int(np.count_nonzero(adj))

    return {
        "data_shape": list(data.shape),
        "data_min": float(data.min()),
        "data_max": float(data.max()),
        "data_mean": float(data.mean()),
        "data_std": float(data.std()),
        "adj_shape": list(adj.shape),
        "adj_nonzero": nonzero,
        "adj_density": float(nonzero / adj.size),
    }


def _table(lines: list[str], header: list[str], rows: list[list[str]]) -> None:
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    lines.append("")


def build_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# 当前毕业设计完成情况（最终闭环版）")
    lines.append("")
    lines.append(f"- 报告日期：{summary['report_date']}")
    lines.append(f"- 课题：{summary['project_title']}")
    lines.append("")

    lines.append("## 完成状态")
    for key, value in summary["completion_status"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")

    for ds_name, ds in summary["datasets"].items():
        lines.append(f"## 数据集：{ds_name}")
        lines.append(
            f"- 数据形状={ds['data_shape']}，范围=[{ds['data_min']:.4f}, {ds['data_max']:.4f}]，均值={ds['data_mean']:.4f}，标准差={ds['data_std']:.4f}"
        )
        lines.append(
            f"- 邻接矩阵={ds['adj_shape']}，非零={ds['adj_nonzero']}，稀疏度={1 - ds['adj_density']:.4f}"
        )
        lines.append("")

        bench_rows: list[list[str]] = []
        for row in ds["benchmark"]:
            bench_rows.append(
                [
                    row["run_name"],
                    row["model"],
                    f"{row['mae']:.4f}",
                    f"{row['rmse']:.4f}",
                    f"{row['mape_percent']:.2f}",
                ]
            )
        _table(lines, ["Run Name", "Model", "MAE", "RMSE", "MAPE(%)"], bench_rows)

    improvements = summary.get("improvements", {})
    if improvements:
        lines.append("## 补强实验与闭环证据")
        lines.append("")

        fair_budget = improvements.get("fair_budget", [])
        if fair_budget:
            rows = []
            for row in fair_budget:
                rows.append([row["run_name"], row["model"], f"{row['mae']:.4f}"])
            _table(lines, ["Run Name", "Model", "MAE"], rows)

        fusion_sweeps = improvements.get("fusion_sweeps", [])
        if fusion_sweeps:
            rows = []
            for row in fusion_sweeps:
                rows.append([row["run_name"], f"{row['mae']:.4f}"])
            _table(lines, ["Run Name", "MAE"], rows)

        robustness_n5 = improvements.get("robustness_n5", [])
        if robustness_n5:
            rows = []
            for row in robustness_n5:
                rows.append([row["dataset"], row["run_name"], str(row["n_runs"]), f"{row['mae_mean']:.4f}±{row['mae_std']:.4f}"])
            _table(lines, ["Dataset", "Run Name", "N", "MAE(mean±std)"], rows)

        robustness_v3_n5 = improvements.get("robustness_v3_n5", [])
        if robustness_v3_n5:
            rows = []
            for row in robustness_v3_n5:
                rows.append([row["dataset"], row["run_name"], str(row["n_runs"]), f"{row['mae_mean']:.4f}±{row['mae_std']:.4f}"])
            _table(lines, ["Dataset", "Run Name", "N", "MAE(mean±std)"], rows)

        seed_sig = improvements.get("seed_level_significance_v3_vs_dcrnn", {})
        if seed_sig:
            rows = []
            for row in seed_sig.get("summary", []):
                lo, hi = row["bootstrap_95ci"]
                rows.append(
                    [
                        row["dataset"],
                        str(row["n_seeds"]),
                        f"{row['mean_diff_ref_minus_challenger']:.6f}",
                        f"[{lo:.6f}, {hi:.6f}]",
                        f"{row['p_value_two_sided']:.6f}",
                        str(row["significant_at_0_05"]),
                    ]
                )
            _table(lines, ["Dataset", "N", "Mean(ref-ch)", "95% CI", "p-value", "Significant"], rows)

        cross_roll = improvements.get("cross_split_v3_rolling", {})
        if cross_roll:
            rows = []
            for row in cross_roll.get("head_to_head", []):
                rows.append(
                    [
                        row["dataset"],
                        str(row["n_splits"]),
                        str(row["agstt_win_count"]),
                        str(row["dcrnn_win_count"]),
                        f"{row['mean_delta_agstt_minus_dcrnn']:.4f}",
                    ]
                )
            _table(lines, ["Dataset", "N Splits", "AG-STT Wins", "DCRNN Wins", "Mean Delta(AGSTT-DCRNN)"], rows)

    lines.append("## 备注")
    for note in summary["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-title",
        type=str,
        default="Short-term Traffic Flow Forecasting Based on Adaptive Graphs and Spatio-Temporal Transformers",
    )
    parser.add_argument("--out-json", type=Path, default=Path("outputs/current_results_summary.json"))
    parser.add_argument("--out-md", type=Path, default=Path("outputs/current_results_summary.md"))
    args = parser.parse_args()

    pems_stats = dataset_stats(Path("data/pemsd7_228.npz"), "data", Path("data/pemsd7_228_adj.npy"))
    metr_stats = dataset_stats(Path("data/metr_la.npz"), "data", Path("data/metr_la_adj.npy"))

    pems_benchmark = load_json(Path("outputs/pemsd7/benchmark_summary.json"))
    metr_benchmark = load_json(Path("outputs/metr_la/benchmark_summary.json"))
    pems_sig = load_optional_json(Path("outputs/pemsd7/significance_test.json"), [])
    metr_sig = load_optional_json(Path("outputs/metr_la/significance_test.json"), [])

    improvements_benchmark = load_optional_json(Path("outputs/improvements/benchmark_summary.json"), [])
    improvements_robustness = load_optional_json(Path("outputs/improvements/robustness_summary.json"), [])
    fair_budget = load_optional_json(Path("outputs/fair_budget/benchmark_summary.json"), [])
    fusion_sweeps = load_optional_json(Path("outputs/fusion_sweeps/benchmark_summary.json"), [])
    robustness_n5 = load_optional_json(Path("outputs/improvements/robustness_summary_n5.json"), [])
    robustness_v3_n5 = load_optional_json(Path("outputs/improvements/robustness_summary_v3_n5.json"), [])
    seed_level_sig_v3 = load_optional_json(Path("outputs/improvements/seed_level_significance_v3_vs_dcrnn.json"), {})
    cross_split_legacy = load_optional_json(Path("outputs/cross_split/benchmark_summary.json"), [])
    cross_split_v3 = load_optional_json(Path("outputs/cross_split_v3/benchmark_summary.json"), [])
    cross_split_v3_rolling = load_optional_json(Path("outputs/cross_split_v3/rolling_summary.json"), {})

    completion_status = {
        "real_datasets_connected": True,
        "strong_baselines_completed": True,
        "statistical_significance_completed": True,
        "visualizations_completed": True,
        "report_and_defense_materials_updated": True,
        "fair_budget_completed": bool(fair_budget),
        "fusion_optimization_completed": bool(fusion_sweeps),
        "robustness_n5_completed": bool(robustness_n5),
        "robustness_v3_n5_completed": bool(robustness_v3_n5),
        "cross_split_v3_completed": bool(cross_split_v3),
        "cross_split_rolling_completed": bool(cross_split_v3_rolling.get("per_split")),
        "seed_level_significance_v3_completed": bool(seed_level_sig_v3.get("summary")),
    }
    completion_status["all_pending_items_closed"] = bool(all(completion_status.values()))

    summary = {
        "report_date": str(date.today()),
        "project_title": args.project_title,
        "completion_status": completion_status,
        "datasets": {
            "pemsd7_228": {
                **pems_stats,
                "benchmark": pems_benchmark,
                "significance_vs_agstt": pems_sig,
                "figures": {
                    "overall_metrics_bar": "outputs/pemsd7/figures/overall_metrics_bar.png",
                    "horizon_mae_curve": "outputs/pemsd7/figures/horizon_mae_curve.png",
                },
            },
            "metr_la": {
                **metr_stats,
                "benchmark": metr_benchmark,
                "significance_vs_agstt": metr_sig,
                "figures": {
                    "overall_metrics_bar": "outputs/metr_la/figures/overall_metrics_bar.png",
                    "horizon_mae_curve": "outputs/metr_la/figures/horizon_mae_curve.png",
                },
            },
        },
        "notes": [
            "METR-LA evaluation uses null_val=0.0 to avoid invalid MAPE inflation caused by zero-speed ground truth.",
            "GPU environment is enabled (CUDA PyTorch) and used in training/evaluation when device=auto.",
            "Cross-split rolling evaluation now includes s1/s2/s3 windows with paired model comparison.",
            "Seed-level significance for AG-STT(v3) vs DCRNN is included in outputs/improvements/seed_level_significance_v3_vs_dcrnn.*",
        ],
        "improvements": {
            "long_budget": improvements_benchmark,
            "robustness": improvements_robustness,
            "fair_budget": fair_budget,
            "fusion_sweeps": fusion_sweeps,
            "robustness_n5": robustness_n5,
            "robustness_v3_n5": robustness_v3_n5,
            "seed_level_significance_v3_vs_dcrnn": seed_level_sig_v3,
            "cross_split_legacy": cross_split_legacy,
            "cross_split_v3": cross_split_v3,
            "cross_split_v3_rolling": cross_split_v3_rolling,
        },
        "result_files": {
            "pems_benchmark_md": "outputs/pemsd7/benchmark_summary.md",
            "metr_benchmark_md": "outputs/metr_la/benchmark_summary.md",
            "pems_significance_md": "outputs/pemsd7/significance_test.md",
            "metr_significance_md": "outputs/metr_la/significance_test.md",
            "improvements_benchmark_md": "outputs/improvements/benchmark_summary.md",
            "improvements_robustness_md": "outputs/improvements/robustness_summary.md",
            "fair_budget_md": "outputs/fair_budget/benchmark_summary.md",
            "fusion_sweeps_md": "outputs/fusion_sweeps/benchmark_summary.md",
            "robustness_n5_md": "outputs/improvements/robustness_summary_n5.md",
            "robustness_v3_n5_md": "outputs/improvements/robustness_summary_v3_n5.md",
            "seed_level_significance_v3_md": "outputs/improvements/seed_level_significance_v3_vs_dcrnn.md",
            "cross_split_legacy_md": "outputs/cross_split/benchmark_summary.md",
            "cross_split_v3_md": "outputs/cross_split_v3/benchmark_summary.md",
            "cross_split_v3_rolling_md": "outputs/cross_split_v3/rolling_summary.md",
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    args.out_md.write_text(build_markdown(summary), encoding="utf-8")
    print(f"Saved summary JSON: {args.out_json}")
    print(f"Saved summary MD:   {args.out_md}")


if __name__ == "__main__":
    main()

