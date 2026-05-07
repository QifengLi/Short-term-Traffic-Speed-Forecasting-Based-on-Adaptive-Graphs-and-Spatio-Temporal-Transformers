from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


SPLIT_RE = re.compile(r"_s(\d+)(?:\D|$)")


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark summary not found: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Expected benchmark summary JSON to be a list.")
    return rows


def infer_dataset(run_name: str, config: str) -> str:
    text = f"{run_name} {config}".lower()
    if "pemsd7" in text:
        return "pemsd7_228"
    if "metr_la" in text:
        return "metr_la"
    return "unknown"


def extract_split(run_name: str, config: str) -> str:
    text = f"{run_name} {config}"
    match = SPLIT_RE.search(text)
    if match is None:
        raise ValueError(f"Cannot extract split id from run/config: {run_name} / {config}")
    return f"s{match.group(1)}"


def build_markdown(
    per_split_rows: list[dict[str, Any]],
    aggregated_rows: list[dict[str, Any]],
    head_to_head_rows: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append("# Cross-Split Rolling Summary")
    lines.append("")
    lines.append("| Dataset | Split | AG-STT MAE | DCRNN MAE | Delta(AGSTT-DCRNN) | Better |")
    lines.append("|---|---|---:|---:|---:|---|")
    for row in per_split_rows:
        lines.append(
            f"| {row['dataset']} | {row['split']} | {row['agstt_mae']:.4f} | {row['dcrnn_mae']:.4f} | "
            f"{row['delta_agstt_minus_dcrnn']:.4f} | {row['better_model']} |"
        )
    lines.append("")
    lines.append("| Dataset | Model | N Splits | Mean MAE | Median MAE | Std MAE | Best MAE | Worst MAE |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in aggregated_rows:
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['n_splits']} | {row['mae_mean']:.4f} | "
            f"{row['mae_median']:.4f} | {row['mae_std']:.4f} | {row['mae_best']:.4f} | {row['mae_worst']:.4f} |"
        )
    lines.append("")
    lines.append("| Dataset | N Splits | AG-STT Wins | DCRNN Wins | Ties | Mean Delta(AGSTT-DCRNN) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in head_to_head_rows:
        lines.append(
            f"| {row['dataset']} | {row['n_splits']} | {row['agstt_win_count']} | {row['dcrnn_win_count']} | "
            f"{row['tie_count']} | {row['mean_delta_agstt_minus_dcrnn']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-json",
        type=Path,
        default=Path("outputs/cross_split_v3/benchmark_summary.json"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("outputs/cross_split_v3/rolling_summary.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("outputs/cross_split_v3/rolling_summary.md"),
    )
    parser.add_argument(
        "--out-csv-per-split",
        type=Path,
        default=Path("outputs/cross_split_v3/rolling_summary_per_split.csv"),
    )
    parser.add_argument(
        "--out-csv-aggregated",
        type=Path,
        default=Path("outputs/cross_split_v3/rolling_summary_aggregated.csv"),
    )
    args = parser.parse_args()

    rows = load_rows(args.benchmark_json)

    table: dict[str, dict[str, dict[str, float]]] = {}
    for row in rows:
        dataset = infer_dataset(str(row.get("run_name", "")), str(row.get("config", "")))
        split = extract_split(str(row.get("run_name", "")), str(row.get("config", "")))
        model = str(row.get("model", "")).lower()
        mae = float(row["mae"])
        table.setdefault(dataset, {}).setdefault(split, {})[model] = mae

    per_split_rows: list[dict[str, Any]] = []
    aggregated_input: dict[tuple[str, str], list[float]] = {}
    head_to_head_rows: list[dict[str, Any]] = []

    for dataset in sorted(table.keys()):
        split_map = table[dataset]
        deltas: list[float] = []
        agstt_wins = 0
        dcrnn_wins = 0
        ties = 0
        for split in sorted(split_map.keys()):
            agstt_mae = split_map[split].get("agstt")
            dcrnn_mae = split_map[split].get("dcrnn")
            if agstt_mae is None or dcrnn_mae is None:
                continue
            delta = agstt_mae - dcrnn_mae
            if delta < 0:
                better = "agstt"
                agstt_wins += 1
            elif delta > 0:
                better = "dcrnn"
                dcrnn_wins += 1
            else:
                better = "tie"
                ties += 1
            deltas.append(delta)
            per_split_rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "agstt_mae": float(agstt_mae),
                    "dcrnn_mae": float(dcrnn_mae),
                    "delta_agstt_minus_dcrnn": float(delta),
                    "better_model": better,
                }
            )

        if deltas:
            head_to_head_rows.append(
                {
                    "dataset": dataset,
                    "n_splits": len(deltas),
                    "agstt_win_count": agstt_wins,
                    "dcrnn_win_count": dcrnn_wins,
                    "tie_count": ties,
                    "mean_delta_agstt_minus_dcrnn": float(np.mean(np.asarray(deltas, dtype=np.float64))),
                }
            )

        for model in ("agstt", "dcrnn"):
            values = [float(split_map[s].get(model)) for s in sorted(split_map.keys()) if model in split_map[s]]
            if values:
                aggregated_input[(dataset, model)] = values

    aggregated_rows: list[dict[str, Any]] = []
    for (dataset, model), values in sorted(aggregated_input.items(), key=lambda x: (x[0][0], np.mean(x[1]))):
        arr = np.asarray(values, dtype=np.float64)
        aggregated_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "n_splits": int(arr.size),
                "mae_mean": float(arr.mean()),
                "mae_median": float(np.median(arr)),
                "mae_std": float(arr.std(ddof=0)),
                "mae_best": float(arr.min()),
                "mae_worst": float(arr.max()),
            }
        )

    payload = {
        "per_split": per_split_rows,
        "aggregated": aggregated_rows,
        "head_to_head": head_to_head_rows,
    }

    for p in [args.out_json, args.out_md, args.out_csv_per_split, args.out_csv_aggregated]:
        p.parent.mkdir(parents=True, exist_ok=True)

    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.out_md.write_text(build_markdown(per_split_rows, aggregated_rows, head_to_head_rows), encoding="utf-8")

    with args.out_csv_per_split.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "split",
                "agstt_mae",
                "dcrnn_mae",
                "delta_agstt_minus_dcrnn",
                "better_model",
            ],
        )
        writer.writeheader()
        writer.writerows(per_split_rows)

    with args.out_csv_aggregated.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "model",
                "n_splits",
                "mae_mean",
                "mae_median",
                "mae_std",
                "mae_best",
                "mae_worst",
            ],
        )
        writer.writeheader()
        writer.writerows(aggregated_rows)

    print(f"Saved JSON: {args.out_json}")
    print(f"Saved MD:   {args.out_md}")
    print(f"Saved CSV:  {args.out_csv_per_split}")
    print(f"Saved CSV:  {args.out_csv_aggregated}")


if __name__ == "__main__":
    main()
