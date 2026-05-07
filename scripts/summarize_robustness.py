from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_metrics(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_group_key(metrics: dict) -> tuple[str, str]:
    dataset = str(metrics.get("dataset", "unknown"))
    run_name = str(metrics.get("run_name", metrics.get("model", "unknown")))
    model = str(metrics.get("model", "unknown"))
    base_run_name = run_name.split("_seed")[0]
    return dataset, f"{base_run_name}|{model}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--out-json", type=Path, default=Path("outputs/improvements/robustness_summary.json"))
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/improvements/robustness_summary.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("outputs/improvements/robustness_summary.md"))
    args = parser.parse_args()

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for p in [Path(x) for x in args.metrics]:
        m = load_metrics(p)
        key = parse_group_key(m)
        grouped[key].append(m)

    rows: list[dict] = []
    for (dataset, run_model), values in grouped.items():
        run_name, model = run_model.split("|", 1)
        maes = np.array([v["overall"]["mae"] for v in values], dtype=np.float64)
        rmses = np.array([v["overall"]["rmse"] for v in values], dtype=np.float64)
        mapes = np.array([v["overall"]["mape_percent"] for v in values], dtype=np.float64)
        rows.append(
            {
                "dataset": dataset,
                "run_name": run_name,
                "model": model,
                "n_runs": int(len(values)),
                "mae_mean": float(maes.mean()),
                "mae_std": float(maes.std(ddof=0)),
                "rmse_mean": float(rmses.mean()),
                "rmse_std": float(rmses.std(ddof=0)),
                "mape_mean": float(mapes.mean()),
                "mape_std": float(mapes.std(ddof=0)),
            }
        )

    rows.sort(key=lambda x: (x["dataset"], x["mae_mean"]))

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)

    args.out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "run_name",
                "model",
                "n_runs",
                "mae_mean",
                "mae_std",
                "rmse_mean",
                "rmse_std",
                "mape_mean",
                "mape_std",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Robustness Summary",
        "",
        "| Dataset | Run Name | Model | N | MAE mean | MAE std | RMSE mean | RMSE std | MAPE mean | MAPE std |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['dataset']} | {r['run_name']} | {r['model']} | {r['n_runs']} | "
            f"{r['mae_mean']:.4f} | {r['mae_std']:.4f} | {r['rmse_mean']:.4f} | {r['rmse_std']:.4f} | "
            f"{r['mape_mean']:.2f} | {r['mape_std']:.2f} |"
        )
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved JSON: {args.out_json}")
    print(f"Saved CSV:  {args.out_csv}")
    print(f"Saved MD:   {args.out_md}")


if __name__ == "__main__":
    main()
