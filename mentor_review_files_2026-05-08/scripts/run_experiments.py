from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml


DEFAULT_CONFIGS = [
    "configs/default.yaml",
    "configs/lstm_baseline.yaml",
    "configs/linear_baseline.yaml",
    "configs/agstt_no_dynamic.yaml",
    "configs/agstt_no_adaptive.yaml",
    "configs/agstt_static_only.yaml",
]


def run_command(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_metrics_path(config_path: Path) -> Path:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = cfg.get("output", {})
    if "metrics_path" in output:
        return Path(output["metrics_path"])
    dataset = cfg["dataset"]["name"]
    model_name = str(cfg["model"].get("name", "agstt")).lower()
    run_name = str(cfg.get("experiment", {}).get("name", model_name)).lower()
    return Path(f"outputs/{dataset}_{run_name}_metrics.json")


def load_metrics(metrics_path: Path) -> dict:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def save_summary(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "benchmark_summary.json"
    csv_path = output_dir / "benchmark_summary.csv"
    md_path = output_dir / "benchmark_summary.md"

    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["config", "run_name", "model", "mae", "rmse", "mape_percent", "metrics_path"],
        )
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Benchmark Summary",
        "",
        "| Config | Run Name | Model | MAE | RMSE | MAPE(%) | Metrics File |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['config']} | {row['run_name']} | {row['model']} | {row['mae']:.4f} | {row['rmse']:.4f} | "
            f"{row['mape_percent']:.2f} | {row['metrics_path']} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved summary JSON: {json_path}")
    print(f"Saved summary CSV:  {csv_path}")
    print(f"Saved summary MD:   {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    rows: list[dict] = []
    for cfg in args.configs:
        cfg_path = Path(cfg)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")

        if not args.skip_train:
            run_command([sys.executable, "train.py", "--config", str(cfg_path)])

        run_command([sys.executable, "evaluate.py", "--config", str(cfg_path)])
        metrics_path = load_metrics_path(cfg_path)
        metrics = load_metrics(metrics_path)
        overall = metrics["overall"]

        rows.append(
            {
                "config": str(cfg_path),
                "run_name": metrics.get("run_name", metrics["model"]),
                "model": metrics["model"],
                "mae": float(overall["mae"]),
                "rmse": float(overall["rmse"]),
                "mape_percent": float(overall["mape_percent"]),
                "metrics_path": str(metrics_path),
            }
        )

    rows.sort(key=lambda x: x["mae"])
    save_summary(rows, Path(args.output_dir))


if __name__ == "__main__":
    main()
