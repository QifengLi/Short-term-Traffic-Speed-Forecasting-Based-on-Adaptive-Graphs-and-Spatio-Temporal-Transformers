from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def plot_overall_metrics(rows: list[dict], out_dir: Path) -> None:
    run_names = [row["run_name"] for row in rows]
    mae = np.array([row["mae"] for row in rows], dtype=np.float32)
    rmse = np.array([row["rmse"] for row in rows], dtype=np.float32)
    mape = np.array([row["mape_percent"] for row in rows], dtype=np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = [("MAE", mae), ("RMSE", rmse), ("MAPE(%)", mape)]
    for ax, (name, values) in zip(axes, metrics):
        ax.bar(run_names, values, color="#4C78A8")
        ax.set_title(name)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "overall_metrics_bar.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_horizon_mae(metric_paths: list[Path], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for path in metric_paths:
        if not path.exists():
            continue
        obj = json.loads(path.read_text(encoding="utf-8"))
        run_name = obj.get("run_name", obj.get("model", path.stem))
        per = obj.get("per_horizon", [])
        if not per:
            continue
        x = [item["horizon"] for item in per]
        y = [item["mae"] for item in per]
        ax.plot(x, y, marker="o", linewidth=1.8, label=run_name)

    ax.set_xlabel("Horizon")
    ax.set_ylabel("MAE")
    ax.set_title("Per-Horizon MAE")
    ax.grid(linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / "horizon_mae_curve.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-json", type=Path, default=Path("outputs/benchmark_summary.json"))
    parser.add_argument(
        "--metrics-jsons",
        nargs="*",
        default=[
            "outputs/demo_agstt_metrics.json",
            "outputs/demo_lstm_metrics.json",
            "outputs/demo_linear_metrics.json",
        ],
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_json(args.benchmark_json)
    rows = sorted(rows, key=lambda x: x["mae"])
    plot_overall_metrics(rows, out_dir)

    metric_paths = [Path(p) for p in args.metrics_jsons]
    plot_horizon_mae(metric_paths, out_dir)


if __name__ == "__main__":
    main()
