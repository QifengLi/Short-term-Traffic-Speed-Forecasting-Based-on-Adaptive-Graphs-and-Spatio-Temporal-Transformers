from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_pred(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    with np.load(path) as npz:
        preds = np.asarray(npz["preds"], dtype=np.float64)
        trues = np.asarray(npz["trues"], dtype=np.float64)
    return preds, trues


def parse_null_val(value: str) -> float | None:
    text = value.strip().lower()
    if text in {"none", "null", ""}:
        return None
    if text == "nan":
        return float("nan")
    return float(text)


def build_valid_mask(trues: np.ndarray, null_val: float | None) -> np.ndarray:
    if null_val is None:
        return np.ones_like(trues, dtype=bool)
    if np.isnan(null_val):
        return ~np.isnan(trues)
    return trues != null_val


def node_mae(preds: np.ndarray, trues: np.ndarray, null_val: float | None) -> np.ndarray:
    abs_err = np.abs(preds - trues)  # [B, T, N, 1]
    mask = build_valid_mask(trues, null_val)
    n = preds.shape[2]
    maes = np.zeros((n,), dtype=np.float64)
    for node in range(n):
        e = abs_err[:, :, node : node + 1, :]
        m = mask[:, :, node : node + 1, :]
        if np.any(m):
            maes[node] = float(e[m].mean())
        else:
            maes[node] = float("nan")
    return maes


def save_topk_csv(mae: np.ndarray, out_csv: Path, k: int) -> None:
    idx = np.argsort(np.nan_to_num(mae, nan=-1.0))[::-1][:k]
    rows = [{"rank": i + 1, "node": int(node), "mae": float(mae[node])} for i, node in enumerate(idx)]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")


def save_single_node_curve(
    agstt_preds: np.ndarray,
    dcrnn_preds: np.ndarray,
    trues: np.ndarray,
    node: int,
    out_png: Path,
    title: str,
    horizon_step: int = 0,
    n_points: int = 240,
) -> None:
    # Use one horizon (default horizon=1) and flatten along batch axis as timeline proxy.
    y_true = trues[:, horizon_step, node, 0]
    y_a = agstt_preds[:, horizon_step, node, 0]
    y_d = dcrnn_preds[:, horizon_step, node, 0]

    n = min(n_points, y_true.shape[0])
    x = np.arange(n)

    plt.figure(figsize=(12, 4))
    plt.plot(x, y_true[:n], label="True", linewidth=1.5)
    plt.plot(x, y_a[:n], label="AG-STT", linewidth=1.2)
    plt.plot(x, y_d[:n], label="DCRNN", linewidth=1.2)
    plt.title(title)
    plt.xlabel("Sample index")
    plt.ylabel("Traffic value")
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["metr_la", "pemsd7"], required=True)
    parser.add_argument("--null-val", type=str, default="none")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/analysis"))
    args = parser.parse_args()

    if args.dataset == "metr_la":
        agstt_path = Path("outputs/metr_la_agstt_predictions.npz")
        dcrnn_path = Path("outputs/metr_la_dcrnn_predictions.npz")
        prefix = "metr_la"
    else:
        agstt_path = Path("outputs/pemsd7_agstt_predictions.npz")
        dcrnn_path = Path("outputs/pemsd7_dcrnn_predictions.npz")
        prefix = "pemsd7"

    null_val = parse_null_val(args.null_val)

    a_pred, a_true = load_pred(agstt_path)
    d_pred, d_true = load_pred(dcrnn_path)
    if a_pred.shape != d_pred.shape or a_true.shape != d_true.shape:
        raise ValueError("AG-STT and DCRNN prediction files have mismatched shapes.")
    if not np.array_equal(a_true, d_true):
        raise ValueError("Ground truth arrays differ between AG-STT and DCRNN files.")

    mae_agstt = node_mae(a_pred, a_true, null_val=null_val)
    mae_dcrnn = node_mae(d_pred, d_true, null_val=null_val)
    mae_gap = mae_agstt - mae_dcrnn

    out_dir = args.out_dir
    save_topk_csv(mae_agstt, out_dir / f"{prefix}_top{args.topk}_hardest_nodes_agstt.csv", args.topk)
    save_topk_csv(mae_dcrnn, out_dir / f"{prefix}_top{args.topk}_hardest_nodes_dcrnn.csv", args.topk)

    gap_idx = np.argsort(np.abs(np.nan_to_num(mae_gap, nan=0.0)))[::-1][: args.topk]
    gap_rows = [
        {
            "rank": i + 1,
            "node": int(node),
            "agstt_mae": float(mae_agstt[node]),
            "dcrnn_mae": float(mae_dcrnn[node]),
            "gap_agstt_minus_dcrnn": float(mae_gap[node]),
        }
        for i, node in enumerate(gap_idx)
    ]
    pd.DataFrame(gap_rows).to_csv(
        out_dir / f"{prefix}_top{args.topk}_largest_agstt_dcrnn_gap_nodes.csv",
        index=False,
        encoding="utf-8",
    )

    node = int(np.nanargmax(np.abs(mae_gap)))
    save_single_node_curve(
        agstt_preds=a_pred,
        dcrnn_preds=d_pred,
        trues=a_true,
        node=node,
        out_png=out_dir / f"{prefix}_single_node_curve_node{node}.png",
        title=f"{prefix}: node {node} (largest |AG-STT - DCRNN| MAE gap)",
    )

    summary = {
        "dataset": args.dataset,
        "null_val": null_val,
        "node_count": int(a_pred.shape[2]),
        "selected_node_for_curve": node,
        "selected_node_agstt_mae": float(mae_agstt[node]),
        "selected_node_dcrnn_mae": float(mae_dcrnn[node]),
        "selected_node_gap_agstt_minus_dcrnn": float(mae_gap[node]),
        "outputs": {
            "agstt_topk_csv": str((out_dir / f"{prefix}_top{args.topk}_hardest_nodes_agstt.csv").as_posix()),
            "dcrnn_topk_csv": str((out_dir / f"{prefix}_top{args.topk}_hardest_nodes_dcrnn.csv").as_posix()),
            "gap_topk_csv": str((out_dir / f"{prefix}_top{args.topk}_largest_agstt_dcrnn_gap_nodes.csv").as_posix()),
            "single_node_curve_png": str((out_dir / f"{prefix}_single_node_curve_node{node}.png").as_posix()),
        },
    }
    (out_dir / f"{prefix}_node_error_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
