from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.data.dataset import create_dataloaders
from src.data.graph import load_adjacency
from src.models.factory import build_model, get_model_name
from src.utils.io import load_config
from src.utils.metrics import masked_mae, masked_mape, masked_rmse


GRAPH_MODELS = {"agstt", "stgcn", "dcrnn"}


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def parse_null_val(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "none", "null"}:
            return None
        if text == "nan":
            return float("nan")
        return float(text)
    raise ValueError(f"Unsupported dataset.null_val type: {type(value)}")


def to_numpy_array(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--metrics-json", type=str, default="")
    parser.add_argument("--allow-unsafe-checkpoint", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    null_val = parse_null_val(cfg["dataset"].get("null_val", None))
    _, _, test_loader, scaler, num_nodes, in_dim = create_dataloaders(cfg)
    model_name = get_model_name(cfg["model"])
    run_name = str(cfg.get("experiment", {}).get("name", model_name)).lower()
    use_static_graph = model_name in GRAPH_MODELS and bool(cfg["model"].get("use_static_graph", True))
    strict_adj_check = bool(cfg["dataset"].get("strict_adj_check", True))
    if use_static_graph:
        static_adj = load_adjacency(
            cfg["dataset"].get("adj_path"),
            num_nodes,
            strict_missing=strict_adj_check,
        )
    else:
        static_adj = torch.eye(num_nodes, dtype=torch.float32)

    model = build_model(
        num_nodes=num_nodes,
        in_dim=in_dim,
        seq_len=cfg["dataset"]["seq_len"],
        pred_len=cfg["dataset"]["pred_len"],
        model_cfg=cfg["model"],
        static_adj=static_adj,
    )

    device = resolve_device(cfg["training"]["device"])
    model = model.to(device)

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = Path(cfg["training"]["checkpoint_dir"]) / f"{cfg['dataset']['name']}_{run_name}_best.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    try:
        ckpt = torch.load(
            ckpt_path,
            map_location=device,
            weights_only=not args.allow_unsafe_checkpoint,
        )
    except pickle.UnpicklingError as exc:
        raise RuntimeError(
            "Safe checkpoint loading failed. The checkpoint may contain unsafe pickled objects. "
            "Please retrain with the latest code or rerun with --allow-unsafe-checkpoint only for trusted files."
        ) from exc

    model.load_state_dict(ckpt["model_state_dict"])
    if "scaler_mean" in ckpt and "scaler_std" in ckpt:
        scaler.mean = to_numpy_array(ckpt["scaler_mean"])
        scaler.std = to_numpy_array(ckpt["scaler_std"])
    model.eval()

    preds_all = []
    trues_all = []
    feature_idx = cfg["dataset"]["feature_idx"]

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Test", leave=False):
            x = x.to(device)
            y = y.to(device)
            pred = model(x)

            pred = scaler.inverse_transform_feature(pred, feature_idx)
            y = scaler.inverse_transform_feature(y, feature_idx)

            preds_all.append(pred.cpu())
            trues_all.append(y.cpu())

    preds = torch.cat(preds_all, dim=0)  # [B, T_out, N, 1]
    trues = torch.cat(trues_all, dim=0)

    horizon = preds.shape[1]
    per_horizon_metrics: list[dict] = []
    print("========== Test Metrics (Per Horizon) ==========")
    for h in range(horizon):
        pred_h = preds[:, h : h + 1]
        true_h = trues[:, h : h + 1]
        mae = masked_mae(pred_h, true_h, null_val=null_val).item()
        rmse = masked_rmse(pred_h, true_h, null_val=null_val).item()
        mape = masked_mape(pred_h, true_h, null_val=null_val).item()
        per_horizon_metrics.append(
            {
                "horizon": h + 1,
                "mae": mae,
                "rmse": rmse,
                "mape_percent": mape,
            }
        )
        print(f"Horizon {h + 1:02d}: MAE={mae:.4f} RMSE={rmse:.4f} MAPE={mape:.2f}%")

    print("========== Test Metrics (Overall) ==========")
    mae = masked_mae(preds, trues, null_val=null_val).item()
    rmse = masked_rmse(preds, trues, null_val=null_val).item()
    mape = masked_mape(preds, trues, null_val=null_val).item()
    print(f"Overall: MAE={mae:.4f} RMSE={rmse:.4f} MAPE={mape:.2f}%")

    metrics = {
        "dataset": cfg["dataset"]["name"],
        "model": model_name,
        "run_name": run_name,
        "checkpoint": str(ckpt_path),
        "overall": {
            "mae": mae,
            "rmse": rmse,
            "mape_percent": mape,
        },
        "per_horizon": per_horizon_metrics,
    }

    if cfg.get("output", {}).get("save_predictions", False):
        default_pred_path = f"outputs/{cfg['dataset']['name']}_{run_name}_predictions.npz"
        out_path = Path(cfg.get("output", {}).get("prediction_path", default_pred_path))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            preds=preds.numpy(),
            trues=trues.numpy(),
        )
        print(f"Saved predictions to: {out_path}")

    if args.metrics_json:
        metrics_path = Path(args.metrics_json)
    else:
        default_metrics_path = f"outputs/{cfg['dataset']['name']}_{run_name}_metrics.json"
        metrics_path = Path(cfg.get("output", {}).get("metrics_path", default_metrics_path))

    if cfg.get("output", {}).get("save_metrics", True) or args.metrics_json:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
