from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from src.data.dataset import create_dataloaders
from src.data.graph import load_adjacency
from src.models.factory import build_model, get_model_name
from src.utils.io import load_config
from src.utils.metrics import masked_mae
from src.utils.seed import set_seed


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


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    grad_clip: float,
    log_interval: int,
    null_val: float | None,
) -> float:
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_loss = 0.0
    total_steps = 0
    desc = "Train" if train_mode else "Valid"

    with torch.set_grad_enabled(train_mode):
        for step, (x, y) in enumerate(tqdm(loader, desc=desc, leave=False), start=1):
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = masked_mae(pred, y, null_val=null_val)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item()
            total_steps += 1

            if train_mode and log_interval > 0 and step % log_interval == 0:
                print(f"[Train] step={step} loss={loss.item():.6f}")

    return total_loss / max(total_steps, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])
    null_val = parse_null_val(cfg["dataset"].get("null_val", None))

    train_loader, val_loader, _, scaler, num_nodes, in_dim = create_dataloaders(cfg)
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
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    checkpoint_dir = Path(cfg["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"{cfg['dataset']['name']}_{run_name}_best.pt"

    best_val = float("inf")
    epochs = cfg["training"]["epochs"]
    grad_clip = cfg["training"]["grad_clip"]
    log_interval = cfg["training"]["log_interval"]

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=grad_clip,
            log_interval=log_interval,
            null_val=null_val,
        )

        val_loss = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            grad_clip=grad_clip,
            log_interval=0,
            null_val=null_val,
        )

        print(f"Epoch {epoch:03d}/{epochs} | train_mae={train_loss:.6f} | val_mae={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "scaler_mean": torch.as_tensor(scaler.mean, dtype=torch.float32),
                    "scaler_std": torch.as_tensor(scaler.std, dtype=torch.float32),
                    "model_name": model_name,
                    "run_name": run_name,
                },
                ckpt_path,
            )
            print(f"Saved best checkpoint to: {ckpt_path}")

    print(f"Training complete ({run_name}). Best val MAE: {best_val:.6f}")


if __name__ == "__main__":
    main()
