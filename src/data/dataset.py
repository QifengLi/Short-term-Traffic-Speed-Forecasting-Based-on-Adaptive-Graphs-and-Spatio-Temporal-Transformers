from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.scaler import StandardScaler


def load_npz_data(path: str, key: str = "data") -> np.ndarray:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    with np.load(data_path) as npz:
        if key in npz:
            data = npz[key]
        else:
            first_key = npz.files[0]
            data = npz[first_key]

    if data.ndim != 3:
        raise ValueError(f"Expected data shape [T, N, F], got {data.shape}")
    return data.astype(np.float32)


def split_data(
    data: np.ndarray, train_ratio: float, val_ratio: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_len = data.shape[0]
    train_end = int(total_len * train_ratio)
    val_end = int(total_len * (train_ratio + val_ratio))

    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]
    return train, val, test


class TrafficWindowDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        seq_len: int,
        pred_len: int,
        feature_idx: int,
    ) -> None:
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_idx = feature_idx
        self.length = data.shape[0] - seq_len - pred_len + 1
        if self.length <= 0:
            raise ValueError(
                f"Time steps ({data.shape[0]}) too short for seq_len={seq_len} and pred_len={pred_len}."
            )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.seq_len]  # [T_in, N, F]
        y = self.data[
            idx + self.seq_len : idx + self.seq_len + self.pred_len,
            :,
            self.feature_idx : self.feature_idx + 1,
        ]  # [T_out, N, 1]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


def create_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader, DataLoader, StandardScaler, int, int]:
    dataset_cfg = cfg["dataset"]
    train_cfg = cfg["training"]

    data = load_npz_data(dataset_cfg["data_path"], dataset_cfg.get("key", "data"))
    start_timestep = int(dataset_cfg.get("start_timestep", 0) or 0)
    if start_timestep < 0:
        raise ValueError(f"dataset.start_timestep must be >= 0, got {start_timestep}")
    if start_timestep >= data.shape[0]:
        raise ValueError(
            f"dataset.start_timestep={start_timestep} is out of range for data length {data.shape[0]}"
        )

    max_timesteps = dataset_cfg.get("max_timesteps")
    if max_timesteps is not None:
        max_timesteps = int(max_timesteps)
        if max_timesteps <= 0:
            raise ValueError(f"dataset.max_timesteps must be > 0, got {max_timesteps}")
        end_timestep = min(start_timestep + max_timesteps, data.shape[0])
        data = data[start_timestep:end_timestep]
    else:
        data = data[start_timestep:]
    train, val, test = split_data(data, dataset_cfg["train_ratio"], dataset_cfg["val_ratio"])

    scaler = StandardScaler()
    scaler.fit(train)
    train = scaler.transform(train)
    val = scaler.transform(val)
    test = scaler.transform(test)

    seq_len = dataset_cfg["seq_len"]
    pred_len = dataset_cfg["pred_len"]
    feature_idx = dataset_cfg["feature_idx"]

    train_dataset = TrafficWindowDataset(train, seq_len, pred_len, feature_idx)
    val_dataset = TrafficWindowDataset(val, seq_len, pred_len, feature_idx)
    test_dataset = TrafficWindowDataset(test, seq_len, pred_len, feature_idx)

    batch_size = train_cfg["batch_size"]
    num_workers = dataset_cfg.get("num_workers", 0)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False
    )

    num_nodes = data.shape[1]
    in_dim = data.shape[2]
    return train_loader, val_loader, test_loader, scaler, num_nodes, in_dim
