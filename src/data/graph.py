from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def _row_normalize(adj: np.ndarray) -> np.ndarray:
    rowsum = adj.sum(axis=1, keepdims=True)
    rowsum = np.where(rowsum < 1e-12, 1.0, rowsum)
    return adj / rowsum


def load_adjacency(path: str | None, num_nodes: int, strict_missing: bool = True) -> torch.Tensor:
    if not path:
        if strict_missing:
            raise ValueError("adj_path is empty. Please set dataset.adj_path or disable strict adjacency check.")
        return torch.eye(num_nodes, dtype=torch.float32)

    adj_path = Path(path)
    if not adj_path.exists():
        if strict_missing:
            raise FileNotFoundError(f"Adjacency file not found: {adj_path}")
        return torch.eye(num_nodes, dtype=torch.float32)

    if adj_path.suffix.lower() == ".npy":
        adj = np.load(adj_path).astype(np.float32)
    elif adj_path.suffix.lower() == ".csv":
        adj = np.loadtxt(adj_path, delimiter=",", dtype=np.float32)
    else:
        raise ValueError(f"Unsupported adjacency file type: {adj_path.suffix}")

    if adj.shape != (num_nodes, num_nodes):
        raise ValueError(
            f"Adjacency shape mismatch. Expected ({num_nodes}, {num_nodes}), got {adj.shape}."
        )

    adj = adj + np.eye(num_nodes, dtype=np.float32)
    adj = _row_normalize(adj)
    return torch.from_numpy(adj).float()
