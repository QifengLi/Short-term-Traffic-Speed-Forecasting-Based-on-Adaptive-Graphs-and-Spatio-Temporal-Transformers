from __future__ import annotations

import numpy as np
import torch


class StandardScaler:
    """Standardize features with statistics from the training split."""

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> None:
        # data: [T, N, F]
        self.mean = data.mean(axis=(0, 1), keepdims=True)
        self.std = data.std(axis=(0, 1), keepdims=True)
        self.std = np.where(self.std < 1e-6, 1.0, self.std)

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler is not fitted.")
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler is not fitted.")
        return data * self.std + self.mean

    def inverse_transform_feature(self, data: torch.Tensor, feature_idx: int) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler is not fitted.")
        mean = torch.as_tensor(
            self.mean[..., feature_idx], dtype=data.dtype, device=data.device
        ).view(1, 1, 1, 1)
        std = torch.as_tensor(
            self.std[..., feature_idx], dtype=data.dtype, device=data.device
        ).view(1, 1, 1, 1)
        return data * std + mean

