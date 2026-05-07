from __future__ import annotations

import math

import torch


def _build_mask(labels: torch.Tensor, null_val: float | None = None) -> torch.Tensor:
    if null_val is None:
        mask = torch.ones_like(labels, dtype=torch.float32)
    elif math.isnan(null_val):
        mask = (~torch.isnan(labels)).float()
    else:
        mask = (labels != null_val).float()

    mask = mask / (mask.mean() + 1e-8)
    return mask


def masked_mae(
    preds: torch.Tensor,
    labels: torch.Tensor,
    null_val: float | None = None,
) -> torch.Tensor:
    mask = _build_mask(labels, null_val)
    loss = torch.abs(preds - labels) * mask
    return torch.nan_to_num(loss, nan=0.0).mean()


def masked_rmse(
    preds: torch.Tensor,
    labels: torch.Tensor,
    null_val: float | None = None,
) -> torch.Tensor:
    mask = _build_mask(labels, null_val)
    loss = ((preds - labels) ** 2) * mask
    loss = torch.nan_to_num(loss, nan=0.0).mean()
    return torch.sqrt(loss)


def masked_mape(
    preds: torch.Tensor,
    labels: torch.Tensor,
    null_val: float | None = None,
) -> torch.Tensor:
    mask = _build_mask(labels, null_val)
    denom = torch.clamp(torch.abs(labels), min=1e-5)
    loss = torch.abs((preds - labels) / denom) * mask
    return torch.nan_to_num(loss, nan=0.0).mean() * 100.0
