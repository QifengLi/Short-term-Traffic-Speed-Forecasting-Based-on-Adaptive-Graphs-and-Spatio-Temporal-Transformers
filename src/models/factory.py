from __future__ import annotations

import torch
import torch.nn as nn

from src.models.agstt import AGSTT
from src.models.baselines import DCRNNBaseline, LSTMBaseline, LinearBaseline, STGCNBaseline


def get_model_name(model_cfg: dict) -> str:
    return str(model_cfg.get("name", "agstt")).lower()


def build_model(
    num_nodes: int,
    in_dim: int,
    seq_len: int,
    pred_len: int,
    model_cfg: dict,
    static_adj: torch.Tensor | None = None,
) -> nn.Module:
    name = get_model_name(model_cfg)

    if name == "agstt":
        return AGSTT(
            num_nodes=num_nodes,
            in_dim=in_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            model_cfg=model_cfg,
            static_adj=static_adj,
        )
    if name == "lstm":
        return LSTMBaseline(
            in_dim=in_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            hidden_dim=int(model_cfg.get("hidden_dim", 64)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
    if name == "linear":
        return LinearBaseline(in_dim=in_dim, seq_len=seq_len, pred_len=pred_len)
    if name == "stgcn":
        return STGCNBaseline(
            in_dim=in_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            num_nodes=num_nodes,
            static_adj=static_adj,
            hidden_dim=int(model_cfg.get("hidden_dim", 64)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
    if name == "dcrnn":
        return DCRNNBaseline(
            in_dim=in_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            num_nodes=num_nodes,
            static_adj=static_adj,
            hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        )

    raise ValueError(f"Unsupported model name: {name}")
