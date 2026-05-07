from __future__ import annotations

import torch
import torch.nn as nn


class TemporalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, D]
        b, t, n, d = x.shape
        x_reshaped = x.permute(0, 2, 1, 3).reshape(b * n, t, d)
        out, _ = self.attn(x_reshaped, x_reshaped, x_reshaped, need_weights=False)
        return out.reshape(b, n, t, d).permute(0, 2, 1, 3)


class GraphDiffusionConv(nn.Module):
    def __init__(self, d_model: int, order: int, dropout: float) -> None:
        super().__init__()
        self.order = order
        self.proj = nn.Linear((order + 1) * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, D], adj: [B, N, N] or [N, N]
        b, _, _, _ = x.shape
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(b, -1, -1)

        supports = [x]
        xk = x
        for _ in range(self.order):
            xk = torch.einsum("bnm,btmd->btnd", adj, xk)
            supports.append(xk)

        out = torch.cat(supports, dim=-1)
        out = self.proj(out)
        return self.dropout(out)


class STTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_hidden: int,
        gcn_order: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.temporal_attn = TemporalSelfAttention(d_model, num_heads, dropout)
        self.graph_conv = GraphDiffusionConv(d_model, gcn_order, dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, d_model),
        )

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.temporal_attn(x)
        x = self.norm1(x + self.dropout(h))

        h = self.graph_conv(x, adj)
        x = self.norm2(x + self.dropout(h))

        h = self.ffn(x)
        x = self.norm3(x + self.dropout(h))
        return x

