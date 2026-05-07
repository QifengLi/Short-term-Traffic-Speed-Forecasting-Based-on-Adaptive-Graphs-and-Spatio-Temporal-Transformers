from __future__ import annotations

import torch
import torch.nn as nn

from src.models.adaptive_graph import AdaptiveGraphGenerator
from src.models.layers import STTransformerBlock


class AGSTT(nn.Module):
    """Adaptive Graph + Spatio-Temporal Transformer."""

    def __init__(
        self,
        num_nodes: int,
        in_dim: int,
        seq_len: int,
        pred_len: int,
        model_cfg: dict,
        static_adj: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        d_model = model_cfg["d_model"]
        ff_hidden = model_cfg["ff_hidden"]
        num_heads = model_cfg["num_heads"]
        num_layers = model_cfg["num_layers"]
        dropout = model_cfg["dropout"]
        gcn_order = model_cfg["gcn_order"]

        if static_adj is None:
            static_adj = torch.eye(num_nodes, dtype=torch.float32)
        self.register_buffer("static_adj", static_adj.float())

        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, 1, d_model))

        self.graph_generator = AdaptiveGraphGenerator(
            num_nodes=num_nodes,
            d_model=d_model,
            use_static_graph=model_cfg.get("use_static_graph", True),
            use_adaptive_graph=model_cfg.get("use_adaptive_graph", True),
            use_dynamic_graph=model_cfg.get("use_dynamic_graph", True),
            fusion_temperature=float(model_cfg.get("fusion_temperature", 1.0)),
            min_static_coeff=float(model_cfg.get("min_static_coeff", 0.0)),
            adaptive_topk=model_cfg.get("adaptive_topk", None),
            dynamic_topk=model_cfg.get("dynamic_topk", None),
            init_weight_static=float(model_cfg.get("init_weight_static", 1.0)),
            init_weight_adaptive=float(model_cfg.get("init_weight_adaptive", 1.0)),
            init_weight_dynamic=float(model_cfg.get("init_weight_dynamic", 1.0)),
        )

        self.blocks = nn.ModuleList(
            [
                STTransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ff_hidden=ff_hidden,
                    gcn_order=gcn_order,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.readout = nn.Linear(d_model, 1)
        self.horizon_proj = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T_in, N, F]
        h = self.input_proj(x) + self.pos_emb
        adj = self.graph_generator(h, self.static_adj)

        for block in self.blocks:
            h = block(h, adj)

        z = self.readout(h).squeeze(-1)  # [B, T_in, N]
        z = z.permute(0, 2, 1)  # [B, N, T_in]
        out = self.horizon_proj(z)  # [B, N, T_out]
        out = out.permute(0, 2, 1).unsqueeze(-1)  # [B, T_out, N, 1]
        return out
