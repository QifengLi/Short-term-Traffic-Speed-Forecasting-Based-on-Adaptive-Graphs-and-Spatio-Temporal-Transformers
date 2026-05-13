from __future__ import annotations

import math

import torch
import torch.nn as nn


class AdaptiveGraphGenerator(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        d_model: int,
        use_static_graph: bool = True,
        use_adaptive_graph: bool = True,
        use_dynamic_graph: bool = True,
        fusion_temperature: float = 1.0,
        min_static_coeff: float = 0.0,
        adaptive_topk: int | None = None,
        dynamic_topk: int | None = None,
        init_weight_static: float = 1.0,
        init_weight_adaptive: float = 1.0,
        init_weight_dynamic: float = 1.0,
    ) -> None:
        super().__init__()
        self.use_static_graph = use_static_graph
        self.use_adaptive_graph = use_adaptive_graph
        self.use_dynamic_graph = use_dynamic_graph
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.fusion_temperature = float(max(fusion_temperature, 1e-4))
        self.min_static_coeff = float(min(max(min_static_coeff, 0.0), 1.0))
        self.adaptive_topk = None if adaptive_topk is None else int(adaptive_topk)
        self.dynamic_topk = None if dynamic_topk is None else int(dynamic_topk)
        self.last_fusion_coeffs: torch.Tensor | None = None

        if use_adaptive_graph:
            self.node_emb1 = nn.Parameter(torch.randn(num_nodes, d_model))
            self.node_emb2 = nn.Parameter(torch.randn(num_nodes, d_model))
            self.weight_adaptive = nn.Parameter(torch.tensor(float(init_weight_adaptive)))
        else:
            self.register_parameter("node_emb1", None)
            self.register_parameter("node_emb2", None)
            self.register_parameter("weight_adaptive", None)

        if use_dynamic_graph:
            self.dynamic_q = nn.Linear(d_model, d_model)
            self.dynamic_k = nn.Linear(d_model, d_model)
            self.weight_dynamic = nn.Parameter(torch.tensor(float(init_weight_dynamic)))
        else:
            self.dynamic_q = None
            self.dynamic_k = None
            self.register_parameter("weight_dynamic", None)

        if use_static_graph:
            self.weight_static = nn.Parameter(torch.tensor(float(init_weight_static)))
        else:
            self.register_parameter("weight_static", None)

    def _adaptive_graph(self) -> torch.Tensor:
        logits = torch.relu(self.node_emb1 @ self.node_emb2.transpose(0, 1))
        return torch.softmax(logits, dim=-1)

    def _dynamic_graph(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, D]
        context = x.mean(dim=1)  # [B, N, D]
        q = self.dynamic_q(context)
        k = self.dynamic_k(context)
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_model)
        return torch.softmax(logits, dim=-1)

    @staticmethod
    def _topk_normalize(mat: torch.Tensor, topk: int | None) -> torch.Tensor:
        if topk is None:
            return mat
        if topk <= 0:
            raise ValueError(f"topk must be > 0, got {topk}")
        if topk >= mat.shape[-1]:
            return mat

        values, indices = torch.topk(mat, k=topk, dim=-1)
        sparse = torch.zeros_like(mat)
        sparse.scatter_(-1, indices, values)
        sparse = sparse / (sparse.sum(dim=-1, keepdim=True) + 1e-8)
        return sparse

    def _apply_min_static_coeff(self, coeffs: torch.Tensor) -> torch.Tensor:
        if not self.use_static_graph or self.min_static_coeff <= 0.0:
            return coeffs

        static_coeff = coeffs[0]
        if static_coeff >= self.min_static_coeff:
            return coeffs

        if coeffs.shape[0] == 1:
            return torch.ones_like(coeffs)

        remaining = coeffs[1:]
        remaining_sum = remaining.sum()
        if remaining_sum <= 0:
            out = torch.zeros_like(coeffs)
            out[0] = 1.0
            return out

        scaled_remaining = remaining / remaining_sum * (1.0 - self.min_static_coeff)
        out = torch.cat(
            [torch.tensor([self.min_static_coeff], device=coeffs.device, dtype=coeffs.dtype), scaled_remaining], dim=0
        )
        return out

    def forward(self, x: torch.Tensor, static_adj: torch.Tensor | None = None) -> torch.Tensor:
        # Return shape: [B, N, N]
        batch_size = x.shape[0]
        mats: list[torch.Tensor] = []
        weights: list[torch.Tensor] = []

        if self.use_static_graph and static_adj is not None:
            mats.append(static_adj.unsqueeze(0).expand(batch_size, -1, -1))
            weights.append(self.weight_static)

        if self.use_adaptive_graph:
            adp = self._adaptive_graph().unsqueeze(0).expand(batch_size, -1, -1)
            adp = self._topk_normalize(adp, self.adaptive_topk)
            mats.append(adp)
            weights.append(self.weight_adaptive)

        if self.use_dynamic_graph:
            dyn = self._dynamic_graph(x)
            dyn = self._topk_normalize(dyn, self.dynamic_topk)
            mats.append(dyn)
            weights.append(self.weight_dynamic)

        if not mats:
            raise RuntimeError("No graph source enabled. At least one graph type must be True.")

        coeffs = torch.softmax(torch.stack(weights) / self.fusion_temperature, dim=0)
        coeffs = self._apply_min_static_coeff(coeffs)
        self.last_fusion_coeffs = coeffs.detach().cpu()
        out = 0.0
        for idx, mat in enumerate(mats):
            out = out + coeffs[idx] * mat

        out = out / (out.sum(dim=-1, keepdim=True) + 1e-8)
        return out
