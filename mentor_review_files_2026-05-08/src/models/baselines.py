from __future__ import annotations

import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    """Node-wise LSTM baseline for multi-step traffic forecasting."""

    def __init__(
        self,
        in_dim: int,
        seq_len: int,
        pred_len: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.readout = nn.Linear(hidden_dim, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T_in, N, F]
        b, _, n, f = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b * n, self.seq_len, f)  # [B*N, T_in, F]
        out, _ = self.encoder(x)
        last_hidden = out[:, -1, :]  # [B*N, H]
        pred = self.readout(last_hidden)  # [B*N, T_out]
        pred = pred.view(b, n, self.pred_len).permute(0, 2, 1).unsqueeze(-1)  # [B, T_out, N, 1]
        return pred


class LinearBaseline(nn.Module):
    """Simple per-node linear baseline from history window to prediction horizon."""

    def __init__(self, in_dim: int, seq_len: int, pred_len: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.proj = nn.Linear(seq_len * in_dim, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T_in, N, F]
        b, t, n, f = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b * n, t * f)
        pred = self.proj(x)  # [B*N, T_out]
        pred = pred.view(b, n, self.pred_len).permute(0, 2, 1).unsqueeze(-1)  # [B, T_out, N, 1]
        return pred


class STGCNBaseline(nn.Module):
    """A compact STGCN-style baseline with temporal conv + graph propagation."""

    def __init__(
        self,
        in_dim: int,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        static_adj: torch.Tensor | None = None,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_layers = max(1, num_layers)

        if static_adj is None:
            static_adj = torch.eye(num_nodes, dtype=torch.float32)
        self.register_buffer("static_adj", static_adj.float())

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.temp_convs1 = nn.ModuleList(
            [nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 1), padding=(1, 0)) for _ in range(self.num_layers)]
        )
        self.temp_convs2 = nn.ModuleList(
            [nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 1), padding=(1, 0)) for _ in range(self.num_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.num_layers)])
        self.dropout = nn.Dropout(dropout)

        self.readout = nn.Linear(hidden_dim, 1)
        self.horizon_proj = nn.Linear(seq_len, pred_len)

    def _graph_propagate(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, N]
        return torch.einsum("nm,bctm->bctn", self.static_adj, x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T_in, N, F]
        h = self.input_proj(x)  # [B, T, N, H]
        h = h.permute(0, 3, 1, 2)  # [B, H, T, N]

        for conv1, conv2, norm in zip(self.temp_convs1, self.temp_convs2, self.norms):
            residual = h
            h = torch.relu(conv1(h))
            h = self._graph_propagate(h)
            h = torch.relu(conv2(h))
            h = self.dropout(h)
            h = h + residual
            h = h.permute(0, 2, 3, 1)
            h = norm(h)
            h = h.permute(0, 3, 1, 2)

        h = h.permute(0, 2, 3, 1)  # [B, T, N, H]
        z = self.readout(h).squeeze(-1)  # [B, T, N]
        z = z.permute(0, 2, 1)  # [B, N, T]
        out = self.horizon_proj(z).permute(0, 2, 1).unsqueeze(-1)  # [B, T_out, N, 1]
        return out


class _GraphGRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate_linear = nn.Linear((input_dim + hidden_dim) * 2, hidden_dim * 2)
        self.candidate_linear = nn.Linear((input_dim + hidden_dim) * 2, hidden_dim)

    @staticmethod
    def _graph_mix(adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D], adj: [N, N]
        return torch.einsum("nm,bmd->bnd", adj, x)

    def _graph_linear(self, adj: torch.Tensor, x: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
        xg = self._graph_mix(adj, x)
        out = linear(torch.cat([x, xg], dim=-1))
        return out

    def forward(self, x: torch.Tensor, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D_in], h: [B, N, H]
        inp = torch.cat([x, h], dim=-1)
        gates = torch.sigmoid(self._graph_linear(adj, inp, self.gate_linear))
        z, r = gates.chunk(2, dim=-1)
        candidate_in = torch.cat([x, r * h], dim=-1)
        h_tilde = torch.tanh(self._graph_linear(adj, candidate_in, self.candidate_linear))
        h_new = (1.0 - z) * h + z * h_tilde
        return h_new


class DCRNNBaseline(nn.Module):
    """A compact DCRNN-style baseline with graph-aware GRU encoder."""

    def __init__(
        self,
        in_dim: int,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        static_adj: torch.Tensor | None = None,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim

        if static_adj is None:
            static_adj = torch.eye(num_nodes, dtype=torch.float32)
        self.register_buffer("static_adj", static_adj.float())

        self.cell = _GraphGRUCell(input_dim=in_dim, hidden_dim=hidden_dim)
        self.readout = nn.Linear(hidden_dim, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T_in, N, F]
        b, t, n, _ = x.shape
        h = torch.zeros(b, n, self.hidden_dim, dtype=x.dtype, device=x.device)
        for i in range(t):
            xt = x[:, i]  # [B, N, F]
            h = self.cell(xt, h, self.static_adj)

        pred = self.readout(h)  # [B, N, T_out]
        pred = pred.permute(0, 2, 1).unsqueeze(-1)  # [B, T_out, N, 1]
        return pred
