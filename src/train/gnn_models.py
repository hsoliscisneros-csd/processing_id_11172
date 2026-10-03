"""Arquitecturas EXP-05: MLP → GCN → GraphSAGE → GAT."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import scatter


class WeightedSAGEConv(torch.nn.Module):
    """GraphSAGE mean con pesos opcionales en aristas."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lin_neigh = Linear(in_channels, out_channels, bias=False)
        self.lin_self = Linear(in_channels, out_channels, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        row, col = edge_index
        if edge_weight is None:
            edge_weight = torch.ones(row.size(0), device=x.device, dtype=x.dtype)
        weighted = x[col] * edge_weight.view(-1, 1)
        agg = scatter(weighted, row, dim=0, dim_size=x.size(0), reduce="sum")
        norm = scatter(edge_weight, row, dim=0, dim_size=x.size(0), reduce="sum").clamp(min=1e-9)
        out = self.lin_neigh(agg / norm.view(-1, 1)) + self.lin_self(x)
        return out


class PhysWeightedSAGEConv(torch.nn.Module):
    """SAGE con agregación softmax guiada por path-loss: w_ij ∝ exp(-α·10·n·log10(1+d))."""

    def __init__(self, in_channels: int, out_channels: int, *, learn_n: bool = True):
        super().__init__()
        self.lin_neigh = Linear(in_channels, out_channels, bias=False)
        self.lin_self = Linear(in_channels, out_channels, bias=True)
        # n path-loss (típico 2–4 indoor/outdoor); α temperatura
        self.log_n = torch.nn.Parameter(torch.tensor(0.693147)) if learn_n else None  # log(2)
        self.n_fixed = 2.0
        self.log_alpha = torch.nn.Parameter(torch.tensor(0.0))  # α=1

    def _n(self) -> torch.Tensor:
        if self.log_n is None:
            return torch.tensor(self.n_fixed)
        return self.log_n.exp().clamp(0.5, 6.0)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        edge_dist: torch.Tensor | None = None,
    ) -> torch.Tensor:
        row, col = edge_index
        if edge_index.numel() == 0:
            return self.lin_self(x)
        if edge_dist is None:
            # fallback: invertir peso de proximidad si existe
            if edge_weight is not None:
                edge_dist = (1.0 / edge_weight.clamp(min=1e-6) - 1.0).clamp(min=0.0)
            else:
                edge_dist = torch.ones(row.size(0), device=x.device, dtype=x.dtype)
        n = self._n().to(device=x.device, dtype=x.dtype)
        alpha = self.log_alpha.exp().clamp(0.05, 10.0).to(device=x.device, dtype=x.dtype)
        logits = -alpha * 10.0 * n * torch.log10(1.0 + edge_dist.clamp(min=0.0))
        # softmax por nodo fuente (row)
        logits = logits - scatter(logits, row, dim=0, dim_size=x.size(0), reduce="max")[row]
        exp_l = logits.exp()
        denom = scatter(exp_l, row, dim=0, dim_size=x.size(0), reduce="sum").clamp(min=1e-9)
        attn = exp_l / denom[row]
        weighted = x[col] * attn.view(-1, 1)
        agg = scatter(weighted, row, dim=0, dim_size=x.size(0), reduce="sum")
        return self.lin_neigh(agg) + self.lin_self(x)


class NodeMLP(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, x, edge_index=None, edge_weight=None, **_kwargs):
        return self.net(x).squeeze(-1)


class NodeGCN(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.head = torch.nn.Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None, **_kwargs):
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x).squeeze(-1)


class NodeSAGE(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.conv1 = WeightedSAGEConv(in_dim, hidden)
        self.conv2 = WeightedSAGEConv(hidden, hidden)
        self.head = torch.nn.Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None, **_kwargs):
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x).squeeze(-1)


class PhysSAGE(torch.nn.Module):
    """Track A: GraphSAGE con agregación path-loss (PhysAgg)."""

    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.3, learn_n: bool = True):
        super().__init__()
        self.conv1 = PhysWeightedSAGEConv(in_dim, hidden, learn_n=learn_n)
        self.conv2 = PhysWeightedSAGEConv(hidden, hidden, learn_n=learn_n)
        self.head = torch.nn.Linear(hidden, 1)
        self.dropout = dropout

    def encode(self, x, edge_index, edge_weight=None, edge_dist=None):
        h = F.relu(
            self.conv1(x, edge_index, edge_weight=edge_weight, edge_dist=edge_dist)
        )
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(
            self.conv2(h, edge_index, edge_weight=edge_weight, edge_dist=edge_dist)
        )
        return h

    def forward(self, x, edge_index, edge_weight=None, edge_dist=None, **_kwargs):
        h = self.encode(x, edge_index, edge_weight=edge_weight, edge_dist=edge_dist)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.head(h).squeeze(-1)

    def path_loss_n(self) -> float:
        return float(self.conv1._n().detach().cpu())


class NodeGAT(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden * heads, hidden, heads=1, concat=False, dropout=dropout)
        self.head = torch.nn.Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None, **_kwargs):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x).squeeze(-1)


class DualHeadDetector(torch.nn.Module):
    """Track B: GNN en nodos conectados + MLP en aislados; cabezal compartido."""

    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.conv1 = WeightedSAGEConv(in_dim, hidden)
        self.conv2 = WeightedSAGEConv(hidden, hidden)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.head = torch.nn.Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None, connected_mask=None):
        h_g = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        h_g = F.dropout(h_g, p=self.dropout, training=self.training)
        h_g = F.relu(self.conv2(h_g, edge_index, edge_weight=edge_weight))
        h_g = F.dropout(h_g, p=self.dropout, training=self.training)
        h_m = self.mlp(x)
        if connected_mask is None:
            deg = torch.zeros(x.size(0), device=x.device, dtype=torch.long)
            if edge_index.numel():
                deg = deg.scatter_add_(
                    0, edge_index[0], torch.ones(edge_index.size(1), device=x.device, dtype=torch.long)
                )
            connected_mask = deg > 0
        mask = connected_mask.to(dtype=h_g.dtype).unsqueeze(-1)
        h = mask * h_g + (1.0 - mask) * h_m
        return self.head(h).squeeze(-1)


def build_model(name: str, in_dim: int) -> torch.nn.Module:
    factories = {
        "mlp": NodeMLP,
        "gcn": NodeGCN,
        "sage": NodeSAGE,
        "gat": NodeGAT,
        "dual": DualHeadDetector,
        "phys_sage": PhysSAGE,
    }
    if name not in factories:
        raise ValueError(f"modelo desconocido: {name}")
    return factories[name](in_dim)
