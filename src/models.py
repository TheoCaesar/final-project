"""Baselines and the proposed node-adaptive directional model.

Design: one layer family, NAMix, covers a 2x2 factorial ablation
                         global gate (shared by all nodes)   node-adaptive gate
  undirected neighbours  SAGE + learned mixing               CAGNN-style (Chen et al. 2024)
  directed in/out        DirGNN-style, learned alpha          NADir (proposed)
so every claim about "direction" or "node adaptivity" is one switch away.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, DirGNNConv
from torch_geometric.utils import to_undirected, scatter


def mean_agg(h, edge_index, n):
    """mean over sources, delivered to destinations (h[src] -> dst)."""
    s, d = edge_index
    return scatter(h[s], d, dim=0, dim_size=n, reduce="mean")


class NAMix(nn.Module):
    def __init__(self, d_in, d_out, directed=True, node_gate=True):
        super().__init__()
        self.directed, self.node_gate = directed, node_gate
        k = 3 if directed else 2  # gates: [self, in, out] or [self, nbr]
        self.k = k
        self.W = nn.ModuleList([nn.Linear(d_in, d_out) for _ in range(k)])
        if node_gate:
            # +2 log-degree inputs so the gate knows when a neighbourhood is empty
            self.gate = nn.Sequential(nn.Linear(k * d_out + 2, d_out), nn.Tanh(), nn.Linear(d_out, k))
        else:
            self.gate_logits = nn.Parameter(torch.zeros(k))
        self.last_gates = None

    def forward(self, h, ei, ei_und, logdeg):
        n = h.size(0)
        parts = [self.W[0](h)]
        if self.directed:
            parts.append(mean_agg(self.W[1](h), ei, n))           # in-neighbours (predecessors)
            parts.append(mean_agg(self.W[2](h), ei.flip(0), n))   # out-neighbours (successors)
        else:
            parts.append(mean_agg(self.W[1](h), ei_und, n))
        P = torch.stack(parts, 1)  # n x k x d
        if self.node_gate:
            g = torch.softmax(self.gate(torch.cat(parts + [logdeg], 1)), -1)
        else:
            g = torch.softmax(self.gate_logits, -1).expand(n, -1)
        self.last_gates = g.detach()
        return (g.unsqueeze(-1) * P).sum(1)


class Net(nn.Module):
    """kind in {mlp, gcn, gat, sage, dirgnn, namix}"""

    def __init__(self, kind, d_in, hid=64, layers=2, dropout=0.5, heads=4,
                 directed=True, node_gate=True, alpha=0.5):
        super().__init__()
        self.kind, self.dropout = kind, dropout
        self.enc = nn.Linear(d_in, hid)
        convs = []
        for _ in range(layers):
            if kind == "mlp":
                convs.append(nn.Linear(hid, hid))
            elif kind == "gcn":
                convs.append(GCNConv(hid, hid))
            elif kind == "gat":
                convs.append(GATConv(hid, hid // heads, heads=heads))
            elif kind == "sage":
                convs.append(SAGEConv(hid, hid))
            elif kind == "dirgnn":  # Rossi et al. 2023, Dir-SAGE
                convs.append(DirGNNConv(SAGEConv(hid, hid, root_weight=False), alpha=alpha, root_weight=True))
            elif kind == "namix":
                convs.append(NAMix(hid, hid, directed, node_gate))
            else:
                raise ValueError(kind)
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList([nn.LayerNorm(hid) for _ in range(layers)])
        self.out = nn.Linear(hid, 2)

    def forward(self, x, ei, ei_und, logdeg):
        h = F.relu(self.enc(x))
        h = F.dropout(h, self.dropout, self.training)
        for conv, norm in zip(self.convs, self.norms):
            if self.kind == "mlp":
                z = conv(h)
            elif self.kind in ("gcn", "gat", "sage"):
                z = conv(h, ei_und)
            elif self.kind == "dirgnn":
                z = conv(h, ei)
            else:
                z = conv(h, ei, ei_und, logdeg)
            h = F.dropout(F.relu(norm(z)), self.dropout, self.training)
        return self.out(h)


def graph_inputs(edge_index, n):
    ei_und = to_undirected(edge_index, num_nodes=n)
    din = torch.bincount(edge_index[1], minlength=n).float()
    dout = torch.bincount(edge_index[0], minlength=n).float()
    logdeg = torch.stack([torch.log1p(din), torch.log1p(dout)], 1)
    return ei_und, logdeg
