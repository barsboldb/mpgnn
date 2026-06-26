import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax as pyg_softmax


class GCNConv(MessagePassing):
    """
    Kipf & Welling (2017)
    h_i = sum_{j in N(i) u {i}} (deg_i * deg_j)^{-1/2} * W * h_j

    Aggregation: add with symmetric normalization.
    Simple and effective for homophilous graphs (where neighbors share the same label).
    Limitation: treats all neighbors equally (no attention).
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, edge_index, batch=None):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5).masked_fill(deg == 0, 0.0)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        x = self.lin(x)
        return self.propagate(edge_index, x=x, norm=norm) + self.bias

    def message(self, x_j, norm):  # type: ignore[override]
        # scale each neighbor message by symmetric normalization coefficient
        return norm.view(-1, 1) * x_j


class SAGEConv(MessagePassing):
    """
    Hamilton et al. (2017) — GraphSAGE
    h_i = W * concat(h_i, mean_{j in N(i)} h_j)

    Aggregation: mean of neighbors, then concat with self-embedding.
    Key difference from GCN: explicit separation of self vs. neighbor transform.
    Works well for inductive (unseen nodes) settings.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr='mean')
        # concat(h_i, mean_neigh) doubles the input to the linear transform
        self.lin = nn.Linear(in_channels * 2, out_channels)

    def forward(self, x, edge_index, batch=None):
        agg = self.propagate(edge_index, x=x)          # mean of neighbors: [N, in_ch]
        return self.lin(torch.cat([x, agg], dim=-1))   # concat self + agg, then project

    def message(self, x_j):  # type: ignore[override]
        return x_j


class GATConv(MessagePassing):
    """
    Veličković et al. (2018) — Graph Attention Network
    e_{ij} = LeakyReLU(a^T [W*h_i || W*h_j])
    h_i = concat_k sum_{j in N(i)+{i}} softmax_j(e^k_{ij}) * W^k * h_j

    Aggregation: learned attention-weighted sum, multi-head.
    out_channels must be divisible by heads.
    Advantage: dynamic, content-based weighting of neighbors.
    """
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1, dropout: float = 0.0):
        super().__init__(aggr='add')
        assert out_channels % heads == 0, f"out_channels ({out_channels}) must be divisible by heads ({heads})"
        self.heads = heads
        self.head_dim = out_channels // heads
        self.out_channels = out_channels
        self.dropout = dropout

        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        # per-head attention vectors (split into src and dst components)
        self.att_src = nn.Parameter(torch.empty(1, heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.empty(1, heads, self.head_dim))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x, edge_index, batch=None):
        N = x.size(0)
        edge_index, _ = add_self_loops(edge_index, num_nodes=N)

        # keep x 2D for propagate; PyG's indexing expects [N, F]
        x = self.lin(x)                                              # [N, H*D]
        x_view = x.view(N, self.heads, self.head_dim)               # [N, H, D]

        alpha_src = (x_view * self.att_src).sum(-1)  # [N, H]
        alpha_dst = (x_view * self.att_dst).sum(-1)  # [N, H]

        row, col = edge_index  # row=src (j), col=dst (i)
        alpha = F.leaky_relu(alpha_src[row] + alpha_dst[col], negative_slope=0.2)  # [E, H]
        alpha = pyg_softmax(alpha, col, num_nodes=N)  # softmax over all incoming edges per node
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = self.propagate(edge_index, x=x, alpha=alpha)  # [N, H*D]
        return out + self.bias

    def message(self, x_j, alpha):  # type: ignore[override]
        # x_j: [E, H*D],  alpha: [E, H]  — reshape, weight, reshape back
        return (x_j.view(-1, self.heads, self.head_dim) * alpha.unsqueeze(-1)).view(-1, self.out_channels)


class GINConv(MessagePassing):
    """
    Xu et al. (2019) — Graph Isomorphism Network
    h_i = MLP((1 + eps) * h_i + sum_{j in N(i)} h_j)

    Aggregation: sum (most expressive for WL-test-distinguishable graphs).
    As powerful as the Weisfeiler-Lehman graph isomorphism test.
    Good for graph-level tasks (graph classification).
    """
    def __init__(self, in_channels: int, out_channels: int, eps: float = 0.0, train_eps: bool = True):
        super().__init__(aggr='add')
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        self.eps = nn.Parameter(torch.tensor(eps)) if train_eps else eps

    def forward(self, x, edge_index, batch=None):
        agg = self.propagate(edge_index, x=x)
        return self.mlp((1 + self.eps) * x + agg)

    def message(self, x_j):  # type: ignore[override]
        return x_j


class GlobalAttnConv(nn.Module):
    """
    Global (all-pairs) multi-head self-attention — no edge restriction.

    Unlike GAT which only attends to graph neighbors, every node attends to
    every other node in the same graph simultaneously (like a Transformer).
    This is what allows O(log N) depth for tasks like graph connectivity
    (Sanford et al. 2024): each layer can double the reachability horizon
    via the doubling trick, rather than extending it by one hop.

    For batched graphs, attention is masked so nodes only attend within
    their own graph.

    local=True restricts attention to graph neighbours (and self) — a node may
    only attend where there is an edge. Reach then grows one hop per layer, so the
    model becomes depth-bounded (the regime where capacity / data-lever effects
    bind), instead of the global all-pairs reach.

    out_channels must be divisible by heads.
    """
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1,
                 dropout: float = 0.0, spd_max_dist: int = 0, local: bool = False):
        super().__init__()
        assert out_channels % heads == 0, f"out_channels ({out_channels}) must be divisible by heads ({heads})"
        self.heads = heads
        self.head_dim = out_channels // heads
        self.out_channels = out_channels
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout
        self.spd_max_dist = spd_max_dist
        self.local = local
        # inspection hook: when True, forward stashes the post-softmax attention
        # weights [N, N, heads] in self.last_attn. Off by default so training/eval
        # pay nothing; inspect_activations.py flips it on.
        self.store_attn = False
        self.last_attn: torch.Tensor | None = None

        self.W_q = nn.Linear(in_channels, out_channels, bias=False)
        self.W_k = nn.Linear(in_channels, out_channels, bias=False)
        self.W_v = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

        # learnable per-head bias for each SPD bucket (0, 1, ..., max_dist)
        # spd_max_dist=0 disables SPD bias entirely
        self.spd_bias = nn.Embedding(spd_max_dist + 1, heads) if spd_max_dist > 0 else None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: torch.Tensor | None = None) -> torch.Tensor:
        N, H, D = x.size(0), self.heads, self.head_dim

        Q = self.W_q(x).view(N, H, D)  # [N, H, D]
        K = self.W_k(x).view(N, H, D)
        V = self.W_v(x).view(N, H, D)

        # all-pairs attention scores: Q_i · K_j / sqrt(d)  →  [N, N, H]
        attn = torch.einsum('ihd,jhd->ijh', Q, K) * self.scale

        # add learned shortest-path distance bias before softmax (Graphormer-style)
        if self.spd_bias is not None:
            from .features import spd_batch
            spd = spd_batch(edge_index, batch, N, self.spd_max_dist)  # [N, N]
            attn = attn + self.spd_bias(spd)                          # [N, N, H]

        # mask out cross-graph pairs so graphs in a batch don't attend to each other
        if batch is not None:
            cross_graph = batch.unsqueeze(0) != batch.unsqueeze(1)  # [N, N]
            attn = attn.masked_fill(cross_graph.unsqueeze(-1), float('-inf'))

        # local mode: a node may attend only to its neighbours and itself (1 hop/layer)
        if self.local:
            adj = torch.zeros(N, N, dtype=torch.bool, device=x.device)
            adj[edge_index[0], edge_index[1]] = True
            adj[torch.arange(N, device=x.device), torch.arange(N, device=x.device)] = True
            attn = attn.masked_fill(~adj.unsqueeze(-1), float('-inf'))

        attn = torch.softmax(attn, dim=1)  # softmax over source nodes j
        if self.store_attn:
            self.last_attn = attn.detach()  # [N, N, heads] for inspection
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        # weighted sum of values
        out = torch.einsum('ijh,jhd->ihd', attn, V)
        return out.reshape(N, self.out_channels) + self.bias
