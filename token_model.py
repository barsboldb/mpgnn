"""Sanford-style graph transformer (Sanford et al. 2024a).

Unlike the node-only path in model.py (where edges enter as an SPD attention bias
or via LPE), here the input sequence for each graph is

    [ vertex tokens ] + [ edge tokens ] + [ one task token ]

so edges are *first-class tokens* the transformer attends over and reasons about.
The prediction is read out from the task token (a learned CLS-style query).

Nodes are given random identities so an edge token can reference its two endpoints
(an edge token is built from the identities of the nodes it connects). LPE, when
enabled, is concatenated onto those identities — this is where LPE earns its keep
for the isomorphism task. There is no SPD bias here; SPD/LPE remain available on
the node-token path (model.GNN / layers.GlobalAttnConv).

This is the representation needed to reproduce Sanford's depth-vs-task results:
connectivity is a parallelizable task expected to need ~log(n) depth when the model
must compute reachability itself (i.e. without SPD/LPE precomputing it).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from config import GNNConfig


class _EncoderBlock(nn.Module):
    """Pre-norm transformer encoder block: multi-head self-attention + FFN."""

    def __init__(self, dim: int, heads: int, dropout: float, ffn_mult: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_mult * dim),
            nn.ReLU(),
            nn.Linear(ffn_mult * dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.dropout(a)
        h = self.norm2(x)
        x = x + self.dropout(self.ffn(h))
        return x


class GraphTokenTransformer(nn.Module):
    TYPE_VERTEX, TYPE_EDGE, TYPE_TASK = 0, 1, 2

    def __init__(self, config: GNNConfig):
        super().__init__()
        self.config = config
        d = config.hidden_channels
        self.node_id_dim = config.node_id_dim
        id_dim = config.node_id_dim + config.lpe_dim  # per-node identity width

        self.vert_proj = nn.Linear(config.in_channels + id_dim, d)
        self.edge_proj = nn.Linear(id_dim, d)          # from summed endpoint identities
        self.task_token = nn.Parameter(torch.randn(d) * 0.02)
        self.type_emb = nn.Embedding(3, d)             # vertex / edge / task

        default_heads = config.layers[0].get("heads", 4) if config.layers else 4
        self.blocks = nn.ModuleList([
            _EncoderBlock(d, lc.get("heads", default_heads), config.dropout)
            for lc in config.layers
        ])
        self.norm = nn.LayerNorm(d)
        self.classifier = nn.Linear(d, config.out_channels)

    def _node_identities(self, n: int, pe: torch.Tensor | None, device) -> torch.Tensor:
        parts = []
        if self.node_id_dim > 0:
            # fresh random identities each forward -> relabeling-invariant in expectation,
            # forces the model to use structure rather than memorize node positions.
            parts.append(torch.randn(n, self.node_id_dim, device=device))
        if self.config.lpe_dim > 0 and pe is not None:
            parts.append(pe)
        return torch.cat(parts, dim=-1) if parts else torch.zeros(n, 0, device=device)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        device = x.device
        N = x.size(0)
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(N, dtype=torch.long, device=device)
        B = int(batch.max().item()) + 1

        ident = self._node_identities(N, getattr(data, "pe", None), device)  # [N, id_dim]

        # undirected edges only (each stored as both directions in edge_index)
        und = edge_index[0] < edge_index[1]
        e_src, e_dst = edge_index[0][und], edge_index[1][und]
        e_batch = batch[e_src]

        type_v = self.type_emb.weight[self.TYPE_VERTEX]
        type_e = self.type_emb.weight[self.TYPE_EDGE]
        type_t = self.type_emb.weight[self.TYPE_TASK]

        seqs, task_pos = [], []
        for g in range(B):
            n_idx = (batch == g).nonzero(as_tuple=True)[0]
            vtok = self.vert_proj(torch.cat([x[n_idx], ident[n_idx]], dim=-1)) + type_v

            emask = e_batch == g
            if emask.any():
                eu, ev = e_src[emask], e_dst[emask]
                etok = self.edge_proj(ident[eu] + ident[ev]) + type_e  # symmetric in u,v
            else:
                etok = torch.zeros(0, vtok.size(-1), device=device)

            ttok = (self.task_token + type_t).unsqueeze(0)
            seq = torch.cat([vtok, etok, ttok], dim=0)  # [L_g, d]
            seqs.append(seq)
            task_pos.append(seq.size(0) - 1)

        # pad ragged sequences into [B, L, d] with a key-padding mask (True = ignore)
        L = max(s.size(0) for s in seqs)
        d = seqs[0].size(-1)
        h = torch.zeros(B, L, d, device=device)
        key_pad = torch.ones(B, L, dtype=torch.bool, device=device)
        for i, s in enumerate(seqs):
            h[i, :s.size(0)] = s
            key_pad[i, :s.size(0)] = False

        for blk in self.blocks:
            h = blk(h, key_pad)
        h = self.norm(h)

        task = h[torch.arange(B, device=device), torch.tensor(task_pos, device=device)]
        return self.classifier(task)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
