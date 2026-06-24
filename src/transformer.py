"""Attention-based graph models.

`GraphTransformer` is the user-facing model selected by `model: transformer`. It
dispatches on `tokenization`:

  - 'node'      — tokens are node-feature vectors (degree / adj_rows / lap / ...);
                  global self-attention with a pooled (or per-node) readout. This
                  uses the shared graph_conv.GraphConvNet engine with global_attn
                  layers, so SPD bias, LPE, input embedding and pooling all apply.
  - 'node_edge' — Sanford-style: vertices + edges + a task token (+ optional CoT),
                  with the prediction read from the task token. Implemented by
                  `_GraphTokenTransformer` below.

──────────────────────────────────────────────────────────────────────────────
Sanford-style node_edge transformer (Sanford et al. 2024a).

The input sequence for each graph is

    [ vertex tokens ] + [ edge tokens ] + [ one task token ]

so edges are *first-class tokens* the transformer attends over and reasons about.
The prediction is read out from the task token (a learned CLS-style query).

Nodes are given random identities so an edge token can reference its two endpoints
(an edge token is built from the identities of the nodes it connects). LPE, when
enabled, is concatenated onto those identities — this is where LPE earns its keep
for the isomorphism task. There is no SPD bias here; SPD/LPE remain available on
the node-token path (graph_conv.GraphConvNet / layers.GlobalAttnConv).

This is the representation needed to reproduce Sanford's depth-vs-task results:
connectivity is a parallelizable task expected to need ~log(n) depth when the model
must compute reachability itself (i.e. without SPD/LPE precomputing it).

Chain-of-thought (config.cot_len > 0): K learnable scratchpad tokens are inserted
between the edge tokens and the task token,

    [ vertex tokens ] + [ edge tokens ] + [ c_1 ... c_K ] + [ task token ]

with a structured attention mask: graph tokens attend only among themselves (a
read-only problem statement), each scratchpad token c_i reads the graph and the
earlier scratchpad tokens (causal) or all of them (full), and the task token reads
everything. This buys *sequential* computation (one round per scratchpad token)
that is orthogonal to depth — see reports/cot-tokens.typ.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import GNNConfig


class _EncoderBlock(nn.Module):
    """Pre-norm transformer encoder block: multi-head self-attention + FFN."""

    def __init__(self, dim: int, heads: int, dropout: float, ffn_mult: int = 4):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_mult * dim),
            nn.ReLU(),
            nn.Linear(ffn_mult * dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor,
                attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        am = None
        if attn_mask is not None:
            # attn_mask is [B, L, L] additive float; MultiheadAttention wants
            # [B*heads, L, L] with batch b, head h at row b*heads + h.
            am = attn_mask.repeat_interleave(self.heads, dim=0)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask,
                         attn_mask=am, need_weights=False)
        x = x + self.dropout(a)
        h = self.norm2(x)
        x = x + self.dropout(self.ffn(h))
        return x


class _GraphTokenTransformer(nn.Module):
    TYPE_VERTEX, TYPE_EDGE, TYPE_TASK, TYPE_COT = 0, 1, 2, 3

    def __init__(self, config: GNNConfig):
        super().__init__()
        self.config = config
        d = config.hidden_channels
        self.node_id_dim = config.node_id_dim
        self.node_id_mode = config.node_id_mode
        self.cot_len = config.cot_len
        self.cot_mask = config.cot_mask
        # node_edge -> vertices + edges + task; node -> vertices + task (no edge tokens),
        # which is how `tokenization: node` gets a scratchpad + task-token readout for CoT.
        self.use_edges = config.tokenization == "node_edge"
        id_dim = config.node_id_dim + config.lpe_dim  # per-node identity width

        # 'learned' identities: an embedding indexed by within-graph node position
        self.node_pos_emb = (
            nn.Embedding(config.max_nodes, config.node_id_dim)
            if config.node_id_dim > 0 and config.node_id_mode == "learned" else None
        )

        # vertex embedding honours config.input_embedding ('mlp' -> 2-layer, else linear)
        vert_in = config.in_channels + id_dim
        if config.input_embedding == "mlp":
            self.vert_proj = nn.Sequential(nn.Linear(vert_in, d), nn.ReLU(), nn.Linear(d, d))
        else:
            self.vert_proj = nn.Linear(vert_in, d)
        self.edge_proj = nn.Linear(id_dim, d) if self.use_edges else None  # summed endpoint identities
        self.task_token = nn.Parameter(torch.randn(d) * 0.02)
        # K learnable scratchpad tokens (chain-of-thought); empty when cot_len == 0
        self.cot = nn.Parameter(torch.randn(self.cot_len, d) * 0.02) if self.cot_len > 0 else None
        self.type_emb = nn.Embedding(4, d)             # vertex / edge / task / cot

        default_heads = config.layers[0].get("heads", 4) if config.layers else 4
        self.blocks = nn.ModuleList([
            _EncoderBlock(d, lc.get("heads", default_heads), config.dropout)
            for lc in config.layers
        ])
        self.norm = nn.LayerNorm(d)
        self.classifier = nn.Linear(d, config.out_channels)

    def _within_graph_position(self, batch: torch.Tensor, N: int, device) -> torch.Tensor:
        """Rank of each node inside its own graph (PyG batches nodes contiguously)."""
        B = int(batch.max().item()) + 1
        counts = torch.zeros(B, device=device).scatter_add_(
            0, batch, torch.ones(N, device=device))
        offsets = torch.cat([torch.zeros(1, device=device), counts.cumsum(0)[:-1]])
        return (torch.arange(N, device=device) - offsets[batch]).long()

    def _node_identities(self, batch: torch.Tensor, N: int,
                         pe: torch.Tensor | None, device) -> torch.Tensor:
        parts = []
        if self.node_id_dim > 0:
            if self.node_pos_emb is not None:            # learned, deterministic
                pos = self._within_graph_position(batch, N, device)
                parts.append(self.node_pos_emb(pos))
            else:                                        # fresh random each forward
                parts.append(torch.randn(N, self.node_id_dim, device=device))
        if self.config.lpe_dim > 0 and pe is not None:
            parts.append(pe)
        return torch.cat(parts, dim=-1) if parts else torch.zeros(N, 0, device=device)

    def _cot_attn_mask(self, seqs, L: int, device) -> torch.Tensor:
        """Boolean [B, L, L] attention mask for the scratchpad layout (True = block).

        Per graph the real positions are [vertices+edges | c_1..c_K | task]:
          - graph tokens attend only among themselves (read-only problem statement),
          - scratchpad c_i attends to the graph and c_{<=i} (causal) or all c (full),
          - the task token attends to everything.
        Padded query rows are left all-False (key_padding_mask removes padded keys),
        so no row is fully masked (which would produce NaNs after softmax). Boolean
        so it matches the boolean key_padding_mask passed to MultiheadAttention.
        """
        B, K = len(seqs), self.cot_len
        mask = torch.zeros(B, L, L, dtype=torch.bool, device=device)

        if self.cot_mask == "causal" and K > 1:
            future = torch.triu(torch.ones(K, K, dtype=torch.bool, device=device), diagonal=1)
        else:
            future = None

        for i, s in enumerate(seqs):
            rl = s.size(0)          # real length
            t = rl - 1              # task position
            c0 = t - K              # first scratchpad position == graph end (exclusive)
            if c0 > 0:              # graph queries: forbid scratchpad + task keys
                mask[i, 0:c0, c0:rl] = True
            mask[i, c0:t, t] = True  # scratchpad queries: forbid the task key
            if future is not None:
                mask[i, c0:t, c0:t] = future  # forbid future scratchpad keys
        return mask

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        device = x.device
        N = x.size(0)
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(N, dtype=torch.long, device=device)
        B = int(batch.max().item()) + 1

        ident = self._node_identities(batch, N, getattr(data, "pe", None), device)  # [N, id_dim]

        type_v = self.type_emb.weight[self.TYPE_VERTEX]
        type_t = self.type_emb.weight[self.TYPE_TASK]
        ctok_base = (self.cot + self.type_emb.weight[self.TYPE_COT]) if self.cot_len > 0 else None

        if self.use_edges:
            # undirected edges only (each stored as both directions in edge_index)
            und = edge_index[0] < edge_index[1]
            e_src, e_dst = edge_index[0][und], edge_index[1][und]
            e_batch = batch[e_src]
            type_e = self.type_emb.weight[self.TYPE_EDGE]

        seqs, task_pos = [], []
        for g in range(B):
            n_idx = (batch == g).nonzero(as_tuple=True)[0]
            vtok = self.vert_proj(torch.cat([x[n_idx], ident[n_idx]], dim=-1)) + type_v

            if self.use_edges:
                emask = e_batch == g
                if emask.any():
                    eu, ev = e_src[emask], e_dst[emask]
                    etok = self.edge_proj(ident[eu] + ident[ev]) + type_e  # symmetric in u,v
                else:
                    etok = torch.zeros(0, vtok.size(-1), device=device)
            else:                                       # node tokenization: no edge tokens
                etok = torch.zeros(0, vtok.size(-1), device=device)

            ttok = (self.task_token + type_t).unsqueeze(0)
            parts = [vtok, etok]
            if ctok_base is not None:                  # [c_1 ... c_K] before the task token
                parts.append(ctok_base)
            parts.append(ttok)
            seq = torch.cat(parts, dim=0)              # [L_g, d]
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

        attn_mask = self._cot_attn_mask(seqs, L, device) if self.cot_len > 0 else None

        for blk in self.blocks:
            h = blk(h, key_pad, attn_mask)
        h = self.norm(h)

        task = h[torch.arange(B, device=device), torch.tensor(task_pos, device=device)]
        return self.classifier(task)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GraphTransformer(nn.Module):
    """Attention-based graph model; `tokenization` selects the token layout.

    - 'node_edge' -> vertices + edges + task token (+ optional CoT), task-token readout.
    - 'node'      -> node-feature tokens through a stack of global-attention layers
                     (the shared GraphConvNet engine with global_attn layers):
                     configurable input embedding, SPD bias, LPE, and pooled (graph)
                     or per-node (node task) readout.

    The 'node' path uses the shared graph_conv.GraphConvNet engine — the same
    scaffold gnn.GNN uses, but with attention layers — so this is a sibling of
    GNN over a common engine, not a wrapper around the GNN class.
    """
    def __init__(self, config: GNNConfig):
        super().__init__()
        self.config = config
        if config.tokenization == "node_edge" or config.cot_len > 0:
            # node_edge, or node tokenization + CoT: token sequence with a task-token
            # readout (the only path with scratchpad tokens). Edge tokens are included
            # only for node_edge (see _GraphTokenTransformer.use_edges).
            self.net: nn.Module = _GraphTokenTransformer(config)
        else:
            from .graph_conv import GraphConvNet
            self.net = GraphConvNet(config)   # node-token attention via the shared engine

    def forward(self, data):
        return self.net(data)

    def embed(self, data):
        if not hasattr(self.net, "embed"):
            raise NotImplementedError("embed() is only available for tokenization: node")
        return self.net.embed(data)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
