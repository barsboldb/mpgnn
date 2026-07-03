"""Graph-native BDH ("Dragon Hatchling"), selected by `model: bdh`.

Adapts the BDH-GPU layer (Kosowski et al. 2025, "The Dragon Hatchling", arXiv
2509.26507; reference implementation in bdh-research-emergency/bdh.py) from a causal
language model into a graph model that plugs into this repo's engine interfaces
(embed / forward / num_parameters), so `model: bdh` is a sibling of gnn.GNN and
transformer.GraphTransformer over the same datasets and readouts.

What is kept from BDH (the signature mechanisms):
  - one shared parameter set (encoder E, encoder_v, decoder) reused for every round
    L = len(config.layers)  — the Universal-Transformer / "4L rounds ↔ depth" design;
  - a wide, ReLU-*positive-sparse* neuron space N = D·mult/nh reached by the encoder,
    where attention actually happens (the opposite of a Transformer's small head dim);
  - *linear* attention (no softmax): scores = QKᵀ with Q=K = the positive sparse
    activations — BDH's Hebbian "edge-reweighting" read;
  - the multiplicative gate  x_sparse ⊙ y_sparse  before the decoder projects back to D;
  - LayerNorm without affine params, and a residual per round.

What changes for graphs (documented deviations from the LM reference):
  - No RoPE and no causal `.tril` mask: a graph is an unordered *set* of nodes, not a
    sequence, so there is no position and no past/future. Attention is bidirectional
    within a graph (the node-token transformer path here makes the same choice).
  - Attention is masked to one graph at a time (batched into [B, T, D] with a validity
    mask) so it never leaks across graphs in a PyG batch.
  - `bdh_adj_mask: true` restricts attention to graph edges (+ self) — BDH as *local*
    message passing, reach one hop per round; the default (false) is all-pairs, the
    graph-transformer reading of BDH.
  - Scores are normalised by the number of attended keys per query (a weighted mean,
    not a raw sum) so activation scale is stable across the wide range of graph sizes
    in these datasets. The LM keeps the raw sum; here variable n would otherwise make
    the scale depend on graph size.

Readouts reuse the repo's conventions: pooled logits (graph task), per-node logits
(node task), and the pairwise reachability readout R̂ = H W Hᵀ (connectivity task),
identical to transformer.GraphTransformer so results are directly comparable.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GNNConfig
from .graph_conv import _build_input_embedding, POOL_MAP


class BDHGraph(nn.Module):
    def __init__(self, config: GNNConfig):
        super().__init__()
        self.config = config
        D = config.hidden_channels
        nh = config.layers[0].get("heads", 4) if config.layers else 4
        N = D * config.bdh_neuron_mult // nh
        assert N > 0, (
            f"bdh: neuron dim N = hidden_channels·bdh_neuron_mult/heads = "
            f"{D}·{config.bdh_neuron_mult}/{nh} = {N} must be > 0; raise hidden_channels "
            f"or bdh_neuron_mult, or lower heads"
        )
        self.D, self.nh, self.N = D, nh, N
        self.n_round = len(config.layers)          # L rounds ↔ Transformer depth
        self.connectivity = config.task == "connectivity"

        # input: raw features (+ LPE) -> D, honouring config.input_embedding.
        raw_in = config.in_channels if config.node_features == "lap" \
            else config.in_channels + config.lpe_dim
        self.input_emb = _build_input_embedding(config)
        # no configured embedding -> a plain linear still has to land in D (BDH works in D)
        self.in_proj = nn.Linear(raw_in, D) if self.input_emb is None else None

        # one shared parameter set, reused for every round (Universal-Transformer style)
        self.encoder = nn.Parameter(torch.zeros(nh, D, N).normal_(std=0.02))
        self.encoder_v = nn.Parameter(torch.zeros(nh, D, N).normal_(std=0.02))
        self.decoder = nn.Parameter(torch.zeros(nh * N, D).normal_(std=0.02))
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.drop = nn.Dropout(config.dropout)

        if self.connectivity:
            # pairwise reachability readout R̂ = H W Hᵀ (Ye et al. 2026), matching
            # transformer.GraphTransformer so the two models are directly comparable.
            self.pair = nn.Linear(D, D, bias=False)
        else:
            self.pool = POOL_MAP.get(config.pooling) if config.task == "graph" else None
            self.classifier = nn.Linear(D, config.out_channels)

    # ── input ────────────────────────────────────────────────────────────────
    def _prepare_input(self, data) -> torch.Tensor:
        x = data.x
        if self.config.lpe_dim > 0 and getattr(data, "pe", None) is not None:
            x = torch.cat([x, data.pe], dim=-1)
        if self.input_emb is not None:
            if self.config.input_embedding == "lookup":
                x = x.squeeze(-1).long()
            x = self.input_emb(x)
        else:
            x = self.in_proj(x)
        return x                                    # [N, D]

    # ── batching helpers: [N, D] <-> padded [B, T, D] ─────────────────────────
    def _pad(self, h: torch.Tensor, batch: torch.Tensor, B: int):
        device = h.device
        counts = torch.bincount(batch, minlength=B)
        T = int(counts.max().item())
        offsets = torch.zeros(B, dtype=torch.long, device=device)
        offsets[1:] = counts[:-1].cumsum(0)
        wpos = (torch.arange(h.size(0), device=device) - offsets[batch])  # rank within graph
        h_pad = h.new_zeros(B, T, h.size(-1))
        h_pad[batch, wpos] = h
        valid = torch.zeros(B, T, dtype=torch.bool, device=device)
        valid[batch, wpos] = True
        return h_pad, valid, wpos, T

    def _adj_allowed(self, edge_index, batch, wpos, B, T, valid) -> torch.Tensor:
        """[B, T, T] boolean: True where query may attend key (graph edges + self)."""
        allowed = torch.zeros(B, T, T, dtype=torch.bool, device=valid.device)
        e0, e1 = edge_index
        allowed[batch[e0], wpos[e0], wpos[e1]] = True     # both directions already in edge_index
        b, t = valid.nonzero(as_tuple=True)
        allowed[b, t, t] = True                           # self-attention
        return allowed

    # ── core: L rounds of the BDH layer over graph tokens ─────────────────────
    def _encode(self, data) -> torch.Tensor:
        """Node embeddings [N, D] after L BDH rounds, in the original PyG node order."""
        h = self._prepare_input(data)                     # [N, D]
        N_nodes = h.size(0)
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(N_nodes, dtype=torch.long, device=h.device)
        B = int(batch.max().item()) + 1

        h_pad, valid, wpos, T = self._pad(h, batch, B)    # [B, T, D]

        if self.config.bdh_adj_mask:
            keymask = self._adj_allowed(data.edge_index, batch, wpos, B, T, valid).float()
        else:
            keymask = valid.float()[:, None, :].expand(B, T, T)   # every valid key visible
        denom = keymask.sum(-1).clamp(min=1.0)            # [B, T] attended keys per query
        keymask = keymask[:, None]                        # [B, 1, T, T] broadcast over heads
        denom = denom[:, None, :, None]                   # [B, 1, T, 1]

        x = self.ln(h_pad).unsqueeze(1)                   # [B, 1, T, D]
        # neuron-space occupancy probe (eval only, so training pays nothing):
        # fraction of nonzero x_sparse activations over valid tokens, averaged
        # across rounds — verifies the positive-*sparse* regime BDH claims.
        track = not self.training
        nz = cnt = 0.0
        vmask = valid[:, None, :, None]                   # [B, 1, T, 1]
        for _ in range(self.n_round):
            x_res = x
            x_sparse = F.relu(x @ self.encoder)           # [B, nh, T, N] positive sparse keys
            if track:
                nz += ((x_sparse > 0) & vmask).sum()
                cnt += valid.sum() * (self.nh * self.N)
            # linear attention (no softmax): Q = K = x_sparse, V = x (dense working vec)
            scores = (x_sparse @ x_sparse.mT) * keymask   # [B, nh, T, T]
            yKV = (scores @ x) / denom                    # [B, nh, T, D] weighted mean
            yKV = self.ln(yKV)
            y_sparse = F.relu(yKV @ self.encoder_v)       # [B, nh, T, N]
            xy = self.drop(x_sparse * y_sparse)           # multiplicative gate in neuron space
            yMLP = xy.transpose(1, 2).reshape(B, 1, T, self.N * self.nh) @ self.decoder
            x = self.ln(x_res + self.ln(yMLP))            # residual round

        if track and torch.is_tensor(cnt) and cnt.item() > 0:
            self.last_sparsity = round((nz / cnt).item(), 4)

        h_out = x.view(B, T, self.D)
        return h_out[batch, wpos]                         # gather back to [N, D]

    # ── public interface (mirrors GraphTransformer) ──────────────────────────
    def embed(self, data) -> torch.Tensor:
        return self._encode(data)

    def forward(self, data):
        if self.connectivity:
            return self._connectivity_forward(data)
        h = self._encode(data)                            # [N, D]
        if self.config.task == "graph":
            if self.config.pooling == "pair":
                raise NotImplementedError(
                    "model: bdh does not support pooling: pair; use mean/add/max"
                )
            batch = getattr(data, "batch", None)
            if batch is None:
                batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
            h = self.pool(h, batch)
        return self.classifier(h)

    def _connectivity_forward(self, data):
        h = self._encode(data)
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        out = []
        for g in range(int(batch.max().item()) + 1):
            hg = h[batch == g]                            # [n_g, D]
            lg = self.pair(hg) @ hg.transpose(0, 1)       # H W Hᵀ -> [n_g, n_g]
            out.append(0.5 * (lg + lg.transpose(0, 1)))   # symmetric, like R
        return out

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
