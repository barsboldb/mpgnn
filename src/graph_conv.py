"""Shared engine: a configurable stack of graph layers over node-feature tokens.

This is the neutral scaffold that both architectures build on — it is *not* a GNN
or a Transformer by itself, just the common "embed -> L x (layer -> residual? ->
norm? -> relu -> dropout) -> readout" pipeline. The layer type decides the
character:

  - message passing (gcn/sage/gat/gin)  -> gnn.GNN
  - all-pairs attention (global_attn)   -> transformer.GraphTransformer (node tokens)

Keeping it here means GNN and GraphTransformer are siblings over a shared engine,
rather than one importing the other.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool

from .layers import GCNConv, SAGEConv, GATConv, GINConv, GlobalAttnConv
from .config import GNNConfig


LAYER_MAP = {
    "gcn":         GCNConv,
    "sage":        SAGEConv,
    "gat":         GATConv,
    "gin":         GINConv,
    "global_attn": GlobalAttnConv,
}

POOL_MAP = {
    "mean": global_mean_pool,
    "add":  global_add_pool,
    "max":  global_max_pool,
}


def _build_input_embedding(config: GNNConfig) -> nn.Module | None:
    """
    linear  — nn.Linear projection for continuous features
    mlp     — two-layer MLP with ReLU, richer transformation
    lookup  — nn.Embedding table for discrete integer node types;
               in_channels is treated as the vocabulary size (num node types)
    None    — no embedding; first conv layer receives raw features directly
    """
    h = config.hidden_channels
    # "lap" uses lpe_dim eigenvectors as the primary features (stored in x, not pe)
    # so lpe_dim must not be added again here
    raw_in = config.in_channels if config.node_features == "lap" \
        else config.in_channels + config.lpe_dim
    if config.input_embedding == "linear":
        return nn.Linear(raw_in, h)
    if config.input_embedding == "mlp":
        return nn.Sequential(
            nn.Linear(raw_in, h),
            nn.ReLU(),
            nn.Linear(h, h),
        )
    if config.input_embedding == "lookup":
        return nn.Embedding(config.in_channels, h)
    return None


def _build_norm(norm_type: str | None, channels: int) -> nn.Module | None:
    if norm_type == "batch":
        return nn.BatchNorm1d(channels)
    if norm_type == "layer":
        return nn.LayerNorm(channels)
    return None


class GraphConvNet(nn.Module):
    """A stack of graph layers over node-feature tokens.

    Architecture:
        input
          -> [input_embedding]
          -> L x [layer -> residual? -> norm? -> relu -> dropout]
          -> [pooling]            (graph task)
          -> linear -> logits

    The layer type is set by config.layers; message-passing and global-attention
    layers run through the identical pipeline.
    """
    def __init__(self, config: GNNConfig):
        super().__init__()
        self.config = config
        self.input_emb = _build_input_embedding(config)

        raw_in = config.in_channels if config.node_features == "lap" \
            else config.in_channels + config.lpe_dim
        in_ch = config.hidden_channels if self.input_emb is not None else raw_in

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        # per-layer residual projection: only needed when in_ch != out_ch
        self.res_projs = nn.ModuleList()

        for layer_cfg in config.layers:
            layer_cls = LAYER_MAP[layer_cfg["type"]]
            out_ch = layer_cfg.get("out_channels", config.hidden_channels)
            kwargs = {k: v for k, v in layer_cfg.items() if k not in ("type", "out_channels")}

            self.convs.append(layer_cls(in_ch, out_ch, **kwargs))
            self.norms.append(_build_norm(config.norm_type, out_ch))  # type: ignore[arg-type]
            # linear projection for residual when dimensions change
            self.res_projs.append(
                nn.Linear(in_ch, out_ch, bias=False) if (config.residual and in_ch != out_ch) else None
            )
            in_ch = out_ch

        self.pool = POOL_MAP.get(config.pooling) if config.task == "graph" else None
        # pair pooling: pool G1 and G2 nodes separately → concat → classify
        # classifier input doubles because we concatenate two pooled vectors
        classifier_in = (2 * in_ch) if config.pooling == "pair" else in_ch
        self.classifier = nn.Linear(classifier_in, config.out_channels)
        self.dropout_p = config.dropout

    def _pair_pool(self, x: torch.Tensor, batch: torch.Tensor, n1_per_graph: torch.Tensor) -> torch.Tensor:
        """Pool G1 and G2 nodes of each graph separately, return [B, 2*hidden].

        n1_per_graph[i] = number of G1 nodes in graph i (G2 nodes follow immediately).
        Builds a split_batch where G1 nodes of graph i map to slot 2*i and
        G2 nodes to slot 2*i+1, then mean-pools to get one vector per component.
        """
        B = n1_per_graph.size(0)
        device = x.device

        node_counts = torch.bincount(batch, minlength=B)          # [B] total nodes per graph
        offsets = torch.zeros(B, dtype=torch.long, device=device)
        offsets[1:] = node_counts[:-1].cumsum(0)                  # start index of each graph

        within_idx = torch.arange(x.size(0), device=device) - offsets[batch]
        is_g2 = within_idx >= n1_per_graph[batch]

        split_batch = batch * 2
        split_batch[is_g2] += 1                                    # G2 nodes → odd slots

        pooled = global_mean_pool(x, split_batch, size=2 * B)       # [2*B, hidden]
        return pooled.view(B, -1)                                   # [B, 2*hidden]

    def _prepare_input(self, data) -> torch.Tensor:
        x = data.x
        if self.config.lpe_dim > 0 and hasattr(data, "pe") and data.pe is not None:
            x = torch.cat([x, data.pe], dim=-1)
        if self.input_emb is not None:
            if self.config.input_embedding == "lookup":
                x = x.squeeze(-1).long()
            x = self.input_emb(x)
        return x

    def forward(self, data):
        x = self._prepare_input(data)
        edge_index = data.edge_index
        batch = getattr(data, "batch", None)

        for i, conv in enumerate(self.convs):
            conv_out = conv(x, edge_index, batch=batch)

            if self.config.residual:
                proj = self.res_projs[i]
                x_skip = proj(x) if proj is not None else x
                conv_out = x_skip + conv_out

            norm = self.norms[i]
            if norm is not None:
                conv_out = norm(conv_out)

            x = F.relu(conv_out)
            x = F.dropout(x, p=self.dropout_p, training=self.training)

        if self.config.pooling == "pair":
            x = self._pair_pool(x, batch, data.n1)
        elif self.pool is not None:
            x = self.pool(x, batch)

        return self.classifier(x)

    def embed(self, data) -> torch.Tensor:
        """Node embeddings after all layers, before the classifier."""
        x = self._prepare_input(data)
        edge_index = data.edge_index
        batch = getattr(data, "batch", None)

        for i, conv in enumerate(self.convs):
            conv_out = conv(x, edge_index, batch=batch)
            if self.config.residual:
                proj = self.res_projs[i]
                x_skip = proj(x) if proj is not None else x
                conv_out = x_skip + conv_out
            norm = self.norms[i]
            if norm is not None:
                conv_out = norm(conv_out)
            x = F.relu(conv_out)
        return x

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
