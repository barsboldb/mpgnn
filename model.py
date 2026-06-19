import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool

from layers import GCNConv, SAGEConv, GATConv, GINConv, GlobalAttnConv
from config import GNNConfig


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
    raw_in = config.in_channels + config.lpe_dim
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


class GNN(nn.Module):
    """
    Configurable GNN that stacks any combination of GCN / SAGE / GAT / GIN / GlobalAttn layers.

    Architecture:
        input
          -> [input_embedding]
          -> L x [conv -> residual? -> norm? -> relu -> dropout]
          -> [pooling]
          -> linear
          -> logits

    Transformer path:  input_embedding=linear, residual=True, norm_type=layer, layers=[global_attn ...]
    mpGNN path:        residual=False, norm_type=batch (or null), layers=[gcn/gat/gin ...]
    """
    def __init__(self, config: GNNConfig):
        super().__init__()
        self.config = config
        self.input_emb = _build_input_embedding(config)

        raw_in = config.in_channels + config.lpe_dim
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

        self.pool = POOL_MAP[config.pooling] if config.task == "graph" else None
        self.classifier = nn.Linear(in_ch, config.out_channels)
        self.dropout_p = config.dropout

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

        if self.pool is not None:
            x = self.pool(x, batch)

        return self.classifier(x)

    def embed(self, data) -> torch.Tensor:
        """Node embeddings after all conv layers, before classifier."""
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


def build_model(config: GNNConfig) -> nn.Module:
    """Pick the model from config.tokenization.

    'node'      -> GNN (message passing, or node-only global attention)
    'node_edge' -> GraphTokenTransformer (Sanford-style edge tokens + task token)
    """
    if getattr(config, "tokenization", "node") == "node_edge":
        from token_model import GraphTokenTransformer
        return GraphTokenTransformer(config)
    return GNN(config)
