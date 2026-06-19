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


class GNN(nn.Module):
    """
    Configurable GNN that stacks any combination of GCN / SAGE / GAT / GIN layers.

    Architecture:
        input -> [conv -> (bn) -> relu -> dropout] x L -> (pooling) -> linear -> logits

    The conv layers are fully determined by config.layers.
    The final linear maps from the last conv output to config.out_channels.
    """
    def __init__(self, config: GNNConfig):
        super().__init__()
        self.config = config
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList() if config.batch_norm else None

        in_ch = config.in_channels
        for layer_cfg in config.layers:
            layer_cls = LAYER_MAP[layer_cfg["type"]]
            out_ch = layer_cfg.get("out_channels", config.hidden_channels)
            # pass any extra keys (heads, dropout, eps, ...) directly to the layer
            kwargs = {k: v for k, v in layer_cfg.items() if k not in ("type", "out_channels")}
            self.convs.append(layer_cls(in_ch, out_ch, **kwargs))
            if self.norms is not None:
                self.norms.append(nn.BatchNorm1d(out_ch))
            in_ch = out_ch

        if config.task == "graph":
            self.pool = POOL_MAP[config.pooling]
        else:
            self.pool = None

        self.classifier = nn.Linear(in_ch, config.out_channels)
        self.dropout_p = config.dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, "batch", None)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, batch=batch)
            if self.norms is not None:
                x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)

        if self.pool is not None:
            x = self.pool(x, batch)

        return self.classifier(x)

    def embed(self, data):
        """Node embeddings after all conv layers, before classifier. Useful for inspection."""
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, "batch", None)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, batch=batch)
            if self.norms is not None:
                x = self.norms[i](x)
            x = F.relu(x)
        return x

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
