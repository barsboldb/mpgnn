"""Architecture dispatcher: pick the model from the config.

The models themselves live in their own modules — gnn.GNN (message passing),
transformer.GraphTransformer (attention) — built on the shared graph_conv engine.
"""
import torch.nn as nn

from .config import GNNConfig


def build_model(config: GNNConfig) -> nn.Module:
    """Pick the architecture from config.model (resolved in GNNConfig.__post_init__).

    'gnn'         -> GNN (message passing: gcn / sage / gat / gin)
    'transformer' -> GraphTransformer (attention; tokenization selects node vs
                     node_edge layout, with CoT available)
    """
    if config.model == "transformer":
        from .transformer import GraphTransformer
        return GraphTransformer(config)
    from .gnn import GNN
    return GNN(config)
