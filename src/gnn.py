"""The message-passing GNN, selected by `model: gnn`."""
from .graph_conv import GraphConvNet


class GNN(GraphConvNet):
    """Message-passing GNN — a GraphConvNet whose layers are local aggregators
    (gcn / sage / gat / gin), enforced by config validation for `model: gnn`."""
