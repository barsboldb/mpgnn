from dataclasses import dataclass, field
from typing import Any
import yaml


@dataclass
class GNNConfig:
    """
    Central config for a GNN experiment.

    layers: list of dicts, each with a 'type' key (gcn | sage | gat | gin)
            plus optional layer-specific kwargs:
              gat:  heads (int), dropout (float)
              gin:  eps (float), train_eps (bool)
            and optional 'out_channels' to override hidden_channels for that layer.

    task:    'node'  — node classification (uses node embeddings directly)
             'graph' — graph classification (global pooling before classifier)

    pooling: 'mean' | 'add' | 'max'  (only used when task='graph')
    """
    in_channels: int
    out_channels: int
    hidden_channels: int = 64
    layers: list[dict[str, Any]] = field(default_factory=lambda: [
        {"type": "gcn"},
        {"type": "gcn"},
    ])
    task: str = "node"
    pooling: str = "mean"
    dropout: float = 0.5
    # norm_type: 'batch' (BatchNorm1d), 'layer' (LayerNorm), or null (no norm)
    # use 'layer' + residual=True for the transformer path
    # use 'batch' + residual=False for the standard mpGNN path
    norm_type: str | None = None
    residual: bool = False
    # input embedding applied before conv layers
    # 'linear'  — nn.Linear(in_channels, hidden_channels)
    # 'mlp'     — two-layer MLP with ReLU
    # 'lookup'  — nn.Embedding for discrete integer node types; in_channels = num_node_types
    # null/None — no embedding; first conv layer takes raw in_channels directly
    input_embedding: str | None = None
    # structural encodings
    # lpe_dim: number of Laplacian eigenvectors concatenated to node features (0 = disabled)
    lpe_dim: int = 0
    # tokenization (graph task only):
    # 'node'      — vertices are the only tokens; edges enter via message passing
    #               (GCN/GIN/GAT) or, for global_attn, via SPD bias / LPE. -> model.GNN
    # 'node_edge' — Sanford-style: tokens = vertices + edges + a task token; edges are
    #               first-class tokens the transformer reasons over. -> GraphTokenTransformer
    tokenization: str = "node"
    # node_id_dim: per-node identity dim for the node_edge transformer, so edge
    # tokens can reference their endpoints. Combined with lpe_dim as the node identity.
    node_id_dim: int = 0
    # node_id_mode: how those identities are produced (node_edge only)
    #   'learned' — nn.Embedding indexed by within-graph node position. Deterministic
    #               and trainable; breaks permutation invariance like an absolute
    #               positional encoding, but actually optimizes. (default)
    #   'random'  — fresh Gaussian vectors each forward. Permutation-invariant in
    #               expectation, but SGD has no stable signal to bind edges to nodes,
    #               so the model tends to ignore them and stalls at chance.
    node_id_mode: str = "learned"
    # max_nodes: size of the learned node-position embedding table (must exceed the
    # largest single-graph node count in the dataset).
    max_nodes: int = 128
    # training
    epochs: int = 200
    lr: float = 0.01
    weight_decay: float = 5e-4
    batch_size: int = 32

    def __post_init__(self):
        assert self.task in ("node", "graph"), f"task must be 'node' or 'graph', got '{self.task}'"
        assert self.pooling in ("mean", "add", "max"), f"unknown pooling '{self.pooling}'"
        assert self.input_embedding in (None, "linear", "mlp", "lookup"), \
            f"unknown input_embedding '{self.input_embedding}'"
        assert self.norm_type in (None, "batch", "layer"), \
            f"norm_type must be 'batch', 'layer', or null — got '{self.norm_type}'"
        assert self.tokenization in ("node", "node_edge"), \
            f"tokenization must be 'node' or 'node_edge' — got '{self.tokenization}'"
        if self.tokenization == "node_edge":
            assert self.node_id_dim + self.lpe_dim > 0, \
                "node_edge tokenization needs node_id_dim > 0 (or lpe_dim > 0) so edge " \
                "tokens can reference their endpoints; otherwise edges are anonymous"
            assert self.node_id_mode in ("learned", "random"), \
                f"node_id_mode must be 'learned' or 'random' — got '{self.node_id_mode}'"
        for i, layer in enumerate(self.layers):
            assert "type" in layer, f"layer {i} is missing 'type'"
            assert layer["type"] in ("gcn", "sage", "gat", "gin", "global_attn"), \
                f"layer {i}: unknown type '{layer['type']}'"

    @classmethod
    def from_yaml(cls, path: str) -> "GNNConfig":
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)

    @classmethod
    def from_dict(cls, d: dict) -> "GNNConfig":
        return cls(**d)

    def describe(self) -> str:
        emb = self.input_embedding or "none"
        lpe = f"LPE({self.lpe_dim})" if self.lpe_dim > 0 else "no LPE"
        lines = [
            f"Tokenization: {self.tokenization}"
            + (f"  |  node_id_dim: {self.node_id_dim}" if self.tokenization == "node_edge" else ""),
            f"Task: {self.task}  |  Pooling: {self.pooling if self.task == 'graph' else '-'}",
            f"In: {self.in_channels}  +{lpe}  ->  [{emb} emb]  ->  Hidden: {self.hidden_channels}  ->  Out: {self.out_channels}",
            # node_edge uses fixed pre-norm residual transformer blocks; norm_type/residual
            # only apply to the node-token GNN path.
            f"Dropout: {self.dropout}  |  Norm: layer (pre-norm)  |  Residual: yes (transformer blocks)"
            if self.tokenization == "node_edge" else
            f"Dropout: {self.dropout}  |  Norm: {self.norm_type or 'none'}  |  Residual: {self.residual}",
            f"Epochs: {self.epochs}  |  LR: {self.lr}  |  Weight decay: {self.weight_decay}  |  Batch size: {self.batch_size}",
            "Layers:",
        ]
        for i, l in enumerate(self.layers):
            extras = {k: v for k, v in l.items() if k not in ("type", "out_channels")}
            out = l.get("out_channels", self.hidden_channels)
            lines.append(f"  [{i}] {l['type'].upper():6s}  out={out}  {extras if extras else ''}")
        return "\n".join(lines)
