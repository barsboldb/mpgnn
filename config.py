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
            f"Task: {self.task}  |  Pooling: {self.pooling if self.task == 'graph' else '-'}",
            f"In: {self.in_channels}  +{lpe}  ->  [{emb} emb]  ->  Hidden: {self.hidden_channels}  ->  Out: {self.out_channels}",
            f"Dropout: {self.dropout}  |  Norm: {self.norm_type or 'none'}  |  Residual: {self.residual}",
            f"Epochs: {self.epochs}  |  LR: {self.lr}  |  Weight decay: {self.weight_decay}  |  Batch size: {self.batch_size}",
            "Layers:",
        ]
        for i, l in enumerate(self.layers):
            extras = {k: v for k, v in l.items() if k not in ("type", "out_channels")}
            out = l.get("out_channels", self.hidden_channels)
            lines.append(f"  [{i}] {l['type'].upper():6s}  out={out}  {extras if extras else ''}")
        return "\n".join(lines)
