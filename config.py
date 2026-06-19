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
    batch_norm: bool = True
    # training
    epochs: int = 200
    lr: float = 0.01
    weight_decay: float = 5e-4
    batch_size: int = 32

    def __post_init__(self):
        assert self.task in ("node", "graph"), f"task must be 'node' or 'graph', got '{self.task}'"
        assert self.pooling in ("mean", "add", "max"), f"unknown pooling '{self.pooling}'"
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
        lines = [
            f"Task: {self.task}  |  Pooling: {self.pooling if self.task == 'graph' else '-'}",
            f"In: {self.in_channels}  ->  Hidden: {self.hidden_channels}  ->  Out: {self.out_channels}",
            f"Dropout: {self.dropout}  |  BatchNorm: {self.batch_norm}",
            f"Epochs: {self.epochs}  |  LR: {self.lr}  |  Weight decay: {self.weight_decay}  |  Batch size: {self.batch_size}",
            "Layers:",
        ]
        for i, l in enumerate(self.layers):
            extras = {k: v for k, v in l.items() if k not in ("type", "out_channels")}
            out = l.get("out_channels", self.hidden_channels)
            lines.append(f"  [{i}] {l['type'].upper():6s}  out={out}  {extras if extras else ''}")
        return "\n".join(lines)
