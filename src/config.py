import os
from dataclasses import dataclass, field, fields
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
    # experiment identity: which dataset to run on and how to generate it, so one
    # YAML fully specifies a run. CLI --dataset takes precedence over `dataset`.
    # dataset_kwargs are forwarded to the generator (and become part of the cache
    # key), e.g. {num_graphs: 8000, max_diameter: 12} for diameter_controlled.
    dataset: str | None = None
    dataset_kwargs: dict[str, Any] = field(default_factory=dict)
    # seed: global RNG seed (python/numpy/torch) set at the start of every run,
    # covering weight init and the per-load "random" node features.
    seed: int = 42
    # train_frac: head/tail split fraction. Generators alternate labels (i % 2),
    # so a sequential split stays class-balanced and is stable across runs.
    train_frac: float = 0.8
    # model: which architecture the front program initializes.
    #   'gnn'         — message passing (gcn/sage/gat/gin)            -> gnn.GNN
    #   'transformer' — attention; tokenization selects the layout    -> transformer.GraphTransformer
    #   'bdh'         — graph-native Dragon Hatchling (Kosowski 2025)  -> bdh.BDHGraph
    #   None          — inferred: node_edge or any global_attn layer => transformer, else gnn
    model: str | None = None
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
    # node_features: how raw graph structure becomes node feature vectors x
    #   "degree"     — normalised degree [n, 1]; safe default
    #   "constant"   — ones [n, 1]; ablation / when structure comes from LPE or edge tokens
    #   "random"     — random vector per node [n, in_channels] (rGIN); symmetry-breaking
    #                  features for reachability/connectivity reasoning
    #   "adj_rows"   — adjacency row padded to max dataset node count [n, max_n]
    #   "membership" — one-hot component flag [n, 2] for isomorphism pairs (needs data.n1)
    #   "lap"        — Laplacian eigenvectors as features [n, lpe_dim]; in_channels must = lpe_dim
    node_features: str = "degree"
    # lpe_dim: number of Laplacian eigenvectors concatenated to node features (0 = disabled)
    lpe_dim: int = 0
    # tokenization (graph task only):
    # 'node'      — vertices are the only tokens; edges enter via message passing
    #               (GCN/GIN/GAT) or, for global_attn, via SPD bias / LPE. -> gnn.GNN
    # 'node_edge' — Sanford-style: tokens = vertices + edges + a task token; edges are
    #               first-class tokens the transformer reasons over. -> transformer.GraphTransformer
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
    # Chain-of-thought. cot_mode selects the mechanism:
    #   'none'           — no CoT (default).
    #   'scratchpad'     — K learnable scratchpad tokens spliced into the encoder
    #                      sequence (the original cot_len experiment; kept for
    #                      comparison). Configured by cot_len / cot_mask below.
    #   'autoregressive' — genuine CoT: the graph is serialized to discrete tokens,
    #                      the dataset supplies a supervised BFS trace, a decoder-only
    #                      transformer (src/cot.py) is teacher-forced on trace+answer
    #                      and decodes greedily at eval. node_features/tokenization
    #                      are unused on this path (the prompt is the edge list).
    cot_mode: str = "none"
    # ── scratchpad knobs (cot_mode: scratchpad) ──
    # cot_len: number K of learnable scratchpad tokens inserted between the edge
    #          tokens and the task token. cot_len > 0 with cot_mode unset implies
    #          cot_mode: scratchpad (backward compat with old configs/checkpoints).
    # cot_mask: attention structure over the scratchpad —
    #   'causal' — c_i attends to the graph and c_{<=i} (BFS-style rounds)
    #   'full'   — c_i attends to the graph and all scratchpad tokens
    #   In both, graph tokens stay isolated (attend only among themselves, becoming
    #   a read-only problem statement) and the task token attends to everything.
    cot_len: int = 0
    cot_mask: str = "causal"
    # ── autoregressive knobs (cot_mode: autoregressive) ──
    # max_trace_len: cap on generated tokens at eval; 0 = answer-only ablation
    #                (the completion is just `ANS YES|NO EOS`, same architecture).
    # max_seq_len:   position-table size and hard bound on prompt+trace length.
    # trace_format:  'bfs_levels' — compact frontier-by-frontier BFS trace;
    #                'bfs_expand' — verbose `EXP parent children` rounds (~2x
    #                longer, every next token locally computable — the target
    #                that survives the Bachmann & Nagarajan next-token pitfall);
    #                'bfs_check' — bfs_expand with every visited-set membership
    #                test emitted as a supervised `v YES|NO` pair (the dense-graph
    #                rung: rejections are silent in bfs_expand, CHANGELOG 07-07);
    #                'wl_expand' — 1-WL colour refinement on isomorphism pairs
    #                (dataset: iso_wl), same local-decomposition trick per round
    #                plus an explicit final-histogram comparison section.
    # permute_node_ids: randomly relabel nodes per graph so contiguous-id layouts
    #                (connectedness_hard blobs, caterpillar backbones) don't leak.
    # cot_pos:       'learned' absolute positions | 'none' (NoPE; length OOD).
    # answer_loss_weight: CE weight on the YES/NO target position.
    # cot_eval_every: greedy-decode eval cadence in epochs (decode is ~L sequential
    #                forwards; teacher-forced answer accuracy is logged every epoch).
    # prompt_roster: include the `N v_0..v_{n-1}` node roster in the prompt. At
    #                fixed n it carries no information but gives every node id a
    #                guaranteed occurrence outside the edge list — distractor mass
    #                for content-matching heads. Set false to drop it (prompt
    #                becomes `E u w .. TRACE`).
    # tie_embeddings: share lm_head with tok_emb (standard LM tying). Tying forces
    #                "match X as a key" and "emit X as output" into one vector,
    #                which can conflict in lookup circuits on tiny vocabularies.
    max_trace_len: int = 64
    max_seq_len: int = 320
    trace_format: str = "bfs_levels"
    permute_node_ids: bool = True
    cot_pos: str = "learned"
    answer_loss_weight: float = 1.0
    cot_eval_every: int = 1
    prompt_roster: bool = True
    tie_embeddings: bool = True
    # BDH (model: bdh) — graph-native Dragon Hatchling (Kosowski et al. 2025).
    # Depth L and per-model head count nh come from `layers` (each entry type: bdh,
    # len(layers) = number of shared-parameter rounds, layers[0].heads = nh).
    # bdh_neuron_mult: neuron-space width multiplier; N = hidden_channels·mult/nh is
    #                  the wide, positive-sparse dimension where BDH's attention happens.
    # bdh_adj_mask:    False — all-pairs attention within a graph (graph-transformer BDH);
    #                  True  — attention restricted to edges + self (local message-passing
    #                          BDH, reach one hop per round).
    bdh_neuron_mult: int = 16
    bdh_adj_mask: bool = False
    # training
    epochs: int = 200
    lr: float = 0.01
    weight_decay: float = 5e-4
    batch_size: int = 32

    def _infer_model(self) -> str:
        if self.tokenization == "node_edge":
            return "transformer"
        if any(l.get("type") == "global_attn" for l in self.layers):
            return "transformer"
        return "gnn"

    def __post_init__(self):
        assert self.task in ("node", "graph", "connectivity"), \
            f"task must be 'node', 'graph', or 'connectivity', got '{self.task}'"
        assert 0.0 < self.train_frac < 1.0, \
            f"train_frac must be in (0, 1), got {self.train_frac}"
        # resolve the model selector (None -> inferred) so it is concrete everywhere
        if self.model is None:
            self.model = self._infer_model()
        assert self.model in ("gnn", "transformer", "bdh"), \
            f"model must be 'gnn', 'transformer', or 'bdh' — got '{self.model}'"
        assert self.pooling in ("mean", "add", "max", "pair"), f"unknown pooling '{self.pooling}'"
        assert self.node_features in ("degree", "constant", "random", "adj_rows", "adj_self", "membership", "lap"), \
            f"unknown node_features '{self.node_features}'"
        if self.task == "connectivity":
            assert self.model in ("transformer", "bdh") and self.tokenization == "node", \
                "task: connectivity needs model: transformer or bdh + tokenization: node " \
                "(node tokens through the encoder, then a pairwise reachability readout)"
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
        assert self.cot_len >= 0, f"cot_len must be >= 0, got {self.cot_len}"
        assert self.cot_mask in ("causal", "full"), \
            f"cot_mask must be 'causal' or 'full' — got '{self.cot_mask}'"
        assert self.cot_mode in ("none", "scratchpad", "autoregressive"), \
            f"cot_mode must be 'none', 'scratchpad', or 'autoregressive' — got '{self.cot_mode}'"
        if self.cot_mode == "none" and self.cot_len > 0:
            self.cot_mode = "scratchpad"   # legacy configs/checkpoints set only cot_len
        if self.cot_mode == "scratchpad":
            assert self.cot_len > 0, "cot_mode: scratchpad needs cot_len > 0"
            assert self.model == "transformer", \
                "scratchpad CoT requires model: transformer (scratchpad lives in the token sequence)"
            # node + CoT and node_edge both run the token transformer; layers only set depth/heads
        if self.cot_mode == "autoregressive":
            assert self.model == "transformer" and self.task == "graph", \
                "cot_mode: autoregressive needs model: transformer and task: graph " \
                "(decoder-only LM over graph tokens with a YES/NO answer)"
            assert self.cot_pos in ("learned", "none"), \
                f"cot_pos must be 'learned' or 'none' — got '{self.cot_pos}'"
            assert self.trace_format in ("bfs_levels", "bfs_expand", "bfs_check",
                                         "wl_expand", "bfs_l1"), \
                f"unknown trace_format '{self.trace_format}' " \
                "(choose 'bfs_levels', 'bfs_expand', 'bfs_check', 'wl_expand', " \
                "or the 'bfs_l1' lookup probe)"
            assert self.max_trace_len >= 0 and self.max_seq_len > 0 and self.cot_eval_every >= 1
        message_passing = ("gcn", "sage", "gat", "gin")
        # node tokenization runs the global_attn conv stack ONLY when there is no
        # scratchpad; with one it runs the token transformer (layers = depth/heads).
        node_conv_stack = self.tokenization == "node" and self.cot_mode != "scratchpad"
        for i, layer in enumerate(self.layers):
            assert "type" in layer, f"layer {i} is missing 'type'"
            t = layer["type"]
            assert t in message_passing + ("global_attn", "bdh"), \
                f"layer {i}: unknown type '{t}'"
            if self.model == "gnn":
                assert t in message_passing, \
                    f"layer {i}: model 'gnn' supports message-passing layers {message_passing}; " \
                    f"got '{t}'. Use model: transformer for global_attn."
            elif self.model == "bdh":
                # layers only supply depth (len) and heads; each must be type: bdh
                assert t == "bdh", \
                    f"layer {i}: model 'bdh' uses 'bdh' layers (they set depth L and heads); got '{t}'."
            elif node_conv_stack and self.task != "connectivity":
                # transformer + node tokenization, no CoT: a stack of global_attn layers
                assert t == "global_attn", \
                    f"layer {i}: model 'transformer' with tokenization 'node' uses global_attn " \
                    f"layers; got '{t}'."
            # task: connectivity runs the same embed -> pairwise readout for ANY layer
            # type (global_attn for the Sanford/Ye transformer, or message-passing
            # gat/gin/gcn for the GNN baseline — same engine, different conv).
            # transformer + node_edge (or node + CoT): layers only supply depth and per-layer heads

    @classmethod
    def yaml_dict(cls, path: str) -> dict:
        """Load a config YAML, resolving `extends: <relative-path>` recursively.
        Child fields wholly replace base fields (including `layers` and
        `dataset_kwargs` — no deep merge, so a config always reads literally)."""
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        base = d.pop("extends", None)
        if base:
            merged = cls.yaml_dict(os.path.join(os.path.dirname(path), base))
            merged.update(d)
            d = merged
        return d

    @classmethod
    def from_yaml(cls, path: str) -> "GNNConfig":
        return cls(**cls.yaml_dict(path))

    @classmethod
    def from_dict(cls, d: dict) -> "GNNConfig":
        """Build from a dict, dropping unknown keys so checkpoints saved with an
        older/newer config schema still load."""
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            print(f"[config] ignoring unknown keys from checkpoint: {sorted(unknown)}")
        return cls(**{k: v for k, v in d.items() if k in known})

    def architecture_family(self) -> str:
        """The reasoning mechanism, independent of the host engine class:
        message-passing GNN (edge-restricted convs), global-attention transformer,
        or BDH. For the connectivity task every run reports model == 'transformer'
        or 'bdh' because the pairwise readout lives in those classes — the layer
        type is what actually distinguishes the architectures."""
        lt = self.layers[0]["type"] if self.layers else None
        if self.model == "bdh" or lt == "bdh":
            return "BDH"
        if lt in ("gcn", "sage", "gat", "gin"):
            return "message-passing GNN"
        if lt == "global_attn":
            return "global-attention transformer"
        return self.model or "gnn"

    def describe(self) -> str:
        emb = self.input_embedding or "none"
        lpe = f"lap({self.lpe_dim}) as features" if self.node_features == "lap" \
            else (f"+LPE({self.lpe_dim})" if self.lpe_dim > 0 else "no LPE")
        lines = [
            f"Dataset: {self.dataset or '(from CLI)'}"
            + (f"  {self.dataset_kwargs}" if self.dataset_kwargs else "")
            + f"  |  Seed: {self.seed}  |  Train frac: {self.train_frac}",
            # 'engine' is the host class (gnn/transformer/bdh); for connectivity the
            # transformer engine may run message-passing (gat/gin) layers — the family
            # line names the actual mechanism so the two never look contradictory.
            f"Architecture: {self.architecture_family()}  |  Engine: {self.model}",
            f"Tokenization: {self.tokenization}"
            + (f"  |  node_id_dim: {self.node_id_dim}" if self.tokenization == "node_edge" else "")
            + (f"  |  CoT: {self.cot_len} scratchpad tokens ({self.cot_mask})"
               if self.cot_mode == "scratchpad" else "")
            + (f"  |  CoT: autoregressive ({self.trace_format}, "
               f"{'trace' if self.max_trace_len > 0 else 'ANSWER-ONLY'}, "
               f"pos={self.cot_pos}, max_seq={self.max_seq_len}, "
               f"permute_ids={self.permute_node_ids})"
               if self.cot_mode == "autoregressive" else ""),
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
