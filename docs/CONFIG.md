# Config Reference

All experiments are configured via a single YAML file passed with `--config config.yaml`.
Every field maps 1-to-1 to `GNNConfig` in `config.py`.

---

## Top-level fields

### Data shape

| Field | Type | Description |
|---|---|---|
| `in_channels` | int | Number of input node features. Must match the dataset (see table below). |
| `out_channels` | int | Number of output classes. |
| `hidden_channels` | int | Width of every hidden conv layer (default: 64). |

Dataset reference:

| Dataset | `in_channels` | `out_channels` |
|---|---|---|
| `connectedness` | 1 (normalised degree) | 2 |
| `isomorphism` | 2 (one-hot graph ID) | 2 |
| `mutag` | 7 (atom type one-hot) | 2 |
| `cora` | 1433 (BoW features) | 7 |

---

### Task and pooling

| Field | Values | Description |
|---|---|---|
| `task` | `node` / `graph` | Node classification vs. graph classification. |
| `pooling` | `mean` / `add` / `max` | Global pooling applied after the conv stack (graph task only). |

---

### Regularisation

| Field | Type | Description |
|---|---|---|
| `dropout` | float (0–1) | Dropout probability applied after each conv layer. |
| `norm_type` | `batch` / `layer` / `null` | Normalisation applied after each conv + residual. `batch` → `BatchNorm1d`, `layer` → `LayerNorm`, `null` → none. |
| `residual` | bool | Whether to add a skip connection `x = x + conv(x)` at each layer. When in/out dims differ, a learnable `Linear` projection is inserted automatically. |

`norm_type: layer` + `residual: true` is the standard Transformer pre-norm pattern.
`norm_type: batch` + `residual: false` is the standard mpGNN pattern.

These fields apply to the **node-token path only**. With `tokenization: node_edge` the
model uses fixed pre-norm residual blocks regardless of these settings (see Tokenization).

---

### Input embedding

| Field | Values | Description |
|---|---|---|
| `input_embedding` | `linear` / `mlp` / `lookup` / `null` | Optional projection applied to raw node features before the first conv layer. |

- `linear` — `Linear(in_channels + lpe_dim, hidden_channels)`. Useful when raw features have a different dimension than `hidden_channels`.
- `mlp` — two-layer MLP with ReLU. More expressive linear.
- `lookup` — `Embedding(in_channels, hidden_channels)`. Use when node features are discrete integer type IDs (`in_channels` = vocabulary size).
- `null` — no projection; first conv layer receives raw features directly.

---

### Structural encodings

| Field | Type | Description |
|---|---|---|
| `lpe_dim` | int | Number of Laplacian eigenvectors concatenated to node features. `0` disables LPE. |

When `lpe_dim > 0`, the eigenvectors are pre-computed and stored in `data.pe` at dataset creation time. The effective input to the first layer becomes `in_channels + lpe_dim`.

---

### Tokenization (graph task)

| Field | Values | Description |
|---|---|---|
| `tokenization` | `node` / `node_edge` | How the graph is presented to the model. |
| `node_id_dim` | int | Random per-node identity width for the `node_edge` model. `0` disables. |

- `node` (default) — vertices are the only tokens. Edges enter via message passing (`gcn`/`gin`/`gat`/`sage`) or, for `global_attn`, via the SPD bias and/or LPE. Built by `model.GNN`.
- `node_edge` — **Sanford et al. 2024a** style. The input sequence is `[vertex tokens] + [edge tokens] + [task token]`, so edges are *first-class tokens* the transformer reasons over. The prediction is read out from the task token. Built by `token_model.GraphTokenTransformer`.

For `node_edge`, each node is given a random identity vector of width `node_id_dim` (concatenated with LPE if `lpe_dim > 0`) so that an edge token can reference its two endpoints. **You must set `node_id_dim > 0` or `lpe_dim > 0`** — otherwise edges are anonymous and the model is rejected by config validation.

> The `node_edge` model always uses pre-norm residual transformer blocks (LayerNorm + residual around attention and FFN) with a 4× FFN. The `norm_type`, `residual`, `input_embedding`, and `pooling` fields apply **only to the `node` path** and are ignored here. The `layers` list is used only for its length (depth) and per-entry `heads`; the `type`/`spd_max_dist` keys are ignored.

This is the representation for reproducing Sanford's depth-vs-task results: connectivity is parallelizable and expected to need ~log(n) depth when the model must compute reachability itself (no SPD/LPE shortcut). Sweep depth by adding/removing entries in `layers`.

---

### Training hyperparameters

| Field | Type | Description |
|---|---|---|
| `epochs` | int | Number of training epochs. |
| `lr` | float | Learning rate for Adam. |
| `weight_decay` | float | L2 regularisation coefficient. |
| `batch_size` | int | Mini-batch size for graph-level tasks. |

---

## `layers` block

A list of conv layer definitions. Each entry must have a `type` key plus any type-specific options.

```yaml
layers:
  - type: gcn
  - type: sage
  - type: gat
    heads: 4
    dropout: 0.1
  - type: gin
    eps: 0.0
    train_eps: true
  - type: global_attn
    heads: 4
    dropout: 0.1
    spd_max_dist: 5
```

Every layer inherits `hidden_channels` as its output width. Override per-layer with `out_channels`:

```yaml
  - type: gcn
    out_channels: 128
```

### Layer types

| Type | Paper | Aggregation | Key options |
|---|---|---|---|
| `gcn` | Kipf & Welling 2017 | Symmetric-normalised sum | — |
| `sage` | Hamilton et al. 2017 | Mean + concat with self | — |
| `gat` | Veličković et al. 2018 | Attention-weighted sum, local neighbours | `heads`, `dropout` |
| `gin` | Xu et al. 2019 | Sum + MLP (1-WL expressive) | `eps`, `train_eps` |
| `global_attn` | Transformer-style | All-pairs self-attention (no edge restriction) | `heads`, `dropout`, `spd_max_dist` |

`spd_max_dist` (global_attn only): adds a learnable per-head bias for each shortest-path distance bucket (Graphormer-style). `0` disables it.

---

## Recipes

### Transformer path
```yaml
input_embedding: linear
norm_type: layer
residual: true
layers:
  - type: global_attn
    heads: 4
  - type: global_attn
    heads: 4
  - type: global_attn
    heads: 4
  - type: global_attn
    heads: 4
```

### Standard mpGNN path
```yaml
input_embedding: null
norm_type: batch
residual: false
layers:
  - type: gin
  - type: gin
  - type: gin
```

### Transformer with SPD bias
```yaml
input_embedding: linear
norm_type: layer
residual: true
lpe_dim: 8
layers:
  - type: global_attn
    heads: 4
    spd_max_dist: 5
  - type: global_attn
    heads: 4
    spd_max_dist: 5
```

### Mixed (local then global)
```yaml
norm_type: batch
residual: false
layers:
  - type: gin
  - type: gin
  - type: global_attn
    heads: 4
```

### Edge-token transformer (Sanford-style)
Vertices + edges + task token; depth sweep for the connectivity reasoning curve.
```yaml
tokenization: node_edge
node_id_dim: 16     # random node identities; raise lpe_dim instead/also for isomorphism
lpe_dim: 0          # 0 = honest connectivity (no precomputed reachability)
hidden_channels: 64
dropout: 0.1
layers:             # number of entries = depth; sweep this
  - type: global_attn
    heads: 4
  - type: global_attn
    heads: 4
  - type: global_attn
    heads: 4
  - type: global_attn
    heads: 4
```
