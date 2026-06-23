# Config Reference

All experiments are configured via a single YAML file passed with `--config config.yaml`.
Every field maps 1-to-1 to `GNNConfig` in `src/config.py`.

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
| `pooling` | `mean` / `add` / `max` / `pair` | Global pooling applied after the conv stack (graph task only). |

- `mean` / `add` / `max` — standard global pooling over all nodes.
- `pair` — **isomorphism pairs only**. Pools G1 and G2 nodes separately using `mean`, then concatenates the two graph-level vectors before the classifier. The classifier sees `[h_G1 ‖ h_G2]` (width `2 × hidden_channels`). Requires `data.n1` to be set (produced automatically for the `isomorphism` dataset).

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

### Node features / tokenization input

`node_features` controls how raw graph structure becomes node feature vectors before the model sees them.

| Value | Shape | Description |
|---|---|---|
| `degree` | `[n, 1]` | Normalised degree `deg(v) / max_deg`. Safe default; breaks symmetry. **Does not work on `connectedness_hard`** — degree distributions are matched by construction. |
| `constant` | `[n, 1]` | All-ones vector. Fully anonymous nodes; useful as a lower bound or when LPE carries all structure. |
| `adj_rows` | `[n, max_n]` | Each node's row in the adjacency matrix, zero-padded to the largest graph in the dataset. Encodes full neighbourhood in one vector — strong structural signal but dimension grows with dataset. |
| `membership` | `[n, 2]` | One-hot component flag: `[1,0]` for nodes in G1, `[0,1]` for nodes in G2. **Isomorphism pairs only** — requires `data.n1`. Tells the model which half of the pair it's looking at. |
| `lap` | `[n, lpe_dim]` | Laplacian eigenvectors used *as* features (not appended). `in_channels` must equal `lpe_dim`. See LPE section below. |

> `in_channels` in the config must match the feature width. For `degree`/`constant`: 1. For `adj_rows`: `max_nodes` of the dataset. For `membership`: 2. For `lap`: equal to `lpe_dim`.

---

### Structural encodings

| Field | Type | Description |
|---|---|---|
| `lpe_dim` | int | Number of Laplacian eigenvectors. `0` disables LPE entirely. |

When `lpe_dim > 0` **and** `node_features != "lap"`: eigenvectors are appended to node features and stored in `data.pe`. The effective input to the first layer becomes `in_channels + lpe_dim`.

When `node_features == "lap"`: eigenvectors *are* the features (`data.x`), not a separate field. Set `in_channels = lpe_dim`.

**Zero-eigenvalue filtering:** the normalized Laplacian of a graph with C connected components has exactly C eigenvalues equal to 0. Each corresponding eigenvector is an indicator function for one component — it directly encodes which component a node belongs to. The LPE implementation skips all eigenvectors whose eigenvalue is below `1e-5`, not just the first one. This means:
- Connected graph (C=1): same as before — skip 1 trivial vector.
- Disconnected graph (C>1): skip all C component-indicator vectors. The remaining eigenvectors encode intra-component geometry only.

Without this fix, `lpe_dim > 0` on `connectedness_hard` would hand the model C−1 component-membership vectors as input features, making the task trivially solvable from the embedding rather than from reasoning.

---

### Tokenization (graph task)

| Field | Values | Description |
|---|---|---|
| `tokenization` | `node` / `node_edge` | How the graph is presented to the model. |
| `node_id_dim` | int | Random per-node identity width for the `node_edge` model. `0` disables. |

- `node` (default) — vertices are the only tokens. Edges enter via message passing (`gcn`/`gin`/`gat`/`sage`) or, for `global_attn`, via the SPD bias and/or LPE. Built by `src.model.GNN`.
- `node_edge` — **Sanford et al. 2024a** style. The input sequence is `[vertex tokens] + [edge tokens] + [task token]`, so edges are *first-class tokens* the transformer reasons over. The prediction is read out from the task token. Built by `src.token_model.GraphTokenTransformer`.

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

### Adjacency-row transformer (honest connectivity, no LPE)
Each node's token is its full adjacency row — who it's connected to — but no positional encoding.
The only information the model has is the local neighbourhood structure.
```yaml
node_features: adj_rows
in_channels: 24        # max_nodes for connectedness_hard
tokenization: node
input_embedding: linear
norm_type: layer
residual: true
lpe_dim: 0             # no LPE: no connectivity leak
task: graph
pooling: mean
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

### Laplacian eigenvector tokenization (isomorphism)
Eigenvectors are the features. Isomorphic graphs share the same eigenspectrum so the
representations are structurally matched across the pair. Uses `pair` pooling so the
classifier sees both halves.
```yaml
node_features: lap
in_channels: 16        # must equal lpe_dim
lpe_dim: 16
tokenization: node
input_embedding: linear
norm_type: layer
residual: true
task: graph
pooling: pair          # pools G1 and G2 separately, then concatenates
layers:
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
