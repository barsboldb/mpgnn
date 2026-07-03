# Config Reference

All experiments are configured via a single YAML file passed with `--config configs/<name>.yaml`.
Every field maps 1-to-1 to `GNNConfig` in `src/config.py`. See `configs/README.md` for the
catalogue of experiments.

Configs can inherit shared defaults with `extends: <relative-path>`: the base file is
loaded first and child fields wholly replace base fields (no deep merge — `layers` and
`dataset_kwargs` are taken from whichever file defines them last). Files starting with
`_` (e.g. `_connectivity_base.yaml`, `_graph_base.yaml`) are bases, not runnable experiments.

---

## Top-level fields

### Experiment identity

| Field | Type | Description |
|---|---|---|
| `dataset` | str / *null* | Which dataset to run on (any `GENERATORS` key, or `cora`/`mutag`/…). CLI `--dataset` takes precedence. |
| `dataset_kwargs` | dict | Forwarded to the generator and included in the cache key, e.g. `{num_graphs: 8000, max_diameter: 12}`. |
| `seed` | int | Global RNG seed (python/numpy/torch) set at run start; covers weight init and `random` node features (default: 42). |
| `train_frac` | float | Head/tail split fraction (default: 0.8). Generators alternate labels, so a sequential split stays balanced. |

Any config field can be overridden from the CLI without editing the YAML:

```bash
python main.py --config configs/connectivity_gat.yaml -o epochs=50 -o lr=0.003 \
    -o dataset_kwargs.num_graphs=500
```

Values are YAML-parsed; one dot reaches into dict fields.

---

### Model selection

| Field | Values | Description |
|---|---|---|
| `model` | `gnn` / `transformer` / *null* | Which architecture the front program (`build_model`) initializes. |

- `gnn` — message passing (`gcn`/`sage`/`gat`/`gin`). Built by `src.gnn.GNN`.
- `transformer` — attention-based. Built by `src.transformer.GraphTransformer`, which dispatches on `tokenization` (see below): `node` runs a stack of `global_attn` layers over node-feature tokens (configurable embedding, SPD bias, LPE, pooling); `node_edge` runs the vertex+edge+task pipeline with optional chain-of-thought.
- *null* (omitted) — **inferred** for backward compatibility: `tokenization: node_edge` or any `global_attn` layer → `transformer`; otherwise `gnn`. The resolved value is recorded in the run JSON.

Validation ties layers to the model: `gnn` accepts only message-passing layers; `transformer` + `tokenization: node` accepts only `global_attn`; `transformer` + `tokenization: node_edge` uses the `layers` list only for depth and per-layer `heads`.

---

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
| `cot_mode` | `none` / `scratchpad` / `autoregressive` | Chain-of-thought mechanism (see below). `cot_len > 0` with `cot_mode` unset implies `scratchpad` (legacy configs keep working). |
| `cot_len` | int | Scratchpad only: number of learnable scratchpad tokens. `0` disables. |
| `cot_mask` | `causal` / `full` | Scratchpad only: attention structure over the scratchpad. |
| `trace_format` | `bfs_levels` / `bfs_expand` | Autoregressive only: compact sorted frontiers vs verbose `EXP parent children` rounds (locally computable next-tokens; ~2x longer). |
| `max_trace_len` | int | Autoregressive only: decode budget at eval. **`0` = answer-only ablation** (same architecture, completion is just `ANS YES\|NO EOS`). |
| `max_seq_len` | int | Autoregressive only: position-table size and hard bound on prompt+trace length (asserted at sequence build). |
| `permute_node_ids` | bool | Autoregressive only: randomly relabel nodes per graph. Keep `true` — generators lay components on contiguous id ranges, which otherwise leaks the label. |
| `cot_pos` | `learned` / `none` | Autoregressive only: absolute positions vs NoPE (for length-OOD runs). |
| `answer_loss_weight` | float | Autoregressive only: extra CE weight on the YES/NO target position (1.0 = off). |
| `cot_eval_every` | int | Autoregressive only: greedy-decode eval cadence in epochs (teacher-forced answer accuracy logs every epoch as `test_tf`). |

- `node` (default) — vertices are the only tokens. With `model: gnn` edges enter via message passing (`gcn`/`gin`/`gat`/`sage`); with `model: transformer` they enter via all-pairs `global_attn` (plus the SPD bias and/or LPE). Built by `src.gnn.GNN` (the shared conv engine, which `GraphTransformer` reuses for `node` tokenization).
- `node_edge` — **Sanford et al. 2024a** style. The input sequence is `[vertex tokens] + [edge tokens] + [task token]`, so edges are *first-class tokens* the transformer reasons over. The prediction is read out from the task token. Built by `src.transformer.GraphTransformer` (requires `model: transformer`).

For `node_edge`, each node is given a random identity vector of width `node_id_dim` (concatenated with LPE if `lpe_dim > 0`) so that an edge token can reference its two endpoints. **You must set `node_id_dim > 0` or `lpe_dim > 0`** — otherwise edges are anonymous and the model is rejected by config validation.

> The `node_edge` model always uses pre-norm residual transformer blocks (LayerNorm + residual around attention and FFN) with a 4× FFN. The `norm_type`, `residual`, `input_embedding`, and `pooling` fields apply **only to the `node` path** and are ignored here. The `layers` list is used only for its length (depth) and per-entry `heads`; the `type`/`spd_max_dist` keys are ignored.

#### Chain-of-thought scratchpad (`cot_mode: scratchpad`)

With `cot_len = K`, `K` learnable scratchpad tokens are inserted before the task token, giving the model **sequential** computation (one round per token) that is orthogonal to depth. They carry no input — blank slots the model fills with intermediate state. Requires `model: transformer`; works with either tokenization:

- `tokenization: node_edge` → `[vertices] + [edges] + [c_1 … c_K] + [task]`
- `tokenization: node` → `[vertices] + [c_1 … c_K] + [task]` (no edge tokens; e.g. adj_rows vertex features — see `configs/adj_cot.yaml`)

Enabling CoT switches `tokenization: node` from the pooled `global_attn` conv stack to this token transformer with a task-token readout (so the `layers` list then supplies only depth + per-layer `heads`, and pooling is unused). The no-CoT (`cot_len: 0`) node path keeps the original pooled conv-stack behavior.

A structured attention mask makes the scratchpad behave like algorithm rounds:
- **Graph tokens** (vertices + edges) attend only among themselves — a read-only problem statement (this differs from the plain `node_edge` model, where attention is fully bidirectional).
- **Scratchpad token `c_i`** reads the graph and, with `cot_mask: causal`, the earlier scratchpad tokens `c_{≤i}` (BFS-style rounds); with `cot_mask: full`, all scratchpad tokens.
- **Task token** reads everything; the answer is read from it.

Under the BFS reading, round `c_k` can hold the `k`-hop reachable set, so connectivity needs `cot_len ≥ diameter`. Sweep `cot_len` (and keep depth shallow) to look for a phase transition at the diameter — keep `lpe_dim: 0` and no SPD so the structural answer isn't precomputed into the features. Full writeup: `reports/cot-tokens.typ`.

This is the representation for reproducing Sanford's depth-vs-task results: connectivity is parallelizable and expected to need ~log(n) depth when the model must compute reachability itself (no SPD/LPE shortcut). Sweep depth by adding/removing entries in `layers`.

> **Status (2026-07-03):** the scratchpad never beat the no-CoT baseline and the reasons are structural — the slots are unsupervised, input-independent, bypassable by the task token, and at depth 1 the causal chain c_1 → c_2 → … is inert (an encoder needs one layer per hop). Kept for comparison; the live CoT experiment is `cot_mode: autoregressive` below. See CHANGELOG 2026-07-03.

#### Autoregressive chain-of-thought (`cot_mode: autoregressive`)

Genuine CoT: the graph is serialized to a discrete token sequence and a decoder-only
transformer (`src/cot.py`, tied embeddings, causal mask) is **teacher-forced on a
supervised BFS trace + answer**, decoding greedily at eval. Every generated token
re-enters the network, so a shallow model gets one full sequential step per token —
the Merrill & Sabharwal 2024 mechanism, as opposed to the scratchpad's single pass.

```
prompt:     N v_0 .. v_{n-1}  E u_1 w_1 .. u_m w_m  TRACE
bfs_levels: l0 SEP l1 SEP .. lk                      ANS YES|NO EOS
bfs_expand: 0 SEP EXP p [children..] EXP p' [..] SEP ..  ANS YES|NO EOS
```

- The trace is the canonical BFS from the lowest node id (levels sorted ascending) —
  deterministic, so teacher forcing has a unique target. `#SEP` = BFS depth, which is
  the sequential-steps quantity for depth-vs-diameter plots.
- `bfs_expand` exists because the compact target hits the **next-token pitfall**
  (Bachmann & Nagarajan 2024): each level's first token is the minimum of the whole
  frontier — a global computation with no partial credit, and the lookup circuit never
  forms (CHANGELOG 2026-07-03). The verbose format makes every next token locally
  computable: parents are a copy of the previous level (induction head), children a
  lookup keyed by the adjacent parent token.
- `node_features` / `tokenization` / `pooling` / `norm_type` are **unused** on this
  path — the prompt is the edge list itself. `layers` supplies depth + `heads` as
  usual; `max_nodes` sizes the vocabulary (node ids `0..max_nodes-1` + 10 specials).
- Training logs: `test` = greedy-decoded answer accuracy (only on `cot_eval_every`
  epochs; best-epoch selection uses it), `test_tf` = teacher-forced answer accuracy
  (cheap upper bound, logged every epoch), `trace_em` = decoded trace exact-match,
  `mean_levels`, `parse_fail`. A by-diameter breakdown and an ER OOD probe are
  attached to the results JSON automatically.
- The matched-parameters baseline is the **same config with `max_trace_len: 0`**
  (answer-only). Configs: `configs/cot_ar.yaml` (compact), `configs/cot_ar_expand.yaml`
  (verbose), `configs/cot_ar_hard.yaml` / `cot_ar_hard_diam.yaml` (adversarial datasets).

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

### Autoregressive chain-of-thought (trace vs answer-only)
Decoder-only LM over graph tokens with a supervised BFS trace; the ablation is the same architecture without the trace. See `configs/cot_ar_expand.yaml`.
```yaml
cot_mode: autoregressive
trace_format: bfs_expand   # local next-tokens; bfs_levels = compact variant
max_seq_len: 224
max_trace_len: 112         # set 0 for the matched answer-only baseline
permute_node_ids: true     # required: contiguous ids leak the label
layers:                    # depth buys per-token circuit capacity, the trace buys steps
  - {type: global_attn, heads: 4}
  - {type: global_attn, heads: 4}
```

### Chain-of-thought scratchpad (sequential-step sweep)
Shallow depth + `K` scratchpad tokens; sweep `cot_len` and look for a phase transition at the graph diameter (`reports/cot-tokens.typ`). See `configs/cot.yaml`. **Superseded by autoregressive CoT** (see above); kept as the comparison baseline.
```yaml
tokenization: node_edge
node_id_dim: 16
lpe_dim: 0          # honest: no precomputed reachability
cot_len: 8          # K scratchpad tokens — the swept variable
cot_mask: causal    # 'causal' (BFS rounds) or 'full' (bidirectional scratchpad)
layers:             # keep depth shallow so gains come from the scratchpad
  - type: global_attn
    heads: 4
  - type: global_attn
    heads: 4
```
