# Experiment configs

Files starting with `_` are shared bases, not runnable experiments. Every other
file is one experiment: `python main.py --config configs/<file>.yaml`. Quick
variants without editing: `-o key=value` (e.g. `-o epochs=50 -o cot_len=12`).

Configs inherit via `extends: <file>` (child fields wholly replace base fields).
The exact resolved config of any past run is embedded in its `results/*.json`.

## Connectivity-matrix family — `extends: _connectivity_base.yaml`

Same encoder engine + same pairwise readout `R̂ = H W Hᵀ`; only the layer type
differs, so these are drop-in architecture comparisons.

| Config | Attention / conv | Reach per layer |
|---|---|---|
| `connectivity_global` | all-pairs global attention | whole graph |
| `connectivity_local`  | neighbour-masked attention | 1 hop |
| `connectivity_gat`    | GAT (message passing)      | 1 hop |
| `connectivity_bdh`    | BDH shared-parameter rounds | mask-dependent |
| `connectivity_hard`   | global attention, on `connectedness_hard` | whole graph |

## Binary graph classification — `extends: _graph_base.yaml`

Default dataset `connectedness_hard` (the plain `connectedness` dataset is
retired: solvable by a local degree shortcut, see docs/CHANGELOG.md).

| Config | Tokenization / features | The experiment |
|---|---|---|
| `adj_transformer`   | node tokens = adjacency rows | depth sweep on hard connectivity |
| `token_transformer` | vertices + edges + task token (Sanford) | depth-vs-task curve |
| `cot`               | node_edge + K scratchpad tokens | sequential axis (sweep `cot_len`) |
| `adj_cot`           | adj-row tokens + scratchpad | CoT on top of adj_rows |
| `connectedness_lap` | Laplacian eigenvectors as features | spectral-geometry probe |
| `mpgnn`             | random features, GIN stack | mpGNN depth sweep |
| `iso_adj` / `iso_lap` / `iso_membership` | pair pooling on `isomorphism` | tokenization comparison |

## Removed configs (2026-07-02 cleanup)

One-off diagnostics whose conclusions are recorded, or presets on the leaky
`connectedness` dataset: `transformer`, `mpgnn_hard` (merged into `mpgnn`),
`cot_adj` (merged into `cot`), `adj_transformer_no_lpe`, `adj_transformer_fixed`,
`adj_yehudai` (reproduce with `--config configs/adj_transformer.yaml
-o dataset=yehudai_connectivity -o in_channels=50`), and the root `config.yaml`.
