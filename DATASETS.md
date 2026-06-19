# Datasets

All datasets are loaded via `--dataset <name>`. Synthetic datasets are generated
once and cached in `data/` so all experiments run on identical graphs.

---

## Cora (`--dataset cora`)

**Task:** Node classification (`task: node`)  
**Source:** `torch_geometric.datasets.Planetoid`

A citation network of academic papers. Nodes are papers, edges are citations.

| Property | Value |
|---|---|
| Graphs | 1 (single large graph) |
| Nodes | 2708 |
| Edges | 10556 |
| Node features | 1433 (bag-of-words: binary word presence) |
| Classes | 7 |
| Train / Val / Test | 140 / 500 / 1000 nodes |

**Classes:** Case Based, Genetic Algorithms, Neural Networks, Probabilistic
Methods, Reinforcement Learning, Rule Learning, Theory.

**Config:**
```yaml
in_channels: 1433
out_channels: 7
task: node
```

---

## MUTAG (`--dataset mutag`)

**Task:** Graph classification (`task: graph`)  
**Source:** `torch_geometric.datasets.TUDataset`

Molecules represented as graphs. Nodes are atoms, edges are bonds.
Label indicates whether the molecule is mutagenic (causes DNA mutations).

| Property | Value |
|---|---|
| Graphs | 188 |
| Node features | 7 (one-hot atom type: C, N, O, F, I, Cl, Br) |
| Classes | 2 (mutagenic / not) |
| Avg nodes per graph | 17.9 |
| Node range | 10 – 28 |
| Class balance | 125 mutagenic / 63 not |
| Split | 80% train / 20% test (shuffled each run) |

**Config:**
```yaml
in_channels: 7
out_channels: 2
task: graph
```

---

## Connectedness (`--dataset connectedness`)

**Task:** Graph classification (`task: graph`)  
**Source:** Synthetic — `dataset.py:make_connectedness_dataset`  
**Cache:** `data/connectedness.pt`

Random Erdős–Rényi graphs labeled by whether the graph is connected.
Edge probability is sampled around the connectivity threshold `log(n)/n`,
producing a natural mix of connected and disconnected graphs.

| Property | Value |
|---|---|
| Graphs | 1000 |
| Node features | 1 (normalised degree: `deg / (n-1)`) |
| Classes | 2 (connected / disconnected) |
| Class balance | ~679 connected / ~321 disconnected |
| Node range | 5 – 20 |
| Diameter range (connected) | 1 – 8, mean 3 |
| Split | Fixed 800 train / 200 test |

**Node features:** Normalised degree rather than a constant, because constant
features cause all nodes to look identical to the model — global attention
produces uniform weights and learns nothing (see CHANGELOG 2026-06-19).

**Config:**
```yaml
in_channels: 1
out_channels: 2
task: graph
```

**Recommended layers:** ≥ 4 layers to cover the p99 diameter of 7.
With fewer layers the model cannot propagate information across the full graph.

---

## Isomorphism (`--dataset isomorphism`)

**Task:** Graph classification (`task: graph`)  
**Source:** Synthetic — `dataset.py:make_isomorphism_dataset`  
**Cache:** `data/isomorphism.pt`

Pairs of graphs encoded as a single disconnected graph. Label indicates
whether the two components are isomorphic.

| Property | Value |
|---|---|
| Samples | 1000 (500 isomorphic / 500 not) |
| Node features | 2 (one-hot: `[1,0]` = Graph 1, `[0,1]` = Graph 2) |
| Classes | 2 (isomorphic / non-isomorphic) |
| Class balance | Exactly 50 / 50 (alternating) |
| Node range per component | 6 – 15 |
| Split | Fixed 800 train / 200 test |

**Encoding:** Each sample merges graph G1 (nodes `0..n-1`) and G2
(nodes `n..2n-1`) into one graph with no cross-edges. The one-hot feature
tells the model which component each node belongs to.

**Positive pairs (label=1):** G2 is G1 with node labels randomly permuted —
structurally identical, different node ordering.

**Negative pairs (label=0):** G1 and G2 generated independently. Verified
to have different degree sequences (necessary condition for non-isomorphism).

**Config:**
```yaml
in_channels: 2
out_channels: 2
task: graph
```

**Why this is hard for GNNs:** Standard GNNs are bounded by the
Weisfeiler-Lehman (WL) graph isomorphism test. GIN (sum aggregation) matches
1-WL expressiveness but still cannot distinguish all non-isomorphic graphs.
Pairs that are non-isomorphic but WL-equivalent will fool GIN and all weaker
layers (GCN, SAGE, GAT).
