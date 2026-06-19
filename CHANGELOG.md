# Experiment Changelog

Record of findings, bugs, and decisions made during experiments.

---

## 2026-06-19

### Constant node features cause symmetry collapse in GlobalAttnConv

**Observation:** Running `global_attn` layers on the connectedness dataset produced
exactly 71.00% test accuracy every epoch — no learning at all. Loss barely moved.

**Cause:** All nodes started with feature `[1]` (constant). In `GlobalAttnConv`,
attention scores are `Q_i · K_j = (W_q · 1) · (W_k · 1)` — identical for every
node pair. Softmax over identical scores gives uniform `1/N` weights, so every
node receives the same output `W_v · [1]` regardless of graph structure.
Mean pooling over identical node embeddings produces the same graph embedding
for every graph. The classifier always saw the same input → always predicted
the majority class (connected = 67.9% of dataset → 71% test acc on test set).

**Fix:** Replaced constant features with normalised node degree `degree / (n-1)`.
Degree gives each node a structural identity — isolated nodes look different
from high-degree nodes. After the fix: 71% → 98% test accuracy.

**Lesson:** Node features must break symmetry for the model to see structure.
This matters especially for global attention (which ignores `edge_index`) but
also applies to local layers when graphs are regular (all nodes same degree).
