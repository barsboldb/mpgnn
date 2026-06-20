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

### Connectedness dataset solvable by a local degree shortcut

**Observation:** A single `global_attn` layer with mean-pool reached 90%+ test
accuracy on the connectedness task even at `hidden_channels=2` — a bottleneck
that should be far too lossy to represent global connectivity. Accuracy held up
even with `lpe_dim=0`.

**Cause:** The dataset builds Erdős–Rényi graphs sampled near the connectivity
threshold `log(n)/n`, where disconnection is almost entirely caused by isolated
vertices. The rule "connected ⇔ min degree ≥ 1 (no isolated node)" already
scores 98.2% (measured, seed=42, 1000 graphs: 94.4% of disconnected graphs have
a degree-0 node, 0% of connected ones do). Since the node feature is degree, the
model just detects "is there a degree-0 node?" — purely local, no global
reasoning. `lpe_dim=16` leaked further: the Laplacian's zero-eigenvalue
eigenvectors are localized on components, pushing runs from ~91% to ~100%.

**Fix:** Added `make_connectedness_hard` (`connectedness_hard`). Both classes are
two dense, internally-connected blobs (every node degree ≥ 2); connected vs.
disconnected differs only by a single bridge edge, with edge counts matched
across classes. The min-degree and mean-degree shortcuts are now both at chance.
Results: single `global_attn` + mean-pool + degree features scores 0.50 at
hidden=2 and ~0.55 at hidden=64 (`lpe=0`); `lpe_dim=16` → 1.00. Accuracy now
tracks model capacity and structural input rather than a dataset artifact.

**Lesson:** A task that looks like it probes global structure can collapse to a
local statistic if the data-generating process correlates the label with a local
feature. Always sanity-check that a trivial rule (majority class, min/mean
degree) doesn't already solve the task before attributing accuracy to the model.
