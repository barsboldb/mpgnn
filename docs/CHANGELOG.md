# Experiment Changelog

Record of findings, bugs, and decisions made during experiments.

---

## 2026-06-25

Long session (work spanned 06-23 → 06-25) centered on graph connectivity: why our
transformer couldn't learn it, and what fixes it. The headline result reframes the
whole project.

### Supervision density is the blocker, not the architecture (the headline)

The **binary** connectedness_hard task (one label per graph: connected?) stalls at
`ln 2` — the model parks at the marginal predictor and never learns. The **exact
same graphs**, retargeted as the **n×n reachability matrix** R (R_ij = 1 iff i, j in
the same component), are learned to **~0.97 exact-match**. The difference is purely
the *density of supervision*: 1 bit/graph starves the algorithm; n² bits/graph feed
it. Producing R requires actually computing reachability for every pair; once you
have R, whole-graph connectedness is a trivial readout (`all(R == 1)`). So the hard
part was never the binary decision — it was computing reachability, and the binary
loss gave gradient descent nothing to grip. **Lesson: train on the dense matrix
target, read the binary answer off it for free.**

### Connectivity-matrix task (Ye et al. 2026) added to the framework

Folded into `GraphTransformer` as `task: connectivity`: input = self-loop-augmented
adjacency rows (`node_features: adj_self` = A+I), encoder = the node-token attention
stack, output = per-graph n×n logits via a **pairwise bilinear readout** `H W Hᵀ`,
target = reachability from per-node component labels, metric = **exact-match** (whole
matrix correct). Config-driven (`configs/connectivity_hard.yaml`), variable-n,
logs/plots like every other run.

### The Ye data lever is a catch-22 on our setups

The paper's lever (train only on within-capacity graphs, diam ≤ 3^L, to suppress the
heuristic) would not engage for us, and the diagnostics (`rho_hard` = fraction of
reachable pairs beyond capacity, + diameter histograms) showed exactly why:

- **ER graphs**: diameters concentrate tightly. At cap 9 they are *all* within
  capacity (filter removes nothing → within ≈ raw), or at lower cap *all* beyond it
  (filter removes everything → no training data). No regime gives beyond-capacity
  mass **and** within-capacity margin at once.
- **Diameter-controlled caterpillars** (new generator) fix filter engagement
  (`rho_hard` 0.04 raw vs 0.00 within) — but caterpillars are degree-uniform, so the
  **degree heuristic never forms**, so there's nothing for the lever to suppress.

So: ER forms the heuristic but can't engage the filter; caterpillars engage the
filter but never form the heuristic. Reproducing the lever needs a distribution with
*both* dense structure and controlled diameter (clustered graphs + bridges — noted as
future work).

### Global attention is not capacity-bound; local (GAT) attention is

The 3^L capacity wall assumes **local** mixing (a node combines with neighbours).
Our `GlobalAttnConv` is **global all-pairs**, so one layer reaches any node — it
solves even diam-18 graphs at capacity 9, and the lever can't bind. Added a
`local: true` option to `GlobalAttnConv` (mask attention to graph neighbours + self
→ 1 hop/layer → depth-bounded reach). With local attention the capacity wall appears
(`test < test_within` once graphs exceed reach). Conceptual finding: **local
attention = GAT = a GNN** — restricting attention to edges slides the model from the
transformer end of the spectrum (complete-graph attention) to the GNN end
(edge attention). For connectivity the **GNN inductive bias is the right one**: it
forces propagation along edges (which *is* reachability), whereas global attention
has too much freedom and finds shortcuts. Rule of thumb that emerged: *more
transformer-like (global) → more shortcutting; more GNN-like (local) → more genuine
computation.*

### OOD generalization: distribution-learning, not algorithm-learning

Model trained on diameter_controlled hits **0.98** in-distribution but collapses OOD:
**0.679** on `connectedness` (= exactly the connected fraction 679/1000) and **0.42**
on `connectedness_hard`. Per-example inspection (`--inspect`) of a failing
disconnected graph showed predicted density = 1.000 and mean cross-component
P(reachable) = **1.000** — i.e. it outputs the **all-ones** matrix, completely
failing to detect disconnection. Mechanism: a dense-blob node's `adj_self` row (many
1s) is far OOD from the sparse caterpillar rows (~2 ones) it trained on, so the
learned "separate components" computation breaks and the bilinear head lights up
everywhere. Clean evidence the model learns reachability *for its training structure*,
not a universal algorithm.

### mpGNN: degree generalizes, expressive random features memorize

On connectedness_hard, a 3-layer GIN reaches **0.975 test** with plain **degree**
features but only **0.53** (memorizes: 0.985 train) with **random** (rGIN) features.
Expressiveness ≠ generalization: 16-dim random features give the model capacity to
memorize each graph via its random fingerprint, with no transferable signal. Caveat:
degree's win comes at depth 3 < diameter (4–8), so it cannot be global reasoning — it
is **local bridge-detection**, a shortcut connectedness_hard's construction failed to
kill. (Full writeup: `reports/random-features-gin.typ`.)

### Idea logged: amplify the weak bridge signal

When the discriminative signal is a tiny minority of entries (the cross-cluster /
bridge pairs in R), up-weight them (focal / weighted loss) to amplify the gradient —
the analog "amplify + filter the weak signal." Can rescue a stalled optimization but
cannot create information; over-gain → overfitting. To be tested against the
no-amplification baseline. (Memory: `amplify-weak-signal-idea`.)

### Tooling / decisions

- **Model persistence**: training saves `checkpoints/<run_id>.pt` (state_dict +
  config); `--eval <ckpt> --dataset X` evaluates on other (OOD) datasets, `--inspect`
  prints failing examples (true vs predicted R + within/cross-component probabilities).
- **`diameter_controlled` dataset generator** (caterpillars with sampled diameter
  2–18, half connected / half two-component) — for capacity / lever experiments.
- **Refactor**: extracted the shared stack-of-layers engine into
  `graph_conv.GraphConvNet`; `GNN` (message passing) and `GraphTransformer`
  (attention) are now siblings over it, neither importing the other; `model.py` is a
  pure dispatcher. Added a `model: gnn|transformer` selector and the `Laplacian`
  zero-eigenvalue filter / lap-via-in_channels work, plus CoT scratchpad tokens.
- **Visualizer**: metric selector (plot any logged series), per-run details modal
  (full config/summary/timing), and an epoch-time chart; per-epoch + inference timing
  now logged for all tasks.
- **Perf note**: `GlobalAttnConv` does all-pairs attention over the *whole batch*
  (O(N_total²), most of it masked cross-graph), so cost scales with batch_size —
  smaller batches are faster. Efficient sparse local attention = use `type: gat`.

---

## 2026-06-21

### Isomorphism task: tokenization analysis and 0.85 accuracy ceiling

**Task setup:** Graph pairs (G1, G2) encoded as one disconnected graph (G1 at nodes
0..n-1, G2 at n..2n-1, n ∈ [6,15]). Label 1 = isomorphic (G2 is a permutation of G1),
label 0 = non-isomorphic (different degree sequences). 1000 pairs, 800/200 train/test.

**Adjacency-row tokenization + pair pooling → 0.85 ceiling:**
Switching from flat mean-pool to pair pooling (G1 and G2 pooled separately, classifier
sees [h_G1 | h_G2]) did not improve over mean-pool. Analysis via `analyze_iso.py`
revealed two distinct failure modes:

- **Wrong non-iso (22/200):** Pairs with small degree-sequence differences
  (avg `deg_seq_diff`=5.4 vs 15.0 for easy cases; 6 pairs had diff=2). The model
  can't resolve near-identical degree distributions from averaged representations.
- **Wrong iso (19/200):** Model predicts non-iso for genuinely isomorphic pairs.
  Root cause: adj_rows column asymmetry — G1 nodes have non-zeros in columns 0..n-1,
  G2 nodes in columns n..2n-1. Even for identical graphs the pair-pooled vectors live
  in different subspaces, so the classifier sees them as different.

**Membership tokenization → ln(2) collapse (0.50):**
One-hot component flag [1,0]/[0,1] is symmetric across G1/G2 but gives every node in
the same component identical features. Global attention with uniform Q/K/V per component
produces the same output for every node → pair pool returns a constant → classifier
always predicts 50/50 → loss pins at ln(2). Same symmetry collapse as constant features
on connectedness.

**Laplacian eigenvector tokenization → worse than adj_rows:**
Eigenvectors are only defined up to sign flips and rotations within degenerate
eigenspaces. Two isomorphic graphs can produce completely different eigenvector matrices.
Pair pool then sees h_G1 ≠ h_G2 for every isomorphic pair, making the task harder not
easier. (Eigenvalues are invariant, but they are graph-level, not node-level features.)

**Fundamental tension in isomorphism tokenization:**
No single tokenization satisfies both requirements simultaneously:
- adj_rows: nodes differ within a component ✓, but G1/G2 live in different column subspaces ✗
- membership: G1/G2 comparable ✓, but all nodes in the same component look identical ✗
- lap: nodes differ ✓, but eigenvectors not canonical across components ✗

**Next direction:** Local adj_rows — remap G2's adjacency rows to columns 0..n-1
(same space as G1) and concatenate with membership flag. Gives each node structural
identity and component identity in a comparable feature space.

---

## 2026-06-20

### Edge-token transformer stalls at the ln 2 plateau on connectedness_hard

**Observation:** The Sanford-style `node_edge` transformer (`token_transformer.yaml`)
sat at train loss `0.693 = ln 2` and 0.50 test for the entire run on
`connectedness_hard` — no learning. It overfit 10 graphs perfectly (1.00 by epoch 25)
but failed on 100+; an overfit sweep (10/50/100/200) showed a sharp cliff. Raising the
learning rate `0.0005 → 0.005` did nothing.

**Cause:** With `node_id_mode: learned`, node identities come from a shared
`nn.Embedding` indexed by within-graph position. The blob split `na` varies per graph,
so position 5 is in blob A for some graphs and blob B for others. Those two cases push
the same embedding row in opposite directions → the gradients cancel → the optimizer
gets ~zero net signal and parks at the max-entropy output `[0.5, 0.5]` (loss `ln 2`).
`lr × 0 = 0`, so a bigger step size can't escape a vanishing gradient. (Full write-up in
[tokenization.typ](../reports/tokenization.typ).)

**Lesson:** A flat loss at exactly `ln 2` is the tell for a saddle, not slow learning —
look for a representational reason the gradient is structurally near-zero, not a tuning
knob.

### Adjacency-rows tokenization trains but does not generalize on our data

**Observation:** Switching to adjacency-row tokens (`adj_transformer.yaml`,
`connectedness_hard_adj`) fixed the gradient (loss → 0.03) but test stalled at ~0.59.
Fixing graph size (`connectedness_hard_adj_fixed`, n=20) only lifted it to ~0.70, with a
large train/test gap.

**Cause:** Adjacency rows are not permutation-invariant — row `i` carries this graph's
arbitrary node numbering, and our varying blob split means "column j" has a different
structural role across graphs. The model memorizes position-specific patterns that don't
transfer. Fixed size alone doesn't help because the split point still varies.

### Yehudai's connectivity is fixed-size; our model aces it — the dataset is the hard part

**Observation:** Reproduced Yehudai et al. 2025's connectivity experiment locally
(`yehudai/run_connectivity.py`). Their `adj_rows` reaches 1.00 and `edge_list` 0.93 with
a single transformer layer (n=50, n_train=100). Verified every graph in their dataset is
exactly n=50. Then ran **our** adjacency-rows global_attn GNN on **their** data
(`yehudai_connectivity_adj`): **1.00 test by epoch 4**, faster and with fewer params than
their own implementation.

**Cause / conclusion:** Same model + training, only the data changes — our pipeline
reproduces their perfect score, so the ~0.6–0.7 ceiling on `connectedness_hard` is purely
the dataset. Their classes come from different generators (gnp/rgg/scale-free/sbm) whose
adjacency-row patterns are separable; ours are degree- and edge-count-matched, differing
by a single bridge edge, which removes that shortcut. (Details and tables in
[yehudai-empirical.md](yehudai-empirical.md).)

**Lesson:** `connectedness_hard` is a genuinely harder probe of global reasoning than the
standard connectivity benchmark — the difference is the adversarial data construction, not
model capacity or graph size. Always run your own model on the reference dataset to
separate a pipeline problem from a dataset-difficulty result.

### Reproduction note: patched two bugs in Yehudai's source

Their `connectivity_adj_mat.py` has a `create_data` NameError (line 207) and a
cache-reload bug that loads the **train** split as val and test (inflating both); torch
≥2.6 also rejects their pickled PyG `Data` under `weights_only=True`. All handled in
`yehudai/run_connectivity.py` without editing their source.

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
