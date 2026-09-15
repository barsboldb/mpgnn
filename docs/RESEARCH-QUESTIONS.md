# Candidate research questions — and a fact-check of the narrative

Written 2026-08-26, after a ~1-month gap. Two parts:

1. **Part A** checks the remembered story against what is actually recorded in this
   repository, flagging what is confirmed, what is imprecise, and what is contradicted.
2. **Part B** lists 10 candidate research questions with their evidence status, decisive
   experiment, and risk.

Supersedes the framing in `docs/QUESTIONS.md`, which assumed the spine was settled.
Nothing is settled; that file is kept for its experiment inventory only.

Literature coverage per Part B question — how crowded the field is, the papers to cite or
defend against, and which ones threaten each claim: `docs/RESEARCH-QUESTIONS-LITERATURE.md`.

---

# Part A — fact-check of the remembered narrative

## A1. "My professor gave me Yehudai et al., *Depth-Width Tradeoffs for Transformers on Graph Tasks*, and I tried to reproduce it."

**CONFIRMED.** `yehudai/` holds their supplementary code; `yehudai/run_connectivity.py`
is a wrapper that runs their connectivity experiment without editing their source.
Full writeup in `docs/yehudai-empirical.md`; changelog 2026-06-20.

Three bugs in their source were patched in the wrapper (documented, worth keeping for
the thesis's reproducibility section):

- `create_data` NameError at `connectivity_adj_mat.py:207`
- cache-reload loads the **train** `.pt` for all three splits, inflating val/test
- torch ≥2.6 rejects their pickled PyG `Data` under `weights_only=True`

Reproduced numbers (n=50, 1 layer, width 64, n_train=100): `adj_rows` **1.000**,
`edge_list` **0.926**.

## A2. "I found there are many flavors of GNNs and decided to compare them all."

**CONFIRMED.** The repo has GIN / rGIN (random features), GAT, `GlobalAttnConv` (global
all-pairs attention, plus a `local: true` neighbour-masked mode), a Sanford-style
`node_edge` edge-token transformer, and BDH (Kosowski et al. 2025). Configs in
`configs/`, engine split across `src/gnn.py` / `src/layers.py` / `src/graph_conv.py`.

Worth remembering: the comparison's own conclusion was that **architecture mattered less
than supervision structure** — BDH/GAT/global all land in one band in-distribution and
all collapse OOD (changelog 06-25, and the disposition notes in `docs/QUESTIONS.md`).

## A3. "Yehudai shows different graph encodings: EdgeList, AdjRows, and LE."

**CONFIRMED**, with one caveat. The three tokenizations are recorded in
`docs/yehudai-empirical.md` §1:

| name | token = | seq length | token dim |
|---|---|---|---|
| `adj_rows` | node's adjacency row ⊕ node features | `n` | `n + d` |
| `edge_list` | `concat(onehot(u)⊕x_u, onehot(v)⊕x_v)` | `#edges` | `2(n+d)` |
| `lap_full` | Laplacian eigenvectors ⊕ node features | `n` | `n + d` |

**Caveat:** `docs/yehudai-empirical.md` §4 lists `lap_full` as **untested** in their
pipeline. The Laplacian leak you remember was found in *our* setup, not theirs — see A4.

## A4. "LE leaks connectivity, so the transformer becomes an is-there-a-0-in-the-array machine."

**CONFIRMED IN SUBSTANCE, BUT TWO SEPARATE LEAKS ARE BEING CONFLATED.**
This matters, because they are different findings and both are useful.

**Leak 1 — the Laplacian leak (this is the LE one).** Our `lpe_dim` Laplacian
positional encoding as node features. The Laplacian's zero-eigenvalue eigenvectors are
localized on connected components, so component structure becomes linearly separable:

- on `connectedness`: ~0.91 → **~1.00** when `lpe_dim` goes 0 → 16
- on `connectedness_hard`: **0.55 → 1.00** with `lpe_dim: 16`

Evidence: changelog 06-19, memory `connectedness-dataset-leak`, `analyze_lap.py`,
`reports/laplacian-eigenvectors.typ`, `figures/lap_*.png`.

**Leak 2 — the degree-0 leak (this is the "is there a 0 in the array" one).** The
original `make_connectedness` builds Erdős–Rényi graphs near the threshold
`p ≈ log(n)/n`, where disconnection is almost entirely caused by *isolated vertices*.
Measured on seed 42, 1000 graphs:

- class balance 67.9% connected (majority baseline 68%)
- rule "connected ⇔ min degree ≥ 1" scores **98.2%**
- 94.4% of disconnected graphs have a degree-0 node; **0%** of connected ones do

Since the node feature *was* normalized degree, the model literally only had to detect a
zero entry. That is the "is-there-a-0" machine — it is the **ER/degree** leak, not the
Laplacian one. Evidence: changelog 06-19, memory `connectedness-dataset-leak`.

There is also **Leak 3**, found later and easy to forget: on token models,
`diameter_controlled` gave connected graphs exactly 23 edges and disconnected 22, so
**sequence length was the label** — the positional embedding of the `TRACE` token read
the answer off prompt length. Invisible to every GNN/encoder baseline. Fixed by padding
to `n−1+k` edges, `k ~ U{1,2,3}`, with diameter-preserving chords (commit `ed47ddd`).

## A5. "With ER graphs it became clear the task wasn't really 'is the graph connected', so I built `connectedness_hard`."

**CONFIRMED.** `make_connectedness_hard` in `dataset.py`: two dense internally-connected
blobs (every node degree ≥ 2), classes differing by **exactly one bridge edge**, edge
counts matched. Verified min-degree and mean-degree shortcuts both at chance. Later
`connectedness_hard_diam` added a diameter axis (two-blob ± bridge, sparse cycle blobs
plus 0–3 chords, diameters 3–13, degree- and edge-count-matched).

**But do not describe `connectedness_hard` as shortcut-free.** Changelog 06-25 records
a residual leak that you flagged yourself and may have forgotten:

> a 3-layer GIN with plain **degree** features reaches **0.975** on `connectedness_hard`
> — at depth 3 < diameter 4–8, so it *cannot* be global reasoning. It is **local
> bridge-detection**, a shortcut the construction failed to kill.

This is important. It means "no model can learn `connectedness_hard`" is false as
stated, and it means the dataset needs one more rung before it can carry a negative
result. See Q2 and Q1 below.

## A6. "Sanford et al. claim connectivity is parallelizable, needing L = O(log N) and m = O(N^ε)."

**CONFIRMED EXACTLY (verified 2026-08-26 against the paper).**

Paper downloaded to `papers/sanford-2024-transformer-reasoning-graph-algorithms.pdf`
(arXiv 2405.18512v1, NeurIPS 2024); companion written at `reports/sanford-2024-notes.typ`.

`L = O(log N)` **and** `m = O(N^ε)` is the verbatim definition of their **LogDepth (LD)**
scaling regime (§3), and connectivity is classed LD (Fig. 2b). The width claim you were
recalling is correct — cite it as *Sanford et al. 2024, §3, LogDepth regime*. Two
precisions to carry into the thesis:

- **ε > 0 is any fixed constant.** The claim is sub-polynomial width, not a specific
  exponent. The four regimes are D1 (`L=1`, `m=O(log N)`), LD, LDP (LD + poly(N) pause
  tokens), LDW (`L=O(log N)`, `m=O(N^{1/2+ε})`).
- **`N` is the input *sequence length* `O(|V|+|E|)`, not the node count.** On a dense
  graph `O(log N) = O(2 log n)` — still logarithmic in `n`, but say "sequence length"
  when you state it.

The underlying result is **Theorem 1**: any `R`-round MPC protocol with `O(N^δ)` local
memory is simulated by a transformer of depth `L = O(R)`, `m = O(N^{δ+ε})` — MLPs do local
computation, one attention layer does communication. That is the formal version of the
depth↔rounds argument in `reports/message-passing.typ`.

Also now sourced: **log depth is *necessary***, conditional on their Conjecture 13
(Thm 3), and **one layer provably cannot** solve connectivity/shortest-path/cycle-detection
without `mH = Ω̃(N)` width (Thm 6).

## A7. "It turns out they used an LLM for the graph task execution."

**CONTRADICTED — the memory is inverted. Do not use this claim.**

§4.3 is titled **"Trained transformers outperform LLM prompting."** The LLM prompting
results are a **baseline they beat**, not their method. Their models are (a) small
autoregressive transformers up to 60M params trained **from scratch**, and (b) a
**fine-tuned** T5-11B — a trained model, not a prompted one. Prompting baselines
(zero-shot / few-shot / CoT / CoT-BAG) lose on every task:

| task | best LLM prompting | 60M transformer (100K) |
|---|---|---|
| connectivity | 84.9 | **98.0** |
| shortest path | 38.6 | **97.1** |
| cycle check | 76.0 | **98.0** |

Saying "they used an LLM" in a defense would invert the paper's headline finding.

**Related correction to a claim this repo has been making loosely:** Sanford is *not* a
blanket "transformers beat GNNs" result. §4.2 reports **MPNN beating the 60M transformer
on node degree (99.8 vs 91.7) and cycle check (100.0 vs 98.0)**, and GNNs dominating in
the low-sample (1K) regime generally. The scoped claim is: transformers win on **global**
tasks (connectivity, shortest path); GNNs win where a **local** inductive bias is correct
and samples are scarce.

**Two new gaps found while reading — both usable (see Q2):**

1. **Their empirical validation runs on small dense ER graphs.** GraphQA is 1,000/500/500
   Erdős–Rényi graphs, **5–20 nodes**, avg degree 5.43. That is the generator family our
   own audit convicted (A4, Leak 2). Nobody in the paper audits for it.
2. **The paper concedes the theory and experiments never meet.** Appendix E notes GraphQA
   graphs "have very small cycles and do not resemble the large-diameter worst-case
   instance" that Theorems 3 and 6 are about. Theory is worst-case; experiments are
   small-diameter average-case. `connectedness_hard_diam` was built to span that gap.

**Terminology trap:** GraphQA "connectivity" is *edge-level* st-connectivity ("is there a
path from u to v"), not our *graph-level* "is the whole graph connected". Our binary label
is a conjunction over all pairs — a strictly harder read-off on the same graphs.

## A8. "My graph transformer couldn't learn `connectedness_hard` no matter its depth and width."

**PARTIALLY CONTRADICTED. This is the claim most in need of repair.**

What *did* fail on the binary label:

| model | result on `connectedness_hard` |
|---|---|
| `node_edge` edge-token transformer (Sanford-style) | pinned at `ln 2 = 0.693`, 0.50 test, entire run |
| `adj_rows` transformer, variable n | ~0.59 (trains, doesn't generalize) |
| `adj_rows` transformer, fixed n=20 | ~0.70 |
| `global_attn` + mean-pool + degree, `lpe=0` | 0.50 @ hidden 2, ~0.55 @ hidden 64 |

The `ln 2` plateau even had a diagnosed mechanism: with `node_id_mode: learned`, the blob
split `na` varies per graph, so the same embedding row gets pushed in opposite directions
across graphs, gradients cancel, and the optimizer parks at max entropy. `lr × 0 = 0`, so
raising lr did nothing (changelog 06-20).

What **did** learn it:

| approach | result | honest reading |
|---|---|---|
| `lpe_dim: 16` | **1.00** | the Laplacian leak — not reasoning |
| 3-layer GIN, degree features | **0.975** | local bridge-detection — not global reasoning |
| n×n reachability-matrix target (Ye-style) | **~0.97** exact-match | dense supervision; genuine |
| AR-CoT with `bfs_check` traces | **0.9972** | supervised BFS execution; genuine |

**Correction (2026-08-26, after reading `results/` directly):** an earlier draft of this
file said no depth × width grid exists. That was wrong. Roughly **20 runs** across
2026-06-19 → 06-23 do cover it, and every one lands between **0.50 and 0.64**:

| depth | width | features | test |
|---|---|---|---|
| 1 | 2 | degree | 0.500 |
| 1 | 64 | degree | 0.545 |
| 1 | 128 | degree | 0.535 |
| 3 | 64 | degree | 0.520 |
| 4 | 64 | degree | 0.520 – 0.545 (incl. a 500-epoch run) |
| 5 | 32 | degree | 0.530 |
| 1 | 20 – 128 | adj_rows | 0.565 – 0.640 |
| 4 | 128 | adj_rows | 0.520 |
| 8 | 32 | adj_rows | 0.585 – 0.600 |

Run IDs: `20260619_221642`, `_221812`, `_224205`, `_224932`, `_234619`, `_234958`,
`_235233`, `20260620_104231`, `_172317`, `20260623_213906`, `_214208`, `_214443`,
`_214814`, `_220326`, `_224939`, `_225849`, `_230552`, `_231306`, `_232307`, `_232954`,
`_234038`, `20260622_180437`, `_181051`.

There is also a **capacity control** that turns this into an argument: same model
(4 × 64, `lpe_dim: 0`), varying only the training-set size —
`overfit10` → **1.00**, `overfit100` → **0.76**, full data → **0.545**
(`20260620_170729_overfit10_*`, `_171029_overfit100_*`, `_172317_*`). The model *can*
fit the data; what fails is generalization.

So the claim is defensible, just narrower than "no matter its depth and width": *the
binary label is not learnable by these encoders at any depth or width I ran*. What is
still missing is that this grid **accumulated across four sessions with drifting
configs** — it is not a pre-registered sweep, and should be re-run as one before it
carries a thesis chapter.

## A9. "Ye et al.'s data statistically leaked connectivity, and their method didn't work on `connectedness_hard`."

**PARTIALLY CONTRADICTED — the recorded finding is different, and more interesting.**

Title confirmed exactly: *Transformers Provably Learn Algorithmic Solutions for Graph
Connectivity, But Only with the Right Data* (`reports/ye-2026-notes.typ`, cited in the
repo as Ye/Fu/Jia/Sharan, ICML 2026).

**Correction (2026-08-26): the "leaked" half of your memory is better supported than an
earlier draft of this file said.** `ye_connectivity.py` includes two adversarial probes —
two disconnected cliques, and the same two cliques joined by one bridge — and the results
are damning:

| regime | train EM | ER test EM | 2 cliques disconnected | + bridge |
|---|---|---|---|---|
| raw (n=60, 3000 train, 300 ep) | 1.000 | 0.578 | **0.000** | 1.000 |
| within (n=60, same) | 1.000 | 0.698 | **0.000** | 1.000 |
| within (n=20, 400 train, 40 ep) | 0.453 | 0.380 | **0.000** | 1.000 |

Receipts: `results/20260625_002255_ye_connectivity_raw.json`,
`_001934_ye_connectivity_within.json`, `_000927_ye_connectivity_within.json`.

**0.0 on disconnected and 1.0 on connected is the exact signature of a model that always
answers "everything is reachable".** It is reading density, not connectivity — and the
within-capacity lever does not change it. This is the sharpest single result in the
project and it cost two runs.

So "their data statistically leaked connectivity" is *defensible*, as long as it is
stated as: on an adversarial out-of-distribution probe, models trained under their recipe
— in **both** regimes — collapse to a density heuristic.

What the repo *also* records (changelog 06-25), which is the separate and more nuanced
half:

1. **Their capacity lever is a catch-22 on our distributions, not a leak claim.** Their
   mechanism is to train only on within-capacity graphs (diameter ≤ 3^L) so the heuristic
   channel is suppressed. Diagnostics (`rho_hard` = fraction of reachable pairs beyond
   capacity, plus diameter histograms) showed:
   - **ER graphs**: diameters concentrate tightly. At cap 9 *everything* is within
     capacity (filter removes nothing) or at lower cap *everything* is beyond it (filter
     removes all training data). No regime gives beyond-capacity mass **and**
     within-capacity margin at once.
   - **Diameter-controlled caterpillars**: the filter engages (`rho_hard` 0.04 raw vs
     0.00 within) — but caterpillars are degree-uniform, so **the degree heuristic never
     forms**, so there is nothing for the lever to suppress.

   → Reproducing the lever needs a distribution with *both* dense structure and
   controlled diameter. Clustered blobs + bridges was logged as the fix and later built
   (`connectedness_hard_diam`), but **the lever was never re-tested on it.** That is a
   concrete, cheap, unfinished experiment.

2. **A second confound: their 3^L capacity wall assumes *local* mixing.** Our
   `GlobalAttnConv` is global all-pairs, so one layer reaches any node — it solves even
   diameter-18 graphs at capacity 9 and the lever cannot bind. Adding `local: true`
   (attention masked to graph neighbours) made the capacity wall appear
   (`test < test_within` once graphs exceed reach). The conceptual finding that fell out:
   *local attention = GAT = a GNN*, and for connectivity the GNN inductive bias is the
   right one, because it forces propagation along edges. Rule of thumb recorded:
   **more transformer-like (global) → more shortcutting; more GNN-like (local) → more
   genuine computation.**

3. **Their *target* worked on `connectedness_hard`.** This is the direct contradiction:
   the Ye-style n×n reachability-matrix task, on the same `connectedness_hard` graphs
   that pin the binary task at `ln 2`, reached **~0.97 exact-match**. That was the June
   headline — *supervision density is the blocker, not the architecture*: 1 bit/graph
   starves the algorithm, n² bits/graph feed it.

So: what failed was **their data lever on our distributions**, not their method. Saying
"their method didn't work on `connectedness_hard`" in a defense would be wrong and
checkable.

## A10. Summary of corrections

| # | claim | verdict |
|---|---|---|
| A1 | Yehudai reproduction | confirmed |
| A2 | many GNN flavors compared | confirmed |
| A3 | EdgeList / AdjRows / LE | confirmed; `lap_full` untested in *their* pipeline |
| A4 | LE leaks connectivity | confirmed, but you are merging the **Laplacian** leak and the **degree-0** leak; there is also a third (sequence-length) |
| A5 | built `connectedness_hard` | confirmed — **but it still has a local bridge-detection shortcut (GIN 0.975)** |
| A6 | Sanford: log-depth | **confirmed exactly — `L=O(log N)`, `m=O(N^ε)` is their LogDepth regime; `N` is sequence length** |
| A7 | Sanford used an LLM | **contradicted — inverted. LLM prompting is a baseline they beat; their models are trained from scratch / fine-tuned** |
| A7b | Sanford = "transformers beat GNNs" | **too broad — MPNN wins on node degree and cycle check; transformers win on *global* tasks** |
| A8 | couldn't learn at any depth/width | **the sweep does exist (~20 runs, all 0.50–0.64) — but four approaches did learn the task, so narrow the claim to the binary label** |
| A9 | Ye's data leaked | **confirmed by the clique probe (0.0 / 1.0 in both regimes)** |
| A9b | Ye's method failed on `connectedness_hard` | **misremembered — the *lever* wouldn't engage; their *matrix target* reached ~0.97** |

---

# Part B — candidate research questions

Each entry: the question, what evidence already exists, the decisive experiment, and the
risk. Ordered roughly from "most of the work is done" to "most open".

---

## Q1 — The representability/trainability separation for connectivity

> *Sanford's construction shows a transformer of depth `O(log N)` can represent
> connectivity. Does gradient descent ever find it, on data with no statistical
> shortcut?*

**Why it's live.** Every result you have been arguing with is about *representation*
(Yehudai's tradeoffs, Sanford's log-depth) or about *a dataset* (Ye's "right data").
Nobody in your reading has established that SGD finds the construction on leak-free data.

**The premise is now fully sourced (2026-08-26).** Sanford Thm 2 gives existence
(parallelizable ⊆ LogDepthPause, LogDepthWide), Thm 3 gives conditional necessity of log
depth, and Thm 6 rules out a single layer below `mH = Ω̃(N)` — see
`reports/sanford-2024-notes.typ`. So the *representability* half of this question is
citable rather than remembered, and our contribution is cleanly the *learnability* half.

**Care with the `ln 2` evidence below.** Thm 6 predicts our narrow `node_edge` runs should
fail on width grounds — but the mechanism we actually diagnosed was learned-node-id
gradient cancellation (A8), a trainability failure. Two different reasons for the same
number; do not merge them when writing this up.

**Existing evidence.** Scattered negatives (A8 table): `ln 2` saddle on edge tokens,
0.59–0.70 on adjacency rows, 0.55 on global attention. Mechanism diagnosed for the `ln 2`
case.

**Decisive experiment.** The grid that does not yet exist:
depth {1,2,4,8} × width {64,128,256} × encoding {edge_list, adj_rows} on
`connectedness_hard` and `connectedness_hard_diam`, trained to saturation, **with**
(a) matched shortcut baselines at chance, (b) `lpe` off, (c) an overfit-N control proving
the model *can* fit the data, so failure is generalization and not capacity.

**Risk.** Negative results at small scale invite "you didn't train long/big enough".
Mitigations: report samples-to-learn curves rather than final numbers, include the
overfit control, and show the same grid *succeeding* on Yehudai's data (you already have
that comparison: your model hits 1.00 on their graphs by epoch 4).

**Blocker.** Q2 first — `connectedness_hard` currently has the GIN-degree shortcut, so a
negative result on it is attackable.

---

## Q2 — A leak taxonomy and audit protocol for graph-reasoning benchmarks

> *How much of the published graph-reasoning literature measures encoding-induced
> shortcuts rather than reasoning — and what is the minimal audit that catches them?*

**Why it's live.** You have found four structurally different leaks in four different
places, which is enough to be a taxonomy rather than an anecdote:

1. **generator leak** — ER near threshold ⇒ "min degree ≥ 1" solves it (98.2%)
2. **encoding leak** — Laplacian zero-eigenvectors encode components (0.55 → 1.00)
3. **serialization leak** — edge count ⇒ sequence length ⇒ label
4. **residual construction leak** — degree-matched blobs still leave local bridge
   detection (GIN 0.975 at depth < diameter)

Plus a methodological move that is itself the protocol: *run your own model on the
reference dataset* to separate a pipeline problem from a dataset-difficulty result
(you did exactly this with Yehudai — 1.00 by epoch 4 on their data, 0.59 on yours).

**Decisive experiment.** Formalize the audit as a checklist with a measured
falsification for each rung, then apply it to the Yehudai, Sanford and Ye setups and
report what survives. **Sanford's rung is now specified (A7):** GraphQA is ER, 5–20 nodes,
avg degree 5.43 — the generator family that gave us the degree-0 leak — and the paper
itself concedes those graphs "do not resemble the large-diameter worst-case instance" its
negative results describe. Measuring the min-degree rule on GraphQA connectivity is a
half-day experiment against the field's reference benchmark. Add the missing rung to `connectedness_hard`: kill the local
bridge-detection shortcut (e.g. bridges whose endpoints are degree-typical and whose
local neighbourhoods are indistinguishable within `k` hops for `k` up to the diameter),
and re-measure the GIN-degree baseline back to chance.

**Risk.** Lowest of the ten. It is a methods contribution rather than a theoretical one —
strong and defensible, but a committee may ask "where is the new science".

**Note.** This is also the prerequisite for Q1 and Q3, so it gets done either way.

---

## Q3 — What is Ye et al.'s "right data" actually doing?

> *Is their data condition a statement about the algorithm, or about which shortcut the
> distribution happens to admit?*

**Why it's live.** Your diagnostics found their lever cannot even engage on two natural
distributions (ER: no regime with both beyond-capacity mass and within-capacity margin;
caterpillars: no heuristic forms to suppress). That is a real, measured statement about
the scope of their condition.

**Existing evidence.** `rho_hard` diagnostics, diameter histograms, the global-vs-local
attention confound (their 3^L wall assumes local mixing; global attention voids it).

**Decisive experiment.** The one that was logged as future work and never run: re-test
the lever on `connectedness_hard_diam`, which was *built* to have both dense structure
and controlled diameter, with `local: true` attention so the capacity wall can bind.
Then state precisely which distributional property is doing the work.

**Risk.** Highest stakes. This is a direct engagement with a recent paper, so every claim
must be airtight and their setup must be reproduced faithfully before any difference is
asserted. Do not write a word of it from memory.

---

## Q4 — Encodings as precomputation oracles

> *How much of a task's computation does each graph encoding perform in preprocessing —
> and what is the exchange rate in layers?*

**Why it's live.** This reframes your professor's paper in exactly the direction your own
findings point. An encoding is not a neutral input format; it is a preprocessing oracle:

- **LE** hands the model transitive closure (components live in the zero eigenspace)
- **SPD / distance bias** hands it all-pairs distances
- **adj_rows** hands it a consistent layout *only when n is fixed* — which is why the
  same tokenization scores 1.00 on Yehudai's fixed n=50 and 0.59 on your variable-n data
- **edge_list** hands it nothing, and correspondingly sits at `ln 2`

**Decisive experiment.** For tasks of known parallel complexity, measure the depth needed
to reach a fixed accuracy under each encoding. The difference is the number of layers the
encoding bought. The fixed-vs-variable-n result is already a clean instance and is
currently under-sold in the writeup.

**Risk.** Low. Tidy, directly extends the assigned paper. Might read as incremental
unless the "exchange rate" is quantified rather than described.

---

## Q5 — Does process supervision buy what depth and width cannot?

> *Same architecture, same data, same optimizer: answer-only label vs. supervised
> intermediate-step trace.*

**Why it's live.** This is the strongest positive result in the repo and the constructive
complement to Q1: depth couldn't buy connectivity, tokens could.

**Existing evidence.** Depth 2, ~430k params:

| dataset | with BFS trace | answer-only control |
|---|---|---|
| `diameter_controlled` (diam 2–18) | 0.964, flat 0.92–0.99 | 0.510 |
| `connectedness_hard_diam` | 0.9975, flat across diam 3–13 | 0.511 |
| `connectedness_hard` (dense) | 0.9972, flat diam 2–8 | — |

Flatness in diameter is the theoretical payload: the trace converts diameter (which no
fixed depth can pay for) into sequence length (which any depth can). Theory that permits
it: Merrill & Sabharwal 2024.

**Decisive experiment.** Mostly done. Missing for publication: seed replicates for error
bars, the depth-1 row at matched data, and the answer-only control on
`connectedness_hard` (only the two other datasets have it).

**Risk.** Low on the result, medium on framing — "CoT helps" is not surprising on its
own. The surprise has to be carried by the *matched control* and the *flatness*, not by
the headline accuracy.

---

## Q6 — Which operations of an algorithm must be tokenized for it to be learnable?

> *If a step the algorithm performs emits no token, is it ever learned?*

**Why it's live.** This is the sharpest and most general claim available, and it
generalizes past graphs: it is a statement about which *reasoning styles* are trainable.

**Existing evidence.** Four instances, one of them predicted-then-fixed:

1. frontier-minimum in `bfs_levels` (level-1 accuracy 0.056 under teacher forcing, while
   every format statistic sat at 0.88–1.00) → fixed by the verbose `bfs_expand` target
2. visited-set subtraction on 8k sparse graphs (stalled ~0.6) → fixed by 4× data
3. visited-set subtraction on 32k *dense* graphs (stalled 0.27/0.24 at levels 2–3) →
   **predicted from the diagnostic, then fixed constructively** by `bfs_check`, which
   makes each membership test an emitted binary token: 0.5342 → **0.9972**, and the
   previously silent op became the model's most reliable one (1.000 over 342,755
   positions)
4. neighbour-colour *gathering* in WL colour refinement (stalled 0.888/0.909) — the
   diagnostic **acquitted** the two ops you predicted would fail and convicted a third;
   fix (`wl_gather`) designed but not run

Instance 3 is the one that makes this a law rather than an observation: failure was
predicted from a per-position diagnostic before the fix existed.

**Decisive experiment.** Run `wl_gather`. If the predicted fix lands on a *different
algorithm*, the law has out-of-sample confirmation and the claim is finished.

**Risk.** Medium. The claim needs a name and a precise statement ("locality is
parameterized by branching factor" is the sharpened version — `bfs_expand` made the
frontier-minimum local but left set-minus implicit, and density is the knob that exposes
it). Also needs honest positioning against Abbe et al.'s globality barrier and Bachmann &
Nagarajan's next-token pitfall, both of which are close.

---

## Q7 — Do standard regularizers prevent algorithmic circuit formation at small scale?

> *Weight decay vs dropout, bisected, on exact-retrieval circuits.*

**Existing evidence.** The atomic-lookup probe went from a 0.15 plateau to 0.95 by epoch
10 / 0.999 by 20 when `weight_decay 0.01 → 0` and `dropout 0.1 → 0`. Bisect: dropout-only
reaches 0.93 by epoch 10 (circuit forms); **weight-decay-only reproduces the plateau**
(0.11 at epoch 30). Mechanism with same-scale support: weight decay → low-rank attention
in 2-layer associative recall (Kobayashi et al. 2024), while retrieval needs high-rank
key–query alignment; under a norm penalty the degenerate marginal-statistics solution is
cheaper (Varma et al. 2024).

**Why it's live.** Published work reports the *opposite sign* at scale ("LMs Grok to
Copy", NAACL 2025), so the contribution is the regime contrast.

**Decisive experiment.** AdamW (decoupled) vs Adam (L2-in-gradient) at matched λ — the
current knob is torch Adam's L2-in-gradient, and the distinction must be stated.

**Risk.** Narrow. Excellent chapter, thin as a whole thesis.

---

## Q8 — Parallel depth vs sequential tokens: which is the cheaper way to buy reachability?

> *A cost accounting — FLOPs, wall-clock, samples-to-learn — over the depth ladder vs the
> trace ladder.*

**Why it's live.** This answers Yehudai's tradeoff question in the *training* regime
rather than the expressivity regime, which is exactly the gap Part A keeps hitting.

**Existing evidence.** ~70% of the data exists across `results/`. The inference-side shape
change is already measured: generation is O(trace length) sequential forwards (~90 for
n=24), decode dominates eval wall-clock. `reports/gat-vs-local-attention.*` is the
microscopic sibling (structural vs post-hoc sparsity).

**Risk.** Low risk, moderate ceiling. Good analytical chapter.

---

## Q9 — Does any of this generalize beyond BFS?

> *Same recipe applied to 1-WL colour refinement, where message-passing GNNs have a
> provable ceiling (Xu et al. 2019).*

**Existing evidence.** Infrastructure built and leak-audited: `iso_wl` (degree-matched
pairs, negatives are degree-preserving double-edge swaps accepted only if 1-WL separates
within `wl_rounds`, no length leak — mean 419.7 vs 420.6 tokens). Real run currently
**stalls** the same way the dense connectivity task did: teacher-forced ≈1.0, loss 0.05,
decoded flat at 0.52, trace-EM 0.003. Diagnostic localizes it to two-hop neighbour-colour
retrieval (see Q6, instance 4).

Also on record: the *current* isomorphism dataset in the older experiments has its own
shortcut — negatives always have different degree sequences — and the analysis found a
genuine tokenization tension (adj_rows: G1/G2 in different column subspaces; membership:
all nodes in a component identical; Laplacian: eigenvectors not canonical across
components).

**Risk.** Higher — this is an open experiment, not a result. Good as the second half of a
thesis whose first half already landed; bad as the main question.

---

## Q10 — Is neural algorithm execution ever distribution-general?

> *Models trained on one graph generator collapse on another. Data-diversity problem or
> mechanism limit?*

**Existing evidence.** The failure is now *localized*, which is more than "it doesn't
generalize":

- caterpillar-trained model on ER: 0.32 answer / 0.01 trace-EM, 21% of decodes never
  form a well-formed answer
- blob-trained (`hard_diam`) lifts the ER probe to 0.594 answer / 0.288 trace-EM
- `bfs_check`-trained: ER trace-EM **0.188** (execution transfers — ~1 in 5 unseen
  distribution graphs gets a token-perfect trace) but answer accuracy **0.389**, *below*
  the always-NO marginal of 0.755

→ **Execution transfers; the verdict read-off does not.** The answer head is calibrated
to blob-shaped traces (mean 6.0 levels on ER vs ~4–5 on blobs) and misfires toward YES.
The standing OOD problem is the *answer-readout circuit*, not "the model".

There is also a keeper failure mode: on `connectedness_hard` the caterpillar model scored
**0.29 — below chance** — because it *faithfully read its own derailed traces*. That is
hallucinated reasoning reproduced in a system small enough to diagnose per token.

**Decisive experiment.** Mixed-generator training sets; separately, retrain only the
answer head on ER traces to test the localization directly.

**Risk.** Open-ended. Strong future-work section, risky as the central question.

---

# Part C — how these assemble

You need one topic, and several of these are chapters rather than theses.

**Option 1 — the expressivity/learnability gap (recommended).**
Q2 → Q1 → Q5 → Q6.

> *Fixed-depth transformers can represent graph connectivity but do not learn it from
> leak-free data at any depth or width we can train; supervising the algorithm's
> intermediate steps makes it learnable at depth 2 — and which steps must be supervised
> is predictable in advance.*

Uses a reproduction (Yehudai), a correction (the leak audit against Sanford/Ye), a
negative result, a positive result, and a mechanism. Covers essentially everything
already run. **Requires:** ~~the Sanford reading companion (A7)~~ *(done 2026-08-26)*, the
depth×width grid (Q1), and closing the GIN-degree shortcut (Q2).

**Option 2 — the benchmark-integrity thesis.**
Q2 → Q4 → Q3. Safest, most immediately useful to the field, least theoretically deep.

**Option 3 — the trainability-laws thesis.**
Q6 → Q9 → Q7, with Q5 as setup. Sharpest single contribution, but leans on `wl_gather`
landing, which is unrun.

---

# Part D — immediate to-do regardless of choice

1. ~~**Read Sanford's experimental section and write `reports/sanford-2024-notes.typ`.**~~
   **DONE 2026-08-26.** Paper at `papers/sanford-2024-transformer-reasoning-graph-algorithms.pdf`,
   companion at `reports/sanford-2024-notes.typ`. A6 confirmed exactly; A7 contradicted
   (inverted). Two new usable gaps found: their empirical graphs are small dense ER (5–20
   nodes) from the generator family our own audit convicted, and the paper itself concedes
   its benchmark "does not resemble the large-diameter worst-case instance" its negative
   results describe. *New follow-on:* we have never tested the `m = O(N^ε)` width axis —
   the A8 grid holds `N` fixed and tops out at width 256.
2. **Close the local bridge-detection shortcut in `connectedness_hard`** (GIN-degree
   0.975 at depth < diameter). Every negative result rests on this dataset.
3. **Run the Ye lever on `connectedness_hard_diam` with `local: true`.** Cheap, logged as
   future work in June, never done, and it decides Q3.
4. **Run `wl_gather`.** Decides whether Q6 is a law or an observation.
5. Note that `results/` contains 137 runs and several are pre-fix and **not comparable**
   — in particular, `diameter_controlled` caches/results from before commit `ed47ddd`
   are invalid (the sequence-length leak).
