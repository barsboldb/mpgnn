# What does it take for a fixed-depth transformer to actually execute a graph algorithm?

*The AR-CoT investigation, 2026-07-03 → 07-05 — findings distilled from `docs/CHANGELOG.md`,
`results/*.json`, and the commit history. Task: graph connectivity; algorithm: BFS;
model: decoder-only transformer, depth 2, hidden 128, ~430k parameters.*

---

## The answer, up front

A depth-2 transformer **can** execute BFS — decoding a full, exactly-correct trace for
80–90% of unseen graphs and reading off connectivity at 0.96–0.99 accuracy, *flat across
graph diameters* that provably exceed any fixed-depth single-pass computation. But none of
that happens by default. It required five ingredients, each isolated by its own
experiment, and **missing any one of them produced chance-level accuracy**:

1. **Supervise the intermediate steps** — unsupervised "thinking slots" do nothing.
2. **Audit the token representation for leaks** — serializing graphs creates observables
   (sequence length!) that never existed for GNN baselines.
3. **Make every next token locally computable** — a trace is a curriculum, not a log.
4. **Turn the regularization off** — weight decay and dropout *prevented circuit
   formation* outright at this scale.
5. **Buy data, not depth** — circuits vs memorization is an economics problem; extra
   layers financed memorization, extra data financed the algorithm.

The final contrast, on two datasets (identical architecture, data, optimizer — the trace
is the *only* difference):

| dataset | with BFS trace | answer-only | by diameter (trace) |
|---|---|---|---|
| `diameter_controlled` (diam 2–18) | **0.964** | 0.510 (chance) | flat 0.92–0.99 |
| `connectedness_hard_diam` (adversarial) | **0.9925** | 0.511 (chance) | flat 0.988–1.000 |

Trace exact-match: 0.80 / 0.90. Every sub-circuit ≥ 0.985 teacher-forced on held-out
graphs. Runs: `results/20260705_005620_*`, `20260705_155408_*` (ans-only),
`20260705_202655_*`, `20260705_224403_*` (ans-only).

---

## 1. Setup

**Task.** Binary graph connectivity — chosen because it is the canonical
depth-limited task: a single-pass model needs parallel rounds ~ log n (Sanford et al.
2024), and our adversarial datasets remove every statistical shortcut
(`connectedness_hard_diam`: two blobs, classes differ by one edge, degree and edge-count
distributions matched exactly).

**Serialization.** The graph becomes a discrete prompt, the model must emit a BFS trace
and answer:

```
prompt:      N 0 1 .. n-1  E u1 w1 u2 w2 ..  TRACE
completion:  0 SEP EXP p [children..] EXP p' [..] SEP ..  ANS YES|NO EOS
```

Node ids are randomly permuted per graph. Teacher forcing at training; greedy
autoregressive decode at eval. The matched baseline is the *same* model trained to emit
only `ANS YES|NO EOS`.

**Why this framing.** Merrill & Sabharwal (2024): chain-of-thought decoding strictly
extends fixed-depth transformer power — each emitted token re-enters the network, buying
one sequential step per token. Depth pays for the *per-step* circuit; tokens pay for the
*number* of steps. The experiments below measure what each actually costs.

---

## 2. The ladder: five failures, five diagnoses

### 2.1 Unsupervised thinking slots are just width

The predecessor design ("scratchpad CoT", `cot_mode: scratchpad`) spliced K *learnable*
tokens into an encoder pass. It never beat the no-CoT baseline (~0.565 vs 0.63 on
`connectedness_hard`, both ≈ chance), for structural reasons: the slots receive no loss,
carry no per-graph content, can be bypassed by the readout token, and at depth 1 the
"causal chain" across slots is inert (an encoder needs one layer per hop — `c₂` only ever
sees `c₁`'s initial, constant value).

> **Finding 1.** CoT gains come from *supervised* (or generated-and-conditioned-on)
> intermediate steps. Unsupervised extra positions in a single forward pass are just
> width. (CHANGELOG 2026-07-03.)

### 2.2 New representation, new leaks: sequence length is an observable

First real AR-CoT run: decoded accuracy 1.0 *by epoch 5* — with trace exact-match 0.0 and
the ER OOD probe at 0.37. Too good to be true, and it was: connected caterpillars had
exactly n−1 edges, disconnected ones n−2. **Edge count was the label**, and the prompt was
2 tokens longer for connected graphs — the learned positional embedding of the `TRACE`
token read the answer off *sequence length*. Invisible to every GNN/encoder baseline;
fatal for a token model. Fixed by padding all graphs to n−1+k edges (k ~ U{1..3},
label-independent) with diameter-safe chords, property-tested to preserve the exact
diameter.

> **Finding 2.** Every model family needs its own leak audit. Changing representation
> (graph → token sequence) creates observables that did not previously exist — sequence
> length first among them. (Run `20260703_162700`; fix in commit `ed47ddd`.)

### 2.3 The next-token pitfall: format learns, computation doesn't

Post-fix, the compact trace (`bfs_levels`: each BFS level as a sorted set) trained to a
hard plateau: loss pinned at ~1.87 at depths 1 *and* 2 with full data (20k steps),
decoded accuracy at chance. The per-position diagnostic (`diag_cot_levels.py`) localized
it: **level-1 accuracy 5.6% under teacher forcing** — the model could not retrieve "the
tokens paired with node 0 in the edge list" even with the entire gold prefix given — while
every *statistical* regularity (separator placement, trace end, answer-given-gold-trace,
EOS) sat at 0.88–1.00. Perfectly shaped traces, garbage membership.

This is Bachmann & Nagarajan (2024)'s pitfall: teacher forcing fits all easy conditionals
and gives the hard computation no partial credit. The compact target makes it maximally
bad — each level's *first* token is the minimum of the entire frontier set, a global
computation graded all-or-nothing.

The fix is a *verbose* trace (`bfs_expand`): `EXP parent children` rounds, in which every
next token is locally computable — parents are a copy of the previous level in order (an
induction head), children a lookup keyed by the token immediately before.

> **Finding 3.** A CoT trace is a curriculum, not a log. Design each next-token to be
> computable from local context, or the gradient never finds the algorithm.
> (Runs `20260703_170616`, `_173939`; diagnostic in `diag_cot_levels.py`.)

### 2.4 Regularization was suppressing circuit formation

The verbose target *still* plateaued — even pure induction copying sat at 0.119 after 20k
steps at depth 2. The knob ladder found the suppressor immediately, and it was nobody's
hypothesis: the **atomic lookup probe** (`trace_format: bfs_l1` — emit only node 0's
sorted neighbours) went from a 0.15 plateau to **0.95 by epoch 10, 0.999 by 20** when
`weight_decay 0.01 → 0` and `dropout 0.1 → 0`, with no other change.

**Bisect (2026-07-06): weight decay alone is the suppressor.** Dropout-only (0.1)
reaches trace-EM 0.93 by epoch 10; weight-decay-only (0.01) reproduces the plateau
(0.11 at epoch 30). Mechanism, with same-scale published support: weight decay induces
low-rank attention in 2-layer transformers on associative recall (Kobayashi et al.
2024), and a retrieval circuit needs high-rank, precise key–query alignment; under a
norm penalty the degenerate marginal-statistics solution is cheaper, so the penalty
selects the wrong circuit (Varma et al. 2024). Note the knob is torch Adam's
L2-in-gradient, not decoupled AdamW decay. This
retroactively explained every earlier plateau: format statistics survive regularization;
content circuits don't. Bonus: the zero-reg lookup circuit was the project's **first
OOD-positive result** — trained on sparse caterpillars, it transferred to dense ER graphs
at trace-EM 0.55.

> **Finding 4.** Regularizers tuned for statistical fitting can be *the* blocker for
> algorithmic circuit formation at small scale. Sweep them to zero before touching the
> representation. (Runs of 2026-07-04; config change in commit `6792721`.)

### 2.5 Circuits vs memorization is an economics problem

Zero-reg full-trace training at depth 2 / 8k graphs produced a new signature: teacher-
forced learning succeeded *in parts* — copy 0.989 and level-1 lookup 0.995 on held-out
graphs — but the third sub-skill (lookup **minus** the visited set, levels 2+) stalled at
~0.6, and decode collapsed under exact-match compounding. The obvious move, more depth,
made things *worse*: depth 4 at 8k drove train loss to literally 0.0000 while decoded
trace-EM *declined* — the added capacity financed memorization of the training set, not
the missing circuit. (Its by-diameter curve even inverted: the model faithfully read its
own derailed traces as "disconnected".)

The correct lever was data: node-id permutation makes every graph a fresh sequence, so
memorization's price grows with dataset size while the circuit's price is fixed. At
**32k graphs, depth 2**, the run groks — trace-EM flat at 0.03 until epoch ~55, phase
transition, 0.80 and climbing by epoch 100; decoded accuracy 0.964.

> **Finding 5.** Depth-4 capacity substituted for generalization; 4× data at depth 2
> tipped the economics to the algorithm. Add data before depth. (Runs
> `20260704_182708` (partial circuits), `_194936` (memorization), `20260705_005620`
> (grokking).)

---

## 3. The result, with its mechanism receipts

**Headline numbers** (best epochs; both datasets, 32k graphs, depth 2, zero reg,
`bfs_expand`):

| | diameter_controlled | connectedness_hard_diam |
|---|---|---|
| decoded answer accuracy | 0.964 | **0.9925** |
| trace exact-match | 0.80 | 0.90 |
| by-diameter accuracy | 0.92–0.99, diam 2–18 | 0.988–1.000, diam 3–13 |
| matched answer-only control | 0.510 | 0.511 |

The by-diameter *flatness* is the theoretical payload: reachability distance is the
quantity that defeats fixed-depth single-pass models, and here it simply doesn't matter —
the trace converts diameter (which no fixed depth can pay for) into sequence length
(which any depth can).

**Sub-circuit verification** (teacher-forced accuracy per position class, held-out
graphs, grokked model):

| position class | accuracy | circuit |
|---|---|---|
| parent (after `EXP`) | 0.998 | induction copy of previous level |
| level-1 children | 1.000 | edge-list lookup |
| level-2+ children | 0.985–0.996 | lookup minus visited set |
| SEP / ANS / EOS | 0.998–1.000 | trace format & stopping |

Nothing partially formed remains; the accuracy claim is backed by a mechanism claim.

**The controls that make it an argument.** The answer-only model — same architecture,
data, optimizer, tokenization, differing *only* in the absence of the trace — never leaves
chance on either dataset (its loss sits at the marginal predictor for all 100 epochs; its
apparent by-diameter "decay" is a class-composition artifact of a NO-biased coin, noted to
prevent over-reading). This echoes the June finding that 1 bit/graph of supervision
starves the same task that n²-bit matrix targets render learnable: **supervision density,
not architecture, has been the binding constraint all along** — the trace is simply
supervision density arranged in time instead of space.

---

## 4. What it costs

- **Depth**: 2 layers suffice — but not 1 (induction requires two; the depth-1 run was
  terminated at chance, consistent with theory). Depth 4 was *harmful* at fixed data.
- **Data**: 8k permuted graphs were not enough for the full circuit; 32k were. The
  transition is grokking-shaped, so intermediate scales may just delay it.
- **Steps**: ~40k gradient steps, zero regularization, lr 1e-3, tied embeddings with
  std-0.02 init (default N(0,1) init on tied embeddings stalls training from the start).
- **Inference**: generation is O(trace length) sequential forwards (~90 for n=24) — the
  compute shape changes from wide-and-shallow to narrow-and-long. No KV cache yet;
  decode dominates eval wall-clock at 32k scale.

## 5. What it does *not* give: distribution transfer

The grokked caterpillar model fails on dense graphs (ER: 0.32 answer / 0.01 trace-EM;
connectedness_hard eval: 0.29 with 42% of decodes never forming a well-formed answer —
*below* chance because the model faithfully misreads its own derailed traces). Two
observations point at training diversity rather than a mechanism limit: the isolated
lookup circuit transferred to ER at 0.55, and the hard_diam-trained model (denser, more
heterogeneous blobs) lifts the ER probe from 0.32 to 0.54. **In-distribution algorithm
execution is demonstrated; distribution-general execution is open** — the obvious next
experiment is a mixed-generator training set.

---

## 6. The checklist, distilled

For a fixed-depth transformer to execute a graph algorithm:

1. Supervise the algorithm's steps, not just its answer *(else: chance)*.
2. Audit what the serialization leaks; permute everything permutable *(else: fake 1.0)*.
3. Make each next token locally computable from the prefix *(else: format without
   content, loss plateau)*.
4. Zero regularization at small scale *(else: circuits never form — copy at 0.12)*.
5. Enough distinct data that memorization costs more than the circuit *(else: loss 0.0,
   decode at chance)*.
6. Depth ≥ the per-step circuit requirement (2, for copy + lookup), and no more than the
   data can discipline.

## 7. Novelty assessment (deep-research pass, 2026-07-06)

A literature search (Claude Research) found **no exact published precedent for any of
the six core claims**; verdicts per claim, with the closest neighbors:

| # | claim | verdict | closest prior work |
|---|---|---|---|
| 1 | traces give diameter-flat accuracy at fixed depth | similar-but-different | Sanford et al. 2024 (log-depth necessary, answer-only); Ye/Fu/Jia/Sharan ICML 2026 (same task, 3^L capacity, no traces); Merrill & Sabharwal 2024 (theory permitting it) |
| 2 | trace locality determines learnability (bfs_levels vs bfs_expand) | similar-but-different; **cleanest novel contribution** | Abbe et al. 2024 "globality barrier / inductive scratchpad"; Bachmann & Nagarajan 2024 (Clever Hans); no BFS/connectivity instance published |
| 3 | **weight decay** prevents circuit formation (bisect 07-06: dropout benign) | **no direct precedent** — scoped to small-model exact-retrieval regime | opposite-sign at scale: Lv et al. "LMs Grok to Copy" (NAACL 2025). Mechanism support: Kobayashi et al. 2024 (wd → low-rank attention, 2-layer recall); Varma et al. 2024 (circuit norm-efficiency) |
| 4 | data threshold + grokking; depth worsens memorization | similar-but-different | Ye et al. (capacity-driven heuristics, no traces); Zhu et al. 2024 (critical data size); depth-hurts direction is the fresh piece |
| 5 | sequence-length serialization leak | no direct precedent for the specific leak | general: seq-length shortcut (2212.08399), positional-embedding leaks (Charformer), degree shortcut (Yehudai et al. 2025) — present as a benchmark-artifact family |
| 6 | role-stratified per-position accuracy as circuit diagnostic | technique exists piecemeal, unnamed | CLRS-Text per-step eval; Nanda et al. progress measures; coin a term ("token-role-stratified accuracy"), don't claim the primitive |

**Consequences for the writeup:**
- Frame Claim 1 as *trading depth for trace length* — exactly what CoT expressivity
  theory (Merrill & Sabharwal; Li et al. 2024) permits; we supply the empirical closure
  for BFS-connectivity at 2 layers.
- Claim 3's bisect is **done** (07-06): weight decay is the suppressor, dropout benign
  — foreground the low-rank-attention mechanism (Kobayashi) and contrast "Grok to Copy"
  on both regime and mechanism. Optional refinement: AdamW (decoupled) vs Adam (L2)
  at the same λ.
- "Supervision density arranged in time" is our framing, not a citable result — present
  as a lens, cross-referencing CLRS hint supervision and arXiv 2503.10542.
- Verify Ye et al.'s experimental scale directly from the paper (the research pass could
  not extract its full text) before asserting differences in print.

## 8. Relation to prior work

- **Sanford et al. 2024** — depth ↔ parallel rounds for graph reasoning; our answer-only
  baselines instantiate the negative side.
- **Merrill & Sabharwal 2024** — CoT strictly extends fixed-depth power; §3's
  diameter-flat curve is that theorem with a price tag.
- **Bachmann & Nagarajan 2024** — the next-token pitfall; §2.3 reproduces it on graphs
  and §2.3's trace redesign is the escape.
- **Olsson et al. 2022** — induction heads as two-layer circuits; the depth-1 failure and
  the parent-copy diagnostics land exactly on that boundary.
- **Power et al. 2022** — grokking; §2.5's data-driven phase transition is the same
  phenomenon with the *regularization sign flipped* (here weight decay prevented rather
  than enabled generalization — worth a discussion section of its own).
- **Ye et al. 2026 / June's connectivity-matrix results** — supervision density as the
  binding constraint; the trace is the sequential twin of their dense spatial target.

---

*Every number above is reproducible: configs in `configs/cot_ar*.yaml`, per-run JSONs in
`results/` (with git provenance embedded), diagnostics via `diag_cot_levels.py`, and the
day-by-day narrative in `docs/CHANGELOG.md` (2026-07-03 → 07-05).*
