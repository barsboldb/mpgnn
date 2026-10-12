# Sanford et al. 2024a — Reproduction & Shortcut Audit

Notes from reproducing the connectivity experiment of Sanford et al., *Understanding
Transformer Reasoning Capabilities via Graph Algorithms* (NeurIPS 2024; §4, App. E;
`papers/sanford-2024-transformer-reasoning-graph-algorithms.pdf`), asked for by the
supervisor on 2026-10-09: "did you reproduce what they claimed and what they did?"

Code: `sanford_repro.py` (sweeps `a1k`, `a100k*`, `bwidth`; `--audit`, `--analyze`).
Runs on Kaggle via `kaggle/width_sweep.ipynb` with `SCRIPT = "sanford_repro.py"`.
Results: `results/sanford/`. Dated log: `docs/CHANGELOG.md` (2026-10-09 → 2026-10-11).
Reading notes on the paper's theory: `reports/sanford-2024-notes.typ`.

## Summary

| What | Theirs | Ours | Status |
|---|---|---|---|
| Data distribution (GraphQA reachability) | 11.9 nodes / 37.0 edges | 12.0 / 37.7 | ✓ reproduced |
| Test accuracy, 1K training graphs | 92.9 | **93.0** | ✓ reproduced |
| Test accuracy, 100K training graphs | 98.0 | **98.2** | ✓ reproduced — at LR 1e-4, not their 5e-4 |
| Their LR 5e-4 at 100K | (works for them) | stuck at ~0.93, NaN | ✗ not reproduced (batch size unknown) |
| "Transformers excel at *global* reasoning" | claimed | a 2-hop classifier scores 99.5 | ✗ not supported by this data |
| Width claim (√N / N^ε) | theory only, never varied | 98 reached at m = 32; hard cases flat in m | ✗ untestable on GraphQA |
| Other GraphQA tasks, GNN and prompting baselines, T5-11B | reported | — | not attempted (connectivity only) |

Added beyond the paper: a shortcut audit (§4), a hard test set, and a width sweep (§6).

---

## 1. Claim vs experiment

- **The claim (theory).** Connectivity is a *parallelizable* task: a transformer of depth
  $O(\log N)$ solves it with width about $\sqrt N$ without pause tokens, or $N^\epsilon$
  with them, where $N = |V| + |E|$ is the number of tokens. One layer provably cannot.
- **The experiment.** One fixed model — decoder-only, $L = 12$, $m = 768$, $H = 12$, GLU,
  ~60M parameters, 1M steps on TPUs — trained on GraphQA connectivity at 1K and 100K
  graphs: test accuracy **92.9** and **98.0**.
- **The gap.** Width is never varied, and $m = 768$ is ~40× the largest graph ($n \le 19$).
  The experiment cannot confirm or refute the width claim.

No model or training code was released; only the dataset code
([google-research/graphqa](https://github.com/google-research/google-research/tree/master/graphqa)).

## 2. What we reproduced

| Part | Source | Ours |
|---|---|---|
| Task | GraphQA `Reachability`: ER, n ~ U{5..19} by size bucket, p ~ U(0,1), one random s–t pair | re-implemented in numpy (same distribution, not the same graphs); mean nodes 12.0 / edges 37.7 vs paper 11.9 / 37.0 |
| Tokens | Fig. 1: vertex tokens, edge tokens, task token | type + node-ID(a) + node-ID(b) + position embeddings; answer read at the task token |
| Model | App. E.2 | pre-LN causal decoder, L=12, m=768, H=12, GLU, dropout 0.1, AdamW: **71M** params |
| Data | 1K / 100K train, 500 dev, 500 test | same train sizes, 500 dev, **2000 test** (±0.3 pt) |

Guessed (not in the paper): batch 64, weight decay 0.01, 1k warm-up, gradient clip 1,
learning-rate schedule. Changed: 20k (1K) and 60k (100K) steps instead of 1M.

## 3. Results — part A

| Run | LR | Test (dev-selected) | Train | Paper |
|---|---|---|---|---|
| `a1k` | 5e-4 constant | **0.930** | 1.000 | 92.9 |
| `a100k` | 5e-4 constant | 0.920, then NaN at ~31k | ~0.92 | 98.0 |
| `a100k_cos` | 5e-4 → cosine | 0.932 | ~0.93 | |
| `a100k_lr1e4` | 1e-4 → cosine | **0.982** | 1.000 | 98.0 |

- **Both numbers reproduce**, but 100K only at 1e-4: their 5e-4 (constant or decayed)
  never fits the training set in our setup and overflows fp16. Likely their batch was
  larger; the paper does not say. A repeat of the constant-5e-4 run failed the same way.
- At 1K the model memorizes the training set by step 1k and generalizes to 93 %.

## 4. Shortcut audit

`python sanford_repro.py bwidth --audit` — rules, then gradient-boosted trees fit on 50k
train-distribution examples, scored on 20k test-distribution examples and on a **hard
set**: 1000 connected pairs ≥ 4 hops apart (the 3-hop rule is always wrong) and 1000
no-path pairs whose endpoints both have edges (the degree rule is always wrong).

| Predictor | Test | Hard: 4+ hops | Hard: no path |
|---|---|---|---|
| always "yes" | 0.833 | 1.000 | 0.000 |
| rule: both endpoints have an edge | 0.973 | 1.000 | 0.000 |
| rule: within 2 hops | 0.932 | 0.000 | 1.000 |
| rule: within 3 hops | 0.981 | 0.000 | 1.000 |
| trees: graph size (n, edges, density) | 0.944 | 0.878 | 0.720 |
| trees: + endpoint degrees | 0.977 | 0.938 | 0.411 |
| **trees: + 2-hop neighbourhoods** | **0.995** | **0.932** | **0.893** |
| reproduced transformer (`a100k_lr1e4`) | 0.982 | — | — |
| same model at 30k steps (`bwidth` m=768) | 0.980 | 0.808 | 0.700 |

- **Graph size alone gives 0.944**, and the model sees it: the token count is n + |E| + 1.
- **A 2-hop learner beats the reproduced 71M transformer** (0.995 vs 0.982). With ≤ 19
  nodes the 2-hop ball usually covers the component.
- **Even the hard set is ~90 % local.** It separates the two rules, but a model shows
  non-local reasoning only by clearly beating 0.93 / 0.89, not 0.5.
- 94 % of connected pairs are ≤ 2 hops apart.

## 5. What it means

1. Their experiment shows transformers *fit* GraphQA connectivity; it does not show they
   *compute* it. A 2-hop classifier does better, so the benchmark cannot support the
   claim that transformers excel at global reasoning here.
2. Their width claim is untested by their own experiment (§1).
3. The same leak class — degree and local statistics on small dense ER graphs — is what
   our own first benchmark had, and why `swap` exists (`reports/width-progress.pdf`, §
   "Making the task honest").

## 6. Part B1 — width on their setup

`bwidth`: their model and 100K graphs at m = 8…768 (L = 12, heads m/64 ≥ 1), 30k steps,
cosine 1e-4, plus 1e-3 for m ≤ 128; seed 0. Best rate per width, final step:

| m | params | test | hard: 4+ | hard: no path |
|---|---|---|---|---|
| 8 | 0.01M | 0.957 | 0.892 | 0.626 |
| 16 | 0.04M | 0.967 | 0.863 | 0.641 |
| 32 | 0.13M | **0.983** | 0.869 | 0.644 |
| 64 | 0.52M | 0.979 | 0.854 | 0.670 |
| 128 | 2.0M | 0.979 | 0.851 | 0.665 |
| 256 | 8.0M | 0.984 | 0.846 | 0.686 |
| 768 | 71M | 0.980 | 0.808 | 0.700 |
| *2-hop trees* | | *0.995* | *0.932* | *0.893* |

(Row per width: the rate with the higher final dev accuracy — 1e-3 for m ≤ 128,
the only rate run above. Hard-set columns ±1.5 pt; one seed.)

- **Their 98.0 needs only m = 32** (130K parameters, ~550× fewer than theirs); test
  accuracy is flat at ~0.98 from m = 32 to 768.
- **No width beats the 2-hop learner on the hard set** — every model sits at 0.81–0.89
  (4+) and 0.63–0.70 (no path), below 0.93 / 0.89, and width does not move it.
- Narrow models don't even finish the degree rule: "no path, isolated" (degree rule
  1.00) is 0.84–0.94 at m ≤ 16 and ≥ 0.98 from m = 32.
- So on GraphQA width only buys the local shortcuts, and they saturate at m ≈ 32 for
  n ≤ 19. There is no connectivity-specific critical width to measure on this data.

## 7. Why GraphQA, and why no audit

The paper doesn't say; this is inference from the paper and the benchmark's origin.

- **Their own benchmark.** GraphQA is from Fatemi, Halcrow & Perozzi, *Talk like a Graph*
  (ICLR 2024) — all three are co-authors here. It already existed, with published
  baselines.
- **Built for LLM prompting, not for algorithmic reasoning.** Graphs are 5–19 nodes so
  they fit in a prompt as text; plain ER, no shortcut control.
- **Their empirical claims are comparisons** (trained transformers vs GNNs vs prompted
  LLMs), and the prompting / GraphToken numbers exist only on GraphQA.
- **The paper is mainly theory;** §4 is framed as "illuminating" the hierarchy.
- **They partly noticed.** App. E: GraphQA graphs "do not resemble the large-diameter
  worst-case instance" behind their lower bounds, and cycle check is "inherently local".
  The same was not quantified for connectivity.
- **Shortcut audits weren't yet standard** for graph-reasoning benchmarks in 2024;
  Abbe et al. (NeurIPS 2024, cycle task) and Ye et al. (2026, degree heuristic) made the
  issue prominent around and after it.
- **Their ranking claims don't need one.** Only "global reasoning" does, and they assert
  it without a direct test. In their favour: local GNNs score below the transformer
  (94.4 vs 98.0) — but a 2-hop classifier beats both, so that gap more likely reflects
  trainability than reach.

Thesis framing: complete their App. E caveat rather than "catch" an error — quantify how
local GraphQA is, show m = 32 suffices there, and test the width claim on audited data.

## 8. Run log

All on Kaggle T4 (2 per session), seed 0, fp16 autocast.

| Sweep | Runs | Steps | Time | Outcome |
|---|---|---|---|---|
| `a1k` | 1 | 20k | 48 min | 0.930 |
| `a100k` | 1 (+1 repeat, not kept) | 60k | 2.1 h each | failed: plateau ~0.92, NaN at ~31k (repeat: ~52k) |
| `a100k_cos` | 1 | 60k | 2.3 h | 0.932, plateau |
| `a100k_lr1e4` | 1 | 60k | 2.5 h | 0.982 |
| `bwidth` | 12 | 30k | ~30 min each, m=768 88 min | §6 |

```sh
python sanford_repro.py a1k --audit        # §4 (any sweep name; ~40 s)
python sanford_repro.py <sweep>            # train (resumes; --shard i/k for workers)
python sanford_repro.py <sweep> --analyze  # tables incl. hard-set buckets
```

## 9. Part B2 — their encoding on audited graphs

`b2`: Q3's setting (q3lr) with Sanford's tokenization instead of adjacency rows.

- **Graphs:** `swap` (chord fraction 0.25, d_min 6), n = 32 / 40 / 48 / 56, node labels
  shuffled per graph, 128k training graphs (one set per n, shared by both seeds),
  400 dev / 400 test.
- **Tokens:** n vertex + |E| edge + **32 query tokens** (uniform random s ≠ t), each read
  out at its own position; test query accuracy estimates Q3's pair accuracy. N = n + |E|
  ≈ 2.25n graph tokens (73–128).
- **Model:** the causal GLU decoder, depth 6, head dim 8, Adam 1e-3 → cosine, no dropout,
  batch 128, 24k steps — Q3's optimizer and size, so only the encoding / read-out differ.
- **Sweep:** m = 8, 16, 24, 32, 48, 64, 96 × 4 n × 2 seeds = 56 runs; critical width where
  test query accuracy crosses 0.95 (`--scaling`), against Q3 (21.9 / 30.2 / 51.1 / 57.9,
  ∝ n^1.85), √N (≈ 9–11) and linear.

**Audit** (`python sanford_repro.py b2 --audit`, 400 test graphs × 32 queries per n):

| Predictor | n=32 | 40 | 48 | 56 |
|---|---|---|---|---|
| always "yes" | 0.749 | 0.748 | 0.757 | 0.760 |
| within 4 hops | 0.751 | 0.693 | 0.639 | 0.602 |
| within 6 hops | 0.917 | 0.870 | 0.840 | 0.809 |
| within 8 hops | 0.998 | 0.981 | 0.961 | 0.933 |
| trees: + 2-hop neighbourhoods | 0.749 | 0.746 | 0.756 | 0.759 |

Local learners sit at the trivial rate (vs 0.995 on GraphQA); 0.95 needs ~7–8 hops of
reach, i.e. computing reachability over most of the diameter.

**Result (Kaggle, 2026-10-12, 56 runs, ~35–50 min each): no run leaves the trivial
solution.** Every width at every n ends at train / test accuracy = the always-"yes" rate
(train 0.745 / 0.749 / 0.754 / 0.756, test 0.739 / 0.751 / 0.758 / 0.761), exact-graph
accuracy 0.50; no critical width exists up to m = 96. Not a bug: labels recomputed from the
decoded tokens match (0 / 9600 mismatches) and the read-out takes the 32 query positions.

The loss shows where it is heading. It sits at the always-"yes" entropy (≈ 0.568 at
n = 32) and drifts below it only late, by an amount ordered by width — final loss at seed 0:

| n | m=8 | 32 | 64 | 96 |
|---|---|---|---|---|
| 32 | 0.567 | 0.567 | 0.556 | **0.532** |
| 40 | 0.562 | 0.561 | 0.559 | 0.548 |
| 48 | 0.557 | 0.557 | 0.552 | 0.539 |
| 56 | 0.552 | 0.551 | 0.546 | 0.537 |

(m = 96 n = 32 seed 0: flat to step 14k, then 0.566 → 0.532 by 24k; seed 1 only 0.559.)
So the widest models are just starting to escape when the budget ends. Under the **same**
budget and depth, adjacency rows (Q3) cross 0.95 at m = 21.9 – 57.9. Their encoding is
therefore *harder* to learn on this task, not easier — the opposite of what Sanford's √N
representation bound would suggest. What B2 cannot yet give is the critical width itself:
that needs a longer budget (or wider models) for the escape to complete.

