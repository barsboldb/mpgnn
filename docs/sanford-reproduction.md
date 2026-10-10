# Sanford et al. 2024a — Reproduction & Shortcut Audit

Notes from reproducing the connectivity experiment of Sanford et al., *Understanding
Transformer Reasoning Capabilities via Graph Algorithms* (NeurIPS 2024; §4, App. E;
`papers/sanford-2024-transformer-reasoning-graph-algorithms.pdf`), asked for by the
supervisor on 2026-10-09: "did you reproduce what they claimed and what they did?"

Code: `sanford_repro.py` (sweeps `a1k`, `a100k*`, `bwidth`; `--audit`, `--analyze`).
Runs on Kaggle via `kaggle/width_sweep.ipynb` with `SCRIPT = "sanford_repro.py"`.
Results: `results/sanford/`. Dated log: `docs/CHANGELOG.md` (2026-10-09, 2026-10-10).
Reading notes on the paper's theory: `reports/sanford-2024-notes.typ`.

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
| reproduced transformer (`a100k_lr1e4`) | 0.982 | pending (`bwidth` m=768) | pending |

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

## 6. Next

- **B1 — width on their setup** (`bwidth`, running): m = 8…768 at L = 12, 100K graphs,
  cosine 1e-4 (and 1e-3 for m ≤ 128). Read the hard-set accuracy against the 2-hop
  trees, not just test accuracy.
- **B2 — their encoding on our graphs** (to build): edge tokens + s–t query on audited
  `swap` graphs at n = 32–56; critical width vs n, against √N and our ~n² (Q3).
