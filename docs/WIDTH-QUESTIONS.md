# Research questions — the effect of width on graph transformers

Thesis: **Analyzing the Effect of Network Width on Graph Transformers** (settled
2026-09-24). Theory and citations behind every prediction below:
[`WIDTH-LITERATURE.md`](WIDTH-LITERATURE.md).

Each question is small: one sweep, one figure, one falsifiable prediction.
Q1–Q4 are the spine; Q5–Q7 are optional depth.

## Shared definitions

- **Width m** — the embedding dimension (`hidden`). Heads H and the FFN multiplier are
  reported alongside; Q5 separates them.
- **Graph transformer** — any transformer over a graph (Müller et al. 2023 taxonomy).
  We test two points on it: *tokens only* (adjacency-row or edge tokens) and
  *attention bias* (Graphormer-style shortest-path bias).
- **Critical width m\*** — the smallest m at which ≥ 2 of 3 seeds reach test
  exact-match ≥ 0.95, with the learning rate chosen on a **validation** split per
  width. Report robustness to the threshold (0.90 / 0.95 / 0.99).
- **Task substrate** — the dense connectivity-matrix target (`src/connectivity.py`):
  the binary connected/disconnected label stalls at ln 2 (CHANGELOG 2026-06-25), the
  n×n reachability matrix trains.

## Spine

### Q1 — Is a fixed learning rate confounding width?
Most width sweeps keep one LR across widths. If the optimal LR drifts with m, a fixed-LR
sweep measures LR mis-tuning, not width.
- **Experiment:** connectivity matrix, fixed n, depth 2, widths 8–256 at fixed head
  dim, LR grid {3e-4, 1e-3, 3e-3, 1e-2}, 3 seeds. "Fixed LR" = the 1e-3 column;
  "tuned" = best-validation LR per width.
- **Prediction:** optimal LR decreases as m grows; the fixed-LR curve under-reports
  small or large widths, shifting m\*.
- **Output:** test acc vs m, fixed vs tuned; heatmap of acc over (m, LR).
- **Status:** done (n=24, depth 2, 2000 train graphs, 3 seeds) — `width_sweep.py q1`,
  `q1b` (q1b adds 5% warm-up + cosine decay, widths 2/4, LR 3e-2).

  | m | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
  |---|---|---|---|---|---|---|---|---|
  | Q1 fixed 1e-3 | – | – | 0.917 | 0.917 | 0.781 | 0.912 | 0.820 | 0.596 |
  | Q1 tuned | – | – | 0.976 | 0.977 | 0.967 | 0.948 | 0.844 | 0.678 |
  | Q1b fixed 1e-3 | 0.497 | 0.483 | 0.759 | 0.911 | 0.709 | 0.623 | 0.489 | 0.478 |
  | Q1b tuned | 0.497 | 0.867 | 0.975 | 0.967 | 0.982 | 0.983 | 0.763 | 0.644 |

  1. **Yes, fixed LR confounds width.** Fixed-LR curves zig-zag and fall early; with
     the schedule, 1e-3 is too low for every width and the fixed curve collapses.
  2. **Optimal LR falls with width** (Q1: 1e-2 at m≤16 → 1e-3 at 256; high LRs
     diverge for wide models without warm-up).
  3. **Warm-up fixes the instability but not the wide-model drop.** With the
     schedule, m=128/256 reach train 1.000 at LR 1e-2 (Q1: 0.74/0.41) yet test stays
     0.76/0.48–0.64 — a genuine generalization gap at 2000 graphs, not optimisation.
  4. **Inverted U.** m=2 never leaves the all-connected predictor (0.50), m=4 underfits
     (train 0.84), m=8–64 ≈ 0.97–0.98, m≥128 memorizes. m\* (tuned) = 8 at 0.95,
     4 at 0.90 — well below n=24.
- **Follow-up (Q1c, done):** width × train size {500, 2000, 8000} at fixed 4800 steps.
  At 8000 graphs every m≥16 reaches ≥0.99 — the m≥128 drop was memorization; width has
  a floor (m≈4–8) and a data-set ceiling. 2000 arm reproduces Q1b exactly.
- **Caveat:** the `hard` generator never relabels nodes (blobs = index ranges) and is
  shallow (diam ≤ 8), so Q1 measures a near-local bridge check. Q3+ use relabelled
  `hard_diam` graphs. See CHANGELOG 2026-10-01.

### Q2 — Does the needed width follow the theory's task hierarchy?
Sanford 2024a: retrieval ≪ parallelizable (connectivity) ≪ search (shortest path).
- **Experiment:** edge existence / degree, connectivity, shortest-path length on the
  same graphs, n fixed, depth 1–2, width sweep.
- **Prediction:** m\*(retrieval) < m\*(connectivity) < m\*(shortest path).
- **Needs:** retrieval and shortest-path generators with leak audits.

### Q3 — How does m\* for connectivity scale with n?
- **Prediction:** at fixed depth, m\* grows with n (Sanford: one layer needs m·H ≳ n);
  at depth ⌈log₂ n⌉ it grows much more slowly.
- **Experiment:** relabelled `swap` graphs (locally indistinguishable classes; graph
  statistics and ≤4-hop heuristics at chance — `audit_width_data.py`; diameter ~log n),
  n ∈ {32, 64, 128}, depth 2 vs ⌈log₂ n⌉ (5/6/7), m ∈ {4 … 128}, LR {1e-3, 3e-3, 1e-2},
  3 seeds, 8000 train graphs, 4800 steps. Pair accuracy logged next to exact-match.
  Pilot 1 on hard_diam failed at n ≥ 32 (diameter ~n/2); see CHANGELOG 2026-10-02.
- **Output:** log–log m\* vs n per depth; the slope is the answer.
- **Status:** with 128 000 graphs (n = 32–56, depth 6, 24 000 steps) fit and generalize
  coincide and the critical width to learn grows ∝ n^1.94 (90% seed bootstrap 1.74–2.14;
  m*/n 0.68 → 1.10). At 32 000 graphs fit ∝ n^1.55, generalize ∝ n^2.27 (data ceiling).
  Step budget checked: 2× steps leaves the critical width unchanged (n = 40/56).
  LR checked: 1e-3 helps wide models at n=56; with tuned values the slope is ≈ n^1.85
  (90% 1.68–2.03). Next: Q6 (is the growth the n-wide read-in?).

### Q4 — Does Graphormer-style structure make width irrelevant?
- **Prediction:** with the SPD attention bias, connectivity m\* stays small and flat in
  n (the bias already encodes reachability); on a task SPD does not leak
  (cycle count, or distance clipped to k) width matters again.
- **Experiment:** repeat Q3 with the SPD bias on.

## Optional

### Q5 — Heads vs embedding: which width matters?
Fix m, vary H ∈ {1, 2, 4, 8, 16}; then fix H, vary m. Theory treats m·H as one product;
per-head dimension m/H says otherwise. Untested on graph tasks.

### Q6 — Is "m\* ≈ n" just an input bottleneck?
Adjacency-row tokens are n-dimensional, so m < n compresses the input.
- **Status:** done. Fixed 96-wide input (A + I)P (random node IDs, ~lossless): critical
  width 37.3 → 73.8 from n = 40 to 56 (∝ n^2.03), vs 30.2 → 57.9 (∝ n^1.93) for adjacency
  rows. The growth is not the read-in; the random code costs ~20–25 % more width.
  Edge tokens remain an optional further variant.

### Q7 — Can depth substitute for width?
- **Q7a (done):** n = 40, depth {2, 3, 4, 6, 8}: critical width 45.3 / 37.9 / 34.5 / 30.1 /
  32.3. Two layers learn at m ≈ 1.1 n (Yehudai's linear-width/constant-depth regime);
  depth saves ~⅓ of the width and saturates by 4–6 layers.
- **Q7c (done):** depth 2 at n = 32 / 48 / 56 — the shallow critical width grows ~n³
  (fit 28 / 44 / 71 / 168 at n = 32–56), steeper than depth 6 (~n^1.85), not linearly;
  at n = 56 no width ≤ 192 generalizes to 0.95.
- **Q7b (dropped 2026-10-13, supervisor):** node count vs path length. Sweeps `q7b_n`,
  `q7b_d0`, `q7b_d10` are defined and audited but not run; in the thesis it is a stated
  limitation (q3's diameter grows 8.7 → 11.8 alongside n).

## E — input encodings (from 2026-10-13, supervisor's direction)

How does the input encoding change the critical width and its growth with n? Same task
(audited `swap`, all-pairs reachability) and protocol as q3lr (128k graphs, 24k steps,
depth 6, LR 1e-3, 2 seeds, n = 32–56).

| Encoding | Sweep | Status |
|---|---|---|
| adjacency rows | q3lr / q3big / q3data | done: m* ∝ n^1.85 |
| fixed random-projection IDs (96 wide) | q6proj | done: ∝ n^2.0, ~20 % above |
| edge-list tokens (Sanford), all-pairs read-out | `enc_edges` | done: no critical width at 24k steps for any n (best: n = 32, m = 48, one seed, exact 0.76); ~3× the width of adjacency rows for equal pair accuracy |
| edge-list tokens, 4× steps (96k), n = 32 / 40 | `enc_edges96k` | done: m* (pair 0.95) 48.2 at n = 32, ≥ 72 at n = 40 — 2.2× / ≥ 2.4× adjacency rows; seeds split (one stalls on a partial solution) |
| adjacency + k-hop reachability rows (k = 2, 4) — k hops for free, diameter / k left | `enc_khop2`, `enc_khop4` | done: m* 13.0 / 18.3 (k = 2, ∝ n^1.0) and 3.6 / 4.5 (k = 4, ∝ n^0.6) at n = 40 / 56 vs 30.1 / 57.9 — free hops collapse width and growth |
| adjacency + RWSE, return probabilities diag(Pʲ), j ≤ 8 | `enc_rwse8` | done: 31.9 / 57.9 — same as adjacency rows; audited clean (`audit_walk_features.py`) |
| adjacency + RWSE j ≤ 16 (GraphGPS default) | `enc_rwse16` | done (positive control; leaks 0.90 / 0.72): 23.5 / 48.2, lower where it leaks more |
| node-ID variants (learned / random / one-hot) | — | optional |
| Laplacian eigenvectors, shortest-path bias | — | positive controls only (they give connectivity away) |

## Timeline (department calendar, week 1 = 2026-09-07)

| Weeks | Work |
|---|---|
| 4–5 | Q1 + a small Q2 — method settled; first result for Явц 1 |
| 6 | Явц 1 |
| 6–9 | Q3, Q4 — the main results |
| 9–11 | Q6 (or Q5); **week 11 research freeze** |
| 13 / 15 / 16 | pre-defense / criticism / defense |
