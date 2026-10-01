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
- **Prediction:** at depth 1, m\*·H grows ≈ linearly in n (Sanford's one-layer bound;
  Yehudai's critical width); at depth ⌈log₂ n⌉ it grows much more slowly.
- **Experiment:** n ∈ {16, 32, 64, (128)}, m ∈ {n/4 … 4n}, depth 1 vs ⌈log₂ n⌉.
- **Output:** log–log m\* vs n; the slope is the answer.

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
Adjacency-row tokens are n-dimensional, so m < n compresses the input. Compare against
edge tokens (`node_edge`), whose token size doesn't grow with n. If Q3's slope
vanishes, Q3 was measuring the read-in. Protects Q3 at the defense.

### Q7 — Can depth substitute for width?
Depth {1, 2, 4, 8} × width grid at fixed n; iso-accuracy contours; test the Loukas-style
collapse of accuracy onto m·H·L.

## Timeline (department calendar, week 1 = 2026-09-07)

| Weeks | Work |
|---|---|
| 4–5 | Q1 + a small Q2 — method settled; first result for Явц 1 |
| 6 | Явц 1 |
| 6–9 | Q3, Q4 — the main results |
| 9–11 | Q6 (or Q5); **week 11 research freeze** |
| 13 / 15 / 16 | pre-defense / criticism / defense |
