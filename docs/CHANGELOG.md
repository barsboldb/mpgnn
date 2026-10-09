# Experiment Changelog

Record of findings, bugs, and decisions made during experiments.

---

## 2026-10-09

### Sanford et al. 2024a reproduction set up; their connectivity data is degree-solvable

Supervisor asked whether Sanford's experiment was reproduced. Their §4/App. E: GraphQA
`Reachability` (ER, n 5–19, p ~ U(0,1), one s–t pair; 1K / 100K train), vertex + edge +
task tokens, decoder-only L=12, m=768, H=12, ~60M params, 1M steps → test 92.9 (1K) /
98.0 (100K). Their experiment fixes m = 768 (≈ 40× n), so it never tests the
sublinear-width claim, which is theory only (√N without pause tokens, N^ε with).

Audit of the same distribution (`sanford_repro.py a1k --audit`, 20k examples; mean
nodes 12.0 / edges 37.7 vs the paper's 11.9 / 37.0): always-yes 0.833; **"yes iff both
endpoints have degree > 0" 0.973**; within 2 hops 0.932, within 3 hops 0.981. A
one-line degree rule matches their 100K transformer.

`sanford_repro.py`: re-implemented generator (same distribution, not the same graphs),
their tokenization, pre-LN causal decoder with GLU (71M params at L=12, m=768), AdamW
5e-4, dropout 0.1; unstated choices (batch 64, wd 0.01, 1k warm-up, clip 1, length-
bucketed batches) in the docstring. Sweeps: `a1k` / `a100k` (part A, 20k / 60k steps
instead of 1M), `bwidth` (part B1: m = 8…768 at L=12, 100K graphs). Kaggle notebook
generalized to run either script (`SCRIPT`, `SWEEPS`).

### Q7c: depth-2 critical width grows ~n³, not linearly

q7c: depth 2 at n = 32 / 48 / 56, same setting as q7depth (`swap`, 128k graphs, 24 000
steps, LR 1e-3), widths 16–192 per n, 2 seeds (34 runs, Kaggle); n = 40 from q7depth.
Critical width (mean of per-seed crossings at 0.95, final epoch):
fit **28.1 / 44.4 / 70.7 / 168.0** (∝ n^3.07; one seed per n: 2.90–3.24);
generalize 28.2 / 45.3 / 92.7 / >192 (∝ n^2.91 over n ≤ 48).

- Not Yehudai's linear width: SGD doesn't find the constant-depth construction as n grows.
  Steeper than depth 6 (∝ n^1.85); depth-2 ÷ depth-6 width 1.3 → 1.5 → 1.8 → >3.3.
- Threshold-robust in direction: fit slope 2.44 at 0.90, 2.79 at 0.93.
- Local slopes 2.1 / 2.6 / 5.6 — the n = 56 point is the least certain (curve flat near
  0.95; m=192 train pair 0.96, seeds 158 / 178).
- Fit/generalize gap reopens at depth 2 even at 128k graphs (n=56, m=192: 0.96 vs 0.92).
- Q7a's "depth trades mildly" holds only at n = 40. (`results/width/q7c.shard*of8.jsonl`;
  `width_sweep.py q7c --scaling --also q7depth`)

## 2026-10-08

### Q7a: depth trades for width only mildly; 2 layers learn at m ≈ 1.1 n

q7depth: n=40 `swap` data (connected diameter ~10), 128k graphs, 24 000 steps, LR 1e-3,
depth ∈ {2, 3, 4, 6, 8}, per-depth widths around the threshold, 2 seeds (60 runs).
Critical width to generalize (mean of per-seed crossings): **45.3 / 37.9 / 34.5 / 30.1 /
32.3** (m*/n 1.13 → 0.75); fit 44.4 / 34.2 / 31.1 / 28.2 / 29.7. Seeds within ~±2.

- Depth 2 learns: m=48 test pair 0.968, m≥64 ≈ 0.99, on diameter-10 graphs — the
  "2^L ≥ diameter" prediction was wrong; the model doesn't trace paths hop by hop.
- Mild, saturating trade-off: 2 → 6 layers saves ~⅓ of the width; 8 = 6.
- L=6 reproduces q3lr's n=40 critical width exactly (30.1).
- Matches Yehudai et al. 2025 (adjacency rows: linear width ⇒ constant depth); measured
  constant ≈ 1.1 n at L=2. Open: does depth-2 m* grow linearly or ~n² across n (Q7c)?
  Kaggle quota exhausted until the Saturday reset. (`results/width/q7depth.shard*of8.jsonl`)

## 2026-10-07

### Q6: a fixed-width input keeps the ~n² growth — not the read-in

q6proj: node input (A + I) P with P a fixed random n × 96 ±1/√96 matrix (per seed,
untrained) instead of the n-wide adjacency row — 96 wide for every n, still ~lossless
(top-degree decoding recovers 99 % of neighbour sets at n = 32/56; k = 32 only 60–70 %,
rejected). Otherwise the tuned 128k baseline: 24 000 steps, LR 1e-3, depth 6, n = 40/56,
m ∈ {16 … 96}, 2 seeds (24 runs). Critical width (mean of per-seed crossings):

| | n=40 | n=56 | slope |
|---|---|---|---|
| baseline adj rows, tuned | 30.2 | 57.9 | 1.93 |
| idproj, generalize | 37.3 (32.3/42.2) | 73.8 (69.8/77.8) | **2.03** |
| idproj, fit | 37.0 | 57.8 | 1.33 |

The growth persists with an input whose size doesn't depend on n, so it isn't the n-wide
read-in. The random code costs ~20–25 % more width at both sizes (it must be decoded)
without changing the slope. Larger seed spread than the baseline. A local pre-check
(n=40, m=48, 16k graphs, 8000 steps) had idproj generalizing better (test pair 0.927 vs
0.835 at equal train fit) — at 128k graphs that advantage is gone.
(`results/width/q6proj.shard*of8.jsonl`)

### Q3 learning rate: 1e-3 helps wide models at n=56; slope ~n^1.85

q3lr: same as the 128k baseline (24 000 steps, depth 6) at LR **1e-3**, n=40/56 on the
q3steps widths, 2 seeds (16 runs). Tuned = rate picked per (width, seed) on final
validation exact-match (the 128k baseline at n=56 comes from q3data, which predates
`val_pair`, so pair accuracy can't be the selector).

| | 3e-3 | 1e-3 | tuned |
|---|---|---|---|
| n=40 fit / generalize | 29.5 / 30.4 | 28.2 / 30.1 | 28.0 / **30.2** |
| n=56 fit / generalize | 59.4 / 61.5 | 52.7 / 57.9 | 52.7 / **57.9** |

n=40: rate doesn't matter. n=56: 3e-3 was too hot for wide models (Q1 again) — m=96 test
pair 0.932 → 0.984; validation picks 1e-3 for every m ≥ 48. Slope 40 → 56: 2.09 → 1.93.
Four-point slope with tuned values at 40/56 (3e-3 at 32/48, the only rate run there):
**n^1.85**, seed bootstrap 90% 1.68–2.03. The superlinear growth survives LR, data and
step-budget checks. `--prepare` worked this time (no shard waited).
(`results/width/q3lr.shard*of8.jsonl`)

### Q3 step budget: 2× steps leaves the critical width unchanged

q3steps: 128 000 graphs, depth 6, LR 3e-3, **48 000** steps (vs 24 000), n=40 with
m ∈ {16, 24, 32, 48} and n=56 with m ∈ {32, 48, 64, 96} (per-n widths around each
threshold), 2 seeds (16 runs). Critical width, mean of per-seed crossings:

| | 24k steps | 48k steps |
|---|---|---|
| n=40 fit / generalize | 29.5 / 30.4 | 26.9 / **29.8** |
| n=56 fit / generalize | 59.4 / 61.5 | 56.3 / **67.4** (seeds 62.7 / 72.0) |

More training lowers the fit width by ~3 and leaves the generalize width unchanged
within seed spread; at n=56 the longer run memorizes a little (m=64 train/test pair
0.972/0.960 → 0.975/0.948, 48 passes over the data). Slope n=40 → 56 at 48k: ~2.2 fit,
~2.4 generalize — consistent with n^1.94. The ~n² width to learn is not a training-time
effect. Shards again waited ~30 min for the n=40 cache (`--prepare` didn't pre-build it;
the lock fix kept them alive). (`results/width/q3steps.shard*of8.jsonl`)

## 2026-10-06

### Q3 at 128k graphs: the width to learn grows ~n² even with ample data

n ∈ {32, 40}, m ∈ {16 … 96}, 128 000 graphs, 24 000 steps, LR 3e-3, 2 seeds (24 runs),
joined with q3data's n = 48/56 at 128k. Critical width, mean of per-seed crossings at
0.95, final epoch:

| | n=32 | n=40 | n=48 | n=56 | slope |
|---|---|---|---|---|---|
| fit | 20.9 | 29.5 | 48.4 | 59.4 | 1.95 |
| generalize | 21.9 | 30.4 | 51.1 | 61.5 | **1.94** (seed bootstrap 90%: 1.74–2.14) |

Fit and generalize coincide at every n (within ~2), so the data ceiling is gone — but the
slope stays near 2 (m*/n: 0.68 → 0.76 → 1.07 → 1.10). This revises the q3data reading
("≈ 1.1 n"): that held at n ≈ 50 only. The superlinear growth is not a data artefact.
Uneven (steepest 40 → 48; n=48 seeds 46.5 / 55.8). Remaining confounds: fixed 24 000
steps, one LR. (`results/width/q3big*.jsonl`)

### Kaggle: a 2 h lock timeout killed six shards

One shard built the n=32, 128k cache itself (> 2 h on Kaggle's shared CPUs; `--prepare`
evidently hadn't produced it), the six waiting shards hit `cached()`'s 7200 s timeout and
died; the notebook's catch-up pass then ran their 17 runs serially into `q3big.jsonl`.
Results complete, ~2 h of GPU quota lost. `cached()` now has no timeout: the lock file
holds the owner's PID, waiters keep waiting while it lives and take over a stale lock.

## 2026-10-05

### Correction: pair accuracy was read at the wrong checkpoint

The `best_val` checkpoint is chosen on validation *exact-match*. When no validation
graph is ever fully right, every epoch ties at 0.5 and epoch 1 wins — so "test pair at
best-val" reported the trivial predictor's pair accuracy, not the trained model's
(q3fine n=48 m=32: 0.755 reported, 0.885 / 0.853 at the final epoch). `--scaling` and
the report now use the final epoch for pair metrics; runs from now on also log
`val_pair`. Impact on q3fine critical widths is small — generalize 23.2 / 38.1 / 56.4 →
22.9 / 37.6 / 57.5 (mean of per-seed crossings), slope 2.19 → **2.27**; fit unchanged
(1.53 pooled, 1.55 seed-mean). The q3fine curves figure changed most: n=48/56 at
m ≤ 32 are 0.75–0.88, not flat at trivial.

### Q3 data check: 128 000 graphs close the fit/generalize gap

n ∈ {48, 56}, m ∈ {32, 48, 64, 96}, 64 000 and 128 000 graphs (nested: the smaller set
is a prefix), same 24 000 steps and LR 3e-3, 2 seeds (32 runs). Critical width, mean of
per-seed crossings (`width_sweep.py q3data --scaling`):

| | 32k | 64k | 128k |
|---|---|---|---|
| n=48 fit / generalize | 31.7 / 57.5 | 40.5 / 54.1 | 48.4 / **51.1** |
| n=56 fit / generalize | 44.7 / >96 | 57.8 / >96 (one seed 81.8) | 59.4 / **61.5** |

At 128k the two nearly meet; n=56 m=64 train/test pair goes 0.984/0.879 → 0.975/0.927
→ 0.972/0.960. The fit width *rises* with data at fixed steps (each graph seen less).
So the 2.27 generalize slope was mostly the data ceiling; with enough data the width to
learn is ≈ 51 / 61 at n = 48 / 56 (≈ 1.1 n). Two sizes 17 % apart can't fix a slope (the
pooled crossing at n=48, 128k, is 69 because one seed dips at m=64 — `--scaling` now also
prints the seed-mean). **Next:** q3big — n=32/40 at 128k for a four-point fit.
(`results/width/q3data.shard*of8.jsonl`)

### Q3 fine grid: critical width grows superlinearly in n

`swap` data, n ∈ {32, 40, 48, 56}, depth 6 (= ⌈log₂ n⌉ for 33–64), m ∈ {16 … 128},
32 000 graphs, 24 000 steps, LR 3e-3, 3 seeds (84 runs). Test pair accuracy (trivial
0.745–0.759): m=24 reaches 0.96 at n=32; n=40 needs m=48 (0.98), n=48 m=64 (0.965),
n=56 gets 0.93 at m=96. Critical width where the seed-mean curve crosses 0.95
(log-interpolated, `width_sweep.py q3fine --scaling --max-width 96`):

| | n=32 | n=40 | n=48 | n=56 | slope |
|---|---|---|---|---|---|
| fit (final train pair) | 18.9 | 23.3 | 31.4 | 44.9 | **n^1.53** |
| generalize (test pair) | 23.2 | 38.1 | 56.4 | >96 | **n^2.19** (32–48) |

Per-seed crossings agree within ~±3. Both exceed linear, far above the ~n^ε width that
suffices to *represent* connectivity at log depth (Sanford 2024a) — this is the width
to *learn* at this budget. The fit/generalize gap widens with n (n=56, m=64: train pair
0.984, test 0.820): the Q1c data ceiling again, since 32 000 graphs is fewer examples
per pair as n grows — the 2.19 is probably inflated by fixing data; 1.53 is the cleaner
number. m=128 is excluded: LR 3e-3 is too hot there (train pair drops to 0.96 at
n=48/56), the Q1 effect. **Next:** q3data — n=48/56, m ∈ {32 … 96}, 64k and 128k graphs
at the same steps. (`results/width/q3fine.shard*of8.jsonl`)

## 2026-10-04

### Q3 trimmed grid: a clean width floor at n=32; nothing learns at n ≥ 64

`swap` data, 32 000 graphs, 12 000 steps, depth ⌈log₂ n⌉, m ∈ {8 … 128}, LR {1e-3,
3e-3}, 2 seeds (60 runs, Kaggle, 8 shards). Tuned test exact-match:

| m | n=32 (L5) | n=64 (L6) | n=128 (L7) |
|---|---|---|---|
| 8 | 0.50 (train 0.27) | 0.50 | 0.50 |
| 16 | 0.49 (train 0.39) | 0.50 | 0.50 |
| 32 | 0.90 | 0.50 | 0.50 |
| 64 | 0.97 | 0.50 | 0.50 |
| 128 | 0.97 | 0.50 | 0.50 |

- **n=32: m\* between 32 and 64 (~1–2·n)** — m=8 can't fit train, m=16 starts
  (pair 0.83–0.88) but stalls. First clean critical width on shortcut-free data (the
  shortcut-laden `hard` data at n=24 needed only m=8).
- **n=64: width decides whether learning starts.** m ≤ 32 never leave the trivial
  predictor; m=64 @ 3e-3 starts (train loss 0.59 → 0.07–0.10, test pair 0.80–0.85);
  m=128 @ 3e-3 gets further (loss 0.03–0.04, pair 0.89–0.90). Exact-match stays 0.5:
  ~4000 pairs must all be right.
- **n=128:** test pair stays at the trivial 0.770 for every m ≤ 128.

Onset width (first m whose test pair clearly leaves trivial): ~16 at n=32, ~64 at
n=64, >128 at n=128 — at least linear in n, the direction of the one-layer gathering
bound, but 2 seeds × 3 sizes is a trend, not a slope. Next: finer n ∈ {32 … 56} where
m\* is reachable, pair-accuracy m\* alongside exact-match.
(`results/width/q3trim.shard*of8.jsonl`)

## 2026-10-02

### Q3 pilot 1: relabelled hard_diam — nothing generalizes at n >= 32

18 runs (seed 0, LR 3e-3, 8000 graphs). n=16 learns with the theory's shape: m=8
fails, m≥32 reaches 0.99 at depth 4 vs 0.87–0.90 at depth 2. At n=32/64 every run
sits at 0.50 test (the all-connected predictor; pair acc = its 0.798/0.817), one
memorizes (n=32 L5 m=128: train 0.98). Cause: hard_diam diameter grows ~n/2
(7 → 14 → 28), so raising n raises both node count (width's axis) and path length
(depth's axis). (`results/width/q3pilot.shard*of8.jsonl`)

### Q3 learnability probe: the clean task is learnable — it was budget (mostly data)

n=32, depth 5, `swap` data, 32 000 graphs, 24 000 steps, m ∈ {64, 128, 256} × LR
{1e-3, 3e-3}, seed 0. Final test exact-match: m=64 0.985/0.983, m=128 **0.998**/0.965,
m=256 0.988/0.007 (3e-3 never trains). Test tracks train (0.998 vs 1.000), pair acc
0.998–0.999 — reachability generalizes on data where statistics and ≤4-hop checks are
at chance. By epoch 12 (~3000 steps, fewer than pilot 2's 4800) test is already
0.54–0.83, so the pilot-2 failure was mainly 8000 graphs (m=128 memorized there:
train 0.99 / test 0.53) — the Q1c data ceiling again (schedules differ, so suggestive,
not controlled). Wider models start later at a fixed LR (epoch 12, LR 3e-3: test 0.83 /
0.40 / 0.07 for m = 64/128/256). (`results/width/q3probe.shard*of8.jsonl`)

### Q3 pilot 2: on shortcut-free data almost nothing learns

`swap` graphs, n ∈ {32, 64, 128}, depth 2 vs ⌈log₂ n⌉, m ∈ {8, 32, 128}, seed 0,
LR 3e-3, 8000 graphs, 4800 steps (18 runs). 17/18 sit at 0.50 test exact-match
(the all-connected predictor; pair acc = its 0.745/0.761/0.770); n=32 L2 m=128
memorizes (train 1.00). The one signal: **n=32, depth 5, m=128 — test 0.532, pair
0.869** vs 0.745 trivial, i.e. partial reachability, not memorization. So the Q1 and
pilot-1 n=16 successes leaned on locality/statistics; with those removed, no width
≤128 learns within this budget even at the theory's depth (cf. Saparov et al. 2025:
transformers struggle to learn search, scale doesn't fix it). Critical width is
undefined until something learns. **Next:** learnability probe — n=32, depth 5,
m ∈ {64, 128, 256}, 5× the steps, 4× the data. (`results/width/q3pilot2.shard*of8.jsonl`)

### Shortcut audit of the candidate generators (`audit_width_data.py`)

Heuristics measured per generator and n: all-ones; k-hop reachability
(exact-match of [dist ≤ k]); gradient boosting on graph statistics (edges, degree
histogram, tr(A³…A⁶)) predicting the connected bit; stats + 4-hop combined.
- **Graph statistics leak the label at small n:** 0.68–0.90 at n=16 for hard,
  hard_diam and sparse blobs; a bridge closes no cycle while the matched
  intra-blob chord closes a short one, visible in tr(A³), tr(A⁴).
- **A first swap version leaked more (0.98 at n=16):** in tiny blobs the far
  pair was already adjacent (edge count 25.76 vs 26.00), and far pairs 2–3 hops
  apart made triangles/4-cycles in the disconnected class only.
- **Small graphs are local:** at n=16 a 6-hop check is exact on 0.6–1.0 of graphs.

**Fix — `swap` generator:** two sparse blobs (cycle + ¼·size chords), far pair
a₁a₂ in A and b₁b₂ in B at ≥ 6 hops; label 0 adds a₁–a₂, b₁–b₂, label 1 adds
a₁–b₁, a₂–b₂ (the 1-cycle-vs-2-cycle idea). Identical degree changes, every new
cycle ≥ 7 edges, failed draws redrawn before the label is applied. Audit at
n = 32/48/64/96/128: stats→y 0.526/0.493/0.512/0.495/0.507, stats+4hop ≈ 0.25,
k-hop exact = 0 for k ≤ 4 (≤ 8 from n=96); diameter (connected) 8.6 → 12.5 → 16.7,
~log n. Not constructible below n≈32 at dmin 6 — Q3 uses n ∈ {32, 64, 128}.

## 2026-10-01

### Thesis re-scoped: the effect of width

Title settled with the supervisor 2026-09-24: **Analyzing the Effect of Network Width
on Graph Transformers**. Thesis-A planning docs removed; questions in
`docs/WIDTH-QUESTIONS.md`, literature in `docs/WIDTH-LITERATURE.md`. "Graph
transformer" is used broadly (Müller et al. 2023 taxonomy): we test tokens-only and
Graphormer-style SPD bias. Width m = embedding dim; head dim fixed at 8 so H = m/8.

### Q1: a fixed learning rate confounds width

Connectivity matrix on `hard`, n=24, depth 2, 2000 train graphs, LR picked on a
new validation split (the old runner selected epochs on test). Widths 8–256 × LR
{3e-4 … 1e-2} × 3 seeds. With LR fixed at 1e-3 the width curve zig-zags (0.917,
0.917, 0.781, 0.912, 0.820, 0.596); tuned per width it is flat at ~0.97 through m=32.
Optimal LR falls with width (1e-2 at m≤16 → 1e-3 at m=256), and wide models diverge
at high LR without warm-up (m=256 @ 1e-2: train 0.41). Even tuned, m=128/256 sit at
0.844/0.678. (`results/width/q1.jsonl`)

### Q1b: warm-up fixes the instability, not the wide-model drop

Same grid + 5% linear warm-up and cosine decay, widths 2/4 and LR 3e-2 added (120 runs,
Kaggle T4 x2, 6 shards). Wide models now fit: m=128/256 reach train 1.000 at LR 1e-2,
yet test stays 0.763/0.644 — a **generalization gap at 2000 graphs, not optimisation**.
Width is an inverted U: m=2 stays at the all-connected predictor (0.50), m=4
underfits (train 0.84), m=8–64 at 0.97–0.98 (warm-up lifted m=64 from 0.948 to 0.983),
m≥128 memorizes. Critical width m* = 8 at 0.95, 4 at 0.90 — far below n=24.
Under the schedule 1e-3 is too low at every width; the fixed-LR curve collapses to 0.48.
(`results/width/q1b.shard*of6.jsonl`)

**Decisions:** every later sweep tunes LR per width under warm-up + cosine; LR grid
{1e-3, 3e-3, 1e-2, 3e-2} covers every Q1b tuned optimum (one m=256 seed chose 1e-3). Next: width × training-set size (does
the upper drop move right with more data?), then Q3 (m* vs n).

### Q1c: the wide-model drop is memorization — data moves the ceiling

Width {8…256} × train size {500, 2000, 8000} × LR {1e-3…3e-2} × 3 seeds, fixed 4800
optimizer steps (216 runs, Kaggle, 8 shards). Tuned test:

| m | 500 | 2000 | 8000 |
|---|---|---|---|
| 8 | 0.629 | 0.975 | 0.988 |
| 16 | 0.708 | 0.967 | 0.998 |
| 32 | 0.398 | 0.982 | 0.993 |
| 64 | 0.423 | 0.983 | 0.998 |
| 128 | 0.460 | 0.763 | 0.998 |
| 256 | 0.444 | 0.644 | 0.991 |

Train acc is ~1.00 everywhere. At 8000 graphs every m≥16 reaches ≥0.99 — the m≥128
drop is gone at equal steps; at 500 everything memorizes and m≥32 falls below the
all-connected predictor. Width therefore has a **floor** (m≈4–8: can't fit) and a
**data-set ceiling** (extra width memorizes). The LR confound is a small-data effect:
at 8000 the fixed 1e-3 is within 0.01 of tuned for m≥16 and optimal LRs drift down
to ~1e-3. The 2000 arm reproduces Q1b to the third decimal (deterministic pipeline,
separate Kaggle run). (`results/width/q1c.shard*of8.jsonl`)

### Caveat: `hard` leaks the component layout through node indices

`make_connectedness_hard_dataset` never relabels nodes: the blobs are always
`[0, na)` and `[na, n)` (200/200 graphs at n=24 have index-contiguous components),
and diameters are small (mean 4.2, max 8). With adjacency-row tokens the split point
is readable and connectivity reduces to "does any row have ones on both sides" — a
near-local bridge check (cf. GIN-degree 0.975, 06-25). Q1/Q1b/Q1c conclusions hold
for this task (LR protocol, data ceiling) but not as evidence about reachability.
**Decision:** Q3 onward uses randomly relabelled nodes and sparse, diameter-spread
blobs (`hard_diam` generator), so pairs need multi-hop paths.

### Infrastructure: sharded sweeps on Kaggle

`width_sweep.py` — resumable sweeps, one JSON line per run, `--shard i/k` with one file
per shard, `--analyze` for width × LR tables, fixed-vs-tuned and m* at 0.90/0.95/0.99.
`kaggle/width_sweep.ipynb` clones the repo and runs k shards across the GPUs; needs a
phone-verified account (GPU and Internet are gated on it). Running locally starved
the laptop; the tiny models gain from parallel shards, not from GPU speed per run.

---

## 2026-07-08

### bfs_check closes the density gap: fail -> diagnose -> supervise -> 0.9972

The cleanest causal chain in the project. connectedness_hard, identical 32k
graphs, model, and optimizer as the failed bfs_expand run (20260707_142843,
decoded 0.5342 flat for 200 epochs); the trace target is the only substantive
difference (batch 32 vs 64 — an OOM concession at max_seq_len 704 — is the
one confound, noted). Result: loss collapses by epoch ~15 (not even
grokking-shaped — three times earlier than hard_diam's transition), decoded
0.933 at epoch 20, best 0.9975 at epoch 60; run killed at 88 as saturated.
Full held-out test split on best weights: **decoded 0.9972, trace_em 0.9044,
parse_fail 0.0002, flat 0.985-1.000 across diameters 2-8**. The dense task
that resisted the previous rung now matches converged hard_diam (0.9975)
exactly. Supervision density is the likely accelerant: check traces grade
~2 tokens per edge-visit where expand left rejections silent.

Diagnostic before/after (the thesis figure): levels 2-3 tf acc 0.272/0.242
(expand) -> 0.999/0.998 (check); the previously silent visited-set membership
tests are now the model's most reliable op — check NO(seen) 1.000 over
n=342,755 positions. Nothing below 0.996 remains. Fourth confirmed instance
of the silent-op law, first constructive one on a task that had already
failed.

OOD decomposition (ER, all 2000, class-imbalanced — always-NO baseline is
0.755): trace_em 0.188 — the best trace transfer yet (0.12 for blob-trained
expand; nearly 1 in 5 ER graphs gets a token-perfect execution on an unseen
distribution) — but answer_acc 0.389, well below the marginal predictor.
Execution transfers; the verdict read-off is calibrated to blob trace shapes
(mean 6.0 levels on ER vs ~4-5 on blobs) and misfires toward YES. The
standing OOD problem is thereby LOCALIZED: not "the model", the answer-
readout circuit. (By-diameter decay on ER is the class-composition artifact
again — do not read it as a diameter effect.)

### hard_diam replicated and converged: 0.9975 / trace_em 0.973

Rerun of the headline config to 200 epochs (20260708_153126, seed 42): grok
at epoch 40-60 (same window), decoded 0.9975, trace_em 0.973 at 200 — the
converged figure the 07-05 entry owed. Seed-robustness runs (seed 43/44 +
one dataset-seed variant + one extra ansonly) remain queued for error bars.

### iso_wl stalls like hard did — and the diagnostic acquits the suspects

The wl_expand run (20260708_000424, 32k pairs): tf ~1.0 / loss 0.05 but
decoded flat ~0.52, trace_em creep stalling at ~0.003 by epoch 160; killed.
WL-aware diag buckets (validated by count identities) ACQUIT both predicted
craters — c_new mint 0.989-0.998, hist(first=min) 0.999, sorting clearly
forms — and convict the neighbour-colour gather: nbr r2 0.888 / r3 0.909
(r1's 1.000 is a decoy — round-1 colours are all 0). The stuck op is a
TWO-HOP retrieval: neighbour ids from the prompt edge list, then each
neighbour's colour from its previous-round record ~150 tokens back. Same law,
finer grain: the trace made WL's multiset HASH local, but left the multiset
GATHERING silent. Fix queued: wl_gather — emit `v c(v)` pairs in neighbour-id
order before the sorted list, so every hop is a single supervised lookup
(each belongs to a circuit measured at ~1.0). First case where the
diagnostic rejected the designer's hypothesis — the instrument localizes,
not confirms.

## 2026-07-07

### Density breaks bfs_expand: the visited-set subtraction is the silent op

The winning recipe (bfs_expand, depth 2, zero reg, 32k graphs) does NOT
transfer to connectedness_hard — the dense extra_p=0.3 blobs (~110 edges,
mean degree ~9). 200 epochs, no grok: decoded flat at ~0.52, trace_em ~0.02
and *declining* late, loss grinding linearly (0.58 -> 0.22) with none of the
collapse that heralded the hard_diam transition. tf_test ~1.0 throughout —
the model reads answers off gold traces fine; free decoding derails. By
diameter: 0.78-0.80 at diam 2-3, decaying to ~0.2 at 5-7. parse_fail ~0 —
the failure is computation, not format. (Run 20260707_142843.)

diag_cot_levels on the best checkpoint localizes it exactly: parent-copy
0.986 and level-1 children 0.998 (induction and pure lookup+sort both fine —
emitting a sorted 9-child list is NOT the problem), but level-2 at 0.272 and
level-3 at 0.242. The wall is specifically the visited-set subtraction: at
level 1 it is "minus {start}"; at levels 2-3 in a dense blob nearly every
neighbor is already visited, and each rejection is a membership test against
the whole trace so far that emits NO token — the hardest op in the dense
regime is unsupervised (Bachmann & Nagarajan again, one level down). Same
sub-skill that stalled at ~0.6 on 8k caterpillars (07-05); there 4x data
fixed it, here visited sets are ~2x larger and 32k does not.

**Lesson: trace locality is not binary — it is parameterized by branching
factor.** bfs_expand made the frontier-min local but left the set-minus
implicit; density is the knob that exposes it. Candidate next rung:
`bfs_check` — expand each parent as an explicit per-neighbor scan with an
emit-or-reject mark, so every membership test is a supervised binary token
(~2x longer traces again; the same move that fixed bfs_levels). Data (64k)
is the fallback lever, but the scaling trend (0.6 @ 8k sparse -> 0.27 @ 32k
dense) says density outpaces data. A second constructive instance of the
globality barrier for the writeup either way.

### Isomorphism chapter opened: iso_wl dataset + wl_expand trace (Q1)

QUESTIONS.md Q1 infrastructure built and smoke-tested. Dataset `iso_wl`
(ladder rung b): degree-matched pairs at fixed n and m — negatives are
degree-preserving double-edge swaps of G1 accepted only if 1-WL separates
the pair within wl_rounds, so labels are provably sound and neither degree
histograms nor sequence length leak (measured: mean len 419.7 vs 420.6).
trace_format `wl_expand`: per round per node `EXP u c_old [sorted neighbor
colors] c_new`, canonical colors shared across the pair, fixed round count
regardless of label, explicit final-histogram comparison before ANS. The
difficulty knob (divergence round) rides the diam metadata channel, so
by-diameter tooling reads as accuracy-by-WL-round. Caveat: negatives
concentrate at divergence round 2 (~95%) — random sparse graphs refine
fast; regular-graph negatives are the widening lever if the round-3 bucket
is too thin. Overfit-32 sanity: decoded 1.0, trace_em 0.94. (Gotcha for
future overfit checks: batch 64 > overfit set = 1 update/epoch — drop
batch_size alongside.) Real run + answer-only control queued.

### Engineering: KV cache, per-epoch result flush, kill-safe runs

generate() now keeps per-layer KV caches (token-identical outputs verified
against the naive loop on the trained hard_diam checkpoint; 3.5x on MPS at
max_new=120, more at longer traces). RunLogger flushes the results JSON
every epoch (atomic replace), and SIGTERM/Ctrl-C now stops the epoch loop
gracefully: best-so-far weights restored, checkpoint + JSON saved,
interrupted_at_epoch recorded, slow tail evals skipped. Killed runs no
longer lose everything — the 07-07 hard run was only allowed to burn its
full 200 epochs because killing it would have.

## 2026-07-06

### The bisect: weight decay alone blocks the circuit; dropout is benign

The regularization finding (07-04) resolved into a sharper claim. On the bfs_l1
lookup probe, 30 epochs each, one knob at a time: **dropout 0.1 alone** —
trace_em 0.93 by epoch 10, 0.98 by 20 (circuit forms, marginally slower and a
hair below the 0.999 clean ceiling); **weight decay 0.01 alone** — trace_em
0.06/0.09/0.11, the original crawling plateau reproduced exactly. Weight decay
is the suppressor; dropout was a bystander.

Mechanism now has published support at the same scale: Kobayashi et al. 2024
show weight decay induces low-rank attention in 2-layer transformers on
associative recall — and a retrieval circuit needs high-rank, precise
key–query alignment. Varma et al.'s circuit-efficiency account supplies the
selection story (under a norm penalty, degenerate marginal statistics are
cheaper than the retrieval circuit). The opposite-sign "LMs Grok to Copy"
(NAACL 2025, wd helps copying at billion scale) is now contrasted on both
regime and mechanism. Nuance for the writeup: our knob is torch Adam's
L2-in-gradient, not decoupled AdamW decay — an AdamW-at-same-λ probe is a
cheap optional follow-up. (Runs 20260706_133219 (dropout-only),
20260706_134019 (wd-only).)

### Novelty assessment (deep-research pass)

Literature search across all six core claims: no exact precedent for any;
per-claim verdicts, closest neighbors, and writeup consequences recorded in
reports/executing-graph-algorithms.md §7. Cleanest novel contribution: the
trace-locality result (constructive instance of Abbe et al.'s globality
barrier). Closest overall prior art: Ye/Fu/Jia/Sharan (ICML 2026) — the paper
this repo has built on since June; answer-only, no traces (verify scale from
the paper before asserting differences in print).

## 2026-07-05

### The adversarial dataset falls too: hard_diam at 0.9925

Same recipe on connectedness_hard_diam (two blobs, single-edge class
difference, matched degrees/edge counts — no statistical shortcut exists):
grokking transition at epoch 40-50, **decoded 0.9925, trace exact-match 0.90,
flat 0.988-1.000 across diameters 3-13** including the 5-10 zone where both
classes coexist. The matched answer-only control (same model/data/optimizer,
`max_trace_len: 0`) sits at **0.5108 for all 100 epochs** — loss pinned at the
marginal predictor, tf_test never leaves 0.50. Trace 0.9925 vs answer-only
0.51 on one dataset, trace the only difference. (The answer-only by-diameter
"decay" is a class-composition artifact of a NO-biased chance model — the
honest statement is flat 0.51 overall, not degradation. An encoder baseline
on hard_diam remains optional garnish; the matched-architecture control is
the one that matters.)

Second signal: the ER OOD probe improved to 0.54 / trace_em 0.12 (vs 0.32 /
0.0075 for the caterpillar-trained model) — denser, more heterogeneous
training blobs transfer better. OOD generalization looks like a
training-diversity problem, not a mechanism limit. Future work.

### AR-CoT works: depth-2, diameter-flat connectivity via grokking at 32k graphs

The run everything built toward: bfs_expand, depth 2, zero regularization,
**32k graphs** (4x data — the anti-memorization lever after depth-4/8k drove
train loss to literally 0.0000 while decode *worsened*). Textbook grokking:
trace_em flat at ~0.03 through epoch 50, phase transition at 60–70, and by
epoch 100 **decoded answer accuracy 0.964, trace exact-match 0.80, still
climbing**. The by-diameter curve is the thesis figure: **0.92–0.99 flat from
diameter 2 to 18 at depth 2** — the diam≥11 buckets that scored 0.25 the night
before (the model faithfully reading its own derailed trace as "disconnected")
included. Sequential decoding substitutes for depth (Merrill & Sabharwal),
measured.

The sub-skill ladder that got here, each rung its own experiment: copy
(induction) and level-1 lookup form at depth 2 once regularization is zero
(both 0.99 test-TF); lookup-minus-visited (levels 2+) stalls at ~0.6 with 8k
graphs; depth 4 at 8k memorizes instead of fixing it (capacity substitutes for
generalization — trace_em *declines* as train loss → 0); 4x data at depth 2
tips the balance to circuits. **Lesson: circuits vs memorization is an
economics problem — permuted-id data makes memorization expensive; capacity
makes it cheap; add data before depth.**

**The contrast (headline table):** the answer-only baseline at identical scale
(same depth-2 model, same 32k graphs, same zero-reg optimizer, `max_trace_len:
0`) sits at **0.51 — flat chance at every diameter for all 100 epochs**. It
never learns at all, echoing 06-25's supervision-density finding (1 bit/graph
gives gradient descent nothing to grip). Trace 0.964 vs answer-only 0.510 with
the trace as the only difference. And the diagnostic on the grokked model shows
the sub-skill ladder complete: copy 0.998, level-1 lookup 1.000, the previously
stuck set-minus levels 0.985–0.996 — nothing partially formed remains.

Caveats: not converged at epoch 100 (rerun longer for the final figure), and
ER OOD still fails (answer 0.32, trace_em 0.01) — the full pipeline does not
yet transfer to dense graphs the way the isolated lookup did (trace_em 0.55
OOD). Checkpoint-eval on connectedness_hard makes the same point harder:
answer 0.29 (below chance = systematic misreading of its own derailed traces,
not noise), trace_em 0.0002, and 42% of decodes never form a well-formed
`ANS` — the trace machinery is calibrated to caterpillar statistics and fails
structurally on 90-edge blobs. In-distribution algorithm execution:
demonstrated. Distribution-general algorithm: open. Remaining grid: depth-1 @ 32k (theory says the trace can't
rescue one layer — induction needs two), length-OOD with cot_pos: none, the
wd-vs-dropout bisect, connectedness_hard_diam headline run.

## 2026-07-04

### Regularization was suppressing circuit formation (the CoT unblocking)

The bfs_l1 probe (atomic lookup: emit the sorted neighbours of node 0, nothing
else) plateaued at trace_em ~0.15 — barely 3x guess rate — at depth 2 with full
data. Sweeping the knob ladder found the culprit immediately: with
**weight_decay 0 + dropout 0** the same probe hits **trace_em 0.95 by epoch 10,
0.999 by 20** — the circuit forms ~10x faster and completely, with no other
change. Mechanism: Adam's L2-in-gradient decay pulls constantly on the precise,
low-redundancy weights a retrieval circuit needs (tied embeddings on a 42-token
vocab have nowhere to hide), and attention dropout injects noise into exactly
the sharp attention patterns being formed; a ~430k-param model has no redundancy
to absorb either. This retroactively explains every AR-CoT plateau of 07-03:
format statistics (SEP/ANS/EOS) survive regularization, content circuits don't.
`cot_ar.yaml` now ships zero regularization for the AR-CoT family. Which of the
two regularizers dominates is not yet isolated (the first bisect attempt
accidentally ran with both off after the config default changed mid-session).
**Lesson: regularizers tuned for statistical fitting can be *the* blocker for
algorithmic circuit formation at small scale — sweep them to zero before
touching the representation.**

**First OOD-positive result of the project:** the zero-reg lookup circuit,
trained only on sparse caterpillars, scores trace_em 0.55 on dense ER graphs
(vs ~0 for every distribution-bound model so far) — evidence of an algorithmic
circuit rather than distribution memorization.

Next: rerun bfs_expand depth-2 with zero regularization (the run everything has
been building toward), then the depth x trace grid.

## 2026-07-03

Session centered on chain-of-thought: why the scratchpad CoT never worked, replacing
it with genuine autoregressive CoT, and the two traps that surfaced immediately — a
dataset leak only a token model can see, and the next-token-prediction pitfall.

### Scratchpad CoT post-mortem: it was never chain-of-thought

The `cot_len` scratchpad (K learnable tokens spliced into the encoder sequence)
never beat the no-CoT baseline, and the reasons are structural, not tuning:

- **No supervision on the scratchpad** — only the task token gets a loss, so nothing
  teaches slot `c_k` to hold the k-hop frontier (that reading was aspiration, not
  objective).
- **Input-independent slots** — the same K learned vectors for every graph: extra
  compute positions (pause/register tokens), not content-carrying steps.
- **Bypassable** — the task token attends to all graph tokens directly; the
  optimizer can (and did) ignore the scratchpad entirely.
- **Inert at depth 1** — in an encoder, `c_2` sees only `c_1`'s *initial* (constant)
  value unless there is one layer per hop. The causal chain the mask draws needs
  depth ≥ chain length, which defeats the "sequential steps orthogonal to depth"
  purpose.

**Lesson: CoT gains come from supervised (or generated-and-conditioned-on)
intermediate steps; unsupervised extra positions in a single forward pass are just
width.** Replaced with `cot_mode: autoregressive`: the graph serialized to discrete
tokens (`N nodes.. E edges.. TRACE`), a canonical BFS trace as the supervised
target, a decoder-only transformer (`src/cot.py`) teacher-forced on trace + answer,
greedy decoding at eval. Answer-only ablation = identical architecture with
`max_trace_len: 0`. Scratchpad kept as `cot_mode: scratchpad` for comparison.
Overfit-32 sanity: decoded accuracy and trace exact-match reach 1.0. (Two
implementation notes: tied embeddings need std-0.02 init — the default N(0,1) blows
tied logits up to initial loss ~12.5 and stalls training; and node ids must be
randomly permuted per graph, see next.)

### Edge count leaked the label in diameter_controlled — via sequence length

First real AR-CoT run: decoded accuracy 1.0 by epoch 5, trace exact-match 0.0,
ER OOD probe at 0.37. Diagnosis: connected caterpillars have exactly n-1 edges, the
two disconnected ones n-2 — **edge count WAS the label**. Invisible to every
GNN/encoder baseline (they don't see edge count as a feature), but the token model's
prompt is 2 tokens longer for connected graphs, so the learned positional embedding
of `TRACE` reads the answer off sequence length. Fixed by padding every graph to
n-1+k edges (k ~ U{1..3}, label-independent) with **diameter-safe chords** (leaf →
backbone neighbour of its own attachment; property-tested over 500 caterpillars that
the exact diameter survives). Pre-fix caches/results are not comparable. **Lesson
(rhymes with 2026-06-19's degree-0 shortcut): every model family gets its own leak
audit — a representation change (graph → token sequence) creates observables that
did not exist before, sequence length first among them.**

### The next-token pitfall: format learns, computation doesn't

Post-fix runs (depth 1 and 2, full 8k data, 20k steps): teacher-forced answer
accuracy 1.0 (the model reads YES/NO off the *gold* trace — the trace does carry the
answer), decoded accuracy at chance, trace loss pinned at ~1.87. A per-position
teacher-forced diagnostic localized the failure: **level-1 accuracy 5.6%** — the
model cannot retrieve "the tokens paired with node 0 in the edge list" even with the
whole gold prefix given — while everything statistical (SEP placement, trace-end,
answer-given-trace, EOS) sits at 0.88–1.0. Inspected decodes confirm it: perfectly
*shaped* traces (ascending levels, plausible sizes, short-before-NO) with garbage
membership. This is Bachmann & Nagarajan 2024's pitfall: teacher forcing fits every
easy conditional and gives the one hard computation no partial credit — the compact
`bfs_levels` target makes it worse because each level's first token is the *minimum
of the whole frontier set* (a global computation, all-or-nothing gradient).

Mitigation: `trace_format: bfs_expand` — verbose `EXP parent children` rounds, every
next token locally computable (parents = copy of the previous level in order, i.e.
an induction head; children = lookup keyed by the adjacent parent token). ~2x longer
traces. A short under-scaled probe (520 steps) showed even parent-copy at 10% —
induction heads form after thousands of steps, so the decisive full-data depth-2 run
is in flight as of this entry. **Lesson: a CoT trace is a curriculum, not a log —
design each next-token to be computable from local context, or the gradient never
finds the algorithm.**

### connectedness_hard_diam: the bridge task gets a diameter axis

`connectedness_hard`'s dense blobs (extra_p=0.3) pin every diameter at ~2-3 —
nothing to stratify. New generator keeps the adversarial single-edge difference
(bridge vs intra-blob chord; edge counts and degrees matched exactly) but builds
blobs as sparse Hamiltonian cycles + 0-3 chords, spreading diameters over ~3-13.
Caveat for figures: diam ≤ 4 is all-disconnected, ≥ 11 all-connected (a
through-bridge path outruns either blob) — class-vs-class comparison lives in the
5-10 overlap zone; `diameter_controlled` remains the instrument for exact sweeps.

### Housekeeping

The `bfs_expand` vocab token (`EXP`) grew the CoT vocabulary 41 → 42: AR-CoT
checkpoints saved before it no longer load (results JSONs unaffected). Everything
through today committed in logical groups (harness rework, config restructure, BDH,
AR-CoT, leak fix, hard_diam, results).

## 2026-07-02

### BDH (Dragon Hatchling) joins the connectivity lineup

Graph-native BDH (Kosowski et al. 2025; reading notes in
`reports/kosowski-2025-notes.typ`): shared-parameter rounds, positive-sparse neuron
space, linear/Hebbian attention; `bdh_adj_mask` toggles all-pairs vs edge-restricted
attention. On the connectivity-matrix task over diameter_controlled, all three
families land in the same band in-distribution — global_attn 0.99-1.0 EM,
gat 0.84-0.95, bdh 0.91-0.95 — and **all collapse on the ER OOD probe**
(EM 0.13-0.24). The 06-25 conclusion (distribution-learning, not
algorithm-learning) survives an architecture swap; BDH's inductive bias does not by
itself buy OOD reachability.

### Infrastructure: one YAML = one run, provenance in every result

- Config schema now owns experiment identity (`dataset`, `dataset_kwargs`, `seed`,
  `train_frac`, explicit `model` selector) and supports `extends:` inheritance; the
  config suite collapsed onto `_graph_base` / `_connectivity_base` with redundant
  variants folded into `-o` overrides (`configs/README.md` is the catalogue).
- RunLogger records provenance (git commit + dirty flag, argv, library versions) and
  parameter counts in every results JSON — a result is now traceable to the code
  that produced it.
- Runner: `-o key=value` overrides, `--limit`/`--overfit` slices, best-epoch
  checkpoints, `--eval`/`--inspect` on saved checkpoints, and an automatic ER OOD
  probe attached to every connectivity run.

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
