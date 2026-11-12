# Experiment Changelog

Record of findings, bugs, and decisions made during experiments.

---

## 2026-11-12

### Baseline at depth 12 (head_dim 8): m* ~ n^1.9, same as depth 6; the edge-bias gap is the encoding, not depth

Colab T4, commit bd32857, `q3lr_d12h8`, 24 runs (complete). The adjacency-rows baseline (q3lr
protocol) at the same depth as the edge-bias curve, head_dim 8 (head_dim 2 failed at every width,
2026-11-09). Final test pair, seeds 0 / 1:

| n | widths: test pair (seed 0 / 1) | m* depth 12 | m* depth 6 (q3lr) | edge bias, depth 12 |
|---|---|---|---|---|
| 32 | 8: 0.743 / 0.755, 16: 0.903 / 0.894, 24: 0.976 / 0.972 | 21.1 | 22.5 | 4.6 |
| 40 | 16: 0.833 / 0.772, 24: 0.904 / 0.930, 32: 0.946 / 0.967 | 28.1 / > 32 | 34.1 | 4.5 |
| 48 | 24: 0.826 / 0.832, 32: 0.916 / 0.877, 48: 0.967 / 0.966 | 43.2 | 44.8 | 4.5 |
| 56 | 32: 0.820 / 0.823, 48: 0.912 / 0.886, 64: 0.954 / 0.950 | 63.1 | 60.0 | 5.6 |

`--scaling`, test pair: slope 1.95 pooled (fit 1.88), against 1.74 at depth 6. Doubling the depth
saves ~6 % of width at n = 32 and nothing at n = 56. Accuracy rises smoothly with width (no
all-or-nothing runs as with edge bias).

At one fixed depth the two encodings now differ by 5x (n = 32) to 11x (n = 56) in m*, and only the
baseline grows with n. This closes the open caveat of the 2026-11-10 report (edge bias at depth 12
compared with a baseline at depth 6).

Operations: T4 data generation is too slow (2 cores; an n = 32 set takes ~4 h), so the datasets
were built locally and uploaded in 50 MB chunks. Shards now keep the train set on the GPU only
(bd32857); a CPU copy per worker nearly filled the T4's 12.7 GB RAM. One VM was lost before
training (11-11) and one container was reset by a broad `pkill -f`. ~11 units used in all;
balance 6.11. (`results/width/q3lr_d12h8.shard*of4.jsonl`)

## 2026-11-10

### n = 128 at depth 12, head_dim 2: m* ~ 6-8.5, so edge-bias m* stays flat from n = 32 to 128

Kaggle T4 x2, commit 1175bbc, `ebd12hd2_128`, 16 runs. Final test pair (max over training in
brackets where higher):

| m | seed 0 | 1 | 2 | 3 | >= 0.95 |
|---|---|---|---|---|---|
| 6 | 0.971 | 0.944 | 0.959 (0.966) | 0.990 | 3/4 |
| 8 | 0.889 | 0.943 | 0.922 | 0.994 (0.998) | 1/4 |
| 12 | 0.990 | 0.996 | 0.999 | 0.987 | 4/4 |
| 16 | 0.993 (0.996) | 0.980 (0.990) | 0.997 | 0.996 (0.997) | 4/4 |

`--scaling`, test pair: m* <= 6 / 8.5 / <= 6 / <= 6 (seed mean 6.6). Seed 0 is not monotone (passes
at 6, fails at 8). Every run at m >= 12 is >= 0.98 and none falls back late, unlike head_dim 8
at the same depth (ebd12_128: 0/4 at m = 12, m* ~ 14.5).

Edge bias at depth 12, head_dim 2, across n: m* 4-6 (n = 32-56), ~7 (72 / 96), ~6-8.5 (128).
No growth with n over a 4x range of n.

This corrects the 2026-11-08 reading that head_dim does not change m* (based on n = 72 / 96,
where both were <= 8): at n = 128, head_dim 2 (3-8 heads) gives m* ~ 7 against ~14.5 for head_dim 8
(1-2 heads). The number of heads matters once n is large. The thesis has to state head_dim as a
fixed setting and show this comparison.

## 2026-11-09

### One depth (12, head_dim 2): edge bias m* ~ 4-6 at n = 32-56; adjacency-rows baseline fails at head_dim 2

Colab, commit 1175bbc. Depth is now a fixed setting (12) for every m* curve; head_dim 2 so that
m < 8 still has several heads.

`ebd12hd2_s` (T4, 24 runs, complete). Edge bias, final test pair, seeds 0 / 1:

| n | m = 2 | m = 4 | m = 6 |
|---|---|---|---|
| 32 | 0.745 / 0.746 | 0.989 / 0.746 | 1.000 / 1.000 |
| 40 | 0.750 / 0.750 | 1.000 / 0.750 | 1.000 / 0.999 |
| 48 | 0.755 / 0.755 | 0.755 / 0.993 | 0.997 / 0.998 |
| 56 | 0.759 / 0.759 | 0.759 / 0.759 | 0.993 / 0.990 |

Runs either solve the task or stay at the trivial pair accuracy (~0.75, the share of pairs that
are connected), nothing in between. m = 6 passes for every n and seed; m = 4 passes for 1 of
2 seeds at n = 32-48 and none at n = 56; m = 2 never. So m* sits between 4 and 6 at n = 32-56,
the same as at depth 6 (eb_adj_hd2, 4-6) and close to ~7 at n = 72 / 96 (ebd12hd2).

`q3lr_d12` (L4, 13 of 24 runs; the VM was lost at 21:43). Adjacency rows without edge bias,
same depth and head_dim:

| n | m | depth 12, head_dim 2 | depth 6, head_dim 8 (q3lr) |
|---|---|---|---|
| 32 | 16 | 0.825 / 0.820 | 0.893 / 0.848 / 0.861 |
| 32 | 24 | 0.911 | 0.983 / 0.960 / 0.956 |
| 40 | 32 | 0.849 | 0.955 / 0.962 / 0.913 |
| 48 | 48 | 0.831 | 0.963 / 0.959 / 0.946 |
| 56 | 48 | 0.792 | 0.905 / 0.933 / 0.901 |

No run reaches 0.95; every cell is below the depth-6 head_dim 8 result. Likely head_dim 2: without
edge bias, attention has to find neighbours by query-key dot products, and 2-dimensional heads
are too small for that, while with edge bias the bias does the routing. The baseline therefore
cannot share head_dim 2; it needs its own head_dim (8, as before) at depth 12.

Colab balance 47.71 -> 17.41 (the L4 alone ~2.3 units / h).

## 2026-11-08

### n = 128: depth 16 does not bring m* down to ~7; runs stay erratic

Kaggle T4 x2, commit 7c49f5c, 18 runs. `ebd16_128` (depth 16, seeds 0-3) and `ebd12_128x`
(depth 12, seeds 2-3, joining ebd12_128 seeds 0-1); eb_adj protocol, head_dim 8. Final test pair
(max over training in brackets where > 0.01 higher):

| depth | m | seed 0 | 1 | 2 | 3 | >= 0.95 |
|---|---|---|---|---|---|---|
| 12 | 8 | 0.869 | 0.946 | 0.958 | 0.876 | 1/4 |
| 12 | 12 | 0.868 | 0.921 | 0.927 | 0.908 | 0/4 |
| 12 | 16 | 0.976 | 0.966 | 0.965 | 0.964 | 4/4 |
| 16 | 8 | 0.840 | 0.869 | 0.994 | 0.848 | 1/4 |
| 16 | 12 | 0.972 | 0.771 | 0.831 | 0.983 | 2/4 |
| 16 | 16 | 0.999 | 0.999 | 0.869 (0.938) | 0.901 (0.985) | 2/4 |

m* per seed: depth 12 14.9 / 14.4 / <= 8 / 14.9; depth 16 11.2 / 15.0 / <= 8 / 10.9.

- 16 layers help a little at m = 12 (2/4 against 0/4) but not at m = 16, where two depth-16 runs
  fell back from 0.94-0.985. m* at n = 128 is ~11-15 at both depths, not the ~7 of n = 72 / 96
  (ebd12hd2). Depth 12 is not obviously short at n = 128.
- Depth 12 at m = 16 is the one reliable cell (4/4, all 0.964-0.976, just above the line).
- Deeper models are less stable: late collapses (n = 96 m = 16 depth 12 seed 0, two n = 128
  depth 16 runs) now cost as much as the plateau did. Gradient clipping or a schedule fix is
  the obvious next control before reading m* at n = 128 as real width growth.

## 2026-11-07

### Edge bias at depth 12, head_dim 2: m* ~ 7 at n = 72 and 96

Colab high-RAM T4, commit fd2d069, ~3 h, ~5.4 compute units. `ebd12hd2_72` / `_96`: eb_adj
protocol, depth 12, head_dim 2, m = 4 / 6 / 8 (2-4 heads), seeds 0 / 1. Final test pair (max over
training in brackets where higher):

| n | seed | m = 4 | 6 | 8 | m* |
|---|---|---|---|---|---|
| 72 | 0 | 0.764 | 0.893 | 0.999 | 7.0 |
| 72 | 1 | 0.764 | 0.905 | 1.000 | 6.9 |
| 96 | 0 | 0.915 (0.978) | 0.996 | 0.975 (0.992) | 4.8 |
| 96 | 1 | 0.768 | 0.862 | 0.994 (0.999) | 7.3 |

With depth no longer the limit, m* is ~ 7 at n = 72 and 96 against 4.1-5.9 at n = 32-56 (depth 6,
same head_dim): close to flat, a slight rise at most. m = 4 stays at always-yes at n = 72 (one
2-wide head pair is too narrow) but n = 96 seed 0 reaches 0.978 there before falling back.

### Kaggle: n = 128 ablation and head_dim 2 at n = 72 / 96, both say depth again

Kaggle T4 x2, commit c9566a5 (after the prepare hang fix), 48 runs. `ebx128_*`: n = 128, m = 24,
seeds 2-5, one change each (as `ebx96_*`). Final test pair:

| condition | seed 2 | 3 | 4 | 5 | >= 0.95 |
|---|---|---|---|---|---|
| base | 0.877 | 0.902 | 0.914 | 0.873 | 0/4 |
| logn | 0.771 | 0.837 | 0.830 | 0.935 | 0/4 |
| ssmax | 0.913 | 0.907 | 0.909 | 0.911 | 0/4 |
| clip | 0.959 | 0.953 | 0.945 | 0.908 | 2/4 |
| wsd | 0.828 | 0.930 | 0.948 | 0.901 | 0/4 |
| all | 0.817 | 0.918 | 0.879 | 0.958 | 1/4 |
| d12 | 0.884 | 0.979 | 0.970 | 0.991 | 3/4 |
| loop2 | **0.990** | **0.988** | **0.980** | **0.998** | **4/4** |

At n = 128 the base never escapes (0/4, n = 96: 1/4); bias and schedule changes do not help,
clipping half helps, 12 layers or the 6 blocks run twice do. Looping (no new parameters) is the
most reliable, as in the surgery.

`eb_adj_hd2_72` / `_96` (head_dim 2, depth 6), final test pair:

| n | seed | m = 4 | 6 | 8 | 12 | 16 | m* |
|---|---|---|---|---|---|---|---|
| 72 | 0 | 0.916 | 0.892 | 0.933 | 0.952 | - | 11.5 |
| 72 | 1 | 0.900 | 0.928 | 0.949 | 0.946 | - | > 12 |
| 96 | 0 | - | 0.856 | 0.926 | 0.926 | 0.932 | > 16 |
| 96 | 1 | - | 0.849 | 0.865 | 0.911 | 0.959 | 15.2 |

More heads do not keep m* at ~6 past n = 56: at depth 6 head_dim 2 gives the same rise as head_dim
8 (13 / 15 at n = 72, 28 / 16 at n = 96). The rise is the depth limit, not the head count.

### Edge bias at depth 12, n = 72-128: m* drops to <= 8 / ~8 / ~14.5, but runs stay erratic

Colab high-RAM T4, commits 176c6c5 (24 runs) and 9b4364f (6 fill-in runs), ~7.5 h,
~9.8 compute units. Sweeps `ebd12_72/96/128`: eb_adj protocol (adjacency tokens, head_dim 8,
bias_init 2), depth 12, seeds 0 / 1. Final test pair:

| n | seed | m = 8 | 12 | 16 | 24 | m* (depth 12) | m* (depth 6, eb_adj_big) |
|---|---|---|---|---|---|---|---|
| 72 | 0 | 0.990 | 0.960 | 0.997 | 0.999 | <= 8 | 13.0 |
| 72 | 1 | 0.995 | 0.995 | 0.986 | 0.994 | <= 8 | 14.9 |
| 96 | 0 | 0.949 | 0.959 | 0.926 | 0.993 | 8.3 | 28.0 |
| 96 | 1 | 0.988 | 0.815 | 0.990 | 0.992 | <= 8 | 15.7 |
| 128 | 0 | 0.869 | 0.868 | 0.976 | 0.941 | 14.9 | > 32 |
| 128 | 1 | 0.946 | 0.921 | 0.966 | 0.994 | 14.4 | 22.8 |

- Depth was the limit: at the same widths, 12 layers pass where 6 stalled; m* at n = 128 falls
  from 23 / > 32 to ~14.5.
- Still not flat: n = 128 needs m ~ 15 where n = 72 / 96 pass at 8 (maybe 12 layers are again
  short at n = 128, or head_dim 8 at m = 8 / 12 means one head).
- Erratic in width: n = 96 seed 1 passes at 8 and 16 but gives 0.815 at 12; n = 96 m = 16 seed 0
  reached 0.987 at epoch 14 and ended at 0.926; n = 128 seed 0 is worse at 24 than at 16. First
  crossings under-describe this; pass rates over more seeds would be the better readout.
- Not deterministic on GPU: n = 72 m = 8 seed 1 ran twice (shard bug below), 0.995 and 0.938.
  The jsonl keeps the first; the .pt on disk is the second run's.

Bug: shards sliced the *pending* runs, so a shard started after another had saved a run took a
shifted slice (xargs queue: 3 runs twice, 6 never). Fixed in 9b4364f (slice the full list, then
skip finished runs); duplicates dropped from `ebd12_72.shard1of2.jsonl`.

## 2026-11-05

### Edge-bias plateau, answer from n = 96 and checkpoint surgery: depth, not bias or schedule

Colab high-RAM T4, commit af20a01, ~6.5 h, ~10.5 compute units.

Checkpoint surgery (`eb_surgery.py`, 6000 steps from the stalled eb_adj_big checkpoints, fresh
Adam; `results/width/eb_surgery.shard*of2.jsonl`), final test pair (start in brackets for the
treatments that change the model):

| checkpoint (start) | ctrl | lr | bias +1.5 | deep +6 blocks | loop x2 |
|---|---|---|---|---|---|
| n = 128 m = 16 s1 (0.855) | 0.857 | 0.857 | 0.855 (0.693) | 0.980 | **0.994** (0.703) |
| n = 128 m = 32 s0 (0.930) | 0.940 | 0.946 | 0.942 (0.858) | 0.983 | **1.000** (0.859) |
| n = 128 m = 32 s1 (0.929) | 0.932 | 0.951 | 0.932 (0.825) | 0.995 | **1.000** (0.906) |
| n = 96 m = 16 s0 (0.925) | 0.930 | 0.931 | 0.933 (0.835) | 0.995 | **0.998** (0.773) |

Ablation `ebx96_*` (n = 96, m = 16, seeds 2-5, full q3lr budget), final test pair:

| condition | seed 2 | 3 | 4 | 5 | >= 0.95 |
|---|---|---|---|---|---|
| base | 0.905 | 0.918 | 0.929 | 0.963 | 1/4 |
| logn (b0 = 3.18) | 0.909 | 0.923 | 0.926 | 0.947 | 0/4 |
| ssmax | 0.857 | 0.934 | 0.965 | 0.923 | 1/4 |
| clip 1.0 | 0.965 | 0.937 | 0.953 | 0.971 | 3/4 |
| wsd | 0.958 | 0.950 | 0.966 | 0.969 | 3/4 |
| logn + clip + wsd | 0.855 | 0.976 | 0.964 | 0.970 | 3/4 |
| depth 12 | 0.994 | 0.993 | 0.999 | 0.993 | **4/4** |
| loop x2 (6 blocks, tied) | 0.999 | 0.988 | 0.998 | 0.945 | 3/4 |

- The stall is a depth limit. A stalled checkpoint run through its own 6 blocks twice, with no
  new parameters, reaches 0.994-1.000 within 6000 steps; adding 6 identity blocks reaches
  0.980-0.995. More training, a re-warmed LR and a larger edge bias leave it on the plateau.
- From scratch, depth 12 at m = 16 passes in 4 / 4 seeds at ~0.99+, loop x2 in 3 / 4. The
  dilution fixes (logn, ssmax) do nothing; clip and WSD help a little (3 / 4, all near 0.95).
- So the eb_adj m* rise past n ~ 56 at depth 6 (8 -> 14 -> 16-28 -> > 32) measures depth, not
  width: the width study has to fix depth relative to n (log n, or a depth grid) for this
  encoding. That fits the diagnostics (stalled models worse at every hop distance, no hard
  12-hop wall: the depth-6 model propagates too weakly, not too short).
- ebx128 (Kaggle, queued) is the same ablation at n = 128.

### Edge bias below m = 8 (head_dim 2): adjacency m* is flat at ~4-6 for n = 32-56

`eb_adj_hd2` / `eb_rid_hd2` (q3lr protocol, head_dim 2 so m = 2-8 has 1-4 heads, seeds 0 / 1),
Kaggle 2x T4, commit 5778050, 64 runs. Models (git-ignored) in `results/width/models/eb_*_hd2/`.
Final test pair, seed 0 / 1:

| n | adj m=2 | adj m=4 | adj m=6 | adj m=8 | rid m=4 | rid m=6 | rid m=8 |
|---|---|---|---|---|---|---|---|
| 32 | 0.745 / 0.746 | 0.745 / 0.746 | 0.991 / 1.000 | 1.000 / 1.000 | 0.858 / 0.994 | 0.995 / 0.998 | 0.993 / 0.992 |
| 40 | 0.750 / 0.750 | 0.750 / 0.750 | 0.966 / 0.993 | 0.994 / 0.995 | 0.859 / 0.838 | 0.986 / 0.865 | 0.931 / 0.975 |
| 48 | 0.755 / 0.755 | 0.977 / 0.919 | 0.985 / 0.985 | 0.988 / 0.991 | 0.824 / 0.767 | 0.950 / 0.935 | 0.874 / 0.947 |
| 56 | 0.759 / 0.759 | 0.941 / 0.759 | 0.979 / 0.884 | 0.971 / 0.975 | 0.804 / 0.828 | 0.853 / 0.928 | 0.945 / 0.923 |

- adj: m* 5.6 / 5.7 / 4.1 / 5.9 at n = 32 / 40 / 48 / 56, slope ~0. m = 2 (one 2-wide head)
  stays at always-yes for every n; the step from failing to ~0.98 is between m = 4 and 6.
  m = 8 matches head_dim 8 (n = 56: 0.971 / 0.975 against 0.974 / 0.867).
- rid: m* 4.6 / 7.8 / > 8 / > 8, rising and non-monotone in m (n = 40: 0.986 at m = 6,
  0.931 at m = 8), the same instability as rid elsewhere.
- The n = 72 / 96 m* (13-16) were head_dim 8, i.e. 1-2 heads at m = 8-16. Queued
  `eb_adj_hd2_72` / `eb_adj_hd2_96` (head_dim 2, m = 4-12 / 6-16) on Kaggle ahead of ebx128
  to see whether m* stays flat once the head count is not the limit.

### Edge-bias plateau: checkpoint diagnostics, and the ablation / surgery runs set up

Why do eb_adj_big runs at n >= 96 stay on a ~0.9 pair-accuracy plateau? A research report
(Claude Research, 2026-11-05) proposed H1 bias dilution (fixed b * (A + I) puts less softmax
mass on neighbours as n grows), H3 cosine schedule running out, H5 6 one-hop layers reaching
~12 hops. Checked first, without a GPU:

- Distances in the data (`swap`, 400 graphs per n): connected pairs median 6 / 7 / 7 hops and
  90th percentile 10 / 11 / 12 at n = 72 / 96 / 128; pairs beyond 12 hops 0.6 / 3.4 / 9.0 % of
  connected-graph pairs. There is no jump between n = 72 and 96, so a 12-hop wall alone cannot
  explain the plateau.
- `eb_diagnose.py` on all 22 saved eb_adj_big models (`results/width/eb_diag/*.json`):
  - b is learned up from 2 to ~3.3-4 in layers 1-2 at every n, but that puts 31-49 % of the
    attention on N(i) + i at n = 128 against 55-63 % at n = 72: dilution is real and the
    model does not raise b far enough to undo it.
  - Middle layers often drop b to ~0 or below (content or global attention), in stalled and
    escaped runs alike.
  - Pair error rises with hop distance in every model (n = 128 m = 24 s1, escaped: 0.4 / 1.5 /
    6 / 11 % at 1-4 / 5-8 / 9-12 / 13+ hops); stalled models are 2-3x worse at every distance,
    not only beyond 12. The clearest marker of a stalled run is the error on disconnected pairs
    (12-32 % against 6 %), with final-layer component separation ~0.7 against ~0.9-1.0.

Set up (commit below):
- `width_sweep.py`: EdgeBiasModel `bias_logn` (b starts at bias_init + log((n - 3.5) / 28.5),
  3.18 at n = 96, 3.47 at n = 128), `bias_scale="logn"` (bias times log n / log 32), `loops`
  (weight-tied repeat of the blocks); training `clip` (grad-norm) and `schedule="wsd"` (warmup,
  constant, linear decay over the last `decay_frac`).
- `ebx{96,128}_{base,logn,ssmax,clip,wsd,all,d12,loop2}`: one change each against base, seeds
  2-5, n = 96 m = 16 and n = 128 m = 24. Escape rate per condition is the readout. ebx128 on
  Kaggle (notebook: queue with at most 4 workers, 2 per T4), ebx96 on Colab.
- `eb_surgery.py`: resume the stalled n = 128 m = 16 s1, m = 32 s0 / s1 and n = 96 m = 16 s0
  checkpoints for 6000 steps with ctrl (LR 1e-4) / bias (+1.5) / lr (re-warm to 1e-3) / deep
  (+6 identity blocks) / loop (blocks twice). Colab.

### Edge bias at n = 96 / 128: past n ~ 96 the limit is escaping a plateau, not width

`eb_adj_big` widened (commit eef41f1), Colab high-RAM T4, two launches. The first VM (4 workers)
was lost after 4 of 10 runs when the laptop dropped off the network; two of its workers had also
hit CUDA OOM loading n = 128 data. The remaining 6 ran on a second VM with 2 workers (commit
1d1e440, ~1.5 h; ~4 compute units in all). Models for all n = 96 / 128 runs except the first 4 in
`results/width/models/eb_adj_big/` (git-ignored). Final test pair (train), seed 0 / 1; `*` marks
runs whose best validation epoch is 1, i.e. graph accuracy never left its starting value:

| m | n = 96 | n = 128 |
|---|---|---|
| 16 | 0.925 (0.918)\* / 0.959 (0.963) | 0.890 (0.891)\* / 0.855 (0.857)\* |
| 24 | 0.930 (0.928) / 0.948 (0.960) | 0.916 (0.915)\* / 0.964 (0.965) |
| 32 | 0.968 (0.972) / 0.968 (0.973) | 0.930 (0.932) / 0.929 (0.929)\* |

- m* (test pair >= 0.95): n = 96 28.0 / 15.7; n = 128 > 32 / 22.8. Accuracy is not monotone in
  m: n = 128 seed 1 passes at m = 24 and fails at m = 32, n = 96 seed 1 passes at 16 and misses at
  24. Runs either escape the 0.85-0.93 plateau or stay on it, and which one happens decides m*.
- Train equals test throughout, so it is not data. A width threshold at n >= 96 needs either
  more seeds per cell or a fix for the plateau (longer training, bias start, LR) first.
- Slope over n = 32-96: 1.12 pooled, 0.95 seed mean, still capped by the m = 8 floor at n <= 48.

### Edge bias at n = 72 / 96: m* grows again, slowly (14 at n = 72, ~16+ at n = 96)

`eb_adj_big` (adjacency rows, bias start 2, head_dim 8, q3lr protocol, 128k graphs), Colab
high-RAM T4, commit 5778050, ~1.6 h, 2.6 compute units. Models (git-ignored) in
`results/width/models/eb_adj_big/`. Final test pair accuracy (train), seed 0 / 1:

| m | n = 72 | n = 96 |
|---|---|---|
| 4 | 0.780 (0.781) / 0.801 (0.798) | 0.808 (0.798) / 0.788 (0.784) |
| 8 | 0.896 (0.901) / 0.909 (0.917) | 0.831 (0.830) / 0.799 (0.803) |
| 12 | 0.935 (0.943) / 0.851 (0.867) | 0.827 (0.818) / 0.828 (0.833) |
| 16 | 0.987 (0.987) / 0.983 (0.987) | 0.925 (0.918) / 0.959 (0.963) |

- m* (test pair >= 0.95): n = 72 13.0 / 14.9; n = 96 > 16 / 15.7. With n = 32-56 (<= 8 up to
  n = 48, <= 8 / 10.9 at n = 56) the width does grow with n once n passes ~50, but at
  n = 72 it is 14 against a train-fit 81 for plain adjacency rows.
- Train equals test at both sizes: the 128k graphs that limit plain adjacency rows at n = 72
  are enough here. The limit is width (or optimisation), not data.
- The script's slope over n = 32-72 (0.66-0.73) is still floor-capped; the sub-8 Kaggle sweep
  (head_dim 2) is needed for a real exponent. n = 96 needs m = 24 / 32 to pin seed 0.

### Edge-bias full sweep: m* at or below 8 up to n = 48, against 45 for adjacency rows

`eb_adj` (adjacency rows, bias start 2) and `eb_rid` (64-wide random IDs, bias start 4),
q3lr protocol, n = 32-56, m = 8 / 12 / 16 / 24, seeds 0 / 1, Kaggle (64 runs, 8 workers,
~20 min each; commit 7b73b36; `results/width/eb_{adj,rid}.shard*of4.jsonl`). Final test pair
accuracy, seed 0 / 1:

| n | eb_adj m = 8 | 12 | 16 | 24 | eb_rid m = 8 | 12 | 16 | 24 |
|---|---|---|---|---|---|---|---|---|
| 32 | .996 / .992 | 1.00 / .996 | 1.00 / 1.00 | 1.00 / .999 | .990 / .995 | .994 / .992 | .995 / .995 | .995 / .996 |
| 40 | .984 / .990 | .993 / .992 | .977 / .998 | .997 / 1.00 | .964 / .972 | .964 / .978 | .986 / .984 | .998 / .989 |
| 48 | .961 / .983 | .979 / .987 | .995 / .998 | .994 / .998 | .928 / .882 | .936 / .938 | .956 / .993 | .929 / .991 |
| 56 | .974 / .867 | .956 / .975 | .996 / .986 | .998 / .994 | .814 / .905 | .899 / .865 | .971 / .985 | .957 / .946 |

Critical width (test pair >= 0.95), seeds 0 / 1; "<= 8" = already passed at the smallest width:

| n | adjacency rows (q3lr) | eb_adj | eb_rid |
|---|---|---|---|
| 32 | 22.4 | <= 8 / <= 8 | <= 8 / <= 8 |
| 40 | 33.1 | <= 8 / <= 8 | <= 8 / <= 8 |
| 48 | 44.7 | <= 8 / <= 8 | 14.7 / 12.8 |
| 56 | 59.6 | <= 8 / 10.9 | 14.7 / 14.7 |
| 72 | train fit 81, test none | – | > 24 / > 24 (eb_rid72) |

- Edge bias on adjacency rows removes almost all of the width cost: one head of width 8
  solves n = 48, and n = 56 needs at most ~11, against 45 / 60 without the bias. The script's
  slopes (0.25-0.40) are meaningless here: the grid floor of 8 caps most crossings.
- Random IDs (no edges in the input at all, read-in 64 wide for every n) need <= 8 up to
  n = 40 and ~14 at n = 48 / 56. Four-size slope 1.25, but two sizes sit on the floor.
- Wider is not reliably better for eb_rid: m = 24 is below m = 16 at n = 48 seed 0 (0.929
  vs 0.956) and at n = 56 (0.957 / 0.946 vs 0.971 / 0.985). Train equals test in every run
  (no overfitting), so this is optimisation, the same failure as the m = 24 collapse in
  eb_rid72.
- Next: widths below 8 need head_dim < 8 (m = 4 is zero heads now); n = 72 for eb_adj;
  a lower LR or bias start for eb_rid at m >= 24 and at n = 72.

---

## 2026-11-04

### Auxiliary k-hop loss over four sizes: exponent 1.74 -> 1.30

`aux_dbl` extended to n = 32 and 48 on Kaggle (16 runs, 8 workers; same shard files, new
logs `aux_dbl.b.shard*.log`). Critical width (test pair >= 0.95, final epoch), seeds 0 / 1:

| n | adjacency rows (q3lr, seeds 0 / 1) | + aux k-hop loss | saving |
|---|---|---|---|
| 32 | 21.9 (20.7 / 23.1) | 21.5 (20.2 / 22.7) | 2 % |
| 40 | 30.1 (30.8 / 29.4) | 26.8 (25.8 / 27.8) | 11 % |
| 48 | 42.4 (41.2 / 43.6) | 33.5 (32.0 / 35.0) | 21 % |
| 56 | 58.0 (59.2 / 56.7) | 45.1 (44.2 / 46.0) | 22 % |
| growth | n^1.74 (seed combinations 1.59–1.90) | n^1.30 (1.13–1.45) | |

- The loss changes the exponent, not just the constant: no saving at n = 32, a fifth of
  the width from n = 48 on. The two bootstrap ranges do not overlap. Fit thresholds
  (train pair) give the same picture: 1.63 -> 1.20.
- The 2-hop rows still lead (n^1.27 at a far lower level, 12.7 / 18.5 at n = 40 / 56), but
  the aux loss reaches a similar exponent with the plain adjacency input and an
  unchanged test-time model. Organising the computation matters for how width grows with
  n; the extra input information mainly lowers the level.

### n = 72 follow-ups (Colab high-RAM T4): more data closes the gap; random-ID edge bias unstable

Commit 018ddb2, ~1.75 h, 2.8 compute units (`results/width/q3n72data*`, `eb_rid72*`).

`q3n72data`, adjacency rows, m = 96, 256k graphs (24k steps, so 12 epochs instead of 24),
final pair accuracy test (train), seeds 0 / 1:

| training graphs | seed 0 | seed 1 |
|---|---|---|
| 128k (q3n72) | 0.947 (0.970) | 0.912 (0.973) |
| 256k | **0.956** (0.974) | **0.933** (0.955) |

- Test rises by 0.01 / 0.02 and the train - test gap shrinks to ~0.02, the n <= 56 level.
  The q3n72 gap was the data, as suspected. Seed 0 now passes 0.95 at m = 96; seed 1 is
  still short, with train itself at 0.955 (half the epochs), so 256k graphs probably need
  more steps too. A generalize m* at n = 72 needs 256k graphs and m = 96 / 128 / 192 runs.

`eb_rid72`, random IDs + edge bias (start 4), q3lr protocol (128k graphs), test (train):

| m | seed 0 | seed 1 |
|---|---|---|
| 12 | 0.793 (0.787) | 0.846 (0.846) |
| 16 | 0.910 (0.912) | 0.945 (0.943) |
| 24 | 0.764 (0.764) | 0.919 (0.932) |

- No run reaches 0.95, and train = test: the limit is optimization, not data.
- m = 24 seed 0 climbed to 0.861 and then collapsed to always-yes for the rest of training
  (from ~epoch 9); wider is not reliably better here. LR 1e-3 with a bias that starts at 4
  may be too aggressive at n = 72. Compare with the Kaggle `eb_rid` sweep (n = 32-56)
  before drawing a width trend.

### Edge-bias pilot round 2: a nonzero start solves n = 32 at m = 16 with either token

Kaggle (`ebp_*` sweeps, 16 runs, ~7–10 min each; `results/width/ebp_*.jsonl`). Same budget as
round 1: n = 32, first 16k graphs of the cached seed-0 set, 3000 steps (12 000 for `_long`),
LR 1e-3, seed 0. Final test pair accuracy (always-yes 0.745):

| tokens, bias start | steps | m = 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| adjacency rows, 0 (round 1 again) | 3000 | 0.828 | 0.865 | 0.931 | – |
| adjacency rows, 2 | 3000 | 0.871 | **0.999** | 0.997 | – |
| random IDs, 0 | 12 000 | – | – | 0.935 | 0.947 |
| random IDs, 2 | 3000 | – | 0.962 | 0.989 | 0.991 |
| random IDs, 2 | 12 000 | – | – | 0.991 | 0.985 |
| random IDs, 4 | 3000 | – | **0.998** | 0.988 | 0.993 |

- Round 1 reproduces exactly (0.828 / 0.865 / 0.931), so the setup is unchanged.
- The zero start was the whole problem for random IDs: at 4x the steps they leave
  always-yes too (0.935 / 0.947), and a start of 2 or 4 learns within the first ~400 steps.
- With a start of 2–4 both token types reach >= 0.99 at m = 16 on an eighth of the data and
  an eighth of the steps of the q3lr baseline, whose m* at n = 32 is 22.4. With random IDs
  the input carries no edges at all, and the read-in is 64 wide for every n.
- m = 8 (one head) is too narrow for adjacency rows (0.871); random IDs were not tried there.
- Next: the full sweep, q3lr protocol, n = 32–56 (and 72, where adjacency rows hit the
  data limit), adjacency rows with start 2 and random IDs with start 4, widths from 8 up.

### n = 72 (Colab T4): training fit stays on the line, generalization breaks

`q3n72` (adjacency rows, q3lr protocol, seeds 0 / 1, m = 64 / 96 / 128) and `q3n72steps`
(m = 96 at 48k steps); the first Colab session ended early (laptop slept), the rerun finished
the rest (`results/width/q3n72*.jsonl`, ~8.1 compute units in all). Final pair accuracy, test
(train), seeds 0 / 1:

| m | 24k steps | 48k steps |
|---|---|---|
| 64 | 0.867 (0.916) / 0.898 (0.925) | – |
| 96 | 0.947 (0.970) / 0.912 (0.973) | 0.960 (0.990) / 0.914 (0.991) |
| 128 | 0.941 (0.978) / 0.936 (0.986) | – |

- Fit threshold (train pair >= 0.95): m* = 80.9 (82.7 / 79.1), where the n^1.63 line from
  n = 32–56 puts it (82.6); pooled fit slope over five sizes 1.65.
- Generalize threshold: no seed reaches test 0.95 at any width at 24k steps; m = 128 is no
  better than m = 96. Doubling the steps takes train to 0.99 but test only to 0.960 / 0.914.
  The gap (train - test 0.04–0.08, against <= 0.02 at n = 56) means the 128k training graphs,
  not the width, limit n = 72: more width or steps fits the training set further without
  carrying over. The generalize m* at n = 72 is undefined under this protocol.
- Every run sits at always-yes (0.764) for the first ~8 of 24 epochs, longer than at n = 56.
- n = 96 (`q3n96`) is not worth running as is: with the same 128k graphs it would hit the
  same wall harder. It needs a larger training set first (e.g. 256k at n = 72 to test
  whether the gap closes).

---

## 2026-11-03

### Edges as attention bias: pilot (n = 32, short budget)

`EdgeBiasModel`: every attention layer adds b[layer, head] · (A + I)_ij to the scores (one
learned scalar per layer and head, start `bias_init`); tokens are adjacency rows ("adj") or
a fixed 64-wide random ID per node position ("rid"). Pilot on a Colab T4: n = 32, the first
16 000 graphs of the cached seed-0 set, 3000 steps, LR 1e-3, one seed. Final test pair
accuracy (always-yes 0.745):

| variant | m = 8 | 16 | 32 | 64 |
|---|---|---|---|---|
| adjacency rows (baseline) | – | 0.747 | 0.764 | – |
| adjacency rows + edge bias, start 0 | 0.828 | 0.865 | 0.931 | – |
| random IDs + edge bias, start 0 | – | 0.745 | 0.745 | 0.745 |

- With adjacency rows the bias speeds learning a lot: the baseline barely leaves always-yes
  at this budget, the biased model reaches 0.93 at m = 32.
- Random IDs did not learn at any width. Likely cause: with a zero start the model sees no
  structure at all. A second round (start 2 and 4 at m = 32 / 64; start 0 at 4x steps;
  adjacency rows with start 2) was stopped before any run finished.
- Next: rerun that second round, then the full sweep (n = 32–56, 2 seeds, q3lr protocol).
- Local pilots at n = 32 are slow to set up: generating swap graphs at n = 32 redraws most
  graphs (dmin = 6 rarely fits small blobs). Use the prefix of the cached 128k set instead.

### Auxiliary k-hop loss: width falls by 11–22 %, far short of the 2-hop rows

`aux_dbl` on Kaggle (22 runs, 8 workers, ~2 h; `results/width/aux_dbl.shard*of8.jsonl`).
Adjacency-row input and the baseline model; during training block L also predicts the
2^L-hop reachability (L = 1, 2, 3 → 2, 4, 8 hops) through a bilinear head, weight 0.3 each.
Heads dropped at test time. Critical width (test pair ≥ 0.95, final epoch, seeds 0 / 1):

| input / training | n = 40 | n = 56 | growth (2 points) |
|---|---|---|---|
| adjacency rows (q3lr, seeds 0 / 1 only) | 30.1 | 57.9 | n^1.94 |
| adjacency rows + aux k-hop loss | 26.8 (25.8 / 27.8) | 45.1 (44.2 / 46.0) | n^1.55 |
| adjacency + 2-hop rows (enc_khop2) | 12.7 | 18.5 | n^1.11 |

- Guiding the layers to compute hops in order lowers m* by 11 % at n = 40 and 22 % at
  n = 56, with the whole pair-accuracy curve shifted: at n = 56, m = 48 reaches 0.967
  against 0.913 without the loss. The saving is a quarter of what the 2-hop rows give,
  so most of the khop effect comes from the information in the input, not from the
  organisation of the computation. The slope over the two sizes (1.55 vs 1.94 for the
  same two baseline points) hints at a smaller exponent, but two sizes cannot fix it.
- The hierarchy is only partly learned. Final pair accuracy of the heads at the widest
  models (m = 48 at n = 40, m = 96 at n = 56): 2-hop 0.98 / 1.00, 4-hop 0.91 / 0.90,
  8-hop 0.97 / 0.91; majority-class rates for the targets are ≈ 0.87 / 0.65 / 0.70
  (2 / 4 / 8 hops), so every head is above trivial, but the 4-hop head stays at ≈ 0.9
  even where connectivity itself is at 0.995. Narrow models learn the heads worse
  (n = 56, m = 32: 0.94 / 0.80 / 0.73) and the main task not at all (0.86).
- Open: the head weight (0.3) and the layer→hop assignment (`aux_lin`: L = 2, 4 → 2, 4
  hops) were not varied; a heavier weight or heads on every layer could move m* further.
  The aux heads cost nothing at test time, so this is a pure training-time lever.

---

## 2026-11-02

### Third seed for the main result and the 2-hop rows: exponents unchanged

Seed 2 ran on Kaggle (28 runs, `q3lr` 12 + `enc_khop2` 16, at the widths that bracket each
threshold; `*.shard{0-3}of4.jsonl`, logs tagged `seed2`). Critical width (test pair ≥ 0.95,
final epoch, mean of per-seed crossings), seeds 0 / 1 / 2:

| input | n = 32 | 40 | 48 | 56 | 72 | growth (90 % seed bootstrap) |
|---|---|---|---|---|---|---|
| adjacency rows | 22.4 | 33.1 | 44.7 | 59.6 | – | n^1.74 (1.53–1.97, 81 combinations) |
| + 2-hop rows | 9.5 | 12.7 | 16.9 | 18.5 | 27.1 | n^1.27 (1.10–1.51, 243 combinations) |

- Both exponents are the same as with 2 seeds (1.74, 1.27). The bootstrap intervals widen a
  little (were 1.59–1.90 and 1.05–1.51) because seed 2 is the most spread: adjacency n = 40
  crosses at m = 39.1 against 30.8 / 29.4 for seeds 0 / 1; all other per-seed crossings
  lie within ±3 of the earlier pair. The two intervals still do not overlap, and the 2-hop
  interval stays above 1.
- Per-seed crossings (generalize): adjacency 20.7 / 23.1 / 23.4, 30.8 / 29.4 / 39.1,
  41.2 / 43.6 / 49.4, 59.2 / 56.7 / 63.1; 2-hop 10.4 / 8.0 / 10.1, 12.5 / 13.5 / 12.0,
  16.7 / 16.7 / 17.4, 19.7 / 16.8 / 19.0, 25.6 / 27.7 / 28.1.
- The bootstrap (one seed per n, every combination, 5th–95th percentile of the log-log slope)
  is not in `width_sweep.py`; computed by hand from the `--scaling` per-seed values.
- Thesis Ch. 3 and the encoding chapter still quote the 2-seed interval 1.59–1.90; to update.

---

## 2026-10-21

### Probing results: connectivity builds up gradually; nodes do not store their reachable sets

- Kaggle trained the 16 `probe` models (pair acc matches q3lr: m* ≈ 32 at n = 40, ≈ 64
  at n = 56), but the user's notebook copy lacked the probe cell, so `probe_layers.py`
  ran locally on the CPU (3 threads, ~2 h). Results: `results/width/probe/*.json`.
- `probe_layers.py --summary` now gives, per k, the first layer whose AUC reaches 0.95.
  The old "largest k decodable, scanning up from k = 1" read 0 exactly where connectivity
  became perfect, because trained models lose 1-hop detail in late layers.
- Pair probe, k = ∞ (connected), seed mean, layers 0 → 6:
  n = 40, m = 48: 0.62 0.76 0.83 0.91 0.97 0.99 1.00; m = 16: 0.57 … 0.81.
  n = 56, m = 96: 0.58 0.67 0.73 0.79 0.87 0.95 1.00; m = 32: 0.56 … 0.84.
  Untrained twins stay at 0.51–0.62. The signal rises steadily over all layers, with no
  single jump; narrow models rise more slowly and end lower, not stalling at one layer.
- Exact distances are not kept: pair-probe AUC for k = 3 rises at layer 1 (≈ 0.86–0.92)
  and stays flat; k = 1 falls from ≈ 1.00 to 0.85–0.95 by the last layer. The model does
  not build a k-hop hierarchy (no doubling of the reach per layer, nor +1 hop) — it builds
  a "same component" representation.
- Set probe: from one node's vector, its connected set is decodable at only AUC 0.60–0.89
  after the last layer (best 0.89 at n = 40, m = 48), and 1-hop falls below the untrained
  twin. Under linear read-out, nodes do not store their reachable sets; the answer comes
  from comparing two nodes' vectors (pair probe 0.99–1.00 for models above m*).
- Narrow read-in loses neighbour detail: at n = 40, m = 16 even layer 0 decodes 1-hop at
  only 0.91 (m < n).

## 2026-10-20

### Probing set up: what does each layer know about paths?

- `width_sweep.py`: sweeps can set `save_models` to save each run's final weights to
  `results/width/models/<name>/` (gitignored). New sweep `probe`: q3lr at n = 40
  (m = 16, 24, 32, 48) and n = 56 (m = 32, 48, 64, 96), 2 seeds — widths bracketing each
  threshold (30.1 / 57.9), so failing and succeeding models are both probed.
- `probe_layers.py`: residual stream after the read-in and every block (layer-normed),
  on fresh swap graphs (500 train / 300 test, seeds 9001 / 9002). Two linear probes per
  layer and k ∈ {1, 2, 3, 4, 6, 8, ∞}, scored by held-out ROC-AUC:
  - pair: bilinear h_iᵀ W h_j (the model's own read-out form) for d(i, j) ≤ k;
  - set: linear h_i → Rⁿ for node i's whole k-hop set (does the node store its set?).
  Every probe is also fitted on the untrained twin (same init seed) as the baseline.
- Local check on a model that failed to learn (n = 40, m = 48, 12k graphs, pair 0.68):
  layer 0 decodes k = 1–2 perfectly by the pair probe in both trained and untrained
  networks (a_i · a_j = 2-hop counts — the probe's own power), the set probe recovers
  k = 1 (m ≥ n), and beyond that trained = untrained (AUC 0.70 at k = 3, 0.62 at ∞). So
  the baseline is necessary and the probe adds no structure of its own.
- MPS is ~18× slower than the CPU for these small fits (26 s vs 1.5 s); the script uses
  CUDA or the CPU. On Kaggle it runs in the notebook after training (new cell, bounded
  by the deadline). Notebook: `SWEEPS = ["probe"]`, `SHARDS = 8`.


### khop4 complete: the growth exponent falls with every free hop

`enc_khop4` rerun on Kaggle (14 runs, n = 48 / 72, `SHARDS = 4`, ~1 h, no errors). Critical
width (test pair ≥ 0.95, final epoch, mean of per-seed crossings), with all encodings:

| input | n = 32 | 40 | 48 | 56 | 72 | growth (90 % seed bootstrap) |
|---|---|---|---|---|---|---|
| adjacency rows | 21.9 | 30.1 | 42.3 | 57.9 | – | n^1.74 (1.59–1.90) |
| + RWSE, j ≤ 8 | 23.3 | 31.9 | 45.2 | 57.9 | – | n^1.65 (1.46–1.84) |
| + 2-hop rows | 9.2 | 13.0 | 16.7 | 18.3 | 26.7 | n^1.27 (1.05–1.51) |
| + 4-hop rows | 3.5 (seed 0) | 3.6 | 3.7 | 4.5 | 6.5 | n^1.03 (0.79–1.25) |

- 0 / 2 / 4 free hops: exponent 1.74 → 1.27 → 1.03. Handing over hops cuts both the
  level and the growth; only with 4 of the ~10–13 hops given is the growth about linear.
- The 4-hop widths (3–7) sit at the one-head floor on a coarse grid (2, 4, 6, 8 …);
  the n^1.03 is the least firm of the four.
- `reports/figures/width-encodings.png` now plots every size where both seeds cross, with
  least-squares slopes; `reports/width-progress.typ` § Encodings — precomputed walks,
  summary, timeline and conclusions updated (2-hop rows no longer called linear).


### More sizes: 2-hop rows grow as n^1.27, not linearly

The 2026-10-18 Kaggle session hit the 12 h limit before its zip cell, but Kaggle kept
the working directory; downloaded as `results.zip`, its `results/width/*.jsonl` are the
full records (the repo's files are exact prefixes of them). Copied in: 70 new runs for
q3lr / enc_rwse8 / enc_khop2, 11 for enc_khop4. (Before the output turned up, the same
70 runs had been parsed back from the notebook log's `--curves` output; they match the
real records to the log's 5e-4 rounding, all 70.)

Critical width (test pair ≥ 0.95, final epoch, mean of per-seed crossings; 90 % seed
bootstrap of the slope):

| input | n = 32 | 40 | 48 | 56 | 72 | growth |
|---|---|---|---|---|---|---|
| adjacency rows, LR 1e-3 | 21.9 | 30.1 | 42.3 | 57.9 | – | n^1.74 (1.59–1.89) |
| + RWSE, j ≤ 8 | 23.2 | 31.9 | 45.2 | 57.9 | – | n^1.65 (1.46–1.84) |
| + 2-hop rows | 9.2 | 13.0 | 16.7 | 18.3 | 26.6 | n^1.27 (1.04–1.50) |

- **2-hop rows are not linear.** The two-point n^1.00 of 2026-10-15 was the flat
  40 → 56 stretch; with five sizes the growth is n^1.27, and the bootstrap interval only
  just reaches 1. Still clearly below adjacency rows (no overlap of the intervals), and
  the gap widens with n: adjacency / 2-hop = 2.4, 2.3, 2.5, 3.2 at n = 32–56.
- **Adjacency rows at one rate: n^1.74.** With LR 1e-3 at all four sizes (before: 3e-3 at
  32 / 48) the slope is 1.74, down from 1.85 (mixed rates) and 1.94 (3e-3). n = 48 drops
  51.1 → 42.3 at the lower rate; n = 32 is unchanged (21.9).
- **RWSE (j ≤ 8) matches adjacency rows at every size** (within the seed spread).
- n = 72 for 2-hop rows: m* 26.6 (seeds 25.5 / 27.7); 1.46× n = 56 for a 1.29× larger n.

### Incident: 64 workers, out of memory, and an unbounded catch-up

- The Kaggle copy ran `SHARDS = 16` → 64 workers, each holding its training set in RAM
  (1.3 GB at n = 72). 38 were killed (-9) by 7.2 h.
- The catch-up cell then ran the remaining runs in one process with no deadline (the
  11 h stop lived only in the worker loop) until Kaggle killed the session at 12 h —
  before the zip cell. The working directory was kept, so nothing was lost, but the
  results had to be fetched from the whole output folder.
- Data build: 4.5 h, of which n = 32 alone took 4.3 h (two 128k sets, 4 CPUs).
- Fixes in `kaggle/width_sweep.ipynb`: one `DEADLINE` (10 h) bounds both the workers and
  the catch-up (`subprocess.run(timeout=...)`); the results are zipped every 30 min; a
  RAM note at `SHARDS` (keep ≤ ~8 workers).
- enc_khop4 is incomplete: seed 0 at n = 32 / 48, one run at n = 72. Rerun drops n = 32
  (threshold at the floor, m* ~3; the n = 32 data costs 4 h): 14 runs at n = 48 / 72,
  `SWEEPS = ["enc_khop4"]`, `SHARDS = 4`, ~1 h. `--scaling` now tolerates runs without
  `train_pair` (prints n/a).

## 2026-10-18

### Encoding comparison: more graph sizes (set up, Kaggle)

The encoding slopes of 2026-10-15 were fitted to two sizes (n = 40 / 56). The same sweeps
get more sizes in place (the runs already on disk are skipped), so `--scaling` fits
every line over one protocol (128k graphs, 24k steps, depth 6, LR 1e-3, 2 seeds):

| sweep | new n | runs | widths |
|---|---|---|---|
| `q3lr` (adjacency rows) | 32, 48 | 20 | 12–48 / 24–96 |
| `enc_rwse8` | 32, 48 | 20 | same as q3lr |
| `enc_khop2` | 32, 48, 72 | 30 | 4–16 / 8–32 / 12–48 |
| `enc_khop4` | 32, 48, 72 | 30 | 2–12 / 2–12 / 4–16 |

- n = 72 only for k-hop rows: their thresholds stay small (khop2 ~24 by the linear fit),
  while adjacency rows would need m ≈ 110 at ~45 min per run.
- swap at n = 72 audited first (`audit_width_data.py --gens swap:0.25:6 --ns 56 72`):
  stats → y 0.506, stats + 4-hop 0.242, k-hop exact-match 0 up to k = 6, 0.08 at k = 8,
  0.64 at k = 12; connected diameter 13.2 [11–18] (n = 56: 11.7). Clean, and the
  diameter keeps rising with n.
- Data build: n = 32 is the slow one (the generator rejects many graphs; ~70 min per
  128k set on one core locally), n = 48 / 72 a few minutes.
- `kaggle/width_sweep.ipynb`: SWEEPS = q3lr, enc_rwse8, enc_khop2, enc_khop4; 2 shards
  each. ~100 runs, ~18 GPU-hours of runs on 8 workers → ~3–4 h wall plus data build.

## 2026-10-15

### Walk encodings: free hops collapse the width; RWSE adds nothing

`enc_khop2`, `enc_khop4`, `enc_rwse8`, `enc_rwse16` on Google Colab (Kaggle quota out;
`colab/width_sweep_colab.ipynb`, 4 workers on 1 GPU, 72 runs, ~7–17 min each). q3lr
protocol, n = 40 / 56, 2 seeds. Critical width (test pair ≥ 0.95, final epoch, mean of
per-seed crossings; per seed in brackets):

| input | n = 40 | n = 56 | growth |
|---|---|---|---|
| adjacency rows (q3lr) | 30.1 (30.8 / 29.4) | 57.9 (59.2 / 56.7) | n^1.95 |
| + RWSE, j ≤ 8 | 31.9 (30.3 / 33.5) | 57.9 (55.3 / 60.6) | n^1.77 |
| + RWSE, j ≤ 16 (leaks) | 23.5 (23.3 / 23.6) | 48.2 (44.4 / 52.1) | n^2.14 |
| + 2-hop rows | 13.0 (12.5 / 13.5) | 18.3 (19.7 / 16.8) | n^1.00 |
| + 4-hop rows | 3.6 (3.7 / 3.5) | 4.5 (4.6 / 4.3) | n^0.63 |

- **Hops left to compute set the width.** Two free hops (remaining diameter ~6 instead
  of ~11) cut the width 2.3× at n = 40 and 3.2× at n = 56 and turn the growth from
  ~n² to linear — the rate Yehudai et al.'s construction predicts. Four free hops
  (~3 left) need width ≈ 4 with almost no growth. Since the swap diameter rises with n
  (8.7 → 11.8 over n = 32–56), much of Q3's ~n² may be path length rather than node
  count — the question Q7b would have asked, answered here at fixed graphs.
- **RWSE (j ≤ 8), the standard structural encoding, changes nothing**: same critical
  width as adjacency rows at both sizes.
- **The leaky RWSE-16 helps where it leaks**: −22 % at n = 40 (audit 0.90), −17 % at
  n = 56 (audit 0.72) — a leaked shortcut lowers the measured width, and less so as the
  leak fades with n.
- Fit ≈ generalize for every encoding (train ≈ test), as in q3lr at 128k graphs.
- Two sizes only: slopes are two-point. Widths below 16 use 1 head (head_dim 8).

`width_sweep.py --scaling`: crossings now use only the widths run at each n. Grids that
differ per n put a NaN between two real widths and blocked the interpolation, giving
"> max" (seen for rwse16 at n = 56: now 48.2). It could only produce NaN, never a wrong
value, so earlier critical widths are unchanged (rechecked q3lr, rwse8, khop2/4,
enc_edges96k). New figure `reports/figures/width-encodings.png`.

### enc_edges96k: with 4x the steps edge tokens learn, at 2-3x the width of adjacency rows

`enc_edges96k` on Kaggle (16 runs, 16 workers on 2 T4s). 11 runs hit the 420 min cap
(`truncated`): m = 96 got 48–54k of 96k steps, m = 48 / 64 at n = 40 84–90k; the
m = 32 runs and n = 32 m = 64 finished. Final test exact / pair, seed 0 · seed 1:

| m | n = 32 | n = 40 |
|---|---|---|
| 32 | 0.52 / 0.939 · 0.50 / 0.915 | 0.37 / 0.856 · 0.22 / 0.762 |
| 48 | 0.74 / 0.972 · 0.36 / 0.816 (T) | 0.14 / 0.757 · 0.41 / 0.856 (T) |
| 64 | **0.995 / 0.999 · 0.897 / 0.990** | 0.59 / 0.934 · 0.11 / 0.788 (T) |
| 96 | 0.835 / 0.979 · 0.960 / 0.996 (T, 54k) | 0.87 / 0.985 · 0.47 / 0.906 (T, 48k) |

Critical width (`--scaling`, test pair ≥ 0.95, final epoch, mean of per-seed crossings —
the measure behind q3lr's 21.9 / 30.2): n = 32 **48.2** (36.5 / 59.9); n = 40 seed 0
72.5, seed 1 > 96 (truncated at 48k steps, still rising) — so ≥ 72. That is 2.2× the
adjacency-row width at n = 32 and ≥ 2.4× at n = 40, with 4× the steps (adjacency rows
don't lower their m* with more steps, q3steps). No slope: n = 40 is a lower bound.
- **A width threshold exists given time.** At n = 32 both seeds of m = 64 generalize
  (exact 0.995 / 0.897); train ≈ test everywhere, no memorization.
- **Seeds split.** Many runs escape the first plateau and then stall on a partial
  solution for 50k+ steps (n = 32 m = 48 seed 1 at pair 0.82; n = 40 m = 64 seed 1 at
  0.79–0.81) while the other seed goes on to solve it — unlike adjacency rows, whose
  seeds agree within a few width units.
- Sanford's representation bound for this encoding is √N ≈ 9–10 (N ≈ 2.25n tokens);
  the width that learns is ≥ 5× that at n = 32.

## 2026-10-14

### enc_edges: edge tokens don't learn at the adjacency-row budget; resolves the B2 confound

`enc_edges` on Kaggle (40 runs, 24k steps, n = 32–56, m = 8–96, 2 seeds, all complete).
No n has a critical width: the only run to clear the trivial exact-match is n = 32, m = 48,
seed 1 (exact 0.76, pair 0.968, still rising at the end; seed 0: 0.10 / 0.875). Final
test pair accuracy, mean of 2 seeds (trivial ≈ 0.75):

| n | m = 16 | 32 | 48 | 64 | 96 | adjacency rows (q3lr) |
|---|---|---|---|---|---|---|
| 32 | 0.745 | 0.820 | 0.921 | — | — | m* 21.9 |
| 40 | 0.750 | 0.758 | 0.790 | 0.851 | — | m = 24: 0.919, m = 48: 0.985 |
| 48 | 0.755 | 0.754 | 0.757 | 0.782 | 0.804 | m* 51.1 |
| 56 | 0.759 | 0.759 | 0.759 | 0.759 | 0.775 | m = 32: 0.821, m = 96: 0.984 |

Same shape as B2: the loss sits at ~0.55 until 7–18k steps (adjacency rows: 3–8k),
escapes in width order, and is still falling at 24k; train ≈ test, so under-training,
not memorization. Exact-match dips *below* 0.50 mid-learning (most graphs get at least
one pair wrong once the model stops predicting the trivial matrix). For equal pair
accuracy edge tokens need roughly 3× the width of adjacency rows at n = 40, more at
n = 56. **B2 confound resolved:** with all n² labels per graph, edge tokens still fail
where adjacency rows succeed — the encoding itself is harder to learn, not only the
label count. Timing: 29–144 min per run, 10 jobs on 2 T4s; all 40 done in one session.

### enc_edges96k set up: does a critical width exist with 4x the steps?

`enc_edges96k`: 96 000 steps, n = 32 / 40, m = 32 / 48 / 64 / 96, 2 seeds (16 runs, one
wave of 16 workers). New config key `max_minutes` (420 here): a run past the cap stops at
its next evaluation and is saved with `"truncated": true`, so a slow run keeps its curve
instead of dying at the session deadline. Expected ~1.5 h data prep + ≤ 7 h training.

### Walk encodings set up: k-hop rows and RWSE; the GraphGPS RWSE default leaks connectivity

`WalkInput` (width_sweep.py): adjacency row plus either the k-hop reachability row
[(A + I)^k > 0] ("khop": k hops for free, leaving connectivity at diameter / k on the
same graphs — also a path-length cut without new graphs, cf. the dropped Q7b) or the
return probabilities diag(P^j), j ≤ K, P = D^-1 (A + I) ("rwse", Dwivedi et al. 2022).
Inputs checked against numpy / shortest paths. Leak audit (`audit_walk_features.py`,
2000 swap graphs, gradient boosting; graph-connected from the RWSE features, chance 0.5;
far pairs (> 4 hops) same-component, base 0.56 / 0.63):

| K | 4 | 8 | 16 | 32 |
|---|---|---|---|---|
| n = 40: graph / pair | 0.503 / 0.543 | 0.536 / 0.540 | **0.896 / 0.806** | **0.997 / 0.965** |
| n = 56: graph / pair | 0.506 / 0.606 | 0.527 / 0.598 | **0.715 / 0.667** | **0.982 / 0.895** |

K ≤ 8 is clean; K = 16, the GraphGPS default, gives connectivity away (long walks
approach deg / vol(component)), less so at larger n. Local check (n = 32, 16k graphs,
3000 steps, exact / pair): adjacency rows m = 32 0.31 / 0.82; rwse8 m = 32 0.26 / 0.81;
khop2 m = 12 0.78 / 0.97, m = 32 0.95 / 0.99; khop4 m = 8 0.97 / 0.998. Two free hops
already change the picture a lot. Sweeps (q3lr protocol, n = 40 / 56, 2 seeds, 72 runs):
`enc_khop2` (m 6–32), `enc_khop4` (m 2–16), `enc_rwse8` (m 16–96), `enc_rwse16` (positive
control, m 8–64). Notebook set to these four, 2 shards each (8 workers).

## 2026-10-13

### Thesis direction: input encodings; Q7b dropped; enc_edges set up

Supervisor: leave the diameter question (Q7b) out and study how the input encoding affects
width. First new encoding, `enc_edges`: Sanford et al.'s vertex + edge tokens
(`EdgeTokenModel`: learned node-ID table, no positions, padded edges masked) with the same
bidirectional encoder and all-pairs bilinear read-out as q3 — B2 without its 32-label
confound. q3lr protocol, n = 32–56, 5 widths per n (8–96), 2 seeds = 40 runs; ~3.2× the
step cost of adjacency rows. Checks: symmetric output, batch-independent (mask works),
smoke test. Learnability: on the 1-hop task (predict A) the edge model sits at the
trivial rate for ≥ 1500 steps, then learns it (edge recall 0.992 at 4000) — a plateau
before it finds ID-matching attention, which adjacency rows get for free; on reachability
at n = 32, m = 48 it is still trivial at 3000 steps where adjacency rows reach pair 0.927.
Notebook: absolute 11 h deadline from session start (workers stopped, catch-up skipped)
so a slow sweep keeps its finished runs.

### Q7b set up: node count vs path length

`width_sweep.py` sweeps `q7b_n` (connected diameter held at ~11 via chord_frac per n:
40/0.1, 48/0.25, 56/0.35, 64/0.35) and `q7b_d0` / `q7b_d10` (n = 56, chord_frac 0 / 0.1:
diameter 17.5 / 14.5 vs q3lr's 11.8), q3lr protocol (128k graphs, 24k steps, depth 6,
LR 1e-3, 2 seeds), 60 runs; new `chord_by_n` config key. At fixed n the swap diameter
range is narrow (n = 40: 10–12; chord_frac >= 0.5 fails d_min 6), hence the n = 56 cut.
Audit (`audit_width_data.py`, 1000 graphs): stats->y 0.47–0.52, k-hop exact 0 up to k = 4
everywhere; k = 8: 0.26–0.48 for diameter-11 settings, 0.005 at cf 0.1, 0 at cf 0 (where
even k = 12 is 0).

### Sanford b2long: the loss escapes but only into memorization; B2 has a supervision confound

`b2long` (8 runs, 100k steps, n = 32 / 48, m = 32 / 96 / 192, LR 1e-3 / 3e-4): every run
whose loss leaves the trivial level (escape at 20k–65k steps, earlier for wider models
and 3e-4) overfits — training loss down to 0.27–0.48, training accuracy 0.75 → ≤ 0.86,
dev/test *below* always-"yes" (0.63–0.68 at m ≥ 96). m = 192 at 1e-3 never moves. No run
generalizes above the trivial rate. Confound: Q3 supervises all n² pairs per graph,
B2 only 32 queries (32–98× fewer labels on the same 128k graphs), so B2 cannot yet
separate "harder encoding" from "less supervision". Details: `docs/sanford-reproduction.md` §9.

## 2026-10-12

### Sanford B2: their encoding stays at "always yes" at every width ≤ 96

`b2` on Kaggle (56 runs, 24k steps): every run ends at the always-"yes" rate (test 0.739 /
0.751 / 0.758 / 0.761 for n = 32…56; exact 0.50). Data and read-out verified (0 label
mismatches from decoded tokens). Loss drifts below the trivial entropy late and in width
order (n = 32 final loss: m ≤ 32 0.567, m = 64 0.556, m = 96 0.532), so the widest models
are only starting to escape. Same budget/depth as Q3, where adjacency rows cross 0.95 at
m ≈ 22–58: Sanford's encoding is harder to learn here. Critical width needs a longer run.
Also: swap data is now built in a spawn pool — fork deadlocked on Kaggle (first B2 attempt
hung ~6.7 h in --prepare). Details: `docs/sanford-reproduction.md` §9.

## 2026-10-11

### Sanford B2 set up: their encoding on swap; local learners at the trivial rate

`sanford_repro.py b2`: q3lr's setting with vertex + edge + 32 query tokens per graph
(uniform s ≠ t, read at each query position) on `swap` n = 32–56, 128k graphs, depth 6,
m = 8…96, 2 seeds (56 runs); `--scaling` gives critical width at query accuracy 0.95.
Script generalized to K queries per graph, an n axis, parallel chunked swap generation
(n = 32 is ~27 ms/graph), shared data seed. Audit: boosted trees on 2-hop features
0.746–0.759 = always-yes (0.748–0.760); 8-hop rule 0.998 / 0.981 / 0.961 / 0.933.
GraphQA path unchanged (smoke numbers identical). Details: `docs/sanford-reproduction.md` §9.

### Sanford B1: their 98 needs only m = 32; no width beats a 2-hop learner on the hard set

`bwidth` (Kaggle, 12 runs, ~30 min each, m=768 88 min): L=12, 100K GraphQA graphs, 30k
steps, cosine 1e-4 (+1e-3 for m ≤ 128). Rate picked on final dev, final test: m=8 0.957,
16 0.967, **32 0.983**, 64 0.979, 128 0.979, 256 0.984, 768 0.980. Hard set (4+ / no path,
both have edges): 0.81–0.89 / 0.63–0.70 at every width, m=768 0.808 / 0.700 — all below
the 2-hop boosted trees (0.932 / 0.893), and flat in width. m ≤ 16 don't fully learn even
the degree rule ("no path, isolated" 0.84–0.94). GraphQA saturates on local shortcuts at
m ≈ 32 for n ≤ 19, so it can't measure a connectivity critical width → B2 (their encoding
on `swap`). Details: `docs/sanford-reproduction.md` §6. (`results/sanford/bwidth.shard*of4.jsonl`)

## 2026-10-10

### GraphQA reachability is solvable from 2-hop features (0.995) — even its hard cases

`sanford_repro.py bwidth --audit`: rules, then gradient-boosted trees fit on 50k
train-distribution examples, scored on 20k test-distribution examples and the hard set
(1000 connected pairs ≥ 4 hops, 1000 no-path pairs with both degrees > 0).

| | test | hard: 4+ | hard: no path |
|---|---|---|---|
| degree rule / 3-hop rule | 0.973 / 0.981 | 1.000 / 0.000 | 0.000 / 1.000 |
| trees: graph size (n, edges, density) | 0.944 | 0.878 | 0.720 |
| trees: + endpoint degrees | 0.977 | 0.938 | 0.411 |
| trees: + 2-hop neighbourhoods | **0.995** | 0.932 | 0.893 |

- Graph size alone gives 0.944 — and the token count reveals n + |E|.
- A 2-hop learner beats the reproduced 71M transformer (0.982). On graphs of ≤ 19 nodes
  the 2-hop ball mostly covers the component, so even the hard set is ~90 % local.
- So GraphQA reachability cannot separate computing connectivity from local statistics;
  Sanford's experiment can't support a global-reasoning claim. The hard set is a
  sharper check than the full test set but must be read against 0.93 / 0.89, not 0.5.

### Sanford part A reproduced at 100K with LR 1e-4 (0.982 vs 98.0) — level with the 3-hop rule

`a100k_lr1e4` (cosine from 1e-4): test **0.982** dev-selected, 0.979 final, train 1.000;
±0.003 at 2000 test graphs, so it matches their 98.0. `a100k_cos` (cosine from 5e-4):
still stuck, train/test ~0.93 — their peak rate is too hot in our setup (batch 64; theirs
unstated), decay alone doesn't fix it. A repeat of the constant-5e-4 run failed the same
way (plateau ~0.93, NaN at ~52k; kept in `kaggle/working/sanford2`, not in results).

Same test split: degree rule 0.971, 3-hop rule 0.980 — the reproduced 98.2 doesn't show
connectivity is computed. `sanford_repro.py` now reports final accuracy per s–t distance
bucket and on a **hard set** (seed 7777, same generator): 1000 connected pairs ≥ 4 hops
apart (3-hop rule always wrong) + 1000 no-path pairs whose endpoints both have edges
(degree rule always wrong). A 2-layer m=16/64 smoke model scores ~0.87–0.90 / ~0.10 there,
i.e. the degree rule. `bwidth` (B1) now at cosine 1e-4 for m = 8…768 plus 1e-3 for
m ≤ 128 (12 runs); its m=768 run gives the hard-set numbers for the reproduced model.
(`results/sanford/a100k_cos.jsonl`, `a100k_lr1e4.jsonl`)

## 2026-10-09

### Sanford part A: 1K reproduced (93.0 vs 92.9); 100K failed to optimize, rerun

Kaggle T4, `sanford_repro.py a1k` / `a100k`, m=768 L=12, seed 0, test = 2000 graphs.
- **a1k: test 0.930 (dev-selected), 0.932 final — their 92.9.** Train 1.000 by step 1k,
  test flat at 0.93 after; memorizes, and stays below the degree rule (0.971 on this
  exact test split). 48 min.
- **a100k: failed.** Train ~0.92, loss flat ~0.19 from step 3k (dip to 0.83 at 6k) —
  never fit, near the 2-hop rule (0.935); NaN at ~31k under fp16, then all-"no" (0.17).
  Best dev-selected test 0.920 vs their 98.0. Likely the constant 5e-4 (schedule not
  stated in the paper) at batch 64.
- Rerun: `a100k_cos` (cosine 5e-4 → 10 %) and `a100k_lr1e4` (cosine 1e-4), one per GPU;
  runs now stop when the loss is mostly non-finite. ~0.13 s/step on a T4.
  (`results/sanford/a1k.jsonl`, `a100k.jsonl`)

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
