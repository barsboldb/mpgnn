// Results since the 10 November report (9-19 Nov 2026): every m* curve at one depth (12).
// Figures: reports/figures/width_report_1119_plot.py (d12-*.png), which also prints every number
// quoted here from results/width/*.jsonl. Earlier context: width-report-2026-11-10.typ.
// Day-by-day record: docs/CHANGELOG.md.

#set page(paper: "a4", margin: (x: 2.2cm, y: 2.2cm), numbering: "1")
#set text(font: ("New Computer Modern", "Linux Libertine"), size: 10.5pt)
#set par(justify: true)
#set heading(numbering: "1.1")
#show table: set par(justify: false)
#show figure.caption: set text(size: 9pt)
#let tbl(..args) = table(stroke: 0.4pt, inset: 4.5pt, align: center, ..args)
#let box2(body) = block(stroke: 0.4pt, inset: 8pt, width: 100%, body)
#let ms = $m^*$
#let pass(a, b, n) = [#a / #b of #n]

#align(center)[
  #v(1cm)
  #text(18pt, weight: "bold")[Width at One Depth: \ Encoding, Not Depth, Sets the Growth]
  #v(0.3cm)
  #text(12pt)[Experiment results, 9 – 19 November 2026]
  #v(0.2cm)
  Follow-up to the report of 10 November · thesis: _Analyzing the Effect of Network Width on Graph Transformers_
  #v(0.6cm)
]

#box2[
  *In one paragraph.* The 10 November report left one caveat open: edge bias was measured at
  depth 12 but the adjacency-rows baseline at depth 6, so part of the gap could have been
  depth. All three encodings have now been run at *depth 12*. The baseline needs the same width
  as at depth 6 ($#ms prop n^(1.95)$, 21 → 63 over $n = 32$–56). With edges as an attention
  bias, $#ms$ stays at *5–6 from $n = 32$ to $n = 128$* (slope 0.16, four seeds from $n = 72$ on).
  At $n = 56$ the gap is 11×, and at $n = 72$ it is more than 20×, because no baseline width
  up to 144 passes there. When the graph enters _only_ through the bias (random-ID tokens),
  $#ms$ grows a little (5–11). Each failed run either never leaves the trivial predictor or
  solves the task and then collapses. All four random-ID runs that collapsed for good fell in
  epochs 2–3, when the learning rate reaches its peak. The growth in width comes from the
  encoding, not from depth.
]

#figure(image("figures/d12-mstar.png", width: 100%),
  caption: [Critical width $m^*$ against graph size, all at depth 12. Markers: $m^*$ where
  the seed-averaged accuracy curve crosses 0.95 on the test set. Pale bars run from the smallest
  grid width at which _some_ seed passes to the one at which _every_ seed passes. Both axes are
  logarithmic, so the slope of a line is the exponent $k$ in $m^* prop n^k$. The dashed grey line
  is the depth-6 baseline from the earlier report.]) <fig-mstar>

#outline(depth: 2, indent: 1.2em)

= Setup in brief

*Task.* Each graph has two sparse clusters, joined by one bridge edge or not, with degree
sequences matched so that local counts cannot give the answer away (the `swap` generator). The
model predicts the full $n times n$ connectivity matrix. *Pair accuracy* is the share of node
pairs it gets right. Predicting "not connected" for every pair already scores 0.745–0.77, so
that value is the *trivial floor*.

*Critical width $m^*$.* The smallest width (model dimension $m$) at which the final-epoch
test pair accuracy reaches 0.95. Between grid widths it is interpolated in $log_2 m$, as
`width_sweep.py --scaling` does. In this report every $m^*$ is recomputed from the raw result
files with all available seeds pooled, so a few values differ from the changelog by up to ≈0.5.

*Fixed settings.* Unless a table says otherwise, every run in this report uses the settings
below. Only $m$ and $n$ vary.

#align(center, tbl(columns: 2, align: (left, left),
  [depth], [12 layers (fixed for every curve; depth is a control, not the variable)],
  [training], [128k graphs, 24k steps (24 epochs × 1000), batch 128],
  [optimiser], [AdamW, peak LR $10^(-3)$, 5 % linear warmup (1 200 steps), cosine decay],
  [evaluation], [400 validation and 400 test graphs, final epoch],
  [head_dim], [8 for the baseline · 2 for edge bias (see @sec-caveats)],
))

*The three encodings.*
/ Adjacency rows (baseline): vertex token $i$ is row $i$ of the adjacency matrix. Attention
  must find neighbours through query–key dot products.
/ Adjacency rows + edge bias: the same tokens, and each attention score $(i, j)$ also gets a
  learned per-layer, per-head scalar $b$ when $(i, j)$ is an edge (start value 2).
/ Random IDs + edge bias: vertex tokens are random 64-dimensional IDs that carry no graph
  information. The graph enters *only* through the attention bias (start value 4).

= Results

== The baseline at depth 12 needs the same width as at depth 6

#figure(tbl(columns: 6,
  [$n$], [widths (test pair, seed 0 / 1)], [#ms depth 12], [#ms depth 6], [#ms edge bias], [gap],
  [32], [16: .903/.894 · 24: .976/.972], [21.1], [22.4], [5.1], [4×],
  [40], [24: .904/.930 · 32: .946/.967], [30.6], [33.1], [5.1], [6×],
  [48], [32: .916/.877 · 48: .967/.966], [43.6], [44.7], [5.1], [9×],
  [56], [48: .912/.886 · 64: .954/.950], [63.3], [59.6], [5.6], [11×],
  [72], [80: .865/.878 · 112: .913/.926 · 144: .814/.892], [> 144], [> 128#super[a]], [6.2], [> 23×],
), caption: [Adjacency-rows baseline at depth 12 (`q3lr_d12h8`, `q3lr_d12h8_72`, head_dim 8,
2 seeds) against depth 6 (`q3lr`, 3 seeds) and edge bias at depth 12. "Gap" is the
baseline-to-edge-bias ratio of #ms. #super[a] Depth 6 at $n = 72$ reaches at most 0.947 (m = 96) and
0.941 (m = 128) with 128k graphs. With 256k graphs or 48k steps, one of two seeds passes at
$m = 96$; there the limit was data, not fitting.]) <tab-base>

Doubling the depth from 6 to 12 saves about 6 % of width at $n = 32$ and nothing at $n = 56$.
The fitted exponent is 1.95 at depth 12 and 1.74 at depth 6, so it is no smaller with more
depth. Accuracy rises smoothly with width and the two seeds agree within ≈0.06. These runs do
not show the all-or-nothing behaviour of edge bias.

A first attempt used head_dim 2, the same as the edge-bias runs (`q3lr_d12`, 13 of 24 runs
before the VM was lost). None of those runs reached 0.95, and every cell fell below its depth-6
counterpart. Without a bias, attention has to find neighbours through query–key dot products,
and 2-dimensional heads seem too small for that. The baseline therefore keeps head_dim 8.

== Edge bias at depth 12: flat from $n = 32$ to $n = 128$

#figure(tbl(columns: 7,
  [$n$], [$m = 2$], [$m = 4$], [$m = 6$], [$m = 8$], [$m = 12$], [#ms],
  [32], [0 / 2], [1 / 2], [*2 / 2*], [], [], [5.1],
  [40], [0 / 2], [1 / 2], [*2 / 2*], [], [], [5.1],
  [48], [0 / 2], [1 / 2], [*2 / 2*], [], [], [5.1],
  [56], [0 / 2], [0 / 2], [*2 / 2*], [], [], [5.6],
  [72], [], [0 / 4], [2 / 4], [*4 / 4*], [], [6.2],
  [96], [], [0 / 4], [2 / 4], [*4 / 4*], [], [6.4],
  [128], [], [0 / 4], [3 / 4], [1 / 4], [*4 / 4*], [5.8],
), caption: [Adjacency rows + edge bias, depth 12, head_dim 2: how many seeds reach 0.95 test
pair accuracy at the final epoch (`ebd12hd2_s`, `_72`, `_96`, `_128`, `_L`, `_128w4`). Bold:
the narrowest width where every seed passes. At $n = 128$, $m = 16$ also passes 4 / 4.]) <tab-eb>

From $n = 72$ on, every cell now has four seeds. The pooled #ms is 6.2, 6.4 and 5.8 at $n = 72$,
96 and 128, a slope of −0.12. Over all of $n = 32$–128 the slope is 0.16, while $n$ grows 4×.
Width 4 (two heads) fails at every $n >= 56$; its best single run reaches 0.915. Between widths
6 and 12 the runs are noisy. At $n = 128$, width 8 does _worse_ than width 6 for three of the
four seeds. The width at which _all_ seeds pass grows from 6 to 8 to 12 across $n = 56$, 72 and
128, so a slow rise in the reliable width cannot be ruled out. Even that rise is 2× over a 2.3×
range of $n$, against $n^(1.95)$ for the baseline.

== Graph only in the attention bias (random-ID tokens)

#figure(tbl(columns: 6,
  [$n$], [$m = 4$], [$m = 6$], [$m = 8$], [$m = 12$], [#ms],
  [32], [1 / 2], [3 / 3], [2 / 3], [], [7.8],
  [40], [2 / 2], [3 / 3], [*4 / 4*], [], [5.3],
  [48], [0 / 0], [], [2 / 3], [3 / *4*], [11.3],
  [56], [0 / 0], [], [2 / 3], [*4 / 4*], [9.4],
), caption: [Random IDs + edge bias, depth 12, head_dim 2, 4 seeds (`ebd12rid`, `ebd12rid_s23`).
Each cell shows the seeds that pass 0.95 at the final epoch / at _any_ epoch, out of 4. Where a
pass count is in bold, every seed passed.]) <tab-rid>

These runs test whether the model needs graph information in its tokens at all. It does not,
for capacity: at every $n$, some seed solves the task at width 6–8. Two things change.

- *Reliability.* At $n = 48$–56, width 8 passes for only half the seeds, where adjacency rows
  pass every seed at width 6. Every seed reaches 0.95 at some epoch at widths 8, 8, 12 and 12;
  with adjacency rows the figure was 6 at every $n$.
- *Growth.* The pooled #ms rises from ≈5–8 to ≈9–11 (slope 0.68). The cell values jump around
  because the outcomes are nearly binary, so with four seeds the exponent is loose. It is still
  well below the baseline's 1.95.

With only two seeds (18 November) it looked as if "width 4 never solves with random IDs". Seeds
2 and 3 solved $n = 32$ and 40 at width 4, so that claim was seed luck and has been withdrawn.

== Every run, seed by seed

#figure(image("figures/d12-seeds.png", width: 100%),
  caption: [Final test pair accuracy of every run in this report. Each dot is one seed and the
  line joins seed means. The baseline (blue) climbs smoothly with width. Edge-bias runs (orange,
  green) cluster either at ≈1.00 or at the trivial floor (≈0.75), with few in between. At
  $n = 72$ the baseline's widest model ($m = 144$) is worse than $m = 112$.]) <fig-seeds>

@fig-seeds shows the main qualitative difference between the encodings. For the baseline,
width works like a dial: more width means steadily higher accuracy. For edge bias it works more
like a switch: below ≈6 almost nothing solves, and above it most runs do. The ones that fail
fail completely. Because of this, the interpolated #ms for edge bias mostly records _which
widths happened to get a failed seed_. The pass counts in @tab-eb and @tab-rid are the more
honest summary.

== Failed random-ID runs: stuck or collapsed at the learning-rate peak

#figure(image("figures/d12-collapse.png", width: 88%),
  caption: [Test pair accuracy over training for five random-ID runs at $m = 8$. The orange
  and yellow runs are at ≈1.00 within the first 2 000 steps and drop to the floor by epoch 3,
  just after warmup ends (1 200 steps) and the learning rate is at its peak of $10^(-3)$. Some
  runs recover later (blue) and some never do.]) <fig-collapse>

The random-ID model starts with bias 4, so at initialisation attention already follows the
edges strongly. *10 of the 48 random-ID runs are above 0.95 after the first epoch.* All 10
drop below 0.95 within the next two epochs, nine of them to the trivial floor, when warmup ends and
the learning rate reaches its peak. Six of them climb back by the end and four do not (final
0.755–0.925). Those four make up most of the "solved then collapsed" cases. The fifth
($n = 56$, $m = 8$, seed 1) reaches 0.967 at epoch 10 and falls back by epoch 14. No
adjacency-row edge-bias run (start value 2, 68 runs) is above 0.95 at epoch 1, and only two
lose a pass late in training ($n = 96$; 0.978 → 0.915 and 0.955 → 0.912).

So with random IDs, final accuracy *understates* what the architecture can do. The remaining
failures look like an *optimisation* problem tied to the learning-rate schedule, not missing
width. This is a hypothesis: it has not been tested by changing the schedule.

== The baseline at $n = 72$: underfitting, not generalisation

#figure(image("figures/d12-n72.png", width: 100%),
  caption: [Depth-12 baseline at $n = 72$ (`q3lr_d12h8_72`, 2-seed means) against depth 6 at
  $m = 96$ (`q3n72`). (a) Training-set pair accuracy: no depth-12 width reaches 0.95, and the
  widest model ($m = 144$) fits worst. (b) Test accuracy follows training within ≈0.03.]) <fig-n72>

Below $n = 56$, and at $n = 72$ for depth 6, the baseline fit the training set and the problem
was generalisation (more data closed the gap). The depth-12 baseline at $n = 72$ is different.
It does not even fit the training set: train pair accuracy is 0.84–0.95, and the $m = 144$
model trains more slowly than $m = 112$. Within the 24k-step budget the larger models have not
finished learning, so $#ms (72) > 144$ is a statement about this budget, not a fitted point.
Read naively, it would give a slope above 3.3 from $n = 56$ to 72. That number should not be
quoted until the longer run (@sec-next) is in.

= What this means for the thesis

#box2[
  + *The width–size exponent is a property of the encoding.* At one depth (12), adjacency rows
    need $#ms prop n^(1.95)$. Adjacency rows plus edge bias need roughly constant width
    ($n^(0.16)$, $#ms approx 5$–6 over $n = 32$–128), and random IDs plus edge bias need
    $n^(0.68)$ with large seed noise.
  + *Depth does not explain the gap.* Going from depth 6 to 12 leaves the baseline's #ms
    unchanged within 6 %. The 10 November caveat is closed.
  + *Edge-bias failures are optimisation failures.* For both edge-bias encodings, some seed
    always solves the task at small width. Runs fail completely, and with random IDs they fail
    at the learning-rate peak. Width beyond ≈6 mainly buys reliability.
]

== Caveats that the thesis text must state <sec-caveats>

- *head_dim differs between encodings*: 8 for the baseline, 2 for edge bias. The baseline
  fails at head_dim 2, and at $n = 128$ edge bias needs ≈14.5 at head_dim 8 against ≈6 at
  head_dim 2 (10 November). The comparison is "each encoding at a working head size", and the
  text must say so.
- *Start value of the bias differs*: 2 with adjacency rows, 4 with random IDs. Some of the
  difference in reliability may come from that, not from the tokens.
- *Seeds*: edge bias at $n = 32$–56 and the depth-12 baseline have two seeds each; edge bias at
  $n >= 72$ and random IDs have four.
- *Threshold on the final epoch*: this choice makes #ms pessimistic for runs that collapse late.

= In flight and next <sec-next>

#tbl(columns: 3, align: (left, left, left),
  [*sweep*], [*what*], [*status*],
  [`q3lr_d12h8_72s`], [baseline, depth 12, $n = 72$, 48k steps, $m = 112, 144$],
    [defined (commit 6ae45c5), not run yet],
  [random IDs, longer warmup / lower peak LR], [tests whether the collapses in @fig-collapse are the schedule],
    [proposed],
  [`ebd12hd2_s` seeds 2–3], [4 seeds for edge bias at $n = 32$–56, like $n >= 72$], [proposed],
)

The 48k-step baseline run decides whether $n = 72$ lands near the $n^(1.95)$ line (≈ 100) or
truly breaks upward. That number is what the thesis can quote for the gap at $n = 72$.

#pagebreak(weak: true)
= Appendix: runs in this report

#tbl(columns: 6, align: (left, left, center, left, center, left),
  [*sweep*], [*encoding*], [*$n$*], [*widths*], [*seeds*], [*where / date*],
  [`q3lr_d12`], [adj rows, head_dim 2], [32–56], [16–48], [1–2], [Colab L4 · 11-09 (13/24, VM lost)],
  [`q3lr_d12h8`], [adj rows], [32–56], [8–64], [2], [Colab T4 · 11-12],
  [`q3lr_d12h8_72`], [adj rows], [72], [80, 112, 144], [2], [Colab L4 · 11-19],
  [`ebd12hd2_s`], [adj + bias], [32–56], [2, 4, 6], [2], [Colab T4 · 11-09],
  [`ebd12hd2_72/_96`], [adj + bias], [72, 96], [4, 6, 8], [2], [Colab · 11-07],
  [`ebd12hd2_128`], [adj + bias], [128], [6–16], [4], [Kaggle T4×2 · 11-10],
  [`ebd12hd2_L`, `_128w4`], [adj + bias], [72–128], [4–8], [+2], [Kaggle T4×2 · 11-17],
  [`ebd12rid`], [rid + bias], [32–56], [4–12], [2], [Kaggle T4×2 · 11-18],
  [`ebd12rid_s23`], [rid + bias], [32–56], [4–12], [+2], [Kaggle T4×2 · 11-19],
)

Raw results: `results/width/<sweep>.shard*.jsonl`. Figures and every number in this report:
`.venv/bin/python reports/figures/width_report_1119_plot.py`.
