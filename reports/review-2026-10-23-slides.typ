#import "@preview/touying:0.6.1": *
#import themes.metropolis: *

// Supervisor meeting 2026-10-23: all experiments, the Sanford et al. 2024a and
// Yehudai et al. 2025 reproductions. Content: reports/width-progress.pdf,
// docs/yehudai-empirical.md, reports/dataset-comparison.pdf.
#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  config-info(
    title: [Analyzing the Effect of Network Width on Graph Transformers],
    subtitle: [All experiments so far, and the reproductions of Sanford et al. 2024a and Yehudai et al. 2025],
    author: [Barsbold Bayarerdene],
    date: [23 October 2026],
  ),
)

#set text(size: 17pt)

#let cG = rgb("#2f8f4e")
#let cB = rgb("#c0392b")
#let cO = rgb("#b9770e")
#let dimc = luma(105)
#let good(b) = text(fill: cG, weight: "semibold")[#b]
#let bad(b) = text(fill: cB, weight: "semibold")[#b]
#let warn(b) = text(fill: cO, weight: "semibold")[#b]
#let rc(b) = text(size: 12pt, fill: dimc)[#b]
#let tbl(..a) = table(
  inset: 7pt,
  stroke: (x, y) => if y == 0 { (bottom: 0.7pt) } else { (bottom: 0.3pt + luma(222)) },
  ..a,
)
#let boxed(b) = align(center, block(inset: 11pt, fill: luma(245), radius: 4pt, width: 92%, b))
#let fig(path) = image(path, width: 100%)

#title-slide()

== The question and the set-up

*How does the width a graph transformer needs to _learn_ connectivity grow with graph
size $n$?*

#grid(columns: (1.1fr, 1fr), column-gutter: 1.2em,
  [
    - *Task:* connectivity matrix: for every pair of nodes, are they in the same
      component?
    - *Model:* encoder, one token per node (its row of $A + I$), pre-LN, pairwise
      read-out $H W H^top$
    - *Width* $m$ = embedding dimension; heads of size 8 ($m\/8$ heads), feed-forward $4m$
  ],
  [
    - *Critical width $m^*$:* smallest $m$ whose final test pair accuracy reaches 0.95,
      interpolated in $log_2 m$ per seed, then averaged
    - *Final protocol:* 128 000 graphs, 24 000 steps, batch 128, LR $10^(-3)$, depth 6,
      2 seeds; LR chosen on validation, numbers reported on test
  ],
)

#boxed[Theory: width ≈ $sqrt(N)$ suffices to *represent* connectivity at depth $log N$
(Sanford et al. 2024a); width $O(n)$ at constant depth for 1-vs-2 cycles (Yehudai et
al. 2025). We measure the width to *learn* it.]

== Headline results

#align(center, tbl(
  columns: 2, align: (center, left),
  table.header([*result*], [*what it says*]),
  [$m^* prop n^(1.74)$], [critical width at depth 6, $n$ = 32–56 (90 %: 1.59–1.90)],
  [$m^* prop n^3$ at depth 2], [training does not find the constant-depth, linear-width solution],
  [$n^(1.27)$ / $n^(1.03)$], [growth when each token gets its 2-hop / 4-hop neighbourhood],
  [2.2–2.5×], [extra width Sanford et al.'s edge tokens need (with 4× the steps)],
  [0.995], [2-hop classifier on Sanford et al.'s GraphQA data (their transformer: 0.98)],
  [0.77], [one number (edge count) on Yehudai et al.'s connectivity data],
  [gradual, pairwise], [probes: "connected" is built up in every layer, not stored per node],
))

= Part 1 — Measuring width honestly

== A fixed learning rate fakes a width effect

#grid(columns: (1.15fr, 1fr), column-gutter: 1em, align: horizon,
  fig("figures/width-q1b-lr.png"),
  [
    - One fixed rate: wide models look up to *0.4 worse*
    - Tuned per width: widths 8–64 all ≈ 0.97–0.98
    - The best rate *falls as width grows* ($10^(-2)$ → $10^(-3)$ at $m = 256$)
    - The remaining drop at $m >= 128$ is memorization: 8 000 graphs remove it (Q1c)

    #v(0.4em)
    #rc[$n = 24$, depth 2, 2000 graphs, 3 seeds (Q1, Q1b, Q1c)]
  ],
)

== The first benchmark leaked; `swap` does not

#align(center, tbl(
  columns: 2, align: (left, left),
  table.header([*shortcut in the old data*], [*evidence*]),
  [node indices reveal the components], [blobs always nodes $[0, n_a)$ and $[n_a, n)$],
  [graph statistics reveal the label], [degree + closed walks $tr A^3 … tr A^6$: 0.68–0.90],
  [small graphs are local], [a 6-hop check is exact on 60–100 % of graphs],
))

*`swap`:* two sparse blobs; pick $a_1, a_2$ and $b_1, b_2$ ≥ 6 hops apart.
Disconnected: add $a_1 a_2$, $b_1 b_2$. Connected: add $a_1 b_1$, $a_2 b_2$. Same
degrees, every new cycle ≥ 7 edges, labels shuffled.

#align(center, tbl(
  columns: 7,
  table.header([$n$], [32], [40], [48], [56], [64], [128]),
  [statistics → label], [0.53], [0.51], [0.49], [0.50], [0.51], [0.51],
  [4-hop check, exact], [0], [0], [0], [0], [0], [0],
  [diameter (connected)], [8.6], [9.8], [10.9], [11.7], [12.5], [16.7],
))

#rc[Trivial predictor: 0.5. On the leaky data width 8 sufficed at $n = 24$.]

= Part 2 — How width must grow with $n$

== Enough data closes the fit–generalize gap

#grid(columns: (1.15fr, 1fr), column-gutter: 1em, align: horizon,
  fig("figures/width-q3-scaling.png"),
  [
    - 32 000 graphs: fit $prop n^(1.55)$, generalize $prop n^(2.27)$; the gap widens with $n$
    - 128 000 graphs at the same 24 000 steps: the two coincide (within ≈ 2)
    - The gap was a data ceiling, as in Q1c
    - Below the threshold nothing learns: $n = 64$ needs $m >= 64$; at $n = 128$
      nothing up to 128 leaves the trivial predictor

    #v(0.4em)
    #rc[depth 6, LR $3 dot 10^(-3)$, 2–3 seeds]
  ],
)

== Main result: $m^* prop n^(1.74)$

#grid(columns: (1.15fr, 1fr), column-gutter: 1em, align: horizon,
  fig("figures/width-q3lr-curves.png"),
  [
    #tbl(
      columns: 4, align: center,
      table.header([$n$], [fit], [generalize], [$m^* \/ n$]),
      [32], [20.9], [21.9], [0.68],
      [40], [28.2], [30.1], [0.75],
      [48], [38.5], [42.4], [0.88],
      [56], [52.7], [57.9], [1.03],
      [growth], [$n^(1.65)$], [$n^(1.74)$], [],
    )
    #v(0.3em)
    #rc[90 % bootstrap range 1.59–1.90 (one seed per $n$; only 16 combinations).
    128 000 graphs, 24 000 steps, LR $10^(-3)$, depth 6.]
  ],
)

== The growth survives every check

#align(center, tbl(
  columns: 3, align: (left, left, left),
  table.header([*could it be…*], [*test*], [*outcome*]),
  [too little data?], [32k → 64k → 128k graphs], [gap closes, #good[growth stays superlinear]],
  [too little training?], [2× the steps (48 000)], [#good[critical width unchanged] (within seed spread)],
  [the learning rate?], [$3 dot 10^(-3)$ vs $10^(-3)$, tuned on validation], [helps wide models at $n = 56$; slope 1.94 → #good[1.74]],
  [the $n$-wide input?], [fixed 96-wide random projection], [#good[still doubles] from $n = 40$ to 56 ($n^(2.03)$)],
  [noise?], [per-seed crossings], [#good[seeds within ≈ ± 3]],
))

#v(0.4em)

#boxed[Over $n = 32$–$56$, the width needed to *learn* connectivity grows faster than
$n$, while the width needed to *represent* it is about $sqrt(N)$.]

= Part 3 — Can depth replace width?

== At $n = 40$, depth saves about a third

#grid(columns: (1.15fr, 1fr), column-gutter: 1em, align: horizon,
  fig("figures/width-q7depth.png"),
  [
    #tbl(
      columns: 6, align: center,
      table.header([$L$], [2], [3], [4], [6], [8]),
      [$m^*$], [45.3], [37.9], [34.5], [30.1], [32.3],
    )
    - *2 layers learn* at $m ≈ 1.1 n$ on graphs of diameter ≈ 10: the model does not
      follow paths hop by hop
    - Beyond 4–6 layers depth buys nothing
    - $L = 6$ repeats the main result exactly (30.1)

    #rc[$n = 40$, 128 000 graphs, LR $10^(-3)$, 60 runs (Q7a)]
  ],
)

== At depth 2 the critical width grows as $n^3$

#grid(columns: (1.15fr, 1fr), column-gutter: 1em, align: horizon,
  fig("figures/width-q7c-scaling.png"),
  [
    #tbl(
      columns: 5, align: center,
      table.header([], [32], [40], [48], [56]),
      [$L = 2$, fit], [28.1], [44.4], [70.7], [168],
      [$L = 6$], [21.9], [30.1], [42.4], [57.9],
      [ratio], [1.3], [1.5], [1.8], [> 3],
    )
    - Fit $prop n^(3.07)$: a linear-width, constant-depth solution exists, but training
      does not find it as $n$ grows
    - At $n = 56$, two layers do not generalize to 0.95 by width 192
    - "Depth trades mildly" holds only at $n = 40$

    #rc[Q7c, 34 runs]
  ],
)

= Part 4 — Reproducing Yehudai et al. 2025

== What Yehudai et al. prove

#tbl(
  columns: 2, align: (left, left),
  table.header([*result*], [*statement*]),
  [Thm 4.1], [1 cycle vs 2 cycles: 2 layers, width $O(n)$, adjacency-row tokens],
  [Thm 4.2], [lower bound: $m p H L = Omega(n)$ for the same task, unconditional],
  [Thm 4.3], [$A^L$ with $O(L)$ layers: one hop per layer (not repeated squaring)],
  [general connectivity], [only a sketch: width $O(n log^3 n)$ via linear sketches (Ahn, Guha, McGregor 2012)],
  [experiments], [*fitting* criterion, 5 000 graphs, depth 1, fixed LR grid],
)

#v(0.4em)

#boxed[Their theory is about *representing*; their experiments measure *fitting*, on
data we show is easy. Our results measure *learning* on audited data.]

== Reproduction: their connectivity data is easy

#grid(columns: (1fr, 1fr), column-gutter: 1.2em,
  [
    *Their code* (two bugs patched: a `NameError`, and the cache reloading the
    *train* split as validation and test)

    #tbl(
      columns: 2, align: (left, center),
      table.header([$n = 50$, 1 layer, 100 graphs], [test]),
      [adjacency rows], [#bad[1.00]],
      [edge list], [0.93],
      [*our* model on *their* data], [#bad[1.00] by epoch 4],
    )
  ],
  [
    *Why it is easy*
    - Classes come from different generators (ER, geometric, scale-free, SBM)
    - Edge count *alone* classifies #bad[77 %]; mean degree too
    - Fixed $n = 50$: column $j$ always means the same node

    #v(0.3em)
    On degree-matched data the same model stalled at 0.59–0.70; on `swap` it needs
    the widths of Part 2.
  ],
)

= Part 5 — Reproducing Sanford et al. 2024a

== Their claim and their numbers reproduce

- *Claim:* connectivity at depth $O(log N)$ with width ≈ $sqrt(N)$ (vertex + edge +
  task tokens, $N = n + |E|$); one layer needs $m H = tilde(Omega)(N)$
- *Experiment:* GraphQA connectivity (ER graphs, $n <= 19$, one $s$–$t$ question),
  decoder $L = 12$, $m = 768$, trained on 1K and 100K graphs. No code released: we
  re-implemented data, tokens and model

#align(center, tbl(
  columns: 3, align: (left, center, center),
  table.header([], [*theirs*], [*ours*]),
  [test accuracy, 1K graphs], [92.9], [#good[93.0]],
  [test accuracy, 100K graphs], [98.0], [#good[98.2] (LR $10^(-4)$)],
  [their LR $5 dot 10^(-4)$ at 100K], [—], [#bad[stuck at 0.93, then NaN]],
))

== But GraphQA connectivity is local

#grid(columns: (1.1fr, 1fr), column-gutter: 1.2em,
  [
    #tbl(
      columns: 4, align: (left, center, center, center),
      table.header([], [test], [hard: ≥ 4 hops], [hard: no path]),
      [degree rule], [0.973], [1.00], [0.00],
      [3-hop rule], [0.981], [1.00], [0.00],
      [trees: size], [0.944], [0.878], [0.720],
      [+ degrees], [0.977], [0.938], [0.411],
      [+ 2-hop], [#bad[0.995]], [0.932], [0.893],
    )
    #rc[Gradient-boosted trees on 11 features within 2 hops of $s$ and $t$, 50k training examples.]
  ],
  [
    - A 2-hop classifier *beats the 71M-parameter transformer* (0.982)
    - 94 % of connected pairs are ≤ 2 hops apart
    - *Width only buys the shortcut* (B1, $m$ = 8–768): 0.98 from $m = 32$, flat to
      768; on the hard set every width sits below the 2-hop classifier
  ],
)

== Their encoding on audited data does not learn

- *B2:* `swap`, $n$ = 32–56, depth 6, 128 000 graphs, 24 000 steps, their tokens with
  32 questions per graph: #bad[no width up to 96 leaves "always yes"]
- *b2long:* 100 000 steps, $m$ up to 192: the loss escapes, but #bad[only into
  memorization] (test below "always yes")
- *Confound:* 32 labels per graph vs $n^2$ = 1024–3136 in our task

#v(0.3em)

*Fix — edge tokens with all-pairs labels* (same read-out as our task):

#align(center, tbl(
  columns: 3, align: center,
  table.header([], [$n = 32$], [$n = 40$]),
  [adjacency rows, 24 000 steps], [21.9], [30.1],
  [edge tokens, 24 000 steps], [no $m^*$ ≤ 96], [no $m^*$ ≤ 96],
  [edge tokens, 96 000 steps], [*48.2* (2.2×)], [*≥ 72* (≥ 2.4×)],
))

#rc[Even with 4× the steps, ≥ 5× their $sqrt(N)$ for this very encoding.]

= Part 6 — Input encodings

== The hops left to compute set the width

#grid(columns: (1.1fr, 1fr), column-gutter: 1em, align: horizon,
  fig("figures/width-encodings.png"),
  [
    #tbl(
      columns: 3, align: (left, center, center),
      table.header([input], [$n = 56$], [growth]),
      [adjacency rows], [57.9], [$n^(1.74)$],
      [+ RWSE ($j <= 8$)], [57.9], [$n^(1.65)$],
      [+ 2-hop rows], [18.3], [$n^(1.27)$],
      [+ 4-hop rows], [4.5], [$n^(1.03)$],
    )
    - 2 free hops cut the width 2.3–3.2×
    - RWSE adds nothing; its usual 16-step version #bad[leaks] connectivity (0.90 at $n = 40$)
    - `swap` diameter grows with $n$, so part of the $n^(1.74)$ is path length
  ],
)

= Part 7 — What does each layer compute?

== Probing: "connected" is built up gradually

#grid(columns: (1.3fr, 1fr), column-gutter: 1em, align: horizon,
  fig("figures/width-probe-layers.png"),
  [
    - Pair-probe AUC for "connected" rises in *every* layer; ≥ 0.95 only at layer 4–6
    - *No hop hierarchy:* 3-hop info appears at layer 1 and stays flat; 1-hop fades
    - *Not stored per node:* one node's vector gives its reachable set at AUC ≤ 0.89;
      a pair of vectors gives 0.99–1.00
    - Models far below $m^*$ fall behind after 1–2 layers

    #rc[16 saved models, $n$ = 40 / 56, linear probes on the residual stream; dashed: untrained]
  ],
)

= What it means

== Takeaways

+ *Measuring width is a protocol problem first.* LR, data and steps each interact with
  width enough to fake or hide an effect.
+ *Benchmarks decide the answer.* Both reproduced papers use data a local rule solves
  (Yehudai: edge count 77 %; Sanford: 2-hop 0.995). An audit is a contribution in itself.
+ *Representation ≠ learning.* The width to learn connectivity grows as $n^(1.74)$ at
  depth 6 and ≈ $n^3$ at depth 2, far above the representation bounds.
+ *Path length is a large part of it.* Free hops lower the exponent: 1.74 → 1.27 → 1.03.
+ *Mechanism:* a component signature compared pairwise, refined in every layer; no path
  doubling.

== Limitations

- Sizes span less than 2× ($n$ = 32–56), 2 seeds; local slopes vary (≈ 1.2–2.9)
- The bootstrap has only 16 seed combinations
- One task (connectivity), one architecture family
- Linear probes show what is easily readable, not what the model uses

== Thesis status and questions

#grid(columns: (1fr, 1fr), column-gutter: 1.2em,
  [
    *Written (Mongolian, department template)*
    - Introduction, Ch. 1 theory, Ch. 2 method, Ch. 3 scaling
    *To write*
    - Ch. 4 depth (Q7a, Q7c)
    - Ch. 5 encodings + both reproductions
    - Probing results, conclusion
  ],
  [
    *Questions*
    + Ch. 4 (depth) as its own short chapter, or part of Ch. 3?
    + Where should the reproductions go: Ch. 5, or a separate chapter?
    + Is the $n$ = 32–56 range enough, or add one larger size?
    + Planned Q2 / Q4 (task hierarchy, Graphormer bias): in scope?
  ],
)
