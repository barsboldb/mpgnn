#import "@preview/touying:0.6.1": *
#import themes.metropolis: *

// Slides for the 2026-11-03 → 11-06 runs: auxiliary loss, n = 72, edge-bias attention and the
// depth limit behind its plateau. Content and numbers: reports/width-runs-2026-11-04.pdf.
#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  config-info(
    title: [Analyzing the Effect of Network Width on Graph Transformers],
    subtitle: [Auxiliary loss, n = 72, edges in the attention, and a depth limit],
    author: [Barsbold Bayarerdene],
    date: [8 November 2026],
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

== Where we are

- *Task:* for every pair of nodes, are they connected? Output an $n times n$ yes/no table.
  `swap` graphs: two blobs, joined or not by two far-apart edges, so no local shortcut works.
- *Width* $m$: the length of each node's vector. *Critical width* $m^*$: the smallest $m$
  whose test pair accuracy reaches 0.95.
- *Baseline* (adjacency rows, depth 6): $m^* prop n^(1.74)$ — doubling $n$ needs 3.3× the width.
- *Question since 10-13:* which encodings of the graph make the transformer narrower?

#boxed[This round: an auxiliary loss, the baseline at $n = 72$, and edges put directly into the
attention — which turned out to be limited by *depth*, not width.]

== Results at a glance

#align(center, tbl(
  columns: 3, align: (left, center, left),
  table.header([*change*], [*$m^*$ at $n = 56$*], [*finding*]),
  [adjacency rows (baseline)], [58], [$n^(1.74)$],
  [\+ 2-hop rows (input)], [18.5], [$n^(1.27)$: lower level, slower growth],
  [\+ auxiliary k-hop loss], [45], [$n^(1.30)$: same input, slower growth],
  [edge bias, adjacency tokens], [#good[≈ 6]], [flat over $n = 32$–56],
  [edge bias, random-ID tokens], [> 8], [rises, unstable],
))

#v(0.4em)
- $n = 72$, baseline: fits training data at the predicted width (81), #warn[limited by data] on test.
- Edge bias past $n approx 96$: training stalls — #bad[a depth limit]. With 12 layers $m^*$ falls
  to #good[≈ 7] at $n = 72$ / 96, #warn[≈ 11–15] at 128 (12 or 16 layers).

= Auxiliary loss and n = 72

== A training-only hint changes the exponent

#grid(columns: (1fr, 1.1fr), column-gutter: 1em, align: horizon,
  [
    Same input and test-time model. In training, layers 1–3 also predict 2-, 4-, 8-hop
    reachability (weight 0.3, dropped at test).

    #v(0.3em)
    #tbl(columns: 4, align: center,
      table.header([$n$], [baseline], [\+ aux], [saving]),
      [32], [21.9], [21.5], [2 %],
      [40], [30.1], [26.8], [11 %],
      [48], [42.4], [33.5], [21 %],
      [56], [58.0], [45.1], [22 %],
      [slope], [1.74], [#good[1.30]], [],
    )
    #rc[$m^*$, seeds 0 / 1 mean. Slope ranges 1.59–1.90 vs 1.13–1.45: no overlap.]
  ],
  fig("figures/width-runs-1104-mstar.png"),
)

#boxed[The saving grows with $n$: the hint changes the *growth*, extra input mostly the *level*.]

== n = 72: the baseline runs out of data, not width

#grid(columns: (1fr, 1fr), column-gutter: 1.2em,
  [
    *128k graphs*, test (train):
    #tbl(columns: 3, align: center,
      table.header([$m$], [seed 0], [seed 1]),
      [64], [.867 (.916)], [.898 (.925)],
      [96], [.947 (.970)], [.912 (.973)],
      [128], [.941 (.978)], [.936 (.986)],
    )
    - Train-fit $m^* = 80.9$; the $n = 32$–56 trend predicts 82.6.
    - Test never reaches 0.95; train–test gap #warn[0.04–0.08].
  ],
  [
    *256k graphs*, $m = 96$:
    #tbl(columns: 3, align: center,
      table.header([graphs], [seed 0], [seed 1]),
      [128k], [.947], [.912],
      [256k], [#good[.956]], [.933],
    )
    - Gap falls to ~0.02, as at $n lt.eq 56$.
    - A test $m^*$ at $n = 72$ needs 256k graphs, more steps, $m = 96$–192.
  ],
)

= Edges in the attention

== The idea

- Normally a node must *learn* which nodes are its neighbours before it can pass information
  along edges — that costs width.
- *Edge bias:* every layer adds a learned bonus $b dot (A + I)_(i j)$ to the attention score
  of neighbours. Only direct edges, no distances.
- Each layer then acts like one step of spreading along the edges; after a few layers every
  node holds a *summary of its component*, and two nodes are connected iff the summaries match.
- Tokens: *adjacency rows* ($n$ wide) or *random IDs* (64 wide, no edges in the input at all).

#boxed[Pilot at $n = 32$ with 1/8 of the budget: both token types reach #good[≥ 0.99 at $m = 16$]
(baseline needs 22 with the full budget). A bonus starting at 0 left random IDs stuck.]

== The needed width stops growing (n = 32–56)

#grid(columns: (1.2fr, 1fr), column-gutter: 1em, align: horizon,
  [
    Final test pair accuracy, seed 0 / 1, head dimension 2:
    #tbl(columns: 5, align: center,
      table.header([$n$], [$m = 2$], [4], [6], [8]),
      [32], [.745 / .746], [.745 / .746], [.991 / 1.00], [1.00 / 1.00],
      [40], [.750 / .750], [.750 / .750], [.966 / .993], [.994 / .995],
      [48], [.755 / .755], [.977 / .919], [.985 / .985], [.988 / .991],
      [56], [.759 / .759], [.941 / .759], [.979 / .884], [.971 / .975],
    )
    #rc[Adjacency tokens. 0.75 = always-yes (nothing learned).]
  ],
  [
    #tbl(columns: 3, align: center,
      table.header([$n$], [baseline], [edge bias]),
      [32], [22.4], [#good[5.6]],
      [40], [33.1], [#good[5.7]],
      [48], [44.7], [#good[4.1]],
      [56], [59.6], [#good[5.9]],
    )
    - Slope ≈ 0: #good[flat].
    - Random IDs: 4.6, 7.8, > 8, > 8 — rising, non-monotone in $m$.
  ],
)

== But it rises again at n ≥ 72 (depth 6)

#grid(columns: (1fr, 1fr), column-gutter: 1em, align: horizon,
  [
    Final test pair, seed 0 / 1 (head dim. 8; † = never left always-yes on exact match):
    #tbl(columns: 4, align: center,
      table.header([$m$], [$n = 72$], [96], [128]),
      [8], [.896 / .909], [.831† / .799†], [–],
      [16], [.987 / .983], [.925† / .959], [.890† / .855†],
      [24], [–], [.930 / .948], [.916† / .964],
      [32], [–], [.968 / .968], [.930 / .929†],
    )
    $m^*$: 13 / 15 at 72; 28 / 16 at 96; > 32 / 23 at 128.
  ],
  [
    - Train = test everywhere: not data.
    - #bad[Not monotone in width:] $n = 128$ seed 1 passes at $m = 24$, fails at $m = 32$.
    - Every run: stuck at always-yes 4–7 epochs, then a *second plateau at 0.85–0.93*. Some
      runs leave it, some don't — that decides $m^*$, not width.
    - Head dim. 2 at $n = 72$ / 96: same rise (11.5 / > 12, > 16 / 15.2) — not the heads.
  ],
)

= Why does it stall?

== Hypotheses and first checks

#grid(columns: (1fr, 1fr), column-gutter: 1.2em,
  [
    *Proposed causes* #rc[(Claude Research report, 11-05)]
    - *H1* the bonus fades: neighbour share of attention 48 % at $n = 32$ → 17 % at 128
    - *H2* escape is luck
    - *H3* the learning rate runs out (cosine to 0)
    - *H4* unbalanced yes / no pairs
    - *H5* too few layers: 6 layers ≈ 12 hops
  ],
  [
    *Data:* pairs beyond 12 hops are 0.6 / 3.4 / 9.0 % at $n = 72$ / 96 / 128 — smooth, #warn[no 12-hop wall].

    *Trained models* (22 checkpoints):
    - $b$ is learned up to 3.3–4, still only 31–49 % neighbour attention at $n = 128$ vs
      55–63 % at 72: dilution is real.
    - Stalled models err 2–3× more at *every* hop distance.
    - Marker of a stalled run: 12–32 % wrong on pairs in different components (6 % escaped).
  ],
)

== Checkpoint surgery: change one thing, train a bit more

Stalled models resumed for 6000 steps (¼ of the budget). Final test pair:

#align(center, tbl(columns: 6, align: center,
  table.header([checkpoint (start)], [as is], [LR $10^(-3)$], [$b + 1.5$], [\+ 6 blocks], [blocks twice]),
  [$n = 128$, $m = 16$ (.855)], [.857], [.857], [.855], [.980], [#good[.994]],
  [$n = 128$, $m = 32$ (.930)], [.940], [.946], [.942], [.983], [#good[1.000]],
  [$n = 128$, $m = 32$ (.929)], [.932], [.951], [.932], [.995], [#good[1.000]],
  [$n = 96$, $m = 16$ (.925)], [.930], [.931], [.933], [.995], [#good[.998]],
))

#boxed[More training, a fresh learning rate, a bigger bonus: #bad[stuck]. Running the model's
*own 6 layers twice* — no new parameters — #good[unsticks every one].]

== From scratch: one change each, 4 seeds

#grid(columns: (1.2fr, 1fr), column-gutter: 1em, align: horizon,
  image("figures/width-runs-1106-ablation.png", width: 100%),
  tbl(columns: 3, align: (left, center, center),
    table.header([change], [$n = 96$ \ $m = 16$], [$n = 128$ \ $m = 24$]),
    [none], [1 / 4], [#bad[0 / 4]],
    [bonus start grows with $n$], [#bad[0 / 4]], [#bad[0 / 4]],
    [bonus × log $n$], [1 / 4], [#bad[0 / 4]],
    [gradient clipping], [3 / 4], [2 / 4],
    [WSD schedule], [3 / 4], [#bad[0 / 4]],
    [all three], [3 / 4], [1 / 4],
    [*depth 12*], [#good[4 / 4]], [3 / 4],
    [6 layers run twice], [3 / 4], [#good[4 / 4]],
  ),
)
#rc[Seeds reaching test pair 0.95. Plot: final test pair per seed, blue $n = 96$, red $n = 128$.]

== Conclusion

- The plateau is a *depth limit*. Each layer moves information about one step; past ~96 nodes
  six layers carry it too weakly, at every distance. Width cannot replace the missing layers.
- So the rise of edge-bias $m^*$ past $n approx 56$ measures *depth*, not width.
- Fits the theory: connectivity needs depth $Theta(log n)$ or width $approx n$.

#boxed[For this encoding, width must be measured with depth tied to $n$ (e.g. $ceil(log_2 n)$),
or as $m^*(n, L)$ on a small depth grid.]

== Width at depth 12

#grid(columns: (1fr, 1.05fr), column-gutter: 1em, align: horizon,
  [
    #tbl(columns: 4, align: center,
      table.header([$n$], [depth 6], [depth 12], [head dim.]),
      [32–56], [#good[4–6]], [–], [2],
      [72], [13 / 15], [#good[7.0 / 6.9]], [2],
      [96], [28 / 16], [#good[4.8 / 7.3]], [2],
      [128], [> 32 / 23], [#warn[≈ 14.5]], [8],
    )
    #rc[$m^*$, seed 0 / 1 (n = 128: 4 seeds, one passes at $m = 8$).]
    - #good[Depth was most of the rise:] $m^* approx 7$ up to $n = 96$.
    - #warn[$n = 128$ still ≈ 2× wider.]
    - #bad[Erratic:] pass at one width, fail at the next; same run twice: .995 / .938.
  ],
  fig("figures/width-runs-1107-mstar.png"),
)

== n = 128: is 12 layers short again?

#grid(columns: (1.15fr, 1fr), column-gutter: 1.2em, align: horizon,
  [
    Seeds reaching 0.95, out of 4 (head dim. 8):
    #tbl(columns: 4, align: center,
      table.header([depth], [$m = 8$], [12], [16]),
      [12], [1 / 4], [#bad[0 / 4]], [#good[4 / 4]],
      [16], [1 / 4], [2 / 4], [#warn[2 / 4]],
    )
    $m^*$: depth 12 ≈ 14.5, depth 16 ≈ 11 (median).
  ],
  [
    - 16 layers do #bad[not] bring $n = 128$ back to ≈ 7.
    - Deeper = less stable: two depth-16 runs reached .94 / .985 and #bad[fell back].
    - Real width growth, or unstable training hiding it? Next control: gradient clipping,
      head dim. 2.
  ],
)

== Next

- $n = 128$ at depth 12–16 with *gradient clipping* and head dim. 2, 4 seeds, readout =
  fraction passing per width: real width growth or unstable training?
- Looping (blocks run twice) as cheap depth: best in both ablations, no new parameters.
- Random IDs: test depth 12 / looping — the instability may be the same limit.
- $n = 72$ baseline: 256k graphs, more steps, $m = 96$–192 for a test $m^*$.
- Thesis ch. 3: add the auxiliary-loss result; update 1.59–1.90 → 1.53–1.97.
