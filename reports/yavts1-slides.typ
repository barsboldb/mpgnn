#import "@preview/touying:0.6.1": *
#import themes.metropolis: *

// Явц 1 progress review (week 6). Content: reports/width-progress.pdf.
#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  config-info(
    title: [Analyzing the Effect of Network Width on Graph Transformers],
    subtitle: [Progress review 1 (Явц 1) — weeks 1–5],
    author: [Barsbold Bayarerdene],
    date: [October 2026],
    institution: [Diploma research],
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

#title-slide()

== The question

*How does a graph transformer's width — its embedding dimension $m$ — decide what it can
learn, and how must width grow with graph size $n$?*

#v(0.3em)

- *Theory* gives width bounds for _representing_ graph algorithms: one attention layer
  needs width ≈ (what each token must gather) — q-sparse averaging, Sanford et al. 2023;
  connectivity at depth $log N$ needs width only just above $sqrt(N)$, with no extra
  tokens ($N$ = vertices + edges; Sanford et al. 2024).
- *Nobody has measured* the width a transformer needs to _learn_ a graph task, how it
  scales with $n$, or how it trades against depth.

#boxed[This thesis measures the *critical width* $m^*$ — the smallest width that learns
the task — and how it moves with $n$, depth, data and training.]

== Set-up

#grid(columns: (1.1fr, 1fr), column-gutter: 1.2em,
  [
    - *Task:* connectivity matrix — for every pair of nodes, are they in the same
      component? (dense supervision; a single yes/no per graph does not train)
    - *Model:* transformer encoder, one token per node (its adjacency row), pairwise
      read-out — task, tokens and read-out from Ye et al. (2026)
    - *Width* $m$ = embedding dimension; the whole model scales with it: $m\/8$
      attention heads of size 8, feed-forward layer $4m$
    - *Metric:* pair accuracy (fraction of pairs right); trivial "all connected" ≈ 0.75
  ],
  [
    - *Critical width $m^*$:* where accuracy crosses 0.95, per seed
    - *Protocol:* choices on validation, results on test; 2–3 seeds
    - *Compute:* sharded sweeps on Kaggle GPUs; ≈ 770 training runs so far
  ],
)

= Part 1 — Measuring width honestly

== Lesson 1: a fixed learning rate fakes a width effect

#grid(columns: (1.15fr, 1fr), column-gutter: 1em, align: horizon,
  image("figures/width-q1b-lr.png", width: 100%),
  [
    - One fixed rate: wide models look up to *0.4 worse*
    - Tuned per width: widths 8–64 all ≈ 0.97–0.98
    - The best rate *falls as width grows*

    #v(0.4em)
    #rc[$n = 24$, depth 2, 2000 graphs, 3 seeds (Q1/Q1b, 192 runs)]
  ],
)

== Lesson 2: width has a floor and a data ceiling

#grid(columns: (1.15fr, 1fr), column-gutter: 1em, align: horizon,
  image("figures/width-q1c-data.png", width: 100%),
  [
    - *Floor:* below ≈ 4–8 the model cannot even fit
    - *Ceiling:* past it, extra width memorizes
    - More data moves the ceiling: at 8000 graphs every $m >= 16$ reaches ≥ 0.99

    #v(0.4em)
    #rc[same steps for every data size (Q1c, 216 runs)]
  ],
)

== Lesson 3: the benchmark was leaking

The first dataset could be solved without tracing a single path:

#align(center, tbl(
  columns: 2, align: (left, left),
  table.header([*shortcut*], [*evidence*]),
  [node indices reveal the components], [blobs always nodes $[0, n_a)$ and $[n_a, n)$ — 200 / 200 graphs],
  [graph statistics reveal the label], [degree + short-cycle counts predict it with 0.68–0.90 accuracy],
  [small graphs are local], [a 6-hop check is exact on 60–100 % of graphs],
))

#v(0.3em)

*Fix — the `swap` generator:* the two classes differ only by swapping two far-apart
edges (the two-cycles-vs-one "cycle task" of Abbe et al., NeurIPS 2024, plus chords
and a dense target); node labels shuffled. Audit: statistics #good[at chance
(0.49–0.53)], 4-hop checks #good[score 0], diameter grows ≈ $log n$.

#rc[On the leaky data width 8 sufficed at $n = 24$ — the shortcuts hid the width requirement.]

= Part 2 — How width must grow

== Width decides whether learning starts at all

#grid(columns: (1.15fr, 1fr), column-gutter: 1em, align: horizon,
  image("figures/width-q3trim-onset.png", width: 100%),
  [
    - $n = 32$: learns from $m ≈ 16$
    - $n = 64$: only $m >= 64$ leave the trivial predictor
    - $n = 128$: *nothing* up to $m = 128$ learns

    #v(0.4em)
    #rc[depth $ceil(log_2 n)$, 32 000 graphs, 12 000 steps]
  ],
)

== Main result: the critical width grows ≈ $n^2$

#grid(columns: (1.15fr, 1fr), column-gutter: 1em, align: horizon,
  image("figures/width-q3fine-scaling.png", width: 100%),
  [
    #tbl(
      columns: 3, align: (center, center, center),
      table.header([$n$], [$m^*$], [$m^* \/ n$]),
      [32], [21.9], [0.68], [40], [30.2], [0.76],
      [48], [51.1], [1.07], [56], [57.9], [1.03],
    )
    #v(0.3em)
    $m^* prop n^(1.85)$ #rc[(90 % range 1.68–2.03)]

    #rc[depth 6, 128 000 graphs, 24 000 steps; table: rate tuned at $n$ = 40 / 56 (green line in the chart: one rate, $n^(1.94)$)]
  ],
)

== The result survives five checks

#align(center, tbl(
  columns: 3, align: (left, left, left),
  table.header([*could it be…*], [*test*], [*outcome*]),
  [too little data?], [32k → 128k graphs], [gap closes, #good[slope stays ≈ 2]],
  [too little training?], [2× the steps], [#good[critical width unchanged]],
  [the learning rate?], [second rate, tuned per width], [slope 1.94 → #good[1.85]],
  [the input encoding?], [fixed 96-wide random node IDs], [#good[still ≈ $n^2$] (2.03)],
  [noise?], [2–3 seeds, per-seed crossings], [#good[seeds within ± 3–5]],
))

#v(0.4em)

#boxed[Over $n = 32$–$56$, the width a transformer needs to *learn* connectivity grows
roughly as $n^2$ — while theory says this depth needs far less width to *represent* it.]

== Depth trades for width only mildly

#grid(columns: (1.15fr, 1fr), column-gutter: 1em, align: horizon,
  image("figures/width-q7depth.png", width: 100%),
  [
    - *2 layers learn* at $m ≈ 1.1 n$ — on graphs of diameter ≈ 10, so the model does
      not trace paths hop by hop
    - 2 → 6 layers saves only ≈ ⅓ of the width; 8 = 6
    - Matches Yehudai et al. 2025: linear width ⇒ constant depth

    #rc[$n = 40$, 60 runs]
  ],
)

= What it means

== Contributions so far

+ *A protocol for measuring width.* Learning rate, data and training budget each
  interact with width enough to fake or hide an effect; we control all three.
+ *An audited benchmark.* Every heuristic we could find sits at or below chance on
  `swap`; the leaky data had hidden the width requirement.
+ *A measured scaling law.* The critical width to learn connectivity grows ≈ $n^(1.9)$
  over $n = 32$–$56$, robust to five confounds.
+ *Representation ≠ learning.* Theory: depth $log N$ needs width just above $sqrt(N)$
  to represent connectivity. Measured: learning needs ≈ $n^2$. Depth helps only mildly.

== Limitations

- Graph sizes span less than a factor of two ($n = 32$–$56$); larger $n$ did not learn
  at any affordable width
- Two seeds for most scaling points; local slopes vary (≈ 1.2–2.9)
- One task (connectivity) and one architecture family so far
- `swap` diameter grows with $log n$: path length and node count still rise together
- Compute: Kaggle's 30 GPU-hours per week caps each sweep

== Plan for weeks 6–7

#align(center, tbl(
  columns: 3, align: (left, left, center),
  table.header([*item*], [*question*], [*when*]),
  [Q7c], [does the *depth-2* critical width grow linearly or ≈ $n^2$?], [week 6],
  [Q7b], [fix $n$, vary diameter: is path length or node count driving the growth?], [week 6],
  [Q4], [Graphormer-style distance bias — on a task it does not give away], [weeks 6–7],
  [Q2], [task hierarchy: retrieval < connectivity < shortest path], [week 7],
))
