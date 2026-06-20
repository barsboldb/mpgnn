#import "@preview/cetz:0.3.4"

#set page(paper: "a4", margin: (x: 2.1cm, y: 2.0cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10.5pt, lang: "en")
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.1")

#show heading.where(level: 1): it => {
  set text(size: 15pt)
  block(above: 1.1em, below: 0.6em)[#counter(heading).display() #it.body]
}
#show heading.where(level: 2): it => {
  set text(size: 12pt)
  block(above: 0.9em, below: 0.45em)[#counter(heading).display() #it.body]
}

// palette
#let cOur = rgb("#1a73e8")
#let cYeh = rgb("#e8710a")
#let cA = rgb("#1a73e8")
#let cB = rgb("#9334e6")
#let cGood = rgb("#137333")
#let cBad = rgb("#c5221f")
#let good(b) = box(fill: rgb("#e6f4ea"), inset: (x: 4pt, y: 1pt), radius: 2pt, text(cGood)[#b])
#let bad(b) = box(fill: rgb("#fce8e6"), inset: (x: 4pt, y: 1pt), radius: 2pt, text(cBad)[#b])

#align(center)[
  #text(size: 19pt, weight: "bold")[Two Connectivity Datasets, Two Different Worlds]
  #v(0.25em)
  #text(size: 11pt)[Our `connectedness_hard` vs. Yehudai et al. 2025 — what each one hides from the transformer]
  #v(0.2em)
  #text(size: 9.5pt, fill: luma(110))[diploma · #datetime.today().display()]
]

#v(0.4em)

Both datasets ask the *same yes/no question* — _is this graph connected?_ — yet one is solved
by a one-layer transformer at 100% and the other defeats it. This document compares them from
every angle: how the graphs are built, how they look, what statistics they carry, how they
become tokens, how those tokens are embedded, and — the crux — *what information each dataset
leaks to the model and what it hides*.

= The two tasks at a glance

#align(center)[
#table(
  columns: (auto, 1fr, 1fr),
  inset: 7pt, align: (left, left, left), stroke: 0.4pt + luma(180),
  table.header([], text(cOur)[*Our `connectedness_hard`*], text(cYeh)[*Yehudai connectivity*]),
  [Graph size], [variable, #good[n ∈ 12–24]], [fixed, #bad[n = 50] (every graph)],
  [Construction], [two dense blobs, hand-matched], [gnp / rgg / scale-free / sbm generators],
  [Connected class], [the two blobs + 1 bridge edge], [1 component (whole graph)],
  [Disconnected class], [the two blobs + 1 *intra*-blob edge], [2–3 components, any sizes],
  [Node features], [normalised degree (1 value)], [constant `x=1` (no signal)],
  [Balance], [500 / 500], [2500 / 2500],
)
]

The designs encode opposite *philosophies*. Yehudai samples graphs from natural random-graph
models and lets the two classes differ however they naturally do. We *engineer* the two classes
to be statistically identical in everything except the one fact that defines the label.

= What each dataset looks like

== Our data: two matched blobs, one edge apart

Every graph is two internally-dense blobs (each built as a Hamiltonian cycle + random chords, so
every node has degree ≥ 2 — *no isolated nodes ever*). The label is decided by a single edge:

#align(center)[
#cetz.canvas(length: 1cm, {
  import cetz.draw: *
  // pure: just compute the 5 ring points (no drawing here)
  let pts(cx, cy) = range(5).map(i => {
    let a = 90deg + i * 72deg
    (cx + 0.5*calc.cos(a), cy + 0.5*calc.sin(a))
  })
  let blob(p, col) = {
    for i in range(5) { line(p.at(i), p.at(calc.rem(i+1,5)), stroke: 0.6pt) }
    line(p.at(0), p.at(2), stroke: 0.6pt)
    line(p.at(1), p.at(3), stroke: 0.6pt)
    for q in p { circle(q, radius: 0.11, fill: col, stroke: none) }
  }
  // connected
  let a1 = pts(0, 0); let b1 = pts(2.6, 0)
  blob(a1, cA); blob(b1, cB)
  line(a1.at(0), b1.at(2), stroke: (paint: cGood, thickness: 1.5pt))
  content((1.3, -1.25), text(8pt)[#good[label 1] = connected])
  content((1.3, -1.75), text(7.5pt, cGood)[+1 *bridge* edge])

  // disconnected
  let a2 = pts(6.4, 0); let b2 = pts(9.0, 0)
  blob(a2, cA); blob(b2, cB)
  line(a2.at(0), a2.at(3), stroke: (paint: cBad, thickness: 1.5pt))
  content((7.7, -1.25), text(8pt)[#bad[label 0] = disconnected])
  content((7.7, -1.75), text(7.5pt, cBad)[+1 *intra*-blob edge])
})
]

The two classes have the *same number of edges* (we add exactly one edge to each), the *same
degree distribution*, and the *same blob structure*. The connected one routes its extra edge
*between* blobs; the disconnected one routes it *inside* a blob. Spot-the-difference at the level
of the whole graph.

== Yehudai's data: natural graphs that differ in bulk

Their connected graphs are drawn from dense random-graph models; their disconnected graphs are a
*union of 2–3 independent* such graphs. Nothing is matched — the classes differ in whatever way
the generators make them differ.

#align(center)[
#cetz.canvas(length: 1cm, {
  import cetz.draw: *
  // connected: one dense cloud
  circle((0, 0), radius: 0.95, fill: cYeh.lighten(75%), stroke: cYeh)
  for i in range(11) {
    let a = i * 32.7deg
    let r = 0.6
    circle((r*calc.cos(a), r*calc.sin(a)), radius: 0.07, fill: cYeh, stroke: none)
  }
  content((0, -1.5), text(8pt)[#good[label 1] · 1 dense piece])
  content((0, -2.0), text(7.5pt)[≈ 285 edges, deg ≈ 11])

  // disconnected: 3 separate clouds
  let cl(cx, cy, r, k) = {
    circle((cx, cy), radius: r, fill: cYeh.lighten(75%), stroke: cYeh)
    for i in range(k) {
      let a = i * (360deg / k)
      circle((cx + 0.55*r*calc.cos(a), cy + 0.55*r*calc.sin(a)), radius: 0.06, fill: cYeh, stroke: none)
    }
  }
  cl(5.0, 0.5, 0.55, 5)
  cl(6.4, -0.4, 0.5, 4)
  cl(5.3, -1.0, 0.42, 3)
  content((5.7, -2.0), text(7.5pt)[≈ 143 edges, deg ≈ 6])
  content((5.7, -1.5), text(8pt)[#bad[label 0] · 2–3 pieces])
})
]

Because a disconnected graph is several *smaller* pieces, it ends up with *far fewer edges* and
*lower average degree* than a single connected graph of the same node count. That difference is
not hidden — it is broadcast in every bulk statistic of the graph.

= The statistics — measured, not assumed

All numbers below are measured on the actual cached datasets (1000 of ours, 4000 of theirs).

#align(center)[
#table(
  columns: (auto, auto, auto, auto, auto),
  inset: 6pt, align: (left, center, center, center, center), stroke: 0.4pt + luma(180),
  table.header([Statistic], [*Our* conn.], [*Our* disc.], [*Yeh.* conn.], [*Yeh.* disc.]),
  [mean \#edges], [39.7], [40.6], [285.0], [143.0],
  [mean degree], [4.49], [4.52], [11.40], [5.72],
  [min degree], [2], [2], [1], [0],
  [mean \#components], [1.00], [2.00], [1.00], [3.10],
)
]

Read the columns in pairs. *Ours* (cols 2–3): connected and disconnected are nearly identical on
every bulk statistic — 39.7≈40.6 edges, 4.49≈4.52 degree, both min-degree 2. *Theirs* (cols 4–5):
connected has *double* the edges and *double* the degree of disconnected, and disconnected graphs
contain isolated nodes (min degree 0).

== How far a single number gets you

For each dataset, take the best possible *single-threshold* classifier on one bulk statistic:

#align(center)[
#cetz.canvas(length: 1cm, {
  import cetz.draw: *
  let bar(x, h, col, lbl, val) = {
    rect((x, 0), (x + 0.8, h*3), fill: col, stroke: none)
    content((x + 0.4, h*3 + 0.25), text(8pt)[#val])
    content((x + 0.4, -0.3), text(7pt)[#lbl])
  }
  // baseline + chance line
  line((-0.3, 0), (9.6, 0), stroke: 0.6pt)
  line((-0.3, 1.5), (9.6, 1.5), stroke: (paint: luma(150), dash: "dashed"))
  content((9.9, 1.5), anchor: "west", text(7pt, luma(110))[chance 0.50])

  content((1.3, 3.0), text(8.5pt, weight: "bold", cOur)[Our hard])
  bar(0.2, 0.519, cOur, "edge-count", "0.52")
  bar(1.3, 0.519, cOur, "mean-deg", "0.52")
  bar(2.4, 0.500, cOur, "isolated?", "0.50")

  content((6.6, 3.0), text(8.5pt, weight: "bold", cYeh)[Yehudai])
  bar(5.5, 0.769, cYeh, "edge-count", "0.77")
  bar(6.6, 0.769, cYeh, "mean-deg", "0.77")
  bar(7.7, 0.507, cYeh, "isolated?", "0.51")
})
]

A single scalar — total edge count, or mean degree — already classifies #bad[77%] of Yehudai's
graphs, but only #good[52%] of ours (chance). Our matched construction drives every cheap
statistic to the coin-flip line; theirs leaves the answer lying in plain sight.

= How a graph becomes tokens

A transformer reads a list of vectors. Three ways to flatten a graph into tokens are relevant
here; all pad to the max node count.

#align(center)[
#cetz.canvas(length: 1cm, {
  import cetz.draw: *
  let tok(x, y, w, body, col) = {
    rect((x, y - 0.3), (x + w, y + 0.3), radius: 2pt, fill: col, stroke: 0.5pt)
    content((x + w/2, y), text(7pt, white)[#body])
  }
  // adj_rows
  content((-0.2, 1.5), anchor: "west", text(8pt, weight: "bold")[adj rows])
  content((-0.2, 1.05), anchor: "west", text(6.5pt, luma(90))[1 token / node])
  tok(2.0, 1.3, 1.5, [$"row"_1$ = 0101…], cOur)
  tok(3.7, 1.3, 1.5, [$"row"_2$ = 1000…], cOur)
  tok(5.4, 1.3, 0.7, […], cOur)
  tok(6.3, 1.3, 1.5, [$"row"_n$ = 0010…], cOur)

  // edge_list
  content((-0.2, 0.2), anchor: "west", text(8pt, weight: "bold")[edge list])
  content((-0.2, -0.25), anchor: "west", text(6.5pt, luma(90))[1 token / edge])
  tok(2.0, 0.0, 1.9, [onehot(u)‖onehot(v)], cYeh)
  tok(4.0, 0.0, 1.9, [onehot(u)‖onehot(v)], cYeh)
  tok(6.0, 0.0, 0.7, […], cYeh)

  // node_edge
  content((-0.2, -1.1), anchor: "west", text(8pt, weight: "bold")[node+edge])
  content((-0.2, -1.55), anchor: "west", text(6.5pt, luma(90))[verts+edges+task])
  tok(2.0, -1.3, 0.9, [$v_1$], cA)
  tok(3.0, -1.3, 0.9, [$v_2$], cA)
  tok(4.0, -1.3, 1.0, [$e_(u v)$], cB)
  tok(5.1, -1.3, 1.0, [$e_(u v)$], cB)
  tok(6.2, -1.3, 1.0, [task], luma(90))
})
]

- *adj rows* — token $i$ is node $i$'s row of the adjacency matrix. Yehudai's winning choice.
- *edge list* — token per edge, each the concatenated one-hot identities of its two endpoints.
- *node+edge* — our Sanford-style sequence: vertex tokens, edge tokens, and a task token.

= How tokens get embedded — and why it matters

After tokenizing, the model applies a linear embedding to each token and (for `adj_rows`)
*mean-pools* over the node tokens before the classifier:

$ "graph vector" = 1/n sum_(i=1)^n W dot "row"_i = W dot underbrace((1/n sum_i "row"_i), "average adjacency row") $

Here is the punchline. The average adjacency row, summed, *is the mean degree*:

$ sum_j (1/n sum_i A_(i j)) = 1/n sum_i underbrace((sum_j A_(i j)), "degree of node "i) = "mean degree". $

So a single linear layer followed by mean-pooling can read out *mean degree directly* — no
attention, no reasoning. On Yehudai's data that scalar already separates the classes 77% (degree
11.4 vs 5.7), and a little attention closes the rest. The 1-layer adj_rows transformer isn't
*reasoning about connectivity* — it is #bad[measuring how dense the graph is].

On our data the same readout returns ≈4.5 for *both* classes. Mean-pooling gives the classifier
nothing. To separate our classes the model must look at *which specific node connects to which* —
it must actually trace whether a path crosses between the blobs. That is real global reasoning,
and it is exactly what we wanted to force.

= What each dataset hides from the transformer

#align(center)[
#cetz.canvas(length: 1cm, {
  import cetz.draw: *
  // Yehudai: leaks
  content((1.6, 2.5), text(9pt, weight: "bold", cYeh)[Yehudai: the answer leaks out])
  let leak(y, lbl) = {
    rect((0, y - 0.25), (3.2, y + 0.25), radius: 3pt, fill: cYeh.lighten(80%), stroke: cYeh)
    content((1.6, y), text(7.5pt)[#lbl])
    line((3.2, y), (4.1, y), mark: (end: ">"), stroke: cGood)
  }
  leak(1.7, [edge count]); leak(1.0, [mean degree]); leak(0.3, [isolated nodes])
  content((4.2, 1.0), anchor: "west", box(fill: rgb("#e6f4ea"), inset: 4pt, radius: 2pt,
    text(8pt, cGood)[label readable\ from bulk stats]))

  // Ours: sealed
  content((1.6, -1.0), text(9pt, weight: "bold", cOur)[Ours: only structure carries it])
  let seal(y, lbl) = {
    rect((0, y - 0.25), (3.2, y + 0.25), radius: 3pt, fill: luma(235), stroke: luma(160))
    content((1.6, y), text(7.5pt, luma(110))[#lbl])
    line((3.2, y), (3.9, y), stroke: cBad)
    line((3.7, y + 0.18), (3.9, y - 0.18), stroke: (paint: cBad, thickness: 1.2pt))
    line((3.7, y - 0.18), (3.9, y + 0.18), stroke: (paint: cBad, thickness: 1.2pt))
  }
  seal(-1.8, [edge count]); seal(-2.5, [mean degree]); seal(-3.2, [isolated nodes])
  content((4.2, -2.5), anchor: "west", box(fill: rgb("#fce8e6"), inset: 4pt, radius: 2pt,
    text(8pt, cBad)[label only in *which*\ nodes connect]))
})
]

#v(0.3em)

#block(fill: luma(245), inset: 9pt, radius: 4pt)[
  *In one line.* Yehudai's dataset lets the model answer "connected?" by asking "how dense?".
  Ours removes density, degree, and isolated-node cues entirely, so the only remaining signal is
  the actual reachability structure — which a position-based tokenization (adjacency rows,
  one-hot edge lists) cannot read off without solving the problem.
]

= Why this matters for the thesis

The empirical chain (see #link("yehudai-empirical.md")[`yehudai-empirical.md`] and
#link("tokenization.typ")[`tokenization.typ`]):

+ *Our model is not the bottleneck.* Our adjacency-rows transformer scores #good[1.00] on
  Yehudai's data (by epoch 4) — it reproduces their result exactly.
+ *The same model scores ≈0.6–0.7 on ours*, even with size fixed.
+ Therefore the gap is the *dataset*. Yehudai's benchmark is separable by bulk statistics that
  flow straight through mean-pooling; ours is engineered so they don't.

This makes `connectedness_hard` a *strictly harder probe of global reasoning*: it is the
connectivity question with every cheap shortcut sealed off, leaving only the genuinely
parallel/depth-bounded computation that the theory (Sanford et al. 2024a) is actually about.

#align(center)[
#table(
  columns: (1fr, auto, auto),
  inset: 6pt, align: (left, center, center), stroke: 0.4pt + luma(180),
  table.header([Model · data], [Tokenization], [Test acc]),
  [our GNN · Yehudai connectivity], [adj rows], text(cGood)[*1.00*],
  [our GNN · our hard (fixed n=20)], [adj rows], [≈0.70],
  [our GNN · our hard (variable n)], [adj rows], [≈0.59],
  [our transformer · our hard], [node+edge], [0.50 (ln 2 plateau)],
)
]
