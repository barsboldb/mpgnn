#import "@preview/touying:0.6.1": *
#import themes.metropolis: *

#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  config-info(
    title: [What the Theory Predicts, and What the Data Actually Rewards],
    subtitle: [Graph connectivity as a probe of transformer reasoning — progress report],
    author: [Barsbold Bayarerdene],
    date: datetime.today().display("[month repr:long] [day], [year]"),
    institution: [Diploma research],
  ),
)

#set text(size: 17pt)
#show raw: set text(size: 0.84em)

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

#title-slide()

== Where I am

*Done.* Reproduced Yehudai's connectivity experiment; read and annotated all three
papers; built a leak-audited dataset; ran ~150 experiments.

#v(0.4em)

*The finding that reorganised the project.* Every benchmark I touched — including two
from published papers — turned out to be solvable by a statistic that has nothing to do
with connectivity.

#v(0.4em)

*What I need guidance on.* There is a gap between what the theory says a transformer
*can represent* and what my training runs actually *find*. I am not sure which side of
it to make the thesis.

= Part 1 — What the papers claim

== Yehudai et al. — fix the depth, grow the width

*Depth-Width Tradeoffs for Transformers on Graph Tasks* (NeurIPS 2025).
A plain transformer encoder, no message passing; the graph is flattened into tokens.

#align(center, tbl(
  columns: 3, align: (left, left, center),
  table.header([*tokenization*], [*one token =*], [*seq len*]),
  [`adj_rows`], [node's adjacency row ⊕ features], [$n$],
  [`edge_list`], [`onehot(u)⊕x_u ‖ onehot(v)⊕x_v`], [\#edges],
  [`lap_full`], [Laplacian eigenvectors ⊕ features], [$n$],
))

#align(center, block(inset: 11pt, fill: luma(245), radius: 4pt, width: 92%)[
  Prior work fixed sub-linear width and asked for depth. They invert it: with *linear*
  width in $n$, *constant* depth suffices — including for #good[graph connectivity].
])

#rc[#warn[Clayton Sanford co-authors both this and the next paper] — a follow-up, not a rival.]

== The width hierarchy, and how width buys connectivity

#align(center, tbl(
  columns: 3, align: (left, center, left),
  table.header([*task*], [*width at fixed depth*], [*result*]),
  [node degree], [constant], [local, trivial],
  [2-cycles, bounded degree $d$], [$O(d log n)$], [Thm 4.4, optimal up to logs],
  [1 vs. 2 cycles], [$O(n)$, 2 layers], [Thm 4.1],
  [connectivity], [$O(n "polylog" n)$], [§4.1, #warn[proof sketch only]],
  [Eulerian cycle verification], [near-quadratic], [Thm 5.1, conditional],
))

#v(0.3em)

Connectivity via *linear sketching* (Ahn et al. 2012): compress each adjacency row to
$O("polylog" n)$, pack all tokens into one, let the MLP finish. High-probability, not
deterministic. Alternative route — Thm 4.3: depth $O(L)$, $m=O(n)$ computes $A^L$.

#rc[Thm 4.2 ($m p H L = Ω(n)$ for 2-cycles on adjacency rows) is #good[unconditional].]

== Their empirical claim

Fixed budget (~100k params), $n = 100$, adjacency rows, splitting the budget as
`(1,125)`, `(2,89)`, `(4,63)`, `(8,45)`, `(10,40)`:

- loss and test accuracy #good[consistent across every configuration]
- shallow-and-wide #good[significantly faster] to train and to run

#v(0.4em)

Their headline is not "wide is more accurate" — it is *"wide is equally accurate and
much cheaper"*.

#v(0.4em)

#warn[Note the discipline:] a *parameter-matched* sweep. My own depth×width grid lacks
exactly this, and I come back to it at the end.

== Sanford et al. 2024 — which scaling regime?

*Understanding Transformer Reasoning Capabilities via Graph Algorithms* (NeurIPS 2024).

#align(center, tbl(
  columns: 3, align: (left, center, left),
  table.header([*regime*], [*depth / width*], [*what lives here*]),
  [D1], [$L=1$, $m=O(log N)$], [retrieval],
  [LogDepth], [$L=O(log N)$, $m=O(N^ε)$], bad[connectivity],
  [LogDepthWide], [$L=O(log N)$, $m=O(N^(1/2+ε))$], [shortest path],
))

#v(0.3em)

*Theorem 1 is the engine.* Any $R$-round MPC protocol with $O(N^δ)$ local memory is
simulated by a transformer of depth $L = O(R)$, width $m = O(N^(δ+ε))$ — MLPs do local
computation, one attention layer does one round of communication.

#v(0.2em)

So tasks sort by *round complexity*, not apparent difficulty.

#rc[$N$ is the input *sequence length* $O(|V|+|E|)$, not the node count. $ε>0$ is any fixed constant.]

== What Sanford says about connectivity

#align(center, tbl(
  columns: 2, align: (left, left),
  table.header([*result*], [*content*]),
  [Theorem 2], [connectivity *is* in LogDepth — the construction exists],
  [Theorem 3], [log depth is *necessary* (conditional on their Conjecture 13)],
  [Theorem 6], [*one layer provably cannot* do it without width $m H = tilde(Omega)(N)$],
))

#v(0.4em)

§3.4 is the one exception: depth drops to $O(log log N)$ — but only with
$O(α N^(1-ε))$ *pause tokens*.

#v(0.3em)

#align(center, block(inset: 11pt, fill: luma(245), radius: 4pt, width: 92%)[
  Depth goes below logarithmic *only* when the computation is moved onto the tape.
  Hold that thought — it is the hinge of this presentation.
])

== Ye et al. 2026, and how the three fit together

*Transformers Provably Learn Algorithmic Solutions for Graph Connectivity, But Only
with the Right Data* (ICML 2026). A depth-$L$ model has reachability capacity $3^L$;
their lever is to train only on graphs *within* capacity, starving the heuristic channel.

#align(center, tbl(
  columns: 3, align: (left, left, left),
  table.header([*paper*], [*fixes*], [*connectivity verdict*]),
  [Sanford 2024], [width $O(N^ε)$], [needs depth $O(log N)$; 1 layer impossible],
  [Yehudai 2025], [depth $O(1)$], [needs width $O(n "polylog" n)$],
  [Ye 2026], [architecture], [learnable *only with the right training data*],
))

#align(center, block(inset: 10pt, fill: luma(245), radius: 4pt, width: 92%)[
  Connectivity is *expensive* — you pay in depth, or in width, or you do not get it.
  All three treat the resource as something the *model* must spend.
])

= Part 2 — Reproducing Yehudai

== It reproduces, and the control clears my pipeline

$n = 50$, 1 layer, width 64, 100 train graphs, 100 epochs:

#align(center, tbl(
  columns: 3, align: (left, left, center),
  table.header([*model*], [*data*], [*test*]),
  [theirs, `adj_rows`], [Yehudai $n=50$], good[1.000],
  [theirs, `edge_list`], [Yehudai $n=50$], [0.926],
  [*mine*, 1 layer, 15,874 params], [Yehudai $n=50$], good[1.000 @ epoch 4],
  [*mine*], [my `connectedness_hard`], bad[≈ 0.59],
))

#v(0.4em)

My pipeline beats their own implementation on their data — so everything that follows is
*about the data*, not about my code.

#v(0.3em)

#warn[Already suspicious:] *one layer* solves connectivity at $n=50$. If the task needed
$O(log N)$ depth, that should not happen.

= Part 3 — The Laplacian result

== The prediction, stated sharply

Both papers rule out the same corner, from opposite directions:

#align(center, tbl(
  columns: 2, align: (left, left),
  table.header([*source*], [*what it forbids*]),
  [Sanford, Thm 6], [one layer needs width $m H = tilde(Omega)(N)$],
  [Yehudai, §4.1 + Thm 4.2], [fixed depth needs width $O(n "polylog" n)$],
))

#v(0.5em)

At $n ≈ 20$ both say a one-layer model needs width on the order of $n$ at minimum, and
realistically in the hundreds.

#v(0.5em)

So — one layer, width 8. What should happen?

== What actually happens

`connectedness_hard`: degree- and edge-count-matched, classes differing by *one bridge
edge*. Left, Laplacian as an add-on; right, Laplacian as the node features:

#align(center, tbl(
  columns: 6, align: (center, center, center, center, center, center),
  table.header([*depth*], [*width*], [*LPE*], [*test*], [*width (LPE-as-features)*], [*test*]),
  [1], [64], [0], bad[0.545], [64], good[1.000],
  [1], [64], [16], good[1.000], [32], good[1.000],
  [1], [24], [16], good[1.000], [16], good[1.000],
  [1], [16], [8], good[1.000], [8], good[1.000],
  [1], [8], [4], good[1.000], [4], [0.915],
))

#v(0.3em)

*Depth 1, width 8, four Laplacian dimensions* — perfect accuracy on the task a depth-8
model cannot beat 0.64 on without them. Degradation only begins at width 4.

#rc[`results/20260619_221812_*`, `_222010_*`, `20260621_011108_*`, `20260623_110250_*` … `_111551_*`]

== Why: the answer is already in the input

$L = D - A$ has one zero eigenvalue *per connected component*, and its zero-eigenvalue
eigenvectors are *indicator vectors on those components*. So the encoding hands over the
component count and the membership directly.

#align(center, image("figures/lap-features.png", height: 7.3cm))

#text(size: 13pt)[
  The $[n times 8]$ matrix the model receives, one row per node. *Disconnected* graphs
  (bottom) show #bad[complementary zero-blocks]; *connected* graphs (top) have none.
]

== And a single scalar separates the classes

#align(center, image("figures/lap-spectrum-top.png", width: 90%))

#text(size: 14pt)[
  Left: the smallest eigenvalue the model sees. Connected graphs sit at
  #good[0.03–0.12], disconnected at #bad[0.25–0.80] — almost no overlap, on the dataset
  whose classes differ by *one edge*. Right: multiplicity of eigenvalue 0 is *exactly*
  the number of components, 500 / 500 on each side.
]

== Credit where it is due — and what is mine

I did not discover this. Yehudai et al., Appendix A:

#align(center, block(inset: 11pt, fill: rgb("#fdf5ea"), radius: 4pt, width: 94%)[
  #text(size: 15pt)[
    "the tokenization *trivializes the connectivity task* because a graph is disconnected
    if and only if its second-smallest eigenvalue is zero; transformers with the
    node-adjacency tokenization require either depth $Ω(log n)$ or width $Ω(n)$ to solve
    the same problem."
  ]
])

#v(0.3em)

They state it as *theory*, and their own encoding experiments (§6.3) are on molecular
property prediction, #warn[not on connectivity]. Mine is the *measurement*: it holds on
an adversarial dataset, at 4 eigenvector dimensions and width 8, with a matched no-LPE
control at 0.545.

#v(0.2em)

#rc[A confirmed prediction is worth reporting — but it is a confirmation, and I present it as one.]

== It does not falsify the theorem — it relocates the work

#align(center, block(inset: 13pt, fill: luma(245), radius: 4pt, width: 94%)[
  #text(size: 18pt)[
    Eigendecomposition is $O(n^3)$ of *preprocessing*. The $O(log N)$ depth has not been
    avoided — it has been *paid for outside the model*. Same structure as Sanford's own
    §3.4 pause-token exception.
  ]
])

#v(0.4em)

An encoding is not a neutral input format. It is an oracle that has already done part of
the task:

#align(center, tbl(
  columns: 3, align: (left, left, center),
  table.header([*encoding*], [*what it hands over*], [*depth needed*]),
  [`lap_full` / LPE], [transitive closure — the components themselves], [1],
  [SPD / distance bias], [all-pairs distances], [low],
  [`adj_rows`], [a consistent layout — *only if $n$ is fixed*], [depends],
  [`edge_list`], [nothing], [$ln 2$ plateau],
))

= Part 4 — Every benchmark leaked

== Two leaks of my own

*The ER dataset I started with.* `connectedness` samples near $p ≈ log(n)/n$, where
disconnection is almost always caused by *isolated vertices*:

#align(center, tbl(
  columns: 2, align: (left, center),
  table.header([*trivial rule*], [*accuracy*]),
  [majority class], [0.679],
  [connected ⇔ min degree ≥ 1], bad[0.982],
))

#v(0.3em)

*My token serialization.* `diameter_controlled` gave connected graphs exactly 23 edges
and disconnected ones 22 — so the prompt was two tokens longer and the positional
embedding read the answer off #bad[sequence length]. Invisible to every GNN baseline.

#rc[Seed 42, 1000 graphs · CHANGELOG 06-19 · serialization leak fixed in `ed47ddd`]

== Leak in Yehudai's own benchmark

#warn[They designed against this.] Appendix E: "to avoid a correlation between
connectivity and edge probability as exists in Erdős–Rényi graphs", they mix four
generators — ER, random geometric, Barabási–Albert, and SBM.

#v(0.3em)

It leaks anyway, because the *classes* remain unmatched within each generator:

#align(center, tbl(
  columns: 3, align: (left, center, center),
  table.header([*single-scalar rule*], [*on Yehudai's data*], [*on mine*]),
  [total edge count], bad[0.77], good[0.52],
  [mean degree], bad[0.77], good[0.52],
  [any isolated node?], [0.51], [0.51],
))

#align(center, block(inset: 9pt, fill: rgb("#fdf5ea"), radius: 3pt, width: 90%)[
  Their dataset lets a model answer *"is it connected?"* by asking *"how dense is it?"*
])

== Leak in Ye's setup, caught by an adversarial probe

Two probes: *two disconnected cliques*, and *the same two cliques plus one bridge*.
Exact-match on the reachability matrix:

#align(center, tbl(
  columns: 5, align: (left, center, center, center, center),
  table.header([*regime*], [*train EM*], [*ER test*], [*2 cliques, disc.*], [*+ bridge*]),
  [raw], [1.000], [0.578], bad[0.000], good[1.000],
  [within-capacity], [1.000], [0.698], bad[0.000], good[1.000],
))

#v(0.3em)

0.0 on disconnected and 1.0 on connected is the signature of a model that always answers
*"everything is reachable"* — reading #bad[density], not connectivity. Their capacity
lever does not change it.

#v(0.2em)

#rc[Sanford's own validation (GraphQA) uses 5–20-node ER graphs — the family Leak 1
convicts — and his Appendix E concedes the graphs "do not resemble the large-diameter
worst-case instance" his theorems are about.]

== The catalogue — including my own dataset

#align(center, tbl(
  columns: (auto, 1fr, auto),
  align: (left, left, center),
  table.header([*where*], [*the shortcut*], [*strength*]),
  [my ER dataset], [is there a degree-0 node?], bad[0.982],
  [Yehudai's benchmark], [how many edges? / mean degree], bad[0.77],
  [Laplacian encoding], [components are literally in the input], bad[1.000],
  [my serialization], [prompt length differs by class], bad[1.000],
  [Ye's setup], [always answer "reachable"; density heuristic], bad[0.0 / 1.0],
  [*my `connectedness_hard`*], [local bridge-detection], bad[0.975],
))

#v(0.3em)

The last row is mine and I have to own it: a 3-layer GIN with degree features scores
*0.975* — at depth 3 on graphs of diameter 4–8, so it cannot be global reasoning. My
construction killed min-degree and mean-degree, but not local bridge-detection.

#rc[`results/20260624_183646_*` · CHANGELOG 06-25]

= Part 5 — What survives the audit

== With the leaks sealed, nothing learns the binary task

`connectedness_hard`, binary label, no Laplacian.

- ~20 runs: depths 1–8, widths 2–128, two encodings, up to 500 epochs
- *generalization* fails, not capacity — same model, varying only training-set size:
  10 graphs → #good[1.00], 100 → 0.76, full → #bad[0.545]

#align(center, tbl(
  columns: 4, align: (center, center, left, center),
  table.header([*depth*], [*width*], [*features*], [*test*]),
  [1 – 5], [2 – 128], [degree], [0.500 – 0.545],
  [1], [20 – 128], [adj\_rows], [0.565 – 0.640],
  [4 – 8], [32 – 128], [adj\_rows], [0.520 – 0.600],
))

== Two things did work — both by changing the supervision

#align(center, tbl(
  columns: 3, align: (left, center, left),
  table.header([*target*], [*result*], [*supervision*]),
  [binary "is it connected?"], bad[stalls at $ln 2$], [1 bit / graph],
  [$n times n$ reachability matrix], good[≈ 0.97 EM], [$n^2$ bits / graph],
  [emit a full BFS trace], good[0.997], [dense, arranged in *time*],
))

#v(0.4em)

The trace result is depth *2* and flat across diameters 2–18, versus 0.510 for a matched
answer-only control on identical data.

#v(0.4em)

Neither is a better architecture. Both are the same lever: *more supervision per graph*.

#rc[CHANGELOG 06-25 (matrix) · `results/20260705_005620_*`, `20260708_180117_*` (traces)]

= Part 6 — The questions

== The one that draws me most

#v(0.5em)

#align(center, block(inset: 14pt, fill: luma(245), radius: 4pt, width: 94%)[
  #text(size: 19pt)[
    Sanford proves the log-depth construction *exists*. My runs say gradient descent does
    not *find* it on data with no statistical shortcut. \
    #v(0.4em)
    *Is there a representability–trainability gap for connectivity, and what closes it?*
  ]
])

#v(0.5em)

Every paper I have read is about what a transformer *can represent*, or about *a
dataset*. None establishes that training finds the construction.

#v(0.3em)

#rc[Thm 2 supplies the premise; my grid plus the capacity control supplies the negative
side; the trace result supplies a positive resolution.]

== Three runners-up

*A. Encodings as precomputation oracles.* Quantify the exchange rate — how many layers
does each encoding buy? The Laplacian result is one clean data point; SPD and fixed-$n$
adjacency rows are two more. Directly extends the paper you gave me.

#v(0.4em)

*B. A leak-audit protocol for graph-reasoning benchmarks.* Six instances across four
setups, two of them published. Least risky, most immediately useful — but a methods
contribution rather than a theoretical one.

#v(0.4em)

*C. Which supervision makes an algorithm learnable?* The matrix target and the BFS trace
both work where the binary label fails, and there is a pattern in *which* intermediate
steps must be made explicit.

#rc[Full write-up with evidence status and decisive experiments in `docs/RESEARCH-QUESTIONS.md`]

== What I still owe

+ *Kill the local bridge-detection shortcut in `connectedness_hard`.* Every negative
  result I have rests on this dataset.
+ *Re-run the depth × width grid parameter-matched*, the way Yehudai §6.1 does. Mine
  accumulated across four sessions with drifting configs and no fixed budget.
+ *Test the width axis properly.* Their claim is $Ω(n)$-to-$O(n "polylog" n)$ width at
  fixed depth; my grid tops out at width 256 with $n$ fixed, so I have never varied width
  *against* $n$ — the only way their bound is falsifiable.
+ *Write the Yehudai reading companion*, as I have for Sanford and Ye.

#v(0.3em)

#rc[Caveats on record: the $ln 2$ plateau predates my run logger (no results JSON); I never
ran `lap_full` inside their pipeline; my reproduction used the supplementary code, which
may predate the arXiv v3 I cite.]

== Summary

#v(0.4em)

+ The theory is a *tradeoff*, not a single bound: connectivity costs $O(log N)$ depth at
  sub-polynomial width (Sanford), or $O(n "polylog" n)$ width at fixed depth (Yehudai).
+ *Every benchmark I audited leaked* — including two published ones and my own.
+ Laplacian encodings solve it at depth 1, width 8 — confirming a trade-off Yehudai
  et al. state theoretically, and showing it survives an adversarial dataset. The
  computation moved into preprocessing; it did not disappear.
+ With the leaks sealed, nothing I trained learned the binary task at any depth or width
  I ran — and the failure is generalization, not capacity.
+ Changing *supervision* (dense matrix target, or a BFS trace) does work.

#v(0.4em)

#rc[Deck source `reports/progress-presentation.typ` · questions
`docs/RESEARCH-QUESTIONS.md` · day-by-day log `docs/CHANGELOG.md`]
