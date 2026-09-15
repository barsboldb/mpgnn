#import "@preview/cetz:0.3.4"

// ============================================================
//  Reading Companion — Airale et al. 2025
//  "Simple Path Structural Encoding for Graph Transformers"
//  ICML 2025 (PMLR 267) · arXiv 2502.09365v2
// ============================================================

// ---- palette ----
#let c-algo   = rgb("#1b6ca8")   // SPSE / path side        (blue)
#let c-heur   = rgb("#c0392b")   // RWSE / walk side        (red)
#let c-accent = rgb("#7d3c98")   // purple accent
#let c-good   = rgb("#1e8449")   // green
#let c-ink    = rgb("#222222")
#let c-faint  = rgb("#f4f6f8")

#set page(
  paper: "a4",
  margin: (x: 1.9cm, top: 2.2cm, bottom: 1.9cm),
  numbering: "1",
  header: context {
    if counter(page).get().first() > 1 [
      #set text(8pt, fill: luma(120))
      #grid(columns: (1fr, 1fr),
        align(left)[Reading Companion],
        align(right)[Airale et al. 2025 · Simple Path Structural Encoding])
      #line(length: 100%, stroke: 0.4pt + luma(200))
    ]
  },
)

#set text(font: ("New Computer Modern", "Linux Libertine"), size: 10pt, fill: c-ink)
#set par(justify: true, leading: 0.62em)
#set heading(numbering: none)
#show heading.where(level: 1): it => {
  v(0.4em)
  block(width: 100%, inset: (y: 6pt), stroke: (bottom: 1.2pt + c-algo),
    text(15pt, weight: "bold", fill: c-algo, it.body))
  v(0.2em)
}
#show heading.where(level: 2): it => {
  v(0.3em); text(11.5pt, weight: "bold", fill: c-ink, it.body); v(0.1em)
}
#show link: it => text(fill: c-algo, it)

// ---- callout boxes ----
#let callout(title, body, col: c-algo, sym: "") = block(
  width: 100%, fill: col.lighten(91%), stroke: (left: 2.5pt + col),
  inset: (x: 9pt, y: 7pt), radius: 2pt, breakable: true,
  [#text(weight: "bold", fill: col, [#sym #title]) #v(-0.3em) #body],
)
#let defn(t, b)    = callout("Definition — " + t, b, col: c-ink, sym: "▸")
#let thm(t, b)     = callout("Proposition — " + t, b, col: c-accent, sym: "◆")
#let note(b)       = callout("My note", b, col: c-good, sym: "✎")
#let idea(b)       = callout("Thesis hook", b, col: rgb("#b9770e"), sym: "💡")
#let ask(b)        = callout("Open question", b, col: c-heur, sym: "?")
#let kbd(b) = box(fill: luma(235), inset: (x: 3pt, y: 1pt), radius: 2pt,
  text(font: "DejaVu Sans Mono", size: 8.5pt, b))

// ============================================================
#align(center)[
  #text(17pt, weight: "bold")[Simple Path Structural Encoding\ for Graph Transformers]
  #v(-0.5em)
  #text(11pt, style: "italic", fill: luma(90))[Count paths, not walks]
  #v(0.3em)
  #text(9.5pt, fill: luma(110))[Airale, Longa, Rigon, Passerini, Passerone (Univ. Trento) · ICML 2025, PMLR 267 · arXiv 2502.09365v2]
  #v(0.2em)
  #text(8.5pt, fill: luma(140))[Reading companion · GNN / Graph-Transformer thesis exploration]
]
#v(0.6em)

// ---------- TL;DR ----------
#callout("TL;DR", [
Graph transformers inject structure into attention through an *edge encoding*.
The incumbent is *RWSE* — stack the $k$-hop random-walk matrices $P_k = (D^(-1)A)^k$
and feed them to the attention layer. RWSE is cheap (closed form) but *ambiguous*:
the authors prove that every node pair in an even-length *cycle* has an RWSE-equivalent
pair in a *path graph*, so walk probabilities cannot separate cycles from chains.
*SPSE* replaces $P_k$ with $S_k$, the count of *simple paths* (no repeated nodes) of
length $k$ between $i$ and $j$. For adjacent pairs this count *is* cycle counting
(Prop. 3), which is exactly the structure molecules are made of. Simple-path counting
is #[#text(style: "italic")[\#P-hard]] in general, so they give an *approximate* counter:
decompose the graph into many DAGs by mixed DFS/BFS traversals, count by matrix powers
on each DAG, keep the running *maximum*. Drop-in replacement, *zero extra parameters*:
improves 21 / 24 benchmark cases, significantly on molecular data.
], col: c-accent, sym: "★")

#grid(columns: (1fr, 1fr), gutter: 8pt,
  callout("The big question", [
    Graph transformers see all pairs, so *distance* is not the problem — *which
    structure* a pair sits inside is. Can an edge encoding be made to carry local
    *cyclic* structure that random walks provably blur away?
  ], col: c-algo, sym: "✦"),
  callout("The one-line answer", [
    Yes — count *simple paths* instead of walks. Path counts between adjacent nodes
    are cycle counts, and an approximate DAG-decomposition counter makes them
    affordable enough to precompute once.
  ], col: c-good, sym: "✦"),
)

= 1 · Setup — where the encoding actually enters

#defn("Walk vs. simple path")[
A *walk* of length $k$ is any node sequence $v_0 ... v_k$ with consecutive nodes
adjacent — *revisits allowed*. A *simple path* is a walk in which *all nodes are
distinct*. A *cycle* is a walk whose nodes are distinct except $v_0 = v_k$.
]

#defn("The two encoding matrices")[
*Random walk matrix* $P_k = (D^(-1) A)^k in RR^(|V| times |V|)$ — landing
*probabilities* of length-$k$ walks. Closed form, cheap.

*Simple path matrix* $S_k in NN^(|V| times |V|)$ — $(S_k)_(i j)$ is the *number of
simple paths* of length $k$ from $i$ to $j$. No closed form; computing it is
expensive (Vassilevska & Williams 2009).
]

Both are stacked over $k = 1 ... K$ and pushed through a small network $phi.alt_0$ to
give an encoding matrix $E in RR^(|V| times |V| times d)$, which enters *both* the
attention logits and the values:

$
a_(i j) &= phi.alt_1 (W^Q x_i, W^K x_j, (E)_(i j)) \
alpha_(i j) &= a_(i j) \/ sum_k a_(i k) \
y_i &= sum_j alpha_(i j) phi.alt_2 (W^V x_j, (E)_(i j))
$

#note[
The encoding is *not* just an attention bias — it is mixed into the *value* path too
(eq. 3). That is what makes this a genuine "structural edge encoding" rather than a
distance prior, and it is why SPSE can be a *drop-in*: swap $E_"RW" arrow.r E_"SP"$ in
equations 1 and 3, change nothing else. Parameter count is *identical*.
]

#idea[
Our #kbd("GlobalAttnConv") sits at the *poorest* end of this family. We add a learned
per-head bias over *shortest-path-distance buckets* (#kbd("spd_bias"),
`src/layers.py:179`, added pre-softmax at `:196`) — a single scalar per (bucket, head),
logits only, never touching values. The ladder is:

#align(center)[
  #text(9pt)[SPD bucket bias #sym.arrow.r #text(fill: c-heur)[RWSE] #sym.arrow.r #text(fill: c-algo)[SPSE]]
]

SPD keeps *one number* (the distance); RWSE keeps a $K$-vector of walk probabilities;
SPSE keeps a $K$-vector of path counts. Each step up strictly refines the previous.
Our SPD bias flattens everything the paper cares about: two pairs at distance 2 get the
same bias whether they sit in a triangle-rich blob or a tree.
]

= 2 · The limitation — RWSE cannot see a cycle

The core negative result. Random walks *normalise*: $P_k$ holds probabilities, not
counts, so the mass leaving a node is always 1 and structural differences can cancel.

#thm("1 (even cycle ≡ path)")[
Let $G$ be an *even-length cycle* on $|V| = 2n$ nodes and $G'$ a *path graph* on
$2n+1$ nodes. Then for *any* pair $(i,j)$ in $G$ there is a pair $(i',j')$ in $G'$ with
$(i,j) eq_"RW" (i',j')$ — i.e. identical $P_k$ entries for *all* $k$.
]

#align(center)[
#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // ---- left: 6-cycle ----
  let cx = 0
  let pts = ()
  for i in range(6) {
    let a = 90deg + i * 60deg
    pts.push((cx + 1.0 * calc.cos(a), 1.0 * calc.sin(a)))
  }
  for i in range(6) {
    line(pts.at(i), pts.at(calc.rem(i + 1, 6)), stroke: 0.7pt + luma(130))
  }
  // highlight one edge
  line(pts.at(0), pts.at(1), stroke: 1.6pt + rgb("#c0392b"))
  for (i, p) in pts.enumerate() {
    circle(p, radius: 0.19, fill: if i < 2 { rgb("#c0392b") } else { white },
           stroke: 0.7pt + luma(70))
  }
  content((cx, -1.75), text(8.5pt)[even cycle $C_6$])

  // ---- middle: equivalence sign ----
  content((3.1, 0), text(11pt, fill: rgb("#c0392b"))[$eq_"RW"$])
  content((3.1, -0.55), text(7.5pt, fill: luma(120))[indistinguishable])

  // ---- right: 7-node path ----
  let y = 0
  let px = ()
  for i in range(7) { px.push((5.0 + i * 0.72, y)) }
  for i in range(6) {
    line(px.at(i), px.at(i + 1), stroke: 0.7pt + luma(130))
  }
  line(px.at(3), px.at(4), stroke: 1.6pt + rgb("#c0392b"))
  for (i, p) in px.enumerate() {
    circle(p, radius: 0.19, fill: if i == 3 or i == 4 { rgb("#c0392b") } else { white },
           stroke: 0.7pt + luma(70))
  }
  content((7.2, -1.75), text(8.5pt)[path $P_7$])
})
]
#align(center)[#text(8.5pt, fill: luma(110))[
  Prop. 1: the marked pairs carry *identical* random-walk encodings at every $k$,
  though one lies on a cycle and the other on a chain. SPSE separates them immediately
  — the cycle pair has two simple paths between its endpoints, the path pair has one.
]]

#thm("2 (general ambiguity)")[
For *any* graph $G$ and pair $(i,j)$ there exists a *non-isomorphic* $G'$ and a pair
$(i',j')$ with $(i,j) eq_"RW" (i',j')$. So an RWSE edge encoding *never* identifies a
graph.
]

#note[
Prop. 2 is weaker than it first sounds — the authors immediately concede it is "also
obviously true for SPSE". No pairwise encoding identifies a graph. The *load-bearing*
result is Prop. 1, which is a *concrete, structured* collapse (cycles $arrow.r$ chains),
not a generic pigeonhole. Read Prop. 2 as framing, Prop. 1 as the actual gap.
]

= 3 · Why path counts are cycle counts

#thm("3 (path counts on an edge = cycle counts)")[
Let $(i,j) in E$ be an *adjacent* pair and $(S_k)_(i j) = m_k$. Then for $k gt.eq 2$
there are *exactly* $m_k$ cycles of length $k+1$ in $G$ that use $(i,j)$ as an edge.
($k=1$ just counts parallel edges.)
]

This is the paper's real engine. For an edge, "how many simple paths of length $k$ join
my endpoints" and "how many $(k+1)$-cycles do I sit on" are *the same number*. So SPSE
hands the attention layer a per-edge *cycle spectrum* for free.

#note[
Crucially Prop. 3 holds *only for adjacent pairs* — $(i,j) in E$. For non-adjacent pairs
the path counts are still informative but no longer a clean cycle count. And no RWSE
analogue exists: landing probabilities on a cycle can be driven arbitrarily low just by
adding nodes to it, so the signal is not scale-stable.
]

#idea[
This is why the paper's benefit concentrates on *molecules* (ZINC, Peptides, PCQM4Mv2):
chemistry *is* cycle structure — aromatic rings, functional groups. Their Fig. 2 shows
SPSE giving identical encodings to the two C=O bonds of two carboxylic acids, and to
edges in two different 6-atom rings. That is chemically correct invariance, learned for
free from the encoding rather than from data.
]

= 4 · Making it computable — DAG decomposition

Exact simple-path counting is intractable (the count itself can grow like
$(|V|-2)! \/ (|V|-k-1)!$). The workaround exploits one fact: *on a DAG, path counts
between all pairs are just powers of the (directed) adjacency matrix*, $O(K |V|^3)$.

So: turn the undirected graph into *many* DAGs, count on each, and keep the best.

#defn("DAGDECOMPOSE (Alg. 2)")[
From a root $r$, build a node ordering $pi$ by *mixing traversals*: run $d_"DFS"-1$
steps of *DFS*, then a *partial BFS* step (expand only a random subset of children),
then *BFS* as far as possible; repeat until all nodes are visited. Orienting every edge
toward the higher index yields a DAG. Sweep $d_"DFS" = 0 ... D_"DFS"$, repeat $N$ times
per depth (random effects), over $R dot |V|$ sampled roots.
]

Why the mixture: *DFS* discovers long paths but commits to one branch and so tends to
find only *one* path between a pair; *BFS* explores breadth-first but *misses long
paths*. The *partial* BFS step is the trick that makes otherwise unreachable paths
traversable (their Fig. 3, on a Circular Skip Link graph).

#align(center)[
#cetz.canvas(length: 1cm, {
  import cetz.draw: *
  let box-(x, title, sub, c) = {
    rect((x - 1.25, -0.65), (x + 1.25, 0.75), radius: 4pt,
         fill: c.lighten(90%), stroke: c + 0.7pt)
    content((x, 0.38), text(8.5pt, weight: "bold", fill: c)[#title])
    content((x, -0.1), text(7pt, fill: luma(110))[#sub])
  }
  box-(-4.2, "undirected " + $G$, "paths intractable", rgb("#c0392b"))
  box-(0, "many DAGs", $pi$ + " orderings, DFS+BFS", rgb("#1b6ca8"))
  box-(4.2, "path counts", "powers of adjacency", rgb("#1e8449"))
  line((-2.85, 0.05), (-1.35, 0.05), mark: (end: "stealth"), stroke: 0.7pt + luma(120))
  line((1.35, 0.05), (2.85, 0.05), mark: (end: "stealth"), stroke: 0.7pt + luma(120))
  content((-2.1, 0.45), text(7pt, fill: luma(120))[decompose])
  content((2.1, 0.45), text(7pt, fill: luma(120))[$A^k$])
  content((0, -1.15), text(7.5pt, fill: luma(110))[running #text(weight: "bold")[max] over DAGs #sym.arrow.r lower bound on true count])
})
]

*Cost.* $O(K R D_"DFS" N |V|^3)$. The $R D_"DFS" N$ factor is "tens to hundreds", so
SPSE is *much* more expensive than RWSE — but it is a *one-time preprocessing step*,
and vastly cheaper than the $2^(|E|)$ possible decompositions. Wall-clock from Table 2:
*1 h* for ZINC, *80 h* for CIFAR10 and PCQM4Mv2 (3.7M graphs).

== Encoding the counts (eq. 4)

Counts explode, so they are squashed by *iterated* logarithms before hitting the network:

$ f: x arrow.r.bar alpha g^n (x) + beta, quad g: x arrow.r.bar ln(1 + x) $

with $g^n$ meaning $g$ composed $n$ times. Per Table 2: denser datasets want *higher $n$*
(more squashing) — $n=1$ for ZINC/PCQM4Mv2, $n=3$ for PATTERN/CLUSTER/MNIST/CIFAR10.

#note[
$n$ is doing real work: it is the knob that keeps a count-based encoding from being
dominated by hub pairs in dense graphs. Compare our own preprocessing — we normalise
nothing on the SPD buckets because a distance is already $O("diam")$. The moment we move
to counts, this compression question becomes ours too.
]

= 5 · Results

== Synthetic — cycle counting (validates Prop. 3)

12,000 graphs (avg. 149 nodes / 190 edges), built by injecting cycles of length 3–8
until each length's count lands in $[0, 14]$; six simultaneous multiclass tasks
("how many $k$-cycles?"). SPSE beats RWSE in *all but one* configuration, on both GRIT
and CSA.

#note[
The one revealing detail: at the *largest* model config, all models get "almost
perfectly", so the gap narrows. Their reading — *deep architectures can compensate for
expressivity limits in the encoding*. That is the same depth-vs-prior trade we keep
hitting: a weaker encoding is survivable if you pay for it in layers.
]

== Real benchmarks

Eight datasets: molecular (ZINC, Peptides-func, Peptides-struct, PCQM4Mv2), SBM
(PATTERN, CLUSTER), superpixel (MNIST, CIFAR10). SPSE swapped into *GRIT*, *CSA*, and
(partially) *GraphGPS*. Retrained on *10 seeds*, *no hyperparameter tuning*.

#align(center)[
#table(
  columns: (auto, auto, auto, auto),
  align: (left, center, center, left),
  stroke: 0.4pt + luma(180),
  inset: 6pt,
  table.header(
    text(weight: "bold")[Benchmark],
    text(weight: "bold", fill: c-heur)[RWSE],
    text(weight: "bold", fill: c-algo)[SPSE],
    text(weight: "bold")[note],
  ),
  [ZINC (GRIT, MAE $arrow.b$)],    [0.065 ± 0.005], [*0.059 ± 0.001*], [significant],
  [ZINC (CSA, MAE $arrow.b$)],     [0.069 ± 0.003], [*0.061 ± 0.003*], [significant],
  [Peptides-func (GRIT, AP $arrow.t$)], [0.6803 ± 0.0085], [*0.6945 ± 0.0113*], [significant],
  [Peptides-struct (GRIT, MAE $arrow.b$)], [0.2486 ± 0.0012], [*0.2449 ± 0.0018*], [significant],
  [CIFAR10 (GRIT, acc $arrow.t$)], [76.246 ± 0.954], [*77.022 ± 0.430*], [significant],
  [PATTERN (CSA, acc $arrow.t$)],  [87.008 ± 0.062], [*87.064 ± 0.052*], [significant],
  [CLUSTER (GRIT, acc $arrow.t$)], [*79.730 ± 0.189*], [79.571 ± 0.122], [no gain — dense],
)
]

*Headline: better in 21 of 24 cases*, statistically significant in 5 of 6 molecular
cases for CSA and GRIT. Gains are largest exactly where cycles carry meaning.

#note[
Two honest asterisks the authors raise themselves:

/ GPS gains are muted: in GraphGPS the encoding is restricted to $E$ and used only for
  *node-level* PE in the MPNN layer. Since parameters are held fixed, adding SPSE there
  effectively *shrinks* the existing edge embedding dimension. The lesson they draw —
  and it matters for us — is that *feeding the full encoding matrix into self-attention
  beats restricting it to real edges*.

/ CLUSTER gets nothing: densely connected SBM graphs defeat the approximate counter.
]

= 6 · Limitations — where the approximation bites

The counter keeps a running *maximum* over DAGs, so the reported counts are *lower
bounds* on the true counts. Worse, some path sets are *unreachable by any single DAG*:
their Fig. 6 shows two length-4 paths between a pair whose simultaneous discovery would
require a *cyclic* orientation — the algorithm can only count them one at a time.

#ask[
Density is the failure axis. In dense graphs (CLUSTER, PATTERN) many paths are missed,
counts are underestimated, and the encoding degrades toward noise. Their own suggestion:
*trade path length for accuracy* — shorter $K$ with exact counts may beat longer $K$
with bad estimates. Untested by them. That is a cheap, well-posed experiment.
]

#ask[
*Appendix B is refreshingly candid:* a Graziani-style argument shows global attention
with SPSE is more expressive than 1-WL — but *no comparison between SPSE and RWSE
expressivity is proven*. Because both aggregate over paths/walks *without enumerating
them*, the Michel et al. (2023) Thm 3.3 proof strategy does not apply. So the paper's
theory establishes *RWSE has a specific blind spot* (Prop. 1) and *SPSE sees cycles*
(Prop. 3) — not a clean separation theorem. The empirical table carries more weight
than the theory here.
]

= 7 · Hooks into our work

#idea[
*1. The `spd_bias` upgrade path is now concrete.* Our #kbd("GlobalAttnConv") already
has the plumbing — an encoding indexed by node pair, added pre-softmax. Moving from SPD
buckets to path counts is a change of *what fills the bucket*, not of architecture. The
cheap intermediate is RWSE ($P_k$ has a closed form, no counting algorithm needed); SPSE
is the expensive ceiling. Worth trying RWSE first precisely because Prop. 1 tells us
exactly what we'd still be missing.
]

#idea[
*2. Cycles are the WL blind spot too.* Our #kbd("iso_wl") experiments sit at the 1-WL
boundary, and WL-indistinguishable graph pairs classically differ in *cycle* structure
(the standard counterexamples are two triangles vs. a 6-cycle). SPSE's per-edge cycle
spectrum is aimed straight at that. This links the message-passing ceiling
(#kbd("reports/message-passing.typ")) to a concrete encoding that breaks it — and note
the mechanism is *not* more depth, it is *more informative tokens*.
]

#note[
*3. But it probably does not help `connectedness_hard`.* Be honest about scope. SPSE
encodes *local cyclic* structure; our hard connectivity task is about *global reach*
across a bridge, and its two classes are built to be locally identical. A cycle spectrum
on a dense blob's edges says nothing about whether a bridge exists. The Sanford/Ye story
(depth $arrow.l.r$ diameter, data lever) governs that task; SPSE governs the
*isomorphism / molecular-structure* side. Different axis — do not conflate them.
]

#ask[
*4. Cost sanity check before adopting.* 80 h of preprocessing for CIFAR10-scale data is
real. Our graphs are small ($n$ in the tens), so $O(K R D_"DFS" N |V|^3)$ should be
minutes — but path counts must be *cached to disk alongside the dataset*
(`data/` is already gitignored and regenerated, so the cache slot exists). Check
whether counting is even necessary at our sizes: for $n approx 32$, *exact* enumeration
at small $K$ may be affordable and sidesteps the whole approximation-quality question.
]

= 8 · One-paragraph summary

RWSE has a provable structural blind spot: even cycles look like paths under random-walk
probabilities. Replacing walk probabilities with *simple-path counts* fixes exactly that
blind spot, and for adjacent pairs the counts are literally cycle counts. The price is
that counting simple paths is intractable, paid down by an approximate DAG-decomposition
counter that yields *lower bounds*, which works well on sparse molecular graphs and
poorly on dense ones. As a drop-in with unchanged parameter count, it wins 21/24
benchmark cases. The theory does not prove SPSE $succ$ RWSE in general — it proves one
concrete gap and demonstrates the rest empirically.

#v(0.8em)
#line(length: 100%, stroke: 0.4pt + luma(200))
#v(0.2em)
#text(8.5pt, fill: luma(110))[
  PDF: #kbd("papers/airale-2025-simple-path-structural-encoding.pdf") ·
  #link("https://arxiv.org/abs/2502.09365")[arXiv:2502.09365v2] ·
  #link("https://proceedings.mlr.press/v267/airale25a.html")[PMLR v267] ·
  Related in this repo: #kbd("message-passing.typ"), #kbd("ye-2026-notes.typ"),
  #kbd("laplacian-eigenvectors.typ")
]
