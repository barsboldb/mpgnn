#import "@preview/cetz:0.3.4"

#set page(paper: "a4", margin: (x: 2.2cm, y: 2.2cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10.5pt, lang: "en")
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.1")

#show heading.where(level: 1): it => {
  set text(size: 15pt)
  block(above: 1.2em, below: 0.7em)[#counter(heading).display() #it.body]
}
#show heading.where(level: 2): it => {
  set text(size: 12pt)
  block(above: 1em, below: 0.5em)[#counter(heading).display() #it.body]
}

#let good(body) = box(fill: rgb("#e6f4ea"), inset: (x: 4pt, y: 1pt), radius: 2pt, text(rgb("#137333"))[#body])
#let bad(body)  = box(fill: rgb("#fce8e6"), inset: (x: 4pt, y: 1pt), radius: 2pt, text(rgb("#c5221f"))[#body])
#let note(body) = block(fill: luma(245), inset: 8pt, radius: 3pt, width: 100%)[#body]

#let blue   = rgb("#1a73e8")
#let purple = rgb("#9334e6")
#let orange = rgb("#e8710a")
#let green  = rgb("#137333")
#let red    = rgb("#c5221f")

#align(center)[
  #text(size: 19pt, weight: "bold")[GAT vs. the Local-Attention Transformer]
  #v(0.3em)
  #text(size: 11pt)[The same one-hop reach, with very different bills]
  #v(0.2em)
  #text(size: 9.5pt, fill: luma(100))[`src/layers.py` — `GATConv` and `GlobalAttnConv(local: true)` · #datetime.today().display()]
]

#v(0.6em)

#note[
  *In one sentence.* GAT and the local-attention transformer restrict a node to its graph
  neighbours in exactly the same way — the difference is that GAT only ever *computes* the
  edge entries (sparse, $O(E)$), while `local: true` computes the *full dense* $N times N$
  attention matrix and then throws away everything that is not an edge — paying the whole
  transformer bill, plus a masking surcharge, to use $#sym.tilde 0.4%$ of what it bought.
]

= Why the repo has both

`local: true` was not built to be a good model. It exists for one controlled experiment
(CHANGELOG 2026-06-25): the global-attention transformer is *not* depth-bounded — one
all-pairs layer reaches any node, so the $3^L$ capacity wall of Ye et al. never binds and
the data-lever experiment cannot engage. To make the wall appear, we needed the *same*
model — same scaled dot-product attention, same $W_Q, W_K, W_V$ parameterisation, same
softmax, same parameter count — with only the *reach* changed from all-pairs to one hop
per layer. That is `GlobalAttnConv(local: true)`: the minimal-diff, depth-bounded
transformer.

GAT would not have been that experiment. It restricts reach the same way, but it also
changes the attention *mechanism* (@sec-mechanisms), so any behavioural difference would be
confounded. The price of the clean comparison is efficiency — and the price is steep
(@sec-why-slow).

= The two mechanisms <sec-mechanisms>

Both compute, for node $i$, a weighted average over the *same* set — the neighbours
$N(i) union {i}$. They differ in how scores are produced and, crucially, over what domain
they are computed.

== GAT: additive attention, computed only on edges

`GATConv` (Veličković et al. 2018) is a PyG `MessagePassing` module. One shared projection
$W$, and a learned *additive* score per edge:

$ e_(i j) = "LeakyReLU"(a_"src"^top W h_j + a_"dst"^top W h_i), quad
  alpha_(i j) = "softmax"_(j in N(i) union {i}) (e_(i j)) $

The implementation never sees a non-edge: scores live in a tensor of shape $[E, H]$
(one row per edge), the softmax is a *segmented* softmax grouped by destination node
(`pyg_softmax`), and aggregation is a scatter-add along edges. Work and memory are
$O(E dot d)$ — for our caterpillars ($E approx 2n$), *linear in nodes*.

== Local attention: dot-product attention, computed on all pairs, then masked

`GlobalAttnConv` is a transformer layer over the node set: separate $W_Q, W_K, W_V$,
scaled dot-product scores for *every pair in the batch*:

$ A_(i j) = (Q_i dot K_j) / sqrt(d_h) quad "for all" i, j in [N]_"batch" $

— an `einsum` producing a dense $[N, N, H]$ tensor where $N$ is the *total* node count of
the batch (32 graphs #sym.times 24 nodes $=$ 768), followed by a cross-graph mask. With
`local: true` it then builds a dense boolean adjacency $[N, N]$ from `edge_index`, masks
every non-neighbour entry to $-infinity$, and softmaxes the full matrix:

$ A_(i j) <- cases(A_(i j) & "if" (j,i) in E "or" i = j, -infinity & "otherwise") $

The *used* entries after masking are exactly GAT's domain. Everything else was computed,
written to memory, masked, exponentiated to $0$ — and discarded.

#figure(
  cetz.canvas(length: 0.42cm, {
    import cetz.draw: *

    // ── the example graph: 6-node caterpillar ────────────────────────────
    let pos = ((0, 1.4), (2, 1.4), (4, 1.4), (6, 1.4), (8, 1.4), (2, 3.4))
    let edges = ((0, 1), (1, 2), (2, 3), (3, 4), (1, 5))

    group(name: "graph", {
      translate((0, 3.5))
      for e in edges {
        line(pos.at(e.at(0)), pos.at(e.at(1)), stroke: luma(120) + 1pt)
      }
      for (i, p) in pos.enumerate() {
        circle(p, radius: 0.55, fill: rgb("#e8f0fe"), stroke: blue + 1pt)
        content(p, text(size: 8pt, fill: blue)[#i])
      }
      content((4, -0.4), text(size: 8pt, fill: luma(100))[the graph ($n=6$, $E=5$)])
    })

    // ── helper: draw a 6x6 attention matrix ──────────────────────────────
    // mode: "gat" (only edge cells exist), "global" (all used),
    //       "local" (all computed, non-edge cells wasted)
    let adj = (
      (0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3), (1, 5), (5, 1),
      (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5),
    )
    let matrix(origin, mode, label, sub) = {
      group({
        translate(origin)
        for i in range(6) {
          for j in range(6) {
            let is-edge = adj.contains((i, j))
            let cell = (j, 5 - i)
            if mode == "gat" {
              if is-edge {
                rect(cell, (cell.at(0) + 1, cell.at(1) + 1), fill: rgb("#ceead6"), stroke: luma(200) + 0.4pt)
              } else {
                rect(cell, (cell.at(0) + 1, cell.at(1) + 1), fill: white, stroke: luma(230) + 0.3pt)
              }
            } else if mode == "global" {
              rect(cell, (cell.at(0) + 1, cell.at(1) + 1), fill: rgb("#d2e3fc"), stroke: luma(200) + 0.4pt)
            } else {
              // local: computed everywhere, kept only on edges
              if is-edge {
                rect(cell, (cell.at(0) + 1, cell.at(1) + 1), fill: rgb("#ceead6"), stroke: luma(200) + 0.4pt)
              } else {
                rect(cell, (cell.at(0) + 1, cell.at(1) + 1), fill: rgb("#fce8e6"), stroke: luma(200) + 0.4pt)
                line((cell.at(0) + 0.18, cell.at(1) + 0.18),
                     (cell.at(0) + 0.82, cell.at(1) + 0.82), stroke: red + 0.5pt)
              }
            }
          }
        }
        rect((0, 0), (6, 6), stroke: luma(120) + 0.6pt)
        content((3, 6.8), text(size: 8.5pt, weight: "bold")[#label])
        content((3, -0.8), text(size: 7.5pt, fill: luma(100))[#sub])
      })
    }

    matrix((-1.0, -6.5), "gat",    [GAT],                    [computes 16 of 36 cells])
    matrix((7.4, -6.5),  "global", [global attention],       [computes 36, uses 36])
    matrix((15.8, -6.5), "local",  [local attention],        [computes 36, #text(fill: red)[keeps 16]])
  }),
  caption: [
    Score computation per layer on a toy 6-node graph (self-loops included).
    #good[green] cells are neighbour scores — the only ones any of the three models
    ultimately *uses* for the local models. GAT (left) never materialises anything else.
    Local attention (right) materialises the full matrix like the global model (middle),
    then #bad[masks out] every non-edge. At batch scale ($N=768$) the kept fraction drops
    from $16\/36$ to $#sym.tilde 0.4%$.
  ]
) <fig-matrices>

= Why local attention is *the slowest of the three* <sec-why-slow>

Measured on identical runs (`conn_diameter_controlled`, depth 2, hidden 64, batch 32,
$n = 24$, Apple MPS):

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, left),
    stroke: 0.4pt + luma(180),
    table.header([*layer*], [*train, s/epoch*], [*inference, ms/graph*], [*score domain*]),
    [GAT (`gat`)],                 [3.3 – 7.4], [0.44 – 0.66], [$[E, H]$: edges only],
    [global (`global_attn`)],      [3.6 – 4.8], [1.60 – 1.96], [$[N, N, H]$: all pairs, all used],
    [local (`local: true`)], [#text(fill: red)[20.7]], [#text(fill: red)[2.50]], [$[N, N, H]$ *plus* dense mask, 0.4% used],
  ),
  caption: [Wall-clock from `results/*.json` timing blocks (runs of 2026-06-25 – 07-02).
    Local attention is #sym.tilde 3–6#sym.times slower per epoch than GAT and — the telling
    part — #sym.tilde 5#sym.times slower than the *global* model it strictly subsets.],
) <tab-timing>

Three compounding reasons, in decreasing order of principle:

== It pays the full all-pairs bill for a sparse result

Per layer, per forward, the batch has $N = 768$ nodes and $E + N approx 2#[,]368$
neighbour-or-self pairs. The dense path computes $N^2 = 589#[,]824$ pair scores
#sym.times 4 heads — then keeps 0.4% of them. GAT's compute *is* the kept set. Asymptotically:
$O(N^2 d)$ vs $O(E d)$ per layer, a factor of $N^2\/E approx 250$ in score work on these
sparse graphs. (Why doesn't the *global* model already suffer? It does the same $N^2$ work
— but as one dense matmul, the operation GPUs are best at, and every computed entry is
used. That is @tab-timing's second row costing only #sym.tilde 3#sym.times GAT's
inference. Dense hardware efficiency hides the asymptotic waste at $n = 24$; it stops
hiding around the point where $[N,N,H]$ stops fitting in cache — and would be fatal at
Yehudai's $n = 50$ or beyond.)

== The mask itself is a second $O(N^2)$ pass — in eager mode, several

`local: true` adds, on top of the global layer's work, *per layer per step*:
a fresh $[N, N]$ boolean adjacency allocation, two advanced-indexing writes into it,
a `masked_fill` over $[N, N, H]$, and a softmax whose rows are mostly $-infinity$.
None of these fuse: each is a separate memory-bound kernel pass over the
$#sym.tilde 9.4$ MB score tensor (vs GAT's #sym.tilde 37 KB edge tensor — a 250#sym.times
memory-traffic gap). On MPS specifically, boolean advanced indexing is a notoriously slow
path (extra synchronisation, occasional CPU fallback), which is the best explanation for
local costing 5#sym.times the global model when it "only" adds masking. The einsum-based
attention also forgoes the fused scaled-dot-product kernel — a deliberate trade for the
`store_attn` inspection hook, which needs the explicit $[N, N, H]$ tensor anyway.

== The waste grows with the *batch*, not the graph

$N$ is the batch-total node count, so doubling `batch_size` quadruples the dense work per
step while the useful (edge) work merely doubles. GAT's cost is batch-linear. The gap in
@tab-timing is at batch 32; larger batches widen it quadratically.

#note[
  *Rule of thumb.* Dense masked attention costs $O(N^2)$ no matter how sparse the mask;
  sparsity only pays when it is *structural* (never compute the entry) rather than
  *post-hoc* (compute, then hide). GAT is structural sparsity; `local: true` is post-hoc
  sparsity. The scratchpad-CoT attention masks in `transformer.py` are post-hoc too — fine
  there, because those sequences genuinely need most pairs.
]

= What is actually different, mechanism-wise

Reach is identical (one hop per layer, self included). Everything else differs slightly —
which is exactly why `local: true` and not GAT is the controlled contrast to the global
transformer:

#table(
  columns: (auto, auto, auto),
  align: (left, left, left),
  stroke: 0.4pt + luma(180),
  table.header([], [*GAT*], [*local attention*]),
  [score form], [additive: $a^top [W h_i || W h_j]$ + LeakyReLU], [scaled dot-product: $Q_i dot K_j \/ sqrt(d_h)$],
  [projections], [one shared $W$; per-head vectors $a$], [separate $W_Q, W_K, W_V$],
  [softmax], [segmented, over incoming edges], [dense rows, non-edges at $-infinity$],
  [compute/memory], [$O(E d)$ / $[E, H]$], [$O(N^2 d)$ / $[N, N, H]$ + $[N, N]$ mask],
  [same as global model?], [no — different mechanism], [*yes* — identical up to the mask],
  [role in the repo], [efficient depth-bounded baseline], [controlled reach-ablation of the transformer],
)

The additive-vs-dot-product distinction is not cosmetic: dot-product scores are bilinear
in the two endpoints (content-content matching), while GAT's additive form factorises into
a source term plus a destination term — a strictly weaker interaction that, e.g., cannot
express "attend to nodes whose key equals my query" without help from $W$. For
connectivity on our datasets both were expressive enough; the choice never mattered for
accuracy, only for the experimental logic.

= Practical guidance

- *Need a depth-bounded model that trains fast?* GAT (or GIN/GCN). Batch-linear, and on
  MPS it is the cheapest attention we have.
- *Need the capacity-wall / reach-ablation comparison?* `local: true` — it is the only
  variant where "the same transformer, one hop per layer" is literally true. Accept the
  #sym.tilde 5#sym.times cost, keep batches small, and don't run it at $n$ much beyond
  50 — the $[N, N, H]$ tensor and the mask passes scale quadratically in batch-total
  nodes.
- *If local attention ever becomes a workhorse* rather than an ablation: implement it as a
  `MessagePassing` module with dot-product scores (i.e., GAT's plumbing with
  `GlobalAttnConv`'s scoring). That keeps the mechanism-controlled comparison *and* the
  $O(E)$ bill — the two are only coupled in the current implementation, not in principle.
