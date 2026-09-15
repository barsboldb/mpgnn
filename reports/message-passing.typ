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
#let bad(body) = box(fill: rgb("#fce8e6"), inset: (x: 4pt, y: 1pt), radius: 2pt, text(rgb("#c5221f"))[#body])
#let code(body) = raw(body)

#align(center)[
  #text(size: 19pt, weight: "bold")[What Message Passing Actually Does]
  #v(0.3em)
  #text(size: 11pt)[One primitive, four layer types, and the ceiling it cannot pass]
  #v(0.2em)
  #text(size: 9.5pt, fill: luma(100))[#code("src/layers.py") · #code("src/graph_conv.py") · #datetime.today().display()]
]

#v(0.6em)

= The one-sentence answer

Message passing is a *fixed number of synchronous rounds of local information exchange*. In each round, every node simultaneously reads the current state of its direct neighbours, compresses what it read into a single vector, and rewrites its own state from that summary plus its old state.

Everything a message-passing GNN can compute follows from that one sentence — including, importantly, everything it *cannot* compute.

#align(center)[
  #box(fill: luma(245), inset: 8pt, radius: 3pt, width: 92%)[
    A node's representation after $L$ rounds is a function of its *$L$-hop neighbourhood* — the subgraph of everything reachable within $L$ edges — and of nothing else.
  ]
]

That single invariant is the lever the whole thesis pushes on. Depth is not a capacity knob here; depth *is* reach.

= The primitive: one round in three steps

Write $h_i^((ell))$ for the state of node $i$ after round $ell$, and $cal(N)(i)$ for its neighbours. One round is:

$
"(1) message"quad & m_(j arrow i) = "MSG"(h_i^((ell)), h_j^((ell)), e_(j i)) \
"(2) aggregate"quad & a_i = plus.o.big_(j in cal(N)(i)) m_(j arrow i) \
"(3) update"quad & h_i^((ell+1)) = "UPD"(h_i^((ell)), a_i)
$

Three design choices, and they are the *only* three: what a message contains, how a multiset of messages collapses to one vector, and how a node folds that summary back into itself.

Two properties are forced by the shape of the formula, not by any particular choice:

- *Permutation invariance.* $plus.o$ runs over a set, so relabelling the graph cannot change the answer. This is the reason GNNs work on graphs at all, and it is bought at the cost of item (2) below.
- *Information loss at the aggregator.* A whole neighbourhood — however large — is squeezed into one fixed-width vector. Nothing downstream can recover what $plus.o$ threw away.

== What "one round" buys you, concretely

Take reachability. Give every node a one-hot marker of itself, use $plus.o = or$ (boolean OR), and $"UPD" = or$. Then $h_i^((ell))$ is exactly the set of nodes within $ell$ hops of $i$. Round by round:

#align(center)[
#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  let nodes = ((0, 0), (1.6, 0), (3.2, 0), (4.8, 0), (6.4, 0))
  let names = ("a", "b", "c", "d", "e")

  // three stacked rows: after 0, 1, 2 rounds
  let rows = (
    (y: 0,    label: "start",     lit: (0,)),
    (y: -1.5, label: "round 1",   lit: (0, 1)),
    (y: -3.0, label: "round 2",   lit: (0, 1, 2)),
  )

  for r in rows {
    content((-1.6, r.y), text(size: 8.5pt, fill: luma(90))[#r.label])
    for (i, p) in nodes.enumerate() {
      let filled = r.lit.contains(i)
      circle((p.at(0), r.y), radius: 0.26,
             fill: if filled { rgb("#137333") } else { white },
             stroke: 0.7pt + luma(60))
      content((p.at(0), r.y),
              text(size: 8pt, fill: if filled { white } else { luma(60) })[#names.at(i)])
    }
    for i in range(4) {
      line((nodes.at(i).at(0) + 0.26, r.y), (nodes.at(i + 1).at(0) - 0.26, r.y),
           stroke: 0.6pt + luma(120))
    }
  }
})
]

The lit frontier advances *exactly one edge per round*. To learn that #code("a") reaches #code("e") on this path you need four rounds; no width, no learning rate, and no amount of data substitutes for the fourth round. This is the entire content of the depth↔diameter story that the connectedness experiments keep running into.

= The four layers in this repo are one template

#code("src/layers.py") implements four message-passing layers. They look different and they are not — each one is the template above with different fills for MSG / $plus.o$ / UPD.

#v(0.4em)
#align(center)[
#table(
  columns: (auto, auto, auto, auto),
  align: (left, left, left, left),
  stroke: 0.4pt + luma(180),
  inset: 6pt,
  table.header(
    text(weight: "bold")[Layer],
    text(weight: "bold")[Message],
    text(weight: "bold")[Aggregate],
    text(weight: "bold")[Update],
  ),
  code("GCNConv"),
  [$c_(i j) dot W h_j$, with $c_(i j) = (d_i d_j)^(-1 slash 2)$],
  [sum],
  [none — self-loop folds $i$ in],

  code("SAGEConv"),
  [$h_j$ (raw)],
  [mean],
  [$W dot [h_i bar.v a_i]$],

  code("GATConv"),
  [$alpha_(i j) dot W h_j$, $alpha$ learned per pair],
  [sum (per head)],
  [none — self-loop folds $i$ in],

  code("GINConv"),
  [$h_j$ (raw)],
  [sum],
  [$"MLP"((1 + epsilon) h_i + a_i)$],
)
]
#v(0.4em)

Reading the table as a progression is the useful part:

- *GCN* fixes the neighbour weights from degree alone. Cheap, and blind to what the neighbour actually contains.
- *SAGE* keeps self and neighbourhood in *separate slots* of the concatenation, so the update can treat "what I am" and "what surrounds me" differently. GCN cannot: the self-loop dissolves the two into one sum.
- *GAT* makes the weights content-dependent — $alpha_(i j)$ is computed from $h_i$ and $h_j$ — so a node can decide, per round, which neighbours matter.
- *GIN* uses sum with an MLP update, which is the choice that maximises what survives step (2). See below.

== Why the aggregator choice is the expressiveness choice

Mean and max are *lossy in a specific way*: mean cannot tell $\{a, a\}$ from $\{a\}$, and max cannot tell $\{a, b\}$ from $\{a, b, b\}$. Sum distinguishes both. Xu et al. (2019) turn this into the sharp statement: with a sum aggregator and an injective update, a message-passing GNN is *exactly as discriminative as the 1-dimensional Weisfeiler–Lehman colour refinement test* — and no message-passing scheme of this form can beat 1-WL.

That is a ceiling, not a floor. It bites in this repo directly: two graphs that 1-WL cannot separate are indistinguishable to #code("GINConv") at any depth and any width. The isomorphism experiments (#code("analyze_iso.py"), the #code("iso_wl") configs) live precisely at that boundary, which is why they need something other than a message-passing prior to move.

= What message passing cannot do

Three distinct failure modes, often conflated. They are separate and have separate fixes.

#v(0.3em)
#align(center)[
#table(
  columns: (auto, 1fr, auto),
  align: (left, left, left),
  stroke: 0.4pt + luma(180),
  inset: 6pt,
  table.header(
    text(weight: "bold")[Failure],
    text(weight: "bold")[Mechanism],
    text(weight: "bold")[Fix that works],
  ),
  [under-reaching],
  [$L$ rounds see $L$ hops; anything farther than the diameter is invisible, period],
  [more depth, or non-local layers],

  [over-squashing],
  [an exponentially growing receptive field is forced through one fixed-width vector at a bottleneck edge],
  [rewiring, virtual nodes, global attention],

  [over-smoothing],
  [repeated averaging is a diffusion; node states converge to a constant as $L$ grows],
  [residuals, normalisation],
)
]
#v(0.3em)

Note the trap: the fix for under-reaching (add layers) *feeds* over-smoothing, and the deeper stack squeezes more of the graph through the same width, feeding over-squashing. This is why "just make it deeper" stops working well before the diameter on real graphs — and why #code("GraphConvNet") applies a residual and an optional norm around every layer (#code("src/graph_conv.py:157")), which buys depth without the diffusion collapse.

= Why this repo contrasts it with global attention

#code("GlobalAttnConv") (#code("src/layers.py:135")) drops into the *same* pipeline slot as the four layers above — same #code("forward(x, edge_index, batch)") signature, same residual/norm/ReLU wrapper — and changes exactly one thing: it removes the restriction that $j$ must be a neighbour of $i$. Every node attends to every node in its graph.

#align(center)[
#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  let ring(cx, cy) = {
    let pts = ()
    for i in range(6) {
      let a = 90deg + i * 60deg
      pts.push((cx + 0.85 * calc.cos(a), cy + 0.85 * calc.sin(a)))
    }
    pts
  }

  // left: message passing — only ring edges
  let L = ring(0, 0)
  for i in range(6) {
    line(L.at(i), L.at(calc.rem(i + 1, 6)), stroke: 0.7pt + luma(120))
  }
  for (i, p) in L.enumerate() {
    circle(p, radius: 0.18, fill: if i == 0 { rgb("#137333") } else { white }, stroke: 0.7pt + luma(60))
  }
  content((0, -1.55), text(size: 8.5pt)[message passing: reach $+1$ hop])

  // right: global attention — all pairs
  let R = ring(5, 0)
  for i in range(6) {
    for j in range(i + 1, 6) {
      line(R.at(i), R.at(j), stroke: 0.35pt + luma(190))
    }
  }
  for (i, p) in R.enumerate() {
    circle(p, radius: 0.18, fill: if i == 0 { rgb("#137333") } else { white }, stroke: 0.7pt + luma(60))
  }
  content((5, -1.55), text(size: 8.5pt)[global attention: reach $times 2$])
})
]

The consequence is a change in the *depth budget*, not in the per-layer cost of reasoning. Message passing extends the reachability horizon *additively* — one hop per round — so it needs $Theta("diameter")$ layers. All-pairs attention can implement the doubling trick, squaring the reachable set each layer, so it needs $O(log N)$ (Sanford et al. 2024). On a graph of diameter 16 that is 16 layers versus 4.

This is the same distinction as PRAM/MPC round complexity, and it is the reason the experiments in this repo hold the task fixed and vary only the layer type: *the question is never "can the architecture represent the answer", it is "how many rounds of communication does the answer need, and does the stack have that many".*

The #code("local=True") flag on #code("GlobalAttnConv") is the controlled middle: attention machinery, but masked to the adjacency, so reach again grows one hop per layer. It isolates "is it the attention or is it the reach?" — the answer being reach.

= Where it lives in the code

#v(0.3em)
#align(center)[
#table(
  columns: (auto, 1fr),
  align: (left, left),
  stroke: 0.4pt + luma(180),
  inset: 6pt,
  code("src/layers.py:8"),   [#code("GCNConv") — degree-normalised sum],
  code("src/layers.py:36"),  [#code("SAGEConv") — mean, self kept separate],
  code("src/layers.py:58"),  [#code("GATConv") — learned per-edge attention, multi-head],
  code("src/layers.py:108"), [#code("GINConv") — sum + MLP, the 1-WL-tight one],
  code("src/layers.py:135"), [#code("GlobalAttnConv") — all-pairs; the non-message-passing contrast],
  code("src/graph_conv.py:72"), [#code("GraphConvNet") — the shared embed → $L times$ layer → readout engine],
  code("src/gnn.py:5"),      [#code("GNN") — #code("GraphConvNet") restricted to local aggregators],
)
]
#v(0.3em)

Steps (1) and (2) of the primitive are not written out by hand: all four layers subclass PyG's #code("MessagePassing") and get them from #code("propagate()"), declaring $plus.o$ via #code("aggr=") in the constructor and MSG via the #code("message()") method. The #code("forward()") body is step (3), the update. Reading the four #code("message()") methods side by side is the fastest way to see that the differences between these architectures are genuinely small.

= Summary

+ Message passing is $L$ synchronous rounds of *message → aggregate → update* over the edge set.
+ After $L$ rounds a node knows its $L$-hop neighbourhood and nothing more. Depth *is* reach.
+ The aggregator is where information is destroyed; sum destroys the least, which is why GIN is 1-WL-tight — and 1-WL is the ceiling for the whole family.
+ Its three failure modes — under-reaching, over-squashing, over-smoothing — are distinct, and the fix for the first aggravates the other two.
+ Global attention is the same pipeline with the neighbour restriction lifted: $O(log N)$ rounds instead of $Theta("diameter")$.

#v(0.8em)
#line(length: 100%, stroke: 0.4pt + luma(200))
#v(0.2em)
#text(size: 9pt, fill: luma(100))[
  Kipf & Welling 2017 (GCN) · Hamilton et al. 2017 (GraphSAGE) · Veličković et al. 2018 (GAT) ·
  Xu et al. 2019 (GIN / 1-WL) · Alon & Yahav 2021 (over-squashing) · Sanford et al. 2024 (depth ↔ MPC rounds)
]
