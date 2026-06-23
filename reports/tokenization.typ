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

#align(center)[
  #text(size: 19pt, weight: "bold")[How Our Graph Transformer Reads a Graph]
  #v(0.3em)
  #text(size: 11pt)[Why the tokenization keeps failing, and what the failures have in common]
  #v(0.2em)
  #text(size: 9.5pt, fill: luma(100))[connectedness#sub[hard] · #datetime.today().display()]
]

#v(0.6em)

= The job a transformer has to do

A transformer does not understand graphs. It understands *sequences of vectors* — "tokens". Every attention layer takes a set of tokens and lets each one look at all the others. So before any learning can happen, we must answer one question:

#align(center)[
  *How do we turn a graph into a list of tokens?*
]

This single choice — the *tokenization* — decides everything. Get it wrong and no learning rate, depth, or epoch count will save you. Everything we have struggled with this week lives inside this one choice.

== The task we are testing on

Our hard connectivity dataset is built so that *only global reasoning* can solve it. Every graph is two dense blobs. The two classes are made deliberately indistinguishable to any local shortcut:

#align(center)[
#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  let blob(cx, cy, col) = {
    for i in range(4) {
      let a = 90deg + i * 90deg
      circle((cx + 0.55 * calc.cos(a), cy + 0.55 * calc.sin(a)),
             radius: 0.13, fill: col, stroke: none)
    }
    // internal edges
    line((cx + 0.55*calc.cos(90deg),  cy + 0.55*calc.sin(90deg)),
         (cx + 0.55*calc.cos(180deg), cy + 0.55*calc.sin(180deg)), stroke: 0.6pt)
    line((cx + 0.55*calc.cos(180deg), cy + 0.55*calc.sin(180deg)),
         (cx + 0.55*calc.cos(270deg), cy + 0.55*calc.sin(270deg)), stroke: 0.6pt)
    line((cx + 0.55*calc.cos(270deg), cy + 0.55*calc.sin(270deg)),
         (cx + 0.55*calc.cos(0deg),   cy + 0.55*calc.sin(0deg)),   stroke: 0.6pt)
    line((cx + 0.55*calc.cos(0deg),   cy + 0.55*calc.sin(0deg)),
         (cx + 0.55*calc.cos(90deg),  cy + 0.55*calc.sin(90deg)),  stroke: 0.6pt)
    line((cx + 0.55*calc.cos(90deg),  cy + 0.55*calc.sin(90deg)),
         (cx + 0.55*calc.cos(270deg), cy + 0.55*calc.sin(270deg)), stroke: 0.6pt)
  }

  // connected
  blob(0, 0, rgb("#1a73e8"))
  blob(3, 0, rgb("#1a73e8"))
  line((0.55, 0), (2.45, 0), stroke: (paint: rgb("#137333"), thickness: 1.4pt))
  content((1.5, -1.3), text(8pt)[#good[label 1] · one bridge → *connected*])

  // disconnected
  blob(7, 0, rgb("#9334e6"))
  blob(10, 0, rgb("#9334e6"))
  line((8.0, 0.45), (8.0, -0.45), stroke: (paint: rgb("#c5221f"), thickness: 1.4pt))
  content((8.5, -1.3), text(8pt)[#bad[label 0] · extra inner edge → *two pieces*])
})
]

Both classes have the *same number of edges* and the *same degree distribution*. The only difference is whether one edge happens to cross between the blobs. To see that, the model must trace reachability across the whole graph. That is the point — it is a pure test of global reasoning.

= Attempt 1 — vertices + edges + a task token

This is the Sanford-style tokenization in `src/token_model.py`. We build one token per vertex, one token per edge, and a single "task" token that reads out the answer:

#align(center)[
#cetz.canvas(length: 1cm, {
  import cetz.draw: *
  let tok(x, body, col) = {
    rect((x, -0.35), (x + 1.15, 0.35), radius: 3pt, fill: col, stroke: 0.5pt)
    content((x + 0.575, 0), text(7.5pt, white)[#body])
  }
  tok(0,    [$v_1$], rgb("#1a73e8"))
  tok(1.25, [$v_2$], rgb("#1a73e8"))
  tok(2.5,  [$dots$], rgb("#1a73e8"))
  tok(3.75, [$e_(12)$], rgb("#e8710a"))
  tok(5.0,  [$e_(34)$], rgb("#e8710a"))
  tok(6.25, [$dots$], rgb("#e8710a"))
  tok(7.5,  [task], rgb("#5f6368"))
  content((4.3, -1.0), text(8pt)[vertices · edges · readout — all attend to each other])
})
]

An edge token has to "name" the two vertices it connects. We do that by giving each vertex an *identity vector* and building the edge token from its two endpoints' identities:

$ "edge"_(u v) = W ("id"_u + "id"_v) $

Here is where it breaks. The identities come from a *shared lookup table* indexed by node position: `nn.Embedding(max_nodes, 16)`. Position 5 in *every* graph reads the same row of that table.

== Why it gets stuck at $ln 2 approx 0.693$

The blob split point varies from graph to graph. So node position 5 lands in blob A in one graph and blob B in another:

#align(center)[
#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // graph A: split at 6 -> pos 5 in blob A
  content((-0.2, 1.2), anchor: "west", text(8pt)[Graph A · split = 6])
  for i in range(10) {
    let col = if i == 5 { rgb("#c5221f") } else if i < 6 { rgb("#1a73e8") } else { rgb("#9334e6") }
    circle((i * 0.5, 0.4), radius: 0.12, fill: col, stroke: none)
    content((i * 0.5, 0.78), text(6pt)[#i])
  }
  content((5.3, 0.4), anchor: "west", text(7.5pt)[pos 5 ∈ #text(rgb("#1a73e8"))[blob A]])

  // graph B: split at 4 -> pos 5 in blob B
  content((-0.2, -0.6), anchor: "west", text(8pt)[Graph B · split = 4])
  for i in range(10) {
    let col = if i == 5 { rgb("#c5221f") } else if i < 4 { rgb("#1a73e8") } else { rgb("#9334e6") }
    circle((i * 0.5, -1.4), radius: 0.12, fill: col, stroke: none)
    content((i * 0.5, -1.05), text(6pt)[#i])
  }
  content((5.3, -1.4), anchor: "west", text(7.5pt)[pos 5 ∈ #text(rgb("#9334e6"))[blob B]])
})
]

When we train, Graph A pushes the embedding of "position 5" to mean *blob A*, and Graph B pushes the *same* embedding to mean *blob B*. The two gradients point in opposite directions and #bad[cancel]:

$ nabla_("id"_5) = underbrace(g_A, "“be blob A”") + underbrace(g_B, "“be blob B”") approx 0 $

The optimizer receives almost no net signal. The model parks at the only safe answer — output $[0.5, 0.5]$ for everything — whose loss is exactly $ln 2 = 0.6931$. That is the flat line we watched for 320 epochs. Raising the learning rate did nothing, because $"lr" times 0 = 0$.

#block(fill: luma(245), inset: 8pt, radius: 3pt)[
  *The tell:* it overfits 10 graphs perfectly but cannot move on 1000. With 10 fixed graphs each position has *one* consistent role, so nothing cancels. Add variety and the shared table tears itself apart.
]

= Attempt 2 — adjacency rows as tokens

Idea: skip the identity table entirely. Let each node's token *be* its row of the adjacency matrix — a binary vector saying exactly who it connects to (padded to `max_nodes = 24`).

#align(center)[
#cetz.canvas(length: 1cm, {
  import cetz.draw: *
  // tiny graph 0-1-2  3
  content((0.6, 1.7), text(8pt)[graph])
  circle((0, 1), radius: 0.12, fill: rgb("#1a73e8"), stroke: none); content((0,1.35), text(6pt)[0])
  circle((1.2, 1), radius: 0.12, fill: rgb("#1a73e8"), stroke: none); content((1.2,1.35), text(6pt)[1])
  circle((0.6, 0.1), radius: 0.12, fill: rgb("#1a73e8"), stroke: none); content((0.6,-0.25), text(6pt)[2])
  line((0,1),(1.2,1), stroke: 0.6pt)
  line((0,1),(0.6,0.1), stroke: 0.6pt)

  // matrix
  content((4.6, 1.7), text(8pt)[tokens = rows of $A$])
  let rows = (("0", "0 1 1 0 …"), ("1", "1 0 0 0 …"), ("2", "1 0 0 0 …"))
  for (k, (lbl, vec)) in rows.enumerate() {
    let y = 1.1 - k * 0.55
    content((3.4, y), anchor: "east", text(7pt)[node #lbl])
    rect((3.5, y - 0.22), (6.6, y + 0.22), radius: 2pt, fill: rgb("#fef7e0"), stroke: 0.5pt)
    content((5.05, y), text(7.5pt, font: "DejaVu Sans Mono")[#vec])
  }
})
]

This #good[fixed the gradient problem] — the loss fell to 0.03, the model clearly learns. But test accuracy stalled around #bad[0.59]. It *memorizes* and refuses to *generalize*. Why?

== Why it cannot transfer

Row $i$ has a 1 in column $j$ purely because of how this particular graph happened to *number* its nodes. The very same graph, relabelled, produces completely different tokens:

#align(center)[
#cetz.canvas(length: 1cm, {
  import cetz.draw: *
  // same structure, two labelings
  let triangle(ox, labels, col) = {
    let pts = ((ox, 0.9), (ox + 1.0, 0.9), (ox + 0.5, 0.05))
    line(pts.at(0), pts.at(1), stroke: 0.6pt)
    line(pts.at(0), pts.at(2), stroke: 0.6pt)
    for (k, p) in pts.enumerate() {
      circle(p, radius: 0.12, fill: col, stroke: none)
      content((p.at(0), p.at(1) + 0.32), text(6.5pt)[#labels.at(k)])
    }
  }
  triangle(0, ("0","1","2"), rgb("#1a73e8"))
  content((0.5, -0.55), text(7.5pt)[bridge = edge (0,2)])
  triangle(3.2, ("7","19","4"), rgb("#9334e6"))
  content((3.7, -0.55), text(7.5pt)[bridge = edge (7,4)])
  content((6.0, 0.45), anchor: "west", text(8pt)[#bad[same shape, different tokens]])
})
]

The network learned "node 3 has a 1 in column 15" — a fact glued to one arbitrary numbering. A new test graph numbers its nodes differently, so none of the learned patterns line up. The information is *there*, but in a form that does not survive relabelling. This property has a name: the representation is *not permutation-invariant*.

= The one problem behind both failures

#align(center)[
#table(
  columns: (auto, auto, auto, 1fr),
  inset: 6pt,
  align: (left, center, center, left),
  stroke: 0.4pt + luma(180),
  table.header([*Tokenization*], [*Trains?*], [*Generalizes?*], [*Failure*]),
  [vertices + edges + learned IDs], [#bad[no]], [—], [shared ID table → gradients cancel → stuck at $ln 2$],
  [adjacency rows], [#good[yes]], [#bad[no]], [tokens tied to node numbering → no transfer],
)
]

Both are two sides of the same coin — *node identity*. A transformer must be able to tell its tokens apart, but for a graph the "names" of nodes are arbitrary. We need identities that are:

#align(center)[
#table(columns: 2, stroke: none, inset: (x: 6pt, y: 3pt), align: left,
  [#good[*unique within a graph*]], [so an edge token can name its two endpoints,],
  [#good[*stable across epochs*]], [so the gradient does not fight itself,],
  [#good[*not shared across graphs*]], [so position 5 in one graph cannot collide with position 5 in another,],
  [#good[*structure-only*]], [so a relabelled copy is treated identically.],
)
]

The learned table violates #bad[#3] (shared across graphs). The adjacency row violates #bad[#4] (carries the arbitrary numbering). No tokenization we have tried satisfies all four at once — and that, not depth or learning rate, is why the model will not learn.

== The direction that satisfies all four

Give each node a *fresh random identity, drawn once when the graph is created and stored with it* (`data.nid` in the dataset, not a lookup table). Random vectors are #good[unique] with probability 1, #good[stable] because they are saved, #good[never shared] because each graph rolls its own, and #good[structure-only] because they carry no positional meaning — they are pure "name tags" the attention layers can use to bind edges to vertices. This is the standard fix in the graph-transformer literature, and it is the natural next experiment.
