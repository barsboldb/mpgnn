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
  #text(size: 19pt, weight: "bold")[Random Features, GIN, and Why Degree Wins]
  #v(0.3em)
  #text(size: 11pt)[what message passing learns on connectedness_hard — and what it only memorises]
  #v(0.2em)
  #text(size: 9.5pt, fill: luma(100))[`src/layers.py` (GINConv) · `src/dataset.py` · #datetime.today().display()]
]

#v(0.6em)

#note[
  *The puzzle.* We expected random node features (rGIN) to help an mpGNN reason about connectivity and plain degree to be useless. The runs said the opposite: 3-layer GIN reached #good[97.5% test] with *degree*, but only #bad[53% test] with *random* features — while *memorising* the training set (98.5% train). This note explains GIN, what random features actually do, why degree generalises here while random does not — and why even degree's win is probably a local shortcut, not connectivity reasoning.
]

= The two results

Both runs: 3 GIN layers, hidden 64, mean pooling, 10k graphs (8k train / 2k test), identical except the node feature.

#align(center)[
#cetz.canvas(length: 1cm, {
  import cetz.draw: *
  // axes
  let x0 = 0; let y0 = 0; let h = 3.4
  line((x0, y0), (x0, y0 + h), stroke: 0.6pt)
  line((x0, y0), (9.5, y0), stroke: 0.6pt)
  // y ticks at 0.5 and 1.0
  for (v, lbl) in ((0.5, "0.5"), (1.0, "1.0")) {
    let y = y0 + v * h
    line((x0 - 0.1, y), (x0, y), stroke: 0.5pt)
    content((x0 - 0.5, y), text(7pt)[#lbl])
  }
  // chance line
  line((x0, y0 + 0.5 * h), (9.5, y0 + 0.5 * h), stroke: (paint: luma(150), dash: "dashed", thickness: 0.5pt))
  content((9.0, y0 + 0.5 * h + 0.22), text(6.5pt, fill: luma(110))[chance])
  // bar helper
  let bar(x, val, col, lbl) = {
    rect((x, y0), (x + 0.7, y0 + val * h), fill: col.lighten(35%), stroke: 0.5pt + col)
    content((x + 0.35, y0 + val * h + 0.22), text(6.5pt)[#lbl])
  }
  // degree group
  bar(1.0, 0.925, blue,  [train .92])
  bar(1.85, 0.955, green, [test .96])
  content((1.6, -0.45), text(8pt, weight: "bold")[degree (in=1)])
  // random group
  bar(5.2, 0.985, blue,  [train .98])
  bar(6.05, 0.53, red,   [test .53])
  content((5.8, -0.45), text(8pt, weight: "bold")[random (in=16)])
  // annotations
  content((3.0, y0 + 0.78 * h), anchor: "west", text(6.5pt, fill: green)[generalises ✓])
  // train–test gap to the right of the random bars
  line((7.0, y0 + 0.985 * h), (7.0, y0 + 0.53 * h), stroke: (paint: red, thickness: 0.6pt), mark: (start: ">", end: ">"))
  content((7.2, y0 + 0.78 * h), anchor: "west", text(6pt, fill: red)[train–test gap])
  content((7.2, y0 + 0.64 * h), anchor: "west", text(6pt, fill: red)[(memorises ✗)])
})
]

The gap is the whole story: with *random* features the model fits the training set almost perfectly (98.5%) but tests at chance — a textbook generalisation failure. With *degree* train and test rise together and stay close. To see why, we need GIN and what each feature feeds it.

= How GIN works

A *Graph Isomorphism Network* layer updates every node from its own value plus the *sum* of its neighbours' values, then passes the result through a small MLP:

$ h_v^((k)) = "MLP"^((k)) ((1 + epsilon) dot h_v^((k-1)) + sum_(u in cal(N)(v)) h_u^((k-1))) . $

#align(center)[
#cetz.canvas(length: 1cm, {
  import cetz.draw: *
  // center node v with 3 neighbours
  let cx = 1.4; let cy = 1.3
  let nb = ((0.2, 2.3), (0.0, 0.5), (0.5, -0.2))
  for p in nb {
    line((cx, cy), p, stroke: 0.6pt + luma(150))
    circle(p, radius: 0.18, fill: blue.lighten(55%), stroke: 0.5pt + blue)
  }
  circle((cx, cy), radius: 0.22, fill: orange.lighten(45%), stroke: 0.6pt + orange)
  content((cx, cy), text(7pt)[$v$])
  content((cx, cy + 0.55), text(6.5pt, fill: orange)[$h_v$])
  // sum box
  content((3.0, 1.3), text(8pt)[$arrow.r.long$])
  rect((3.4, 0.9), (6.3, 1.7), radius: 3pt, fill: purple.lighten(85%), stroke: 0.5pt + purple)
  content((4.85, 1.3), text(7.5pt)[$(1{+}epsilon) h_v + sum_(u in cal(N)(v)) h_u$])
  // MLP
  content((6.5, 1.3), text(8pt)[$arrow.r$])
  rect((6.9, 0.9), (7.9, 1.7), radius: 3pt, fill: green.lighten(80%), stroke: 0.5pt + green)
  content((7.4, 1.3), text(7pt)[MLP])
  content((8.1, 1.3), text(8pt)[$arrow.r$])
  circle((8.6, 1.3), radius: 0.22, fill: orange.lighten(45%), stroke: 0.6pt + orange)
  content((8.6, 1.3), text(6.5pt)[$h'_v$])
})
]

Two design choices matter:

/ #text(blue)[*Sum*, not mean or max]: summation preserves *multiplicity* — "three neighbours valued 1" differs from "one neighbour valued 1". This is what makes GIN as discriminative as the *Weisfeiler–Lehman* graph-isomorphism test (Xu et al. 2019); mean and max throw multiplicity away and so cannot tell some structures apart.
/ #text(green)[*An MLP after aggregation*]: a learnable nonlinearity over the aggregated multiset, so the layer can compute rich functions of a neighbourhood, not just a linear average.

After $k$ layers a node's vector summarises its *$k$-hop neighbourhood*. Crucially, information travels exactly *one hop per layer* — remember this for §5.

#note[
  *What the feature feeds in.* GIN computes functions of the *node features it is given*. With $h_v^((0)) = $ degree, it builds functions of the degree pattern around each node. With $h_v^((0)) = $ a random vector, it builds functions of a random pattern. Same machinery, completely different raw material — that is the whole experiment.
]

= How random node features work

Plain GINs are blind to symmetry: two structurally-identical nodes (same neighbourhood up to isomorphism) get the *same* features forever, so the network literally cannot tell them apart. On `connectedness_hard` the two blobs are near-symmetric, so constant or matched-degree features give the model almost nothing to distinguish nodes.

*Random node initialisation* (rGIN; Sato et al. 2021, Abboud et al. 2021) breaks the symmetry by hand: give every node an independent random vector. Now distinct nodes are distinct, and sum-aggregation can build a *reachable-set fingerprint* — a node accumulates the (random) signatures of everyone it can reach:

#align(center)[
#cetz.canvas(length: 1cm, {
  import cetz.draw: *
  let nodes = ((0,0),(0.9,0.5),(0.9,-0.5),(1.8,0))
  let edges = ((0,1),(0,2),(1,3),(2,3))
  let draw(ox, lbl, merged) = {
    for e in edges {
      let a=nodes.at(e.at(0)); let b=nodes.at(e.at(1))
      line((ox+a.at(0), a.at(1)),(ox+b.at(0), b.at(1)), stroke: 0.5pt+luma(160))
    }
    for (i,p) in nodes.enumerate() {
      let c = if merged { green } else { (blue, purple, orange, red).at(i) }
      circle((ox+p.at(0), p.at(1)), radius: 0.16, fill: c.lighten(35%), stroke: 0.5pt+c)
    }
    content((ox+0.9, -1.0), text(6.5pt)[#lbl])
  }
  draw(0, [round 0: distinct random], false)
  content((3.0, 0), text(9pt)[$arrow.r.long$])
  draw(3.7, [round $gt.eq$ diameter: shared fingerprint], true)
})
]

Nodes in the *same component* converge to the *same* fingerprint (they reach the same set); nodes in *different* components keep different fingerprints. In principle that is exactly the connectivity signal — which is why we tried it.

== The catch: expressiveness is not generalisation

Random features make the network *more expressive* — but expressive power cuts both ways. With 16 random dimensions and sum-aggregation, the model can hand every training graph a *near-unique fingerprint* and simply memorise "this fingerprint $arrow.r$ this label". That is precisely what happened: #bad[98.5% train, 53% test]. The random values carry *no signal that transfers* — a test graph arrives with brand-new random numbers the model has never seen, so its memorised lookup is useless and it falls back to chance.

#note[
  Our `node_features: random` samples the vectors *once per load* (fixed for the run). That is the most memorisable setting: the model sees the same random pattern for each training graph every epoch. The stronger rGIN variant *resamples every forward pass*, which denies the model any stable values to memorise and forces it to use randomness only as a symmetry-breaker. Worth trying — but it does not change the deeper point below.
]

= Why degree generalises and random does not

#align(center)[
#table(
  columns: (auto, 1fr, 1fr),
  inset: 6pt, align: (left, left, left),
  stroke: 0.4pt + luma(180),
  table.header([], [#text(green)[*degree*]], [#text(red)[*random*]]),
  [what it is], [a real structural quantity, shared across all graphs], [per-graph noise, different every graph],
  [dimensionality], [1 — tiny capacity to memorise], [16 — ample capacity to memorise],
  [transfers to test?], [yes: the same structural pattern recurs], [no: test graphs have unseen values],
  [result], [#good[train .92 / test .96]], [#bad[train .98 / test .53]],
)
]

Generalisation needs the feature to carry signal that is *the same kind of thing* in train and test. Degree does: a degree-pattern that distinguishes the classes in training graphs distinguishes it in test graphs too. Random features do not: whatever the model learns about specific random values is meaningless on fresh draws. High capacity + no transferable signal = *memorise the train set, fail the test set*. This is the same shortcut-vs-signal lesson as the rest of these experiments, seen from the feature side.

= The twist: degree's win is (probably) a local shortcut

Before crowning degree, look at the depth. The connected-class *diameter is 4–8* (median 5), but degree succeeded with only *3 GIN layers* — a 3-hop receptive field. Information cannot cross a diameter-8 graph in 3 hops, so the model *cannot* be tracing global reachability. It must be reading something *local*.

#align(center)[
#cetz.canvas(length: 1cm, {
  import cetz.draw: *
  // two blobs joined by a bridge; mark a node and its 3-hop ball
  let blob(cx, cy, col) = {
    for a in range(5) {
      let ang = a/5*2*3.14159
      circle((cx+0.45*calc.cos(ang), cy+0.45*calc.sin(ang)), radius: 0.1, fill: col.lighten(40%), stroke: none)
    }
  }
  blob(0.6, 0, blue); blob(3.4, 0, purple)
  line((1.05, 0),(2.95, 0), stroke: 1.2pt + green)
  content((2.0, 0.32), text(6.5pt, fill: green)[bridge])
  content((2.0, -0.95), text(7pt)[a 3-hop ball around a bridge endpoint already])
  content((2.0, -1.3), text(7pt)[spans *both* dense regions — a *local* tell])
})
]

In `connectedness_hard` the classes differ by exactly one edge: a *bridge* between the blobs (connected) versus an extra edge *inside* a blob (disconnected). A bridge endpoint's few-hop neighbourhood looks different from an internal node's — two dense regions stitched together vs. one — and because the blobs are dense, that difference shows up *within ~3 hops*. So GIN+degree learns to *detect the bridge motif locally*, which on this dataset happens to decide connectivity — without any global reasoning.

#note[
  *Consequence for the thesis.* `connectedness_hard` killed the obvious degree-statistic shortcut (min/mean degree are matched), but it did *not* kill a subtler *local-structure* shortcut: a few-hop GIN can spot the bridge. Degree's 97.5% is real generalisation, but most likely of a *local pattern*, not of connectivity *reasoning*. The clean way to tell them apart:
  - *Depth ablation:* if 2–3 hops already suffice (they do), the model is local — true reachability would need depth $gt.eq$ diameter.
  - *Size generalisation:* train on $n lt.eq 20$, test on $n = 40$–$50$. A local bridge-detector keeps working (the motif is size-independent); a model that truly needed global reachability would degrade. Strong transfer here would *confirm* it is the local shortcut.
]

= Summary

#align(center)[
#table(
  columns: (auto, 1fr),
  inset: 7pt, align: (left, left),
  stroke: 0.4pt + luma(180),
  table.header([*Concept*], [*What to remember*]),
  [GIN], [sum-aggregate neighbours + self, then MLP; sum keeps multiplicity (1-WL power); 1 hop per layer],
  [random features], [break node symmetry so MP can build reachable-set fingerprints; add expressive power],
  [expressive ≠ general], [16-dim random features let GIN memorise each graph (train .98) but carry no transferable signal (test .53)],
  [why degree wins], [a low-dim, shared structural feature transfers train→test; the model learns a recurring pattern, not noise],
  [the catch], [3 hops $<$ diameter, yet 97.5% — so degree detects the bridge *locally*; a shortcut, not global connectivity reasoning],
  [how to check], [depth ablation + train-small/test-large size generalisation],
)
]

The headline is counter-intuitive but clean: *more expressive features made the model worse*, because expressiveness without transferable signal is just a licence to memorise — and the humble degree feature won by carrying real, reusable structure. Whether that structure amounts to *reasoning* or merely *local bridge-spotting* is the next thing to pin down.
