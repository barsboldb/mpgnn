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
  #text(size: 19pt, weight: "bold")[Chain-of-Thought Tokens]
  #v(0.3em)
  #text(size: 11pt)[Giving a fixed-depth transformer room to think one step at a time]
  #v(0.2em)
  #text(size: 9.5pt, fill: luma(100))[scratchpad reasoning · `src/transformer.py` · #datetime.today().display()]
]

#v(0.6em)

#note[
  *In one sentence.* A transformer's forward pass is a *parallel* circuit of fixed depth; chain-of-thought (CoT) tokens let it spend extra *sequential* steps, writing intermediate results into new token positions and reading them back — turning a shallow network into one that can unroll an iterative algorithm like BFS over many rounds.
]

= Two budgets: depth and sequential steps

A single forward pass of an $L$-layer transformer is a bounded-depth computation. Every token is refined $L$ times in parallel; information can cross at most $L$ "hops" of attention before the answer must be read out. For a problem whose answer depends on a long chain of dependencies — *is node $s$ connected to node $t$?* needs reachability across the whole graph — a shallow model simply runs out of rounds.

There are two ways to buy more computation:

#align(center)[
#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // LEFT: depth — stacked layers, parallel
  content((1.1, 3.0), text(8pt, weight: "bold")[pay with #text(blue)[depth]])
  for l in range(4) {
    let y = l * 0.62
    rect((0, y), (2.2, y + 0.42), radius: 2pt, fill: blue.lighten(78%), stroke: 0.5pt + blue)
    content((1.1, y + 0.21), text(6.5pt)[layer #(l + 1)])
  }
  line((1.1, -0.25), (1.1, 0), mark: (end: ">"), stroke: 0.5pt)
  content((1.1, -0.5), text(6.5pt)[graph tokens])
  content((1.1, -0.95), text(7pt)[4 layers · 1 pass])

  // arrows showing all positions advance together
  content((3.4, 1.3), text(11pt)[vs])

  // RIGHT: CoT — one layer, sequential scratchpad tokens
  content((6.6, 3.0), text(8pt, weight: "bold")[pay with #text(purple)[CoT steps]])
  rect((4.7, 1.6), (8.5, 2.05), radius: 2pt, fill: green.lighten(80%), stroke: 0.5pt + green)
  content((6.6, 1.82), text(6.5pt)[1 shared layer, applied repeatedly])
  for s in range(4) {
    let x = 4.7 + s * 0.95
    rect((x, 0.4), (x + 0.8, 0.85), radius: 2pt, fill: purple.lighten(75%), stroke: 0.5pt + purple)
    content((x + 0.4, 0.625), text(6.5pt)[$c_#(s + 1)$])
    if s > 0 { line((x - 0.15, 0.625), (x, 0.625), mark: (end: ">"), stroke: 0.5pt) }
  }
  content((6.6, -0.5), text(6.5pt)[each step reads all earlier ones])
  content((6.6, -0.95), text(7pt)[1 layer · 4 sequential steps])
})
]

Depth is *parallel* time; CoT is *serial* time. Crucially they are interchangeable for iterative problems: a computation that needs $D$ rounds of propagation can be paid for with depth $D$ *or* with $D$ chain-of-thought steps on a shallow model. This is the lever the thesis is missing — Sanford et al.'s depth$arrow.l.r$MPC-rounds story covers the first axis; CoT is the second. See [[thesis-sanford-graph-reasoning]].

= What a chain of thought actually is

The idea comes from large language models: prompting a model to *"show its work"* — emit intermediate reasoning before the final answer — sharply improves performance on multi-step problems (Wei et al. 2022; Nye et al. 2021, *scratchpads*). Nothing about the architecture changes. What changes is that the model is allowed to use *its own output positions as a scratchpad*.

The mechanism has two distinct ingredients:

/ #text(purple)[Extra serial computation]: each generated token is produced by another full forward pass. Ten CoT tokens means ten more passes' worth of compute spent *before* committing to the answer.
/ #text(orange)[External memory]: the generated tokens persist in the context. Position $t$ can attend to everything written at positions $< t$. The fixed-width residual stream is a cramped working memory; the growing sequence of tokens is an unbounded tape the model writes to and reads back.

#align(center)[
#cetz.canvas(length: 1cm, {
  import cetz.draw: *
  let cell(x, w, body, col) = {
    rect((x, 0), (x + w, 0.55), radius: 2pt, fill: col, stroke: 0.5pt + luma(150))
    content((x + w / 2, 0.275), text(6.5pt)[#body])
  }
  cell(0,    2.4, [input / prompt tokens], blue.lighten(80%))
  cell(2.5,  0.7, [$c_1$], purple.lighten(75%))
  cell(3.25, 0.7, [$c_2$], purple.lighten(75%))
  cell(4.0,  0.7, [$c_3$], purple.lighten(75%))
  cell(4.75, 1.3, [answer], green.lighten(78%))
  // read-back arrows
  let top = 0.55
  line((3.6, top), (3.6, top + 0.45), (2.95, top + 0.45), (2.95, top), stroke: 0.5pt + purple, mark: (end: ">"))
  line((4.35, top), (4.35, top + 0.75), (2.5, top + 0.75), (2.5, top), stroke: 0.5pt + purple, mark: (end: ">"))
  content((3.0, -0.4), text(6.5pt)[each $c_t$ reads the input and all earlier $c_(<t)$])
})
]

#note[
  *Why this adds real power.* Each forward pass is a constant-depth, highly parallel circuit — formally weak (it sits in the complexity class $sans("TC")^0$). One pass cannot, by itself, perform an inherently sequential computation of unbounded length. But a *chain* of $T$ passes composes $T$ such circuits in series, and that composition can simulate a $T$-step algorithm. Theory makes this precise: transformers with a polynomial number of CoT steps recognise far larger language classes than transformers without (Merrill & Sabharwal 2024; Feng et al. 2023). CoT trades width-bounded parallel weakness for depth-in-time.
]

= CoT in a graph transformer

Our `node_edge` model (Sanford-style) already lays each graph out as a sequence and reads the answer from a dedicated task token:

$ underbrace([v_1, dots, v_n], "vertex tokens") + underbrace([e_1, dots, e_m], "edge tokens") + underbrace([sans("task")], "readout") $

Adding chain-of-thought means inserting $K$ learnable *scratchpad* tokens between the graph and the task token. They carry no input information of their own — they are blank slots the model learns to fill with intermediate state.

#align(center)[
#cetz.canvas(length: 1cm, {
  import cetz.draw: *
  let cell(x, w, body, col) = {
    rect((x, 0), (x + w, 0.6), radius: 2pt, fill: col, stroke: 0.5pt + luma(150))
    content((x + w / 2, 0.3), text(7pt)[#body])
  }
  // vertices
  cell(0,   0.6, [$v_1$], blue.lighten(75%))
  cell(0.65, 0.6, [$v_2$], blue.lighten(75%))
  cell(1.3, 0.6, [$dots$], blue.lighten(75%))
  cell(1.95, 0.6, [$v_n$], blue.lighten(75%))
  // edges
  cell(2.7, 0.6, [$e_1$], orange.lighten(70%))
  cell(3.35, 0.6, [$dots$], orange.lighten(70%))
  cell(4.0, 0.6, [$e_m$], orange.lighten(70%))
  // scratchpad
  cell(4.75, 0.6, [$c_1$], purple.lighten(72%))
  cell(5.4, 0.6, [$dots$], purple.lighten(72%))
  cell(6.05, 0.6, [$c_K$], purple.lighten(72%))
  // task
  cell(6.8, 0.95, [task], green.lighten(76%))

  // braces / labels
  content((1.3, 1.0), text(7pt, fill: blue)[vertices])
  content((3.35, 1.0), text(7pt, fill: orange)[edges])
  content((5.4, 1.0), text(7pt, fill: purple)[scratchpad (new)])
  content((7.27, 1.0), text(7pt, fill: green)[readout])
})
]

== The attention mask is where the algorithm lives

Plain encoder attention is bidirectional — every token sees every token. For the scratchpad to behave like a sequence of computation steps, we constrain who attends to whom. The natural choice mirrors BFS rounds:

#grid(columns: (1fr, 1fr), gutter: 10pt,
[
  - #text(blue)[*Graph tokens*] attend among themselves (the fixed problem statement). They do *not* read the scratchpad, so the encoding of the graph stays stable.
  - #text(purple)[*Scratchpad token $c_i$*] reads all graph tokens *and* every earlier scratchpad token $c_(<=i)$ — but not later ones. This causal constraint forces $c_i$ to be computed from $c_(i-1)$, making the slots a genuine sequence of rounds.
  - #text(green)[*Task token*] reads everything, and the answer is taken from it after the last block.
],
[
#align(center)[
#cetz.canvas(length: 1cm, {
  import cetz.draw: *
  let toks = ($v_1$, $v_2$, $v_3$, $e_1$, $c_1$, $c_2$, $c_3$, $sans(T)$)
  let grp(i) = if i < 4 { 0 } else if i < 7 { 1 } else { 2 }  // 0=G 1=C 2=T
  let attends(q, k) = {
    let g = grp(q)
    if g == 0 { grp(k) == 0 }
    else if g == 1 { (grp(k) == 0) or (grp(k) == 1 and k <= q) }
    else { true }
  }
  let s = 0.46
  for q in range(8) {
    for k in range(8) {
      let x = k * s
      let y = -q * s
      let on = attends(q, k)
      let col = if not on { luma(243) } else if grp(q) == 0 { blue.lighten(55%) } else if grp(q) == 1 { purple.lighten(50%) } else { green.lighten(55%) }
      rect((x, y), (x + s, y - s), fill: col, stroke: 0.3pt + luma(190))
    }
    content((-0.32, -q * s - s / 2), text(5.5pt)[#toks.at(q)])
    content((q * s + s / 2, 0.26), text(5.5pt)[#toks.at(q)])
  }
  content((4 * s, 0.62), text(6pt)[key (attended to)])
  content((-0.95, -4 * s + 0.1), text(6pt)[query])
})
]
]
)

The lower-triangular #text(purple)[purple] block is the whole trick: it is the causal mask *restricted to the scratchpad*, while the graph stays fully visible to it. Stacking blocks lets each round refine the state further.

== Reading the scratchpad as BFS

With this mask, the cleanest hypothesis for what the model *can* learn is a frontier expansion. Suppose the task is "is the graph connected from a source $s$." Round $c_k$ can hold the set of nodes reachable within $k$ hops; each round attends to the graph and to the previous frontier and grows it by one ring:

#align(center)[
#cetz.canvas(length: 1cm, {
  import cetz.draw: *
  let nodes = ((0, 0), (1, 0.55), (1, -0.55), (2, 0), (3, 0.55), (3, -0.55), (4, 0))
  let edges = ((0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (3, 5), (4, 6), (5, 6))
  let level = (0, 1, 1, 2, 3, 3, 4)  // BFS distance from node 0
  let drawg(ox, kmax, lbl) = {
    for e in edges {
      let a = nodes.at(e.at(0))
      let b = nodes.at(e.at(1))
      line((ox + a.at(0) * 0.42, a.at(1) * 0.7), (ox + b.at(0) * 0.42, b.at(1) * 0.7),
           stroke: 0.5pt + luma(170))
    }
    for (i, p) in nodes.enumerate() {
      let reached = level.at(i) <= kmax
      let col = if i == 0 { green } else if reached { blue } else { luma(220) }
      circle((ox + p.at(0) * 0.42, p.at(1) * 0.7), radius: 0.12, fill: col, stroke: none)
    }
    content((ox + 0.84, -1.0), text(6.5pt)[#lbl])
  }
  drawg(0,   1, [after $c_1$: 1 hop])
  drawg(2.5, 2, [after $c_2$: 2 hops])
  drawg(5.0, 3, [after $c_3$: 3 hops])
  drawg(7.5, 4, [after $c_4$: all reached])
})
]

The #text(green)[green] node is the source; #text(blue)[blue] nodes are reached so far; grey nodes are not yet. The reachable set grows by one ring per scratchpad step, so $K$ steps cover everything within distance $K$. To decide connectivity the model needs $K >= "diameter"$ of the graph — *exactly the prediction we want to test*.

= The experiment this enables

#note[
  *The phase transition.* If CoT genuinely performs frontier expansion, accuracy should be at chance while $K < "diameter"$ and jump once $K >= "diameter"$. A model that instead exploits an embedding shortcut would not show this dependence on $K$ at all. That contrast is the experiment.
]

#align(center)[
#cetz.canvas(length: 1cm, {
  import cetz.draw: *
  // axes
  line((0, 0), (8.2, 0), mark: (end: ">"), stroke: 0.6pt)
  line((0, 0), (0, 3.0), mark: (end: ">"), stroke: 0.6pt)
  content((8.2, -0.35), text(7pt)[$K$ (CoT steps)])
  content((-0.1, 3.3), text(7pt)[test accuracy])
  // chance baseline
  line((0, 0.75), (8, 0.75), stroke: (paint: luma(150), dash: "dashed", thickness: 0.5pt))
  content((7.4, 0.95), text(6.5pt, fill: luma(110))[chance 0.5])
  // diameter marker
  line((4, 0), (4, 2.9), stroke: (paint: red, dash: "dashed", thickness: 0.6pt))
  content((4, 3.1), text(6.5pt, fill: red)[$K = "diameter"$])
  // step-like accuracy curve
  let pts = ((0, 0.75), (1, 0.76), (2, 0.78), (3, 0.82), (4, 1.05), (5, 2.4), (6, 2.7), (7, 2.78), (8, 2.8))
  for i in range(pts.len() - 1) {
    line(pts.at(i), pts.at(i + 1), stroke: 1.4pt + blue)
  }
  for p in pts { circle(p, radius: 0.06, fill: blue, stroke: none) }
  content((6.4, 2.5), text(6.5pt, fill: blue)[$approx 1.0$])
})
]

*Controls that make it clean.*
- Strip the structural shortcuts: no Laplacian PE, no SPD bias. Otherwise the answer is handed to the model in its features and $K$ becomes irrelevant (the failure mode documented for the connectedness task — see [[connectedness-dataset-leak]]).
- Sweep $K$ on graphs of *known, controlled diameter*; the transition point should track the diameter.
- Compare against a depth sweep at matched parameter count: does one CoT step buy what one extra layer buys? This directly probes the depth$arrow.l.r$serial-step equivalence.

= Supervised vs latent scratchpads

There are two ways to train the scratchpad, with very different difficulty:

#align(center)[
#table(
  columns: (auto, 1fr, 1fr),
  inset: 6pt, align: (left, left, left),
  stroke: 0.4pt + luma(180),
  table.header([], [#text(green)[*Supervised CoT*]], [#text(purple)[*Latent CoT*]]),
  [idea], [generate ground-truth traces (e.g. the BFS frontier at each round) and train each $c_k$ to predict round $k$], [insert blank scratchpad tokens and train only on the final label; the model self-organises the slots],
  [signal], [strong, per-step supervision], [weak — gradient only from the answer],
  [cost], [must synthesise correct traces; ties the model to one algorithm], [trivial to set up; nothing to label],
  [risk], [model may copy the trace format without internalising it], [optimisation is hard; slots may collapse to #bad[no-ops] (filler tokens that don't compute)],
)
]

The latent form is the more interesting scientific test (does sequential capacity *alone* unlock the task?), and it connects to recent results that even content-free *filler* or *pause* tokens can add usable computation when the task structure rewards it (Goyal et al. 2023; Pfau et al. 2024). The supervised form is the stronger baseline if latent training stalls.

= Implementation in `GraphTransformer`

The change is small and local to `src/transformer.py`. Three edits:

+ *Append $K$ scratchpad tokens* per graph, between the edge tokens and the task token. A single learnable parameter bank, plus a fourth `type_emb` entry:
  ```python
  self.cot = nn.Parameter(torch.randn(config.cot_len, d) * 0.02)  # [K, d]
  self.type_emb = nn.Embedding(4, d)   # vertex / edge / task / cot
  # in the per-graph loop, before the task token:
  ctok = self.cot + self.type_emb.weight[self.TYPE_COT]            # [K, d]
  seq  = torch.cat([vtok, etok, ctok, ttok], dim=0)
  ```

+ *Build an additive attention mask* `[L, L]` encoding the rules above (graph block full, scratchpad lower-triangular over the graph, task row all-visible). `nn.MultiheadAttention` accepts an additive `attn_mask` with $-infinity$ on forbidden entries:
  ```python
  mask = torch.zeros(L, L)
  mask[is_graph][:, is_cot | is_task] = -inf      # graph ignores scratchpad/task
  mask[is_cot]  = build_causal_over_scratchpad()  # see the grid figure
  # task row stays all-zero (sees everything)
  ```

+ *Thread the mask through the blocks.* `_EncoderBlock.forward` currently takes only `key_padding_mask`; add an `attn_mask` argument and pass it to `self.attn(...)`. The readout is unchanged — still `h[task_pos]`.

#note[
  *Two config knobs.* `cot_len` ($K$, the number of scratchpad steps — the swept variable) and a switch for whether blocks *share* the mask (they do; the same causal structure applies at every layer). Keep `lpe_dim: 0` and no SPD so the only way to solve the task is through the scratchpad. With ragged batches, build the mask at the padded length $L$ and combine it with the existing `key_padding_mask`.
]

= Pitfalls

/ #bad[Collapsed slots]: with only end-of-sequence supervision, the optimiser may leave the scratchpad as dead weight (the same saddle that stalls anonymous-identity training). Mitigations: curriculum on $K$, auxiliary per-step losses, or starting from supervised traces.
/ #bad[Leak re-entry]: any positional encoding that already encodes reachability (LPE's near-zero modes) makes $K$ irrelevant and fakes success. The CoT experiment is only meaningful with the structural shortcuts removed.
/ #bad[Diameter confound]: if graph size and diameter co-vary, a $K$-dependence might reflect size, not reasoning depth. Control diameter explicitly.
/ #bad[Attention dilution]: long scratchpads add tokens every block; with small width the task token's attention can wash out. Watch that accuracy gains track $K$ rather than raw parameter count.

= Summary

#align(center)[
#table(
  columns: (auto, 1fr),
  inset: 7pt, align: (left, left),
  stroke: 0.4pt + luma(180),
  table.header([*Concept*], [*What to remember*]),
  [CoT token], [a blank scratchpad position the model fills with intermediate state; one extra forward pass of serial compute each],
  [depth vs CoT], [parallel rounds vs sequential rounds — interchangeable for iterative tasks; CoT is the second axis the depth$arrow.l.r$MPC story omits],
  [attention mask], [graph full-visible, scratchpad causal over the graph, task sees all — the mask *is* the algorithm],
  [BFS reading], [round $c_k$ holds the $k$-hop reachable set; connectivity needs $K >= "diameter"$],
  [the test], [sweep $K$; a phase transition at $K = "diameter"$ is evidence of genuine sequential reasoning, not an embedding shortcut],
  [where], [`src/transformer.py`: add scratchpad params, a fourth token type, an `attn_mask`, and thread it through `_EncoderBlock`],
)
]

Chain-of-thought tokens turn the question *"can the transformer reason through connectivity?"* into a measurable one: give it a scratchpad, deny it the structural shortcut, and watch whether the number of thinking steps it needs matches the number of hops the problem actually requires.

#v(0.8em)
#text(size: 8.5pt, fill: luma(110))[
*References (pointers).* Wei et al. 2022, _Chain-of-thought prompting_. Nye et al. 2021, _Show your work: scratchpads_. Feng et al. 2023, _Towards revealing the mystery behind CoT_. Merrill & Sabharwal 2024, _The expressive power of transformers with chain of thought_. Goyal et al. 2023, _Think before you speak: pause tokens_. Pfau et al. 2024, _Let's think dot by dot: filler tokens_. Sanford et al. 2024a (thesis basis).
]
