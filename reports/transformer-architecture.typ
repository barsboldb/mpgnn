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
  #text(size: 19pt, weight: "bold")[How a Transformer Works]
  #v(0.3em)
  #text(size: 11pt)[Attention, multi-head, blocks, positional information, and how to read the output]
  #v(0.2em)
  #text(size: 9.5pt, fill: luma(100))[architecture primer · `src/transformer.py` · `src/layers.py` · #datetime.today().display()]
]

#v(0.6em)

#note[
  *In one sentence.* A transformer turns a set of token vectors into a better set of token vectors by repeating one idea — *let every token gather information from every other token, weighted by how relevant they are* (attention), then *let each token think on its own* (a small MLP) — and stacks that idea many times, reading the answer off whichever token(s) you choose.
]

= The shape of a transformer

A transformer operates on a *sequence (or set) of tokens*, each a vector of width $d$. Everything in between the input and the output is a stack of identical *blocks*; each block has two sub-layers — *multi-head self-attention* (tokens exchange information) and a *feed-forward network* (each token is transformed on its own). The output is read from one or more tokens, depending on the task.

#align(center)[
#cetz.canvas(length: 1cm, {
  import cetz.draw: *
  let tok(x, lbl, col) = {
    rect((x, 0), (x + 0.85, 0.6), radius: 2pt, fill: col.lighten(78%), stroke: 0.5pt + col)
    content((x + 0.425, 0.3), text(7pt)[#lbl])
  }
  // input tokens
  for (i, l) in (([$x_1$], [$x_2$], [$x_3$], [$dots$], [$x_n$])).enumerate() {
    tok(i * 0.95, l, blue)
  }
  content((2.05, -0.45), text(7.5pt)[input tokens $x_1 dots x_n$])
  // arrow up into the stack
  line((2.1, 0.75), (2.1, 1.4), mark: (end: ">"), stroke: 0.6pt)
  // block stack
  let by = 1.5
  for b in range(3) {
    let y = by + b * 0.95
    rect((0.0, y), (4.2, y + 0.7), radius: 3pt, fill: orange.lighten(85%), stroke: 0.5pt + orange)
    content((2.1, y + 0.35), text(6.5pt)[block #(b + 1)  ·  attention + FFN])
  }
  content((4.45, by + 1.4), anchor: "west", text(7pt)[$N times$ identical])
  // readout
  line((2.1, by + 2.85), (2.1, by + 3.25), mark: (end: ">"), stroke: 0.6pt)
  rect((1.3, by + 3.25), (2.9, by + 3.85), radius: 3pt, fill: green.lighten(80%), stroke: 0.5pt + green)
  content((2.1, by + 3.55), text(7pt)[readout])
  content((3.1, by + 3.55), anchor: "west", text(7.5pt)[$arrow.r$ logits])
})
]

Two facts shape everything else:

/ #text(blue)[Tokens never disappear]: the block maps $n$ tokens to $n$ tokens of the same width. Information accumulates in a *residual stream* — each sub-layer *adds* to the running token vectors rather than replacing them.
/ #text(purple)[Attention is permutation-equivariant]: shuffle the input tokens and the output is the same, just shuffled. The architecture has *no built-in notion of order or position* — that has to be supplied separately (§9).

= The residual stream: a shared workspace

Picture each token as a horizontal "wire" carrying a $d$-dimensional vector through the depth of the network. A block never overwrites a wire; it computes an update and *adds* it on:

$ x <- x + "Attention"(x), quad x <- x + "FFN"(x) . $

#align(center)[
#cetz.canvas(length: 1cm, {
  import cetz.draw: *
  // three residual-stream wires
  for (i, name) in (([tok 1], [tok 2], [tok 3])).enumerate() {
    let y = 2 - i * 0.9
    line((0, y), (9, y), stroke: 0.8pt + luma(140))
    content((-0.15, y), anchor: "east", text(7pt)[#name])
  }
  // attention block: mixes across wires (vertical coupling)
  rect((1.6, -0.6), (2.5, 2.6), radius: 3pt, fill: blue.lighten(85%), stroke: 0.5pt + blue)
  content((2.05, -0.95), text(6.5pt, fill: blue)[attention])
  content((2.05, 2.9), text(6.5pt)[mixes tokens])
  // FFN block: per-wire (no coupling)
  for i in range(3) {
    let y = 2 - i * 0.9
    rect((4.6, y - 0.22), (5.5, y + 0.22), radius: 2pt, fill: orange.lighten(80%), stroke: 0.5pt + orange)
  }
  content((5.05, -0.95), text(6.5pt, fill: orange)[FFN (per token)])
  content((5.05, 2.9), text(6.5pt)[each token alone])
  // "+" residual markers
  for x in (3.4, 6.4) {
    for i in range(3) {
      let y = 2 - i * 0.9
      circle((x, y), radius: 0.12, fill: white, stroke: 0.5pt)
      content((x, y), text(7pt)[+])
    }
  }
})
]

This split is the whole division of labour: *attention is the only place tokens talk to each other*; the *FFN is the only place a token is nonlinearly transformed*. Stacking blocks alternates "communicate" and "compute" — and the additive stream means gradients flow cleanly to the bottom, which is what lets transformers be very deep.

= Self-attention: the core mechanism

The intuition is a *soft, content-based lookup*. Each token asks a question (a *query*), every token advertises a label (a *key*), and carries a payload (a *value*). A token's output is the average of all payloads, weighted by how well each key answers its query.

From each token vector $x_i$ we make three projections:

$ q_i = W_Q x_i, quad k_i = W_K x_i, quad v_i = W_V x_i . $

Token $i$ scores every token $j$ by the dot product $q_i dot k_j$ (high when query and key point the same way), turns the scores into weights with a softmax, and returns the weighted sum of values:

$ alpha_(i j) = "softmax"_j (frac(q_i dot k_j, sqrt(d_k))), quad "out"_i = sum_j alpha_(i j) thin v_j . $

#align(center)[
#cetz.canvas(length: 1cm, {
  import cetz.draw: *
  // query
  rect((0, 1.1), (1.0, 1.6), radius: 2pt, fill: purple.lighten(70%), stroke: 0.5pt + purple)
  content((0.5, 1.35), text(7pt)[$q_i$])
  content((0.5, 1.95), text(6.5pt, fill: purple)[query])
  // keys column
  content((2.2, 2.55), text(6.5pt, fill: blue)[keys])
  for j in range(4) {
    let y = 2.0 - j * 0.55
    rect((1.9, y - 0.2), (2.6, y + 0.2), radius: 2pt, fill: blue.lighten(70%), stroke: 0.4pt + blue)
    content((2.25, y), text(6.5pt)[$k_#(j+1)$])
    // dot-product arrows
    line((1.0, 1.35), (1.9, y), stroke: 0.4pt + luma(150))
  }
  content((3.5, 2.55), text(6.5pt)[score = $q dot k$])
  // scores -> softmax
  let sc = (0.7, 0.15, 0.1, 0.05)
  for (j, w) in sc.enumerate() {
    let y = 2.0 - j * 0.55
    content((3.5, y), text(6.5pt)[#str(w)])
  }
  content((4.5, 1.1), text(7pt)[$arrow.r.long$])
  content((4.5, 1.5), text(6pt)[softmax])
  // weighted values
  content((6.3, 2.55), text(6.5pt, fill: green)[values])
  for j in range(4) {
    let y = 2.0 - j * 0.55
    let w = sc.at(j)
    rect((5.7, y - 0.2), (6.4, y + 0.2), radius: 2pt, fill: green.lighten(90% - w * 60%), stroke: 0.4pt + green)
    content((6.05, y), text(6.5pt)[$v_#(j+1)$])
    content((6.95, y), text(6pt)[$times #str(w)$])
  }
  // sum -> output
  line((7.5, 1.1), (8.2, 1.1), mark: (end: ">"), stroke: 0.6pt)
  content((7.85, 1.45), text(6pt)[$sum$])
  rect((8.3, 0.85), (9.3, 1.35), radius: 2pt, fill: orange.lighten(65%), stroke: 0.5pt + orange)
  content((8.8, 1.1), text(7pt)[$"out"_i$])
})
]

The output token is a *blend of the value vectors of whichever tokens it found relevant*. Nothing about which tokens are relevant is fixed — it is recomputed from the content every forward pass. That dynamism is the source of the transformer's power.

== The whole thing at once: the attention matrix

Stacking all queries and keys, the scores form an $n times n$ matrix; the softmax is taken along each row, so *every row sums to 1*. It is a data-dependent "mixing matrix" that says how much each token reads from each other token:

$ "Attention"(Q, K, V) = underbrace("softmax"(frac(Q K^top, sqrt(d_k))), [n times n] "weights") V . $

#align(center)[
#cetz.canvas(length: 1cm, {
  import cetz.draw: *
  let n = 6
  let s = 0.5
  // a plausible attention matrix (rows ~ sum to 1), darker = more weight
  let w = (
    (0.5, 0.2, 0.1, 0.1, 0.05, 0.05),
    (0.1, 0.6, 0.1, 0.1, 0.05, 0.05),
    (0.05, 0.1, 0.5, 0.2, 0.1, 0.05),
    (0.05, 0.05, 0.2, 0.5, 0.15, 0.05),
    (0.05, 0.05, 0.1, 0.2, 0.5, 0.1),
    (0.1, 0.05, 0.05, 0.1, 0.2, 0.5),
  )
  for r in range(n) {
    for c in range(n) {
      let x = c * s
      let y = -r * s
      rect((x, y), (x + s, y - s), fill: blue.lighten(100% - w.at(r).at(c) * 90%), stroke: 0.3pt + luma(200))
    }
    content((-0.3, -r * s - s / 2), text(6pt)[$q_#(r+1)$])
    content((r * s + s / 2, 0.28), text(6pt)[$k_#(r+1)$])
  }
  content((n * s + 0.7, -1.2), anchor: "west", text(7pt)[each #text(blue)[row] = how query $q_i$])
  content((n * s + 0.7, -1.6), anchor: "west", text(7pt)[spreads its attention])
  content((n * s + 0.7, -2.0), anchor: "west", text(7pt)[over the keys (sums to 1)])
})
]

= Why attention is so useful

#align(center)[
#table(
  columns: (auto, 1fr),
  inset: 6pt, align: (left, left),
  stroke: 0.4pt + luma(180),
  table.header([*property*], [*why it matters*]),
  [#good[content-based]], [which tokens interact is decided by the data, not by a fixed kernel or a fixed graph — the model *learns what to look at*,],
  [#good[global in one step]], [any token can read any other in a single layer; long-range dependencies do not have to be relayed hop-by-hop (unlike RNNs or message passing),],
  [#good[dynamic routing]], [the mixing matrix is recomputed per input, so the same weights handle very different structures,],
  [#good[permutation-equivariant]], [it acts on a *set*; perfect for tokens, pixels, or graph nodes with no natural order,],
  [#good[parallel]], [all positions are processed at once (no sequential recurrence), which is what makes training on modern hardware fast.],
)
]

The contrast with earlier architectures is the point: a convolution mixes a *fixed local window*, an RNN mixes *sequentially through time*, a message-passing GNN mixes *along fixed edges*. Attention mixes *all-pairs, content-dependently, in parallel*. (In this codebase, `GlobalAttnConv` in `src/layers.py` is exactly all-pairs attention over graph nodes — a graph "conv" with no edge restriction.)

== A detail that matters: the $1 \/ sqrt(d_k)$ scale

Dot products of two $d_k$-dimensional vectors grow in magnitude like $sqrt(d_k)$. Left unscaled, large scores push the softmax into a near one-hot spike with vanishing gradients. Dividing by $sqrt(d_k)$ keeps the scores in a sane range so the softmax stays soft and trainable. Small, but the model does not learn well without it.

= Multi-head attention

One attention operation can only express *one* notion of relevance at a time. *Multi-head* attention runs several attentions in parallel, each in its own learned subspace, so different heads can specialise — one might track syntactic neighbours, another long-range coreference, another positional offset.

Split the width $d$ into $h$ heads of size $d \/ h$; each head has its own $W_Q, W_K, W_V$; attend independently; concatenate; project back to $d$:

$ "head"_ell = "Attention"(X W_Q^ell, X W_K^ell, X W_V^ell), quad "MHA"(X) = "Concat"("head"_1, dots, "head"_h) thin W_O . $

#align(center)[
#cetz.canvas(length: 1cm, {
  import cetz.draw: *
  // input
  rect((0, 1.0), (1.1, 1.6), radius: 2pt, fill: luma(235), stroke: 0.5pt)
  content((0.55, 1.3), text(7pt)[$X$])
  // split into heads
  let cols = (blue, purple, orange, green)
  for hh in range(4) {
    let y = 2.6 - hh * 0.78
    line((1.1, 1.3), (2.2, y), stroke: 0.4pt + luma(160))
    rect((2.2, y - 0.27), (3.9, y + 0.27), radius: 2pt, fill: cols.at(hh).lighten(80%), stroke: 0.5pt + cols.at(hh))
    content((3.05, y), text(6.5pt)[head #(hh+1): $"Attn"(Q_#(hh+1), K_#(hh+1), V_#(hh+1))$])
    line((3.9, y), (5.0, 1.3), stroke: 0.4pt + luma(160))
  }
  content((3.05, 3.05), text(6.5pt)[$h$ parallel heads, each a different subspace / relation])
  // concat + project
  rect((5.0, 1.0), (6.3, 1.6), radius: 2pt, fill: luma(235), stroke: 0.5pt)
  content((5.65, 1.3), text(6pt)[concat])
  line((6.3, 1.3), (6.8, 1.3), mark: (end: ">"), stroke: 0.6pt)
  rect((6.8, 1.0), (7.7, 1.6), radius: 2pt, fill: luma(235), stroke: 0.5pt)
  content((7.25, 1.3), text(6.5pt)[$W_O$])
  line((7.7, 1.3), (8.4, 1.3), mark: (end: ">"), stroke: 0.6pt)
  content((8.9, 1.3), text(7pt)[out])
})
]

Crucially the heads cost almost nothing extra: $h$ heads of size $d\/h$ have the same total parameter count and FLOPs as one head of size $d$. You get the diversity of several attention patterns for the price of one. (In `_EncoderBlock`, this is the single `nn.MultiheadAttention(dim, heads, ...)` call; in `GlobalAttnConv` it is the explicit `[N, N, H]` score tensor.)

= The transformer block

Each block wraps the two sub-layers with *layer normalization* and *residual connections*, and the FFN is a small two-layer MLP that widens to $4d$ and back. Modern transformers put the norm *before* each sub-layer (*pre-norm*), which trains far more stably at depth.

#align(center)[
#cetz.canvas(length: 1cm, {
  import cetz.draw: *
  let box(x, y, w, body, col) = {
    rect((x, y), (x + w, y + 0.55), radius: 2pt, fill: col.lighten(82%), stroke: 0.5pt + col)
    content((x + w / 2, y + 0.275), text(6.5pt)[#body])
  }
  let arr(x0, y0, x1, y1) = line((x0, y0), (x1, y1), mark: (end: ">"), stroke: 0.6pt)
  // main spine
  content((0, 0.275), text(7pt)[$x$])
  arr(0.3, 0.275, 0.9, 0.275)
  box(0.9, 0, 1.4, [LayerNorm], purple)
  arr(2.3, 0.275, 2.7, 0.275)
  box(2.7, 0, 2.0, [multi-head attn], blue)
  // residual add 1
  circle((5.1, 0.275), radius: 0.14, fill: white, stroke: 0.5pt); content((5.1, 0.275), text(7pt)[+])
  arr(4.7, 0.275, 4.96, 0.275)
  arr(5.24, 0.275, 5.7, 0.275)
  box(5.7, 0, 1.4, [LayerNorm], purple)
  arr(7.1, 0.275, 7.5, 0.275)
  box(7.5, 0, 1.7, [FFN ($4 d$)], orange)
  circle((9.6, 0.275), radius: 0.14, fill: white, stroke: 0.5pt); content((9.6, 0.275), text(7pt)[+])
  arr(9.2, 0.275, 9.46, 0.275)
  arr(9.74, 0.275, 10.1, 0.275)
  content((10.35, 0.275), text(7pt)[out])
  // residual skip arcs
  line((0.6, 0.275), (0.6, 1.05), (5.1, 1.05), (5.1, 0.42), stroke: (paint: green, thickness: 0.6pt), mark: (end: ">"))
  content((2.8, 1.25), text(6pt, fill: green)[residual skip])
  line((5.4, 0.275), (5.4, -0.5), (9.6, -0.5), (9.6, 0.12), stroke: (paint: green, thickness: 0.6pt), mark: (end: ">"))
  content((7.5, -0.7), text(6pt, fill: green)[residual skip])
})
]

This is exactly `_EncoderBlock.forward` in `src/transformer.py`:
```python
h = self.norm1(x);  a = self.attn(h, h, h, ...);  x = x + a   # attention sub-layer
h = self.norm2(x);  x = x + self.ffn(h)                       # feed-forward sub-layer
```
The residual adds are *not optional decoration* — they are what carries the original signal forward and keeps gradients healthy, which is why every transformer has them regardless of how the config exposes other knobs.

= Positional information

Because attention is permutation-equivariant, a plain transformer cannot tell "dog bites man" from "man bites dog" — it sees a *set*. Position must be injected. The main families:

#align(center)[
#table(
  columns: (auto, 1fr),
  inset: 6pt, align: (left, left),
  stroke: 0.4pt + luma(180),
  table.header([*method*], [*idea*]),
  [absolute (sinusoidal / learned)], [add a position vector to each token embedding (original Transformer, BERT, GPT),],
  [relative], [bias attention by the *offset* $i - j$ rather than absolute index — generalises better across lengths,],
  [RoPE / ALiBi], [rotate queries/keys by position, or add a distance-decaying bias to the scores — the dominant choices in modern LLMs,],
  [attention bias], [add a learned bias per relation directly to the score matrix (Graphormer). #text(blue)[In `GlobalAttnConv` this is the shortest-path-distance bias `spd_bias`],],
  [graph Laplacian PE], [for graphs, use Laplacian eigenvectors as positional coordinates — see the companion `laplacian-eigenvectors.typ`.],
)
]

= Masking: who is allowed to attend to whom

The attention matrix can be *masked* (set entries to $-infinity$ before the softmax) to forbid certain token pairs. Two patterns dominate, and structured masks build the rest:

#grid(columns: (1fr, 1fr), gutter: 14pt,
[
  #align(center)[
  #cetz.canvas(length: 1cm, {
    import cetz.draw: *
    let s = 0.4
    for r in range(5) { for c in range(5) {
      rect((c*s, -r*s), (c*s+s, -r*s - s), fill: blue.lighten(55%), stroke: 0.3pt + luma(200))
    }}
    content((2.5*s, 0.3), text(7pt)[bidirectional])
    content((2.5*s, -5*s - 0.3), text(6pt)[every token sees every token])
  })
  ]
],
[
  #align(center)[
  #cetz.canvas(length: 1cm, {
    import cetz.draw: *
    let s = 0.4
    for r in range(5) { for c in range(5) {
      let on = c <= r
      rect((c*s, -r*s), (c*s+s, -r*s - s), fill: if on { purple.lighten(50%) } else { luma(243) }, stroke: 0.3pt + luma(200))
    }}
    content((2.5*s, 0.3), text(7pt)[causal])
    content((2.5*s, -5*s - 0.3), text(6pt)[token $t$ sees only $1 dots t$])
  })
  ]
]
)

*Bidirectional* (encoder) lets every token use full context — right for classification and for our graph tasks. *Causal* (decoder) forbids looking ahead — required for left-to-right generation, where the model must predict token $t$ from only $1 dots t-1$. Padding masks hide filler tokens in ragged batches; and arbitrary *structured* masks encode task rules — e.g. the scratchpad mask in `cot-tokens.typ`, where the graph is read-only, the scratchpad is causal, and a task token sees everything.

= How to read the output

A block returns $n$ refined token vectors. *Which* of them you turn into a prediction is the readout, and it depends on the task:

#align(center)[
#cetz.canvas(length: 1cm, {
  import cetz.draw: *
  let toks(ox, hl) = {
    for i in range(5) {
      let on = hl == i
      circle((ox + i * 0.5, 0), radius: 0.15,
             fill: if on { green } else { blue.lighten(40%) }, stroke: none)
    }
  }
  // per-token
  toks(0, -1)
  for i in range(5) { line((i*0.5, -0.25), (i*0.5, -0.7), mark: (end: ">"), stroke: 0.4pt) }
  content((1.0, -1.0), text(6.5pt)[per-token])
  content((1.0, -1.35), text(6pt)[node / sequence labels])

  // pooled
  toks(4.2, -1)
  for i in range(5) { line((4.2 + i*0.5, -0.25), (5.2, -0.7), stroke: 0.4pt + luma(160)) }
  circle((5.2, -0.85), radius: 0.13, fill: orange, stroke: none)
  content((5.2, -1.2), text(6.5pt)[pooled (mean/max/sum)])
  content((5.2, -1.55), text(6pt)[one vector per graph])

  // CLS / task token: one input token (green) gathers from all, you read it
  toks(8.2, 0)   // first token is the green CLS/task token
  for i in range(5) { line((8.2 + i*0.5, -0.25), (9.2, -0.7), stroke: 0.4pt + luma(160)) }
  circle((9.2, -0.85), radius: 0.13, fill: green, stroke: none)
  content((9.2, -1.2), text(6.5pt)[task / CLS token])
  content((9.2, -1.55), text(6pt)[attends to all; read this])
})
]

/ #text(blue)[Per-token]: feed each token to a shared classifier. Used for sequence labelling and *node classification* (our `task: node`).
/ #text(orange)[Pooling]: average / max / sum all tokens into one vector, then classify. Used for sentence- and *graph-level* tasks (`pooling: mean` on the node path).
/ #text(green)[CLS / task token]: prepend (or append) one special learnable token whose only job is to gather a summary via attention; read the answer off it. This is the *task token* in our `node_edge` and CoT models.
/ #text(purple)[Autoregressive]: in a decoder, the output at each position predicts the *next* token; generation feeds predictions back in one at a time.

= Families of transformers

The same block composes into three architectures, distinguished by masking and by whether two sequences are involved:

#align(center)[
#cetz.canvas(length: 1cm, {
  import cetz.draw: *
  let stack(ox, lbl, col, sub) = {
    for b in range(3) {
      rect((ox, b * 0.5), (ox + 2.0, b * 0.5 + 0.4), radius: 2pt, fill: col.lighten(82%), stroke: 0.5pt + col)
    }
    content((ox + 1.0, 1.75), text(7.5pt, weight: "bold")[#lbl])
    content((ox + 1.0, -0.4), text(6.5pt)[#sub])
  }
  stack(0, [encoder-only], blue, [bidirectional · BERT])
  stack(3.2, [decoder-only], purple, [causal · GPT])
  stack(6.4, [encoder–decoder], green, [+ cross-attention · T5])
  // cross-attention arrow for enc-dec
  line((6.3, 0.7), (6.4, 0.7), mark: (end: ">"), stroke: 0.5pt + green)
})
]

/ Encoder-only (BERT-style): bidirectional; great for understanding/classification. *Our graph models are encoders.*
/ Decoder-only (GPT-style): causal; generates text token by token. The dominant LLM design.
/ Encoder–decoder (T5-style): an encoder reads the input, a decoder generates the output and pulls from the encoder via *cross-attention* (queries from the decoder, keys/values from the encoder).

== Attention variants (beating the $O(n^2)$ cost)

Full attention compares every pair, so it costs $O(n^2 d)$ time and memory — fine for hundreds of tokens, painful for very long sequences. The efficiency literature trades exactness for scale:

#align(center)[
#table(
  columns: (auto, 1fr),
  inset: 6pt, align: (left, left),
  stroke: 0.4pt + luma(180),
  table.header([*variant*], [*idea*]),
  [local / windowed], [each token attends only to a nearby window (Longformer, sliding-window),],
  [sparse], [attend to a fixed sparse pattern + a few global tokens (BigBird, Sparse Transformer),],
  [linear / kernel], [approximate softmax so attention is $O(n)$ (Performer, Linear Transformer),],
  [low-rank], [project keys/values to a small set before attending (Linformer),],
  [flash attention], [exact, but a fused kernel that never materialises the $n times n$ matrix — faster and lower-memory.],
)
]

For graphs there is a further axis: *graph transformers* either restrict attention to edges, or keep it global and inject structure through positional encodings (Laplacian PE) and attention biases (SPD) — which is exactly the design space these experiments live in.

= Complexity at a glance

For $n$ tokens of width $d$ with an FFN factor of 4:

#align(center)[
#table(
  columns: (auto, auto, 1fr),
  inset: 6pt, align: (left, left, left),
  stroke: 0.4pt + luma(180),
  table.header([*sub-layer*], [*cost*], [*dominates when*]),
  [self-attention], [$O(n^2 d)$], [sequences are long ($n$ large),],
  [feed-forward], [$O(n d^2)$], [the model is wide ($d$ large),],
  [total per block], [$O(n^2 d + n d^2)$], [],
)
]

The quadratic-in-$n$ attention term is what the efficient variants above attack; the quadratic-in-$d$ FFN term is usually the bigger share at typical sizes.

= Summary

#align(center)[
#table(
  columns: (auto, 1fr),
  inset: 7pt, align: (left, left),
  stroke: 0.4pt + luma(180),
  table.header([*Concept*], [*What to remember*]),
  [token + residual stream], [tokens are vectors carried through depth; sub-layers *add* updates, never overwrite,],
  [self-attention], [content-based soft lookup: $"softmax"(Q K^top \/ sqrt(d_k)) V$; rows of the weight matrix sum to 1,],
  [why useful], [global, content-based, dynamic, permutation-equivariant, parallel,],
  [multi-head], [several attentions in parallel subspaces for the price of one; different heads, different relations,],
  [the block], [pre-norm + residual around (attention, then FFN); attention mixes tokens, FFN transforms each,],
  [position], [must be added (absolute / relative / RoPE / ALiBi / bias / Laplacian PE) — attention alone is order-blind,],
  [masking], [bidirectional vs causal vs structured; controls who attends to whom,],
  [readout], [per-token, pooled, or a CLS/task token — pick by task,],
  [families], [encoder (understand) · decoder (generate) · encoder–decoder (translate, via cross-attention),],
  [cost], [$O(n^2 d)$ attention drives the efficient-attention zoo.],
)
]

A transformer is, at bottom, *one block applied many times*: let tokens gather what they need from each other, let each token think, add it to the stream, repeat — then read the answer off whichever token the task points to.
