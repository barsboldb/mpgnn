// ============================================================
//  Explainer — q-sparse averaging (qSA)
//  Sanford, Hsu, Telgarsky 2023, "Representational Strengths and
//  Limitations of Transformers", NeurIPS 2023 · arXiv 2306.02896
// ============================================================

// ---- palette (shared with the other reading notes) ----
#let c-algo   = rgb("#1b6ca8")   // constructions / upper bounds (blue)
#let c-heur   = rgb("#c0392b")   // lower bounds / limits        (red)
#let c-accent = rgb("#7d3c98")   // theorems                     (purple)
#let c-good   = rgb("#1e8449")   // green
#let c-ink    = rgb("#222222")

#set page(
  paper: "a4",
  margin: (x: 1.9cm, top: 2.2cm, bottom: 1.9cm),
  numbering: "1",
  header: context {
    if counter(page).get().first() > 1 [
      #set text(8pt, fill: luma(120))
      #grid(columns: (1fr, 1fr),
        align(left)[Explainer],
        align(right)[q-sparse averaging · Sanford, Hsu, Telgarsky 2023])
      #line(length: 100%, stroke: 0.4pt + luma(200))
    ]
  },
)

#set text(font: ("New Computer Modern", "Linux Libertine"), size: 10pt, fill: c-ink)
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.")
#show heading.where(level: 1): it => {
  v(0.4em)
  block(width: 100%, inset: (y: 6pt), stroke: (bottom: 1.2pt + c-algo),
    text(14pt, weight: "bold", fill: c-algo, [#counter(heading).display() #it.body]))
  v(0.2em)
}
#show heading.where(level: 2): it => {
  v(0.3em); text(11pt, weight: "bold", fill: c-ink, it.body); v(0.1em)
}
#show link: it => text(fill: c-algo, it)

#let callout(title, body, col: c-algo, sym: "") = block(
  width: 100%, fill: col.lighten(91%), stroke: (left: 2.5pt + col),
  inset: (x: 9pt, y: 7pt), radius: 2pt, breakable: false,
  [#text(weight: "bold", fill: col, [#sym #title]) #v(-0.3em) #body],
)
#let defn(t, b) = callout("Definition — " + t, b, col: c-ink, sym: "▸")
#let thm(t, b)  = callout("Theorem — " + t, b, col: c-accent, sym: "◆")
#let note(b)    = callout("Intuition", b, col: c-good, sym: "✎")
#let idea(b)    = callout("Thesis hook", b, col: rgb("#b9770e"), sym: "→")
#let warn(b)    = callout("Careful", b, col: c-heur, sym: "!")

// ============================================================
#align(center)[
  #text(18pt, weight: "bold")[q-Sparse Averaging]
  #v(-0.4em)
  #text(11pt, fill: luma(80))[What it is, why width must grow with $q$, and what it means for this thesis]
  #v(-0.2em)
  #text(9pt, fill: luma(110))[
    Source: Sanford, Hsu, Telgarsky, _Representational Strengths and Limitations of
    Transformers_, NeurIPS 2023,
    #link("https://arxiv.org/abs/2306.02896")[arXiv:2306.02896] — Definition 4,
    Theorems 2–4, 10, 11 and Appendix B.3.
  ]
]

#v(0.4em)
#block(width: 100%, inset: 9pt, fill: luma(245), radius: 2pt)[
  *One-line summary.* In q-sparse averaging every token must fetch and average the
  values of $q$ _other_ tokens that it names. A single attention layer can do this
  exactly when its embedding width $m$ is about $q$ (up to $log N$ and precision
  factors): enough width is *sufficient* (Theorem 2) and, with $p$-bit numbers,
  $m p gt.tilde q$ is *necessary* (Theorem 4). Fully-connected networks and RNNs need
  size growing with the sequence length $N$ instead. The width needed scales with *how
  much each token must pull in*, not with how long the input is.
]

= The task

#defn("q-sparse average (qSA), Def. 4")[
  Input: $N$ tokens $x_i = (z_i; y_i; i)$, where
  - $z_i in RR^(d')$ (unit ball) is the token's *data*,
  - $y_i subset.eq [N]$ with $|y_i| = q$ is a set of $q$ positions the token *points to*,
  - $i$ is its index.

  Output for token $i$: the average of the data at the positions it points to,
  $ "qSA"(X)_i = 1/q sum_(j in y_i) z_j . $
  A function $f$ *$epsilon$-approximates* qSA if $max_i norm(f(X)_i - "qSA"(X)_i)_2 <= epsilon$
  for every input $X$.
]

== A toy instance ($N = 6$, $q = 2$, scalar data)

#let hl(x) = table.cell(fill: c-algo.lighten(75%))[#x]
#figure(
  grid(columns: (auto, auto), column-gutter: 1.6em, align: horizon,
    table(
      columns: 4, align: center, stroke: 0.4pt + luma(170), inset: 5pt,
      [*$i$*], [*$z_i$*], [*$y_i$*], [*qSA$(X)_i$*],
      [1], [3], [{2, 3}], [$(1+4)\/2 = 2.5$],
      [2], [1], [{1, 6}], [$(3+9)\/2 = 6$],
      [3], [4], [{4, 5}], [$(1+5)\/2 = 3$],
      [4], [1], [{1, 2}], [$(3+1)\/2 = 2$],
      [5], [5], [{3, 6}], [$(4+9)\/2 = 6.5$],
      [6], [9], [{2, 5}], [$(1+5)\/2 = 3$],
    ),
    table(
      columns: 7, align: center, stroke: 0.4pt + luma(170), inset: 4.5pt,
      [], [*1*], [*2*], [*3*], [*4*], [*5*], [*6*],
      [*1*], [], hl[½], hl[½], [], [], [],
      [*2*], hl[½], [], [], [], [], hl[½],
      [*3*], [], [], [], hl[½], hl[½], [],
      [*4*], hl[½], hl[½], [], [], [], [],
      [*5*], [], [], hl[½], [], [], hl[½],
      [*6*], [], hl[½], [], [], hl[½], [],
    ),
  ),
  caption: [Left: the task. Right: the attention matrix that solves it — row $i$ puts
  weight $1\/q$ on each position in $y_i$ and $0$ elsewhere. The output is then
  (attention matrix) $times$ (column of $z$'s).],
)

== Why this is "the" attention task

A self-attention unit outputs, for each token, a *convex combination of value
vectors*: $f(X)_i = sum_j "softmax"(A)_(i j) dot v_j$. qSA asks for exactly such a
combination — uniform weight $1\/q$ on an arbitrary, input-dependent set of $q$
positions. So qSA isolates the one thing attention is built to do (content-based
routing to arbitrary positions) and asks: *how wide must the key/query space be for
the routing to be precise?* Everything else (the values, the averaging) is free.

#note[
  The pointer set $y_i$ can be _any_ $q$-subset of $[N]$ — not a local window. A
  convolution or RNN that mixes nearby positions has no cheap way to do this; attention
  can, because the query of token $i$ can be made to "match" the keys of exactly the
  positions in $y_i$.
]

= Upper bound: width $m approx q log N$ is enough

#thm("Fixed precision, Thm. 2")[
  For any $N$, any $m >= Omega(d' + q log N)$, any $epsilon in (0, 1)$ and precision
  $p = Omega(log(q/epsilon dot log N))$ bits, there is a single self-attention unit (with
  an input MLP) of embedding dimension $m$ that $epsilon$-approximates qSA.
]

#thm("Infinite precision, Thm. 3")[
  With real-valued (unbounded-precision) arithmetic, $m >= Omega(d' + q)$ suffices —
  the $log N$ factor disappears.
]

== How the construction works

+ *Values carry the data:* $v_j = z_j$. Then any attention row that is uniform on $y_i$
  outputs exactly $"qSA"(X)_i$.
+ *Keys name positions:* each position $j$ gets a fixed key vector $k_j in RR^m$
  (in the paper: vertices of a polytope built from random binary vectors; Theorem 3
  uses a cyclic polytope).
+ *Queries name sets:* token $i$'s query $q_i$ is chosen so that $chevron.l q_i, k_j chevron.r$
  is a fixed large value if $j in y_i$ and noticeably smaller otherwise. After a large
  softmax temperature, the weights collapse to $≈ 1\/q$ on $y_i$ and $≈ 0$ elsewhere.

The paper builds the queries with dual certificates from compressed sensing. The
following simplified picture reproduces the same scaling:

#note[
  Take random keys $k_j in {plus.minus 1\/sqrt(m)}^m$, and let the query be the sum of
  the keys it should hit: $q_i = sum_(l in y_i) k_l$. Then
  $ chevron.l q_i, k_j chevron.r = cases(
      1 + "noise" quad & j in y_i,
      "noise" & j in.not y_i,
    ) quad "noise" approx cal(N)(0, q\/m). $
  The routing is correct if the noise is well below $1$ *simultaneously for all $N$
  positions*. The largest of $N$ Gaussians of variance $q\/m$ is about
  $sqrt(2 q log N \/ m)$, so we need
  $ m gt.tilde q log N . $
  Each of the $q$ keys summed into the query adds interference, and $log N$ pays for
  the union over all positions. A softmax temperature $beta gt.tilde log N$ then
  pushes the weight of non-members to zero; the numbers must resolve scores of that
  size, consistent with the paper's precision $p = Omega(log((q\/epsilon) log N))$.
]

The learned solution looks like this: Figure 2 in the paper shows a self-attention unit
trained on qSA ($q = 3$) converging to the predicted attention pattern, uniform on
each $y_i$.

= Lower bound: $m p gt.tilde q$ is necessary

#thm("Thm. 4")[
  For sufficiently large $q$, any $N >= 2q + 1$ and any $d' >= 1$, there is a universal
  constant $c$ such that if $m p <= c q$, *no* one-layer, single-head transformer
  (attention plus MLPs, $T^(1,1)_(d,m,d',p)$) $1/(2q)$-approximates qSA.
]

Combined with Theorem 2 (choose $p = O(log(q log N))$), the width is optimal up to
log factors in $q$ and doubly-log factors in $N$: *the necessary and sufficient width
both scale linearly with $q$.*

== The proof idea: attention is a communication channel

In one attention layer, token $i$ influences the computation over the other tokens
*only through its query vector* $Q(x_i) in RR^m$. That vector is $m$ numbers of $p$
bits: *$m p$ bits*. If token $i$'s output requires information from its own input that
cannot be compressed below $q$ bits, then $m p >= Omega(q)$.

The paper makes this exact with a reduction from *set disjointness* (DISJ): Alice holds
$a in {0,1}^q$, Bob holds $b in {0,1}^q$, and they must decide whether some $i$ has
$a_i = b_i = 1$. Any deterministic protocol needs $q$ bits of communication (Yao 1979).

#figure(
  table(
    columns: (auto, 1fr), align: (left, left), stroke: 0.4pt + luma(170), inset: 6pt,
    [*Who*], [*What they set or do*],
    [Alice (holds $a$)],
      [The pointer set of token $2q+1$: for each $i in [q]$, point at one of a pair of
       positions, the "1" slot if $a_i = 1$ and the "0" slot if $a_i = 0$.],
    [Bob (holds $b$)],
      [The data of positions $1 … 2q$: the "1" slot of pair $i$ holds $+e_1$ if
       $b_i = 1$, every other slot holds $-e_1$.],
    [Then],
      [$"qSA"(X)_(2q+1) = (|{i: a_i b_i = 1}| - |{i: a_i b_i = 0}|) \/ q dot e_1$, which
       equals $-e_1$ *iff the sets are disjoint*.],
    [Protocol],
      [Alice sends her query $Q(x_(2q+1))$ — $O(m p)$ bits. Bob knows every other
       token, so he computes the attention output himself and reads off the answer
       (threshold at $-1 + 1\/q$; a $1\/(2q)$-approximation is accurate enough).],
  ),
  caption: [The reduction behind Theorem 4 (Appendix B.3), re-indexed for readability.
  A qSA-solving layer would give an $O(m p)$-bit DISJ protocol, so $m p >= Omega(q)$.],
)

#warn[
  The bound is for *one layer with one head*. It says nothing directly about deeper
  models, and heads enter the related bounds as a product ($m p H$) — the paper's Match3
  result (Theorem 7) is the multi-head version of the same argument.
]

= Contrast: other architectures pay in $N$

#figure(
  table(
    columns: (auto, auto, auto), align: (left, left, left), stroke: 0.4pt + luma(170),
    inset: 6pt,
    [*Model*], [*Size needed for qSA*], [*Source*],
    [one self-attention unit], [$m approx q log N$ (fixed precision); $m approx q$ (infinite)], [Thms 2, 3, 4],
    [fully-connected network], [first-layer width $>= N d' \/ 2$ (for $q <= N\/2$)], [Thm 10],
    [recurrent / memory-bounded], [memory $>= (N - 1)\/2$ bits, even for $q = 1$], [Thm 11],
  ),
  caption: [The attention cost depends on $q$ (what each token must gather); the
  others depend on $N$ (how long the input is).],
)

The intuition for the gap: attention can route *any* position to *any* position in one
step, so the cost is only the precision of the routing. An MLP must keep all $N d'$
inputs linearly separable in its first layer; an RNN reading left to right must
remember everything it might later be asked to fetch.

= What it means for "width"

- *Width buys addressing capacity.* $m$ is the dimension in which keys and queries
  live; it sets how many positions a single query can pick out precisely. Width needed
  $approx$ (number of things to fetch) $times$ (bits to name one among $N$).
- *Width and precision trade.* Only the product $m p$ appears in the lower bound — a
  lower-precision model needs more width, and vice versa.
- *Length is cheap.* For attention, $N$ enters only through $log N$ (and not at all with
  infinite precision).

#idea[
  *Connection to our graph task.* With adjacency-row tokens, node $i$'s token is
  essentially a pointer set $y_i$ = its neighbours. One layer of "average my
  neighbours' states" is literally qSA with $q = deg(i)$ — so one hop of message
  passing is cheap in width ($approx "deg" dot log n$; our `swap` graphs have degree
  $approx 2$–$4$).

  Connectivity is different: after a few hops a node's *reach set* has size up to $n$,
  and gathering it in one step is qSA with $q$ growing towards $n$. That is the same
  communication logic behind Sanford et al. 2024a's one-layer bound for connectivity
  ($m H = tilde(Omega)(N)$, with $N$ = vertices + edges, their input length), and why
  depth (pointer doubling, $log N$ layers) can replace width: at that depth, width just
  above $sqrt(N)$ suffices with no extra tokens. Our Q3 measures exactly this: the critical width $m^*$ at depth
  $ceil(log_2 n)$ versus $n$.
]

#warn[
  Don't read our measured widths against these formulas directly: the theorems hide
  constants, assume a specific precision $p$, and concern _representation_ (does a
  solution exist), not _learning_ (does SGD find it). Our Q3 probe shows the gap is
  real — a width that can represent connectivity may still fail to learn it with too
  little data.
]

= Summary

#table(
  columns: (auto, 1fr), stroke: none, inset: (x: 4pt, y: 3pt),
  [*Task*], [each token averages the data of $q$ positions it names],
  [*Enough*], [one attention layer with $m approx d' + q log N$ (fixed precision)],
  [*Necessary*], [$m p >= c q$ for one layer, one head (via set disjointness)],
  [*Lesson*], [attention width scales with information fetched per token, not with $N$],
  [*For us*], [one hop of message passing is cheap; gathering whole reach sets is not — the depth/width trade-off Q3 measures],
)
