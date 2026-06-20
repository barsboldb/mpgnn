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
  #text(size: 19pt, weight: "bold")[Laplacian Eigenvector Tokenization]
  #v(0.3em)
  #text(size: 11pt)[Giving graph nodes a coordinate system the way sines and cosines give one to a line]
  #v(0.2em)
  #text(size: 9.5pt, fill: luma(100))[positional encoding · `features.laplacian_positional_encoding` · #datetime.today().display()]
]

#v(0.6em)

#note[
  *In one sentence.* A transformer needs to tell its tokens apart, but a graph's node numbers are arbitrary. Laplacian eigenvectors hand each node a short vector of coordinates that depend only on *where it sits in the graph's shape* — not on how it was numbered — so two relabelled copies of the same graph get the same encoding.
]

= Why a graph needs positional encoding at all

A transformer sees a *set of tokens* and lets every token attend to every other. By itself, attention is permutation-invariant: shuffle the tokens and you get the same answer shuffled. For text that is wrong — "dog bites man" $!=$ "man bites dog" — so transformers add a *positional encoding*: a vector glued to each token that says *where it is*.

For a sentence "where" is easy — position $1, 2, 3, dots$ along a line, encoded with sines and cosines of different frequencies. For a graph there is no such line. A node has no natural index; the only thing that is real is *the pattern of connections around it*. So the question becomes:

#align(center)[
  *What is the graph-shaped analogue of "sine and cosine of position"?*
]

The answer is the *eigenvectors of the graph Laplacian*. They are, quite literally, the natural vibration modes of the graph — the graph's own sines and cosines.

= Building block 1 — the Laplacian matrix

Start from two matrices you already know. For a graph on $n$ nodes:

#align(center)[
#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // little path graph 1-2-3-4
  content((1.5, 1.6), text(8pt)[a path graph $1!-!2!-!3!-!4$])
  let xs = (0, 1, 2, 3)
  for (i, x) in xs.enumerate() {
    circle((x, 0.7), radius: 0.16, fill: blue, stroke: none)
    content((x, 0.7), text(7pt, white)[#(i+1)])
    if i > 0 { line((x - 1, 0.7), (x, 0.7), stroke: 0.7pt) }
  }

  // adjacency
  content((6.4, 1.6), text(8pt)[adjacency $A$])
  let A = (
    (0,1,0,0),
    (1,0,1,0),
    (0,1,0,1),
    (0,0,1,0),
  )
  for (r, row) in A.enumerate() {
    for (c, v) in row.enumerate() {
      let x = 5.4 + c * 0.5
      let y = 1.0 - r * 0.5
      rect((x - 0.24, y - 0.24), (x + 0.24, y + 0.24),
           fill: if v == 1 { blue.lighten(60%) } else { luma(245) }, stroke: 0.4pt)
      content((x, y), text(7pt)[#v])
    }
  }
})
]

/ #text(blue)[Adjacency $A$]: $A_(i j) = 1$ if nodes $i$ and $j$ share an edge, else $0$. It records *who connects to whom*.
/ #text(orange)[Degree $D$]: a diagonal matrix, $D_(i i) = sum_j A_(i j) =$ the number of edges at node $i$. It records *how busy each node is*.

The *combinatorial Laplacian* is simply their difference:

$ L = D - A $

That tiny formula is doing something specific. Apply $L$ to a vector $x$ that assigns a number $x_i$ to each node, and look at row $i$:

$ (L x)_i = D_(i i) x_i - sum_(j tilde i) x_j = sum_(j tilde i) (x_i - x_j) $

So $(L x)_i$ measures *how different node $i$'s value is from its neighbours'* — a discrete second derivative, the graph version of $-nabla^2$. This is why $L$ is called the Laplacian: on a grid it literally becomes the finite-difference Laplace operator.

== The quadratic form: $L$ measures smoothness

The single most important fact about $L$ is what it computes in the quadratic form $x^top L x$:

$ x^top L x = sum_((i,j) in E) (x_i - x_j)^2 $

#note[
  Read this slowly. Hand the graph any assignment of numbers to nodes. Then $x^top L x$ is the *total squared disagreement across edges*. It is large when neighbours hold very different values (a jagged signal) and small when neighbours agree (a smooth signal). $L$ is the graph's built-in *roughness meter*.
]

Because it is a sum of squares, $x^top L x >= 0$ always: $L$ is *positive semi-definite*. Combined with $L$ being real and symmetric, this guarantees everything we need in the next step.

= Building block 2 — eigenvectors as vibration modes

Because $L$ is real, symmetric, and PSD, the spectral theorem gives us a full set of $n$ real eigenvectors $u_1, dots, u_n$ that are mutually orthogonal, with non-negative eigenvalues

$ 0 = lambda_1 <= lambda_2 <= dots <= lambda_n . $

Each eigenvector solves $L u = lambda u$. Rearranged through the quadratic form, the eigenvalue *is* the roughness of its eigenvector:

$ lambda_k = u_k^top L u_k = sum_((i,j) in E) (u_k (i) - u_k (j))^2 quad (||u_k|| = 1). $

So eigenvectors are ordered *from smoothest to roughest*. This is exactly the physics of a vibrating string or drum: low modes wobble slowly across the whole object, high modes oscillate rapidly. On a path graph the eigenvectors are literally cosine waves of increasing frequency:

#align(center)[
#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  let n = 8
  // sample three "modes" as cosines over n nodes
  let mode(freq) = range(n).map(i => calc.cos(freq * (i + 0.5) / n * 3.14159))

  let plot(ox, oy, vals, col, lbl) = {
    // baseline
    line((ox, oy), (ox + (n - 1) * 0.42, oy), stroke: (paint: luma(180), dash: "dashed", thickness: 0.4pt))
    let pts = ()
    for (i, v) in vals.enumerate() {
      let x = ox + i * 0.42
      let y = oy + v * 0.45
      pts.push((x, y))
    }
    for k in range(pts.len() - 1) {
      line(pts.at(k), pts.at(k + 1), stroke: (paint: col, thickness: 1.2pt))
    }
    for p in pts { circle(p, radius: 0.07, fill: col, stroke: none) }
    content((ox + (n - 1) * 0.21, oy - 1.05), text(7.5pt)[#lbl])
  }

  plot(0,   0, mode(0), green,  [$lambda_1 = 0$ · constant])
  plot(3.6, 0, mode(1), blue,   [$lambda_2$ · 1 oscillation])
  plot(7.2, 0, mode(3), purple, [$lambda_k$ · many oscillations])
})
]

The leftmost mode is flat — every node gets the same value. The next splits the graph gently into "one side high, one side low". Higher modes carve the graph into ever finer alternating regions. *These alternating regions are the coordinates we are after.*

= Why the first eigenvector is thrown away

The smallest eigenvalue is always $lambda_1 = 0$, and for a connected graph its eigenvector is the *constant* vector $u_1 = (1, 1, dots, 1) \/ sqrt(n)$. Check it: $L dot bold(1) = (D - A) bold(1) = 0$ because each row of $A$ sums to that node's degree.

A constant assigns *the same coordinate to every node*, so it tells you nothing about position. That is why the implementation skips it:

```python
# features.py — symmetric normalized Laplacian, drop the trivial mode
_, eigvecs = torch.linalg.eigh(L)   # ascending eigenvalues
pe = eigvecs[:, 1:k + 1]            # skip column 0 (constant), keep next k
```

#note[
  *Aside — counting connected components.* The multiplicity of eigenvalue $0$ equals the number of connected components. If the graph splits into two pieces there are *two* independent constant-like modes (one per piece). This is the spectral fingerprint of connectivity — the very property the #good[connectedness] task is about — and a reason the Laplacian spectrum is a natural language for these experiments. See [[connectedness-dataset-leak]].
]

= The normalized Laplacian we actually use

`features.py` does not use $L = D - A$ directly. It uses the *symmetric normalized* Laplacian:

$ L_"sym" = I - D^(-1/2) A D^(-1/2) . $

The normalization rescales each node by $1\/sqrt(deg)$, which keeps high-degree hubs from dominating the spectrum and pins all eigenvalues into the clean range $lambda in [0, 2]$. It is the same operator that sits inside a GCN layer, so the positional encoding speaks the same dialect as message passing.

#align(center)[
#cetz.canvas(length: 1cm, {
  import cetz.draw: *
  let box(x, body, col) = {
    rect((x, -0.4), (x + 2.3, 0.4), radius: 3pt, fill: col.lighten(80%), stroke: 0.5pt + col)
    content((x + 1.15, 0), text(8pt)[#body])
  }
  let arrow(x) = line((x, 0), (x + 0.55, 0), mark: (end: ">"), stroke: 0.6pt)
  box(0,    [graph edges], luma(120))
  arrow(2.3)
  box(2.85, [$A, D$], blue)
  arrow(5.15)
  box(5.7,  [$L_"sym" = I - D^(-1/2)A D^(-1/2)$], orange)
  content((6.85, -0.85), text(7pt)[eigh →])
  box(9.0,  [$u_2 dots u_(k+1)$], green)
})
]

The rest of the story — skip $u_1$, keep the next $k$ — is unchanged. The output is an $n times k$ matrix: *$k$ structural coordinates per node*.

= From eigenvectors to tokens

Stack the chosen eigenvectors as columns. Reading the matrix *by rows* gives, for each node, its coordinates in the graph's natural basis:

#align(center)[
#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  content((1.4, 2.5), text(8pt)[$U = [u_2 | u_3 | u_4]$, columns = modes])
  // a 5x3 matrix; highlight row 3
  for r in range(5) {
    for c in range(3) {
      let x = c * 0.7
      let y = 1.8 - r * 0.55
      let hl = r == 2
      rect((x - 0.32, y - 0.24), (x + 0.32, y + 0.24),
           fill: if hl { orange.lighten(55%) } else { luma(247) }, stroke: 0.4pt)
      content((x, y), text(6.5pt)[#calc.round(calc.cos((r + c) * 0.9), digits: 2)])
    }
  }
  // column labels
  for (c, lbl) in (([$u_2$], [$u_3$], [$u_4$])).enumerate() {
    content((c * 0.7, 2.2), text(7pt, blue)[#lbl])
  }
  content((-0.55, 0.7), anchor: "east", text(7pt, orange)[node 3 →])

  // arrow to token
  line((2.4, 0.7), (3.4, 0.7), mark: (end: ">"), stroke: 0.6pt)

  // the token
  content((5.2, 2.5), text(8pt)[token for node 3])
  rect((3.6, 0.45), (6.8, 0.95), radius: 3pt, fill: orange.lighten(70%), stroke: 0.5pt + orange)
  content((5.2, 0.7), text(7.5pt)[$["pe"_2, "pe"_3, "pe"_4]$ ⊕ features])
  content((5.2, -0.1), text(7pt)[3 numbers locating node 3 in the graph])
})
]

These $k$ numbers are concatenated onto (or added to) each node token before it enters the transformer. Now attention can distinguish nodes *by their structural position*, and an edge token can tell its two endpoints apart by their coordinates — without ever consulting an arbitrary node index.

== Why this is the encoding the other failures were missing

The companion note on tokenization (`tokenization.typ`) ended with four properties a good node identity must have. Laplacian PE hits all four:

#align(center)[
#table(columns: 2, stroke: none, inset: (x: 6pt, y: 3pt), align: left,
  [#good[*unique within a graph*]], [distinct nodes almost surely get distinct coordinate vectors,],
  [#good[*stable across epochs*]], [computed once from the graph, never trained — no gradient fight,],
  [#good[*not shared across graphs*]], [coordinates come from this graph's own spectrum, not a global table,],
  [#good[*structure-only*]], [depends only on edges, so a relabelled copy gets the identical encoding.],
)
]

That last point is the crucial one and deserves a proof sketch.

== Permutation equivariance — the property we actually wanted

Relabel the nodes with a permutation $P$. Then $A |-> P A P^top$ and $L |-> P L P^top$. If $L u = lambda u$, then

$ (P L P^top)(P u) = P L u = lambda (P u), $

so the eigenvectors simply get *permuted the same way the nodes did*. Node $i$'s coordinates travel with node $i$ no matter what number you paint on it. This is exactly the #good[structure-only] property that adjacency-row tokens lacked — there, relabelling produced completely different tokens for the same shape.

= The one real catch: sign and basis ambiguity

Eigenvectors are not perfectly unique, and this is the thing that trips people up.

== Sign flips

If $u$ is a unit eigenvector, so is $-u$ — they solve $L u = lambda u$ equally well. The solver picks a sign *arbitrarily*, and it may pick differently for the same graph on another run or another machine.

#align(center)[
#cetz.canvas(length: 1cm, {
  import cetz.draw: *
  let n = 7
  let vals = range(n).map(i => calc.cos(3.14159 * (i + 0.5) / n))
  let plot(ox, sign, col, lbl) = {
    line((ox, 0), (ox + (n - 1) * 0.4, 0), stroke: (paint: luma(180), dash: "dashed", thickness: 0.4pt))
    let pts = vals.enumerate().map(((i, v)) => (ox + i * 0.4, sign * v * 0.5))
    for k in range(pts.len() - 1) { line(pts.at(k), pts.at(k+1), stroke: (paint: col, thickness: 1.2pt)) }
    for p in pts { circle(p, radius: 0.06, fill: col, stroke: none) }
    content((ox + (n - 1) * 0.2, -1.0), text(7.5pt)[#lbl])
  }
  plot(0,   1, blue, [$u$])
  content((3.0, 0), text(11pt)[vs])
  plot(3.6, -1, red, [$-u$ · #bad[same mode, mirrored]])
})
]

Both encode the identical structure, but a model that memorized "$"pe"_2 > 0$ means left side" breaks when the sign flips. Standard fixes:

/ Random sign flipping: during training, multiply each eigenvector by a random $plus.minus 1$. The model is forced to learn features invariant to sign — the cheap, common default.
/ Sign-invariant networks: feed both $u$ and $-u$ through a shared net and combine symmetrically (SignNet), so the output cannot depend on the choice.

== Repeated eigenvalues

When $lambda_k = lambda_(k+1)$ the eigenvectors are only defined *up to rotation within that eigenspace* — there is no canonical choice at all, not even up to sign. Highly symmetric graphs (cycles, grids) have many such repeats. This is a deeper ambiguity than sign flips and is the main theoretical wrinkle of Laplacian PE; sign-flip augmentation handles the common case, and basis-invariant architectures handle the rest.

#note[
  *Practical defaults baked into `features.py`.* (1) Use $L_"sym"$ so eigenvalues live in $[0,2]$. (2) Always skip $u_1$. (3) Keep a small $k$ (the lowest, smoothest modes carry the most global structure); pad with zeros when the graph has fewer than $k+1$ nodes. (4) Apply random sign flips at train time. With these four habits, Laplacian PE is a robust, permutation-aware coordinate system for graph tokens.
]

= A worked micro-example

Take the path $1!-!2!-!3$. Its Laplacian and the two non-trivial modes:

$ L = mat(1, -1, 0; -1, 2, -1; 0, -1, 1), quad
  u_2 = 1/sqrt(2) vec(1, 0, -1), quad
  u_3 = 1/sqrt(6) vec(1, -2, 1). $

Read the rows to get each node's 2-D coordinate (we keep $k = 2$):

#align(center)[
#table(
  columns: (auto, auto, auto, 1fr),
  inset: 6pt, align: (center, center, center, left),
  stroke: 0.4pt + luma(180),
  table.header([*node*], [$"pe"_1 = u_2$], [$"pe"_2 = u_3$], [*reading*]),
  [1], [$+0.71$], [$+0.41$], [an end node — extreme on the slow mode],
  [2], [$0.00$],  [$-0.82$], [the centre — zero on $u_2$, the dip of $u_3$],
  [3], [$-0.71$], [$+0.41$], [the other end — mirror of node 1],
)
]

Node 2 sits at the *centre of mass* of the slow mode ($"pe"_1 = 0$) while the two ends are pushed to opposite extremes — the encoding has discovered the geometry of the path with no coordinates ever supplied. Notice also that nodes 1 and 3 share $"pe"_2$ but differ in $"pe"_1$: it takes *both* coordinates together to separate the symmetric endpoints, which is exactly why we keep several eigenvectors rather than one.

= Summary

#align(center)[
#table(
  columns: (auto, 1fr),
  inset: 7pt, align: (left, left),
  stroke: 0.4pt + luma(180),
  table.header([*Concept*], [*What to remember*]),
  [Laplacian $L$], [$D - A$; its quadratic form $x^top L x = sum_(i j in E)(x_i - x_j)^2$ measures roughness],
  [Eigenvectors], [the graph's natural vibration modes, ordered smooth → rough by eigenvalue],
  [Drop $u_1$], [the constant mode (eigenvalue 0) carries no position; multiplicity = \# components],
  [$L_"sym"$], [normalized variant used in code; eigenvalues in $[0,2]$, matches GCN],
  [Token], [each node's row across the kept eigenvectors = its structural coordinates],
  [Equivariance], [relabel the graph and the encoding permutes with it — #good[structure-only]],
  [Ambiguity], [sign flips and degenerate eigenspaces; fix with sign-flip augmentation / SignNet],
)
]

Laplacian eigenvector tokenization is, at heart, one idea: *let the graph tell you where its nodes are, in its own basis of vibration modes,* instead of imposing arbitrary numbers from outside. That is precisely the permutation-aware, structure-only identity the earlier tokenizations were reaching for and missing.
