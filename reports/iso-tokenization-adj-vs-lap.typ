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
  #text(size: 19pt, weight: "bold")[Adjacency Rows vs. Laplacian Eigenvectors]
  #v(0.3em)
  #text(size: 11pt)[Why the "weaker" encoding won the isomorphism task — and what each one really tells the model]
  #v(0.2em)
  #text(size: 9.5pt, fill: luma(100))[isomorphism · `src.dataset.tokenize_dataset` · `src.features.laplacian_positional_encoding` · #datetime.today().display()]
]

#v(0.6em)

#note[
  *The puzzle.* Adjacency-row tokenization tops out at #good[\~0.85] test accuracy. Laplacian-eigenvector tokenization — the "real" structural encoding — never clears #bad[0.60]. The intuition that structural encoding $>$ raw adjacency was reasonable, but it is *backwards for this dataset*. The reason is not that Laplacian encodings are weak in general; it is that *this task is secretly a degree-sequence comparison*, and the two encodings sit on opposite sides of that fact: adjacency rows hand degree to the model for free, while the normalized Laplacian deliberately throws degree away.
]

= The task is not the task you think it is

Each sample is a *pair* $(G_1, G_2)$ packed into one disconnected graph: $G_1$ on nodes $0 dots n-1$, $G_2$ on nodes $n dots 2n-1$. The label is whether they are isomorphic. The architecture (one global-attention layer, then `pair` pooling that averages $G_1$ nodes and $G_2$ nodes separately and concatenates $[h_(G_1) | h_(G_2)]$) means the classifier's whole job is: *compare a summary of $G_1$ against a summary of $G_2$.*

Now look at how the negatives are made (`make_isomorphism_dataset`, `src/dataset.py`):

#note[
  - *label 1 (iso):* $G_2$ is a random *permutation* of $G_1$. #good[Same degree sequence by construction.]
  - *label 0 (non-iso):* $G_2$ is regenerated until its degree sequence #bad[*differs*] from $G_1$ (the generator loops up to 200 times to force this).
]

Measured on the actual cached 1000-pair dataset:

#align(center)[
#table(
  columns: (auto, auto),
  align: (left, left),
  stroke: 0.5pt + luma(180),
  inset: 6pt,
  [*Class*], [*Pairs whose two components share a degree sequence*],
  [iso (500)], [#good[500 / 500]],
  [non-iso (500)], [#bad[0 / 500]],
)
]

So the degree sequence is a *perfect* separator. A model that does nothing but "compute each component's degree multiset and check whether they match" would score 100%. The task never asks the model to solve the genuinely hard part of isomorphism (telling apart non-isomorphic graphs that *share* a degree sequence — cospectral mates, regular graphs, etc.). It only ever asks: #good[*do the degree sequences agree?*]

This reframes everything. The winning encoding is whichever one makes *degree* easiest to read and compare.

= What an adjacency-row token looks like

Each node becomes its row of the adjacency matrix, zero-padded to the dataset-wide maximum width (here $max 2n = 30$, so each token is a 30-dim binary vector). Node $i$ has a $1$ in column $j$ iff $i tilde j$.

#align(center)[
#cetz.canvas(length: 1cm, {
  import cetz.draw: *
  let row(y, label, bits, col) = {
    content((-2.4, y), text(9pt)[#label])
    for (k, b) in bits.enumerate() {
      let fill = if b == 1 { col } else { white }
      rect((k*0.42, y - 0.18), (k*0.42 + 0.38, y + 0.2), fill: fill, stroke: 0.4pt + luma(150))
      if b == 1 { content((k*0.42 + 0.19, y), text(7pt, white)[1]) }
    }
  }
  // G1 node 0: neighbors 1,3  (columns within 0..n-1 block)
  row(2.0, [$G_1$ node 0], (0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0), blue)
  row(1.3, [$G_1$ node 1], (1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0), blue)
  // G2 node n (=index 0 in its block) neighbors put in the n.. block
  row(0.4, [$G_2$ node $n$], (0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0), orange)
  content((4.2, -0.4), text(8pt, fill: luma(100))[#sym.dots.h, padded to width 30])
  // brackets for the two id blocks
  line((0,2.55),(8.0,2.55), stroke: blue+0.8pt)
  content((4.0, 2.85), text(8pt, blue)[columns 0 .. n#sym.minus 1  (the $G_1$ id-block)])
})
]

#v(0.4em)

Two facts about this token decide the whole story.

/ #good[It leaks degree, for free.]: The number of $1$s in a row #emph[is] that node's degree. A single linear layer can read $sum_j x_(i j) = deg(i)$. `pair` pooling then averages rows within a component; column $j$ of the pooled $G_1$ vector becomes $deg(j) slash n$. The classifier receives, in effect, both components' degree profiles side by side — exactly the separating signal.

/ #bad[It hides the correspondence between the two graphs.]: $G_1$'s $1$s live in columns $0 dots n-1$; $G_2$'s live in columns $n dots 2n-1$. The two components occupy *disjoint column blocks*, so the raw vectors are never directly comparable — a node and its image under the isomorphism have orthogonal token vectors. The model cannot match node-to-node; it can only compare *permutation-invariant summaries* of each block. Fortunately degree is exactly such a summary, and it is all the task needs.

That tension is the ceiling. The model reliably extracts and compares degree distributions (#good[easy, leaked]) but cannot recover the true vertex correspondence (#bad[hidden]). It therefore lands near the accuracy of a degree-sequence test — about 0.85, missing only the non-iso pairs whose degree sequences differ by a hair (`analyze_iso.py` confirms the errors concentrate at small `deg_seq_diff`).

= What a Laplacian-eigenvector token looks like

Each node becomes its row of the matrix of the $k=16$ smallest non-trivial eigenvectors of the *symmetric normalized* Laplacian $L = I - D^(-1 slash 2) A D^(-1 slash 2)$. The promise: a coordinate system that depends only on graph shape, so isomorphic graphs get matching encodings. Three things break that promise *on this particular dataset*.

== Break 1 — normalization erases degree

The $D^(-1 slash 2)$ factors are there precisely to make the Laplacian scale-free: they divide out degree. Empirically, on a real pair from the dataset, the correlation between a node's eigenvector-row norm and its degree is

#align(center)[ #bad[$"corr"("LPE row norm", deg) = 0.086 approx 0$.] ]

The one feature that perfectly solves the task is the one feature the normalized Laplacian is *designed* to remove. The encoding spends its 16 dimensions describing *relative position* — a question the task never asks — and says almost nothing about *degree* — the only question it does ask.

== Break 2 — the pair is disconnected, so the low eigenvectors are arbitrary

The number of zero eigenvalues of $L$ equals the number of connected components. The packed pair $G_1 union.sq G_2$ has (at least) *two* components, so eigenvalue $0$ has multiplicity $gt.eq 2$. `laplacian_positional_encoding` skips only eigenvector #0 and keeps eigenvector #1 as its first column — but eigenvector #1 lives *inside* that degenerate null space, where `eigh` returns an *arbitrary orthonormal basis*. On a real iso pair from the cache, the kept "first eigenvector" came out as:

#align(center)[
#cetz.canvas(length: 1cm, {
  import cetz.draw: *
  let g1 = (0.23,0.27,0.23,0.38,0.27,0.40,0.23,0.30,0.19,0.27,0.23,0.23,0.30)
  let g2 = (0,0,0,0,0,0,0,0,0,0,0,0,0)
  content((-2.0, 0.9), text(9pt, blue)[$G_1$ rows])
  content((-2.0, 0.2), text(9pt, orange)[$G_2$ rows])
  for (k, v) in g1.enumerate() {
    let h = v * 1.6
    rect((k*0.5, 0.9), (k*0.5+0.4, 0.9+h), fill: blue, stroke: none)
  }
  for (k, v) in g2.enumerate() {
    rect((k*0.5, 0.2), (k*0.5+0.4, 0.22), fill: orange, stroke: none)
  }
  line((-0.1,0.9),(6.7,0.9), stroke: 0.5pt+luma(120))
  content((3.2, -0.35), text(8pt, fill: luma(100))[same column, on $G_2$: every value #sym.approx 0])
})
]

The vector is nonzero on $G_1$ and #bad[*exactly zero on $G_2$*]: `eigh` happened to align the 2-D null space so one basis vector sits entirely on $G_1$, the other entirely on $G_2$. So the model's "first structural coordinate" is really a *component indicator*, carrying zero information about the other graph — and which graph it lands on is an arbitrary numerical accident that flips from pair to pair.

== Break 3 — isomorphic components are cospectral, so *every* eigenspace is degenerate

When $G_1$ and $G_2$ are isomorphic they have identical spectra, so the union's eigenvalues come in *exact duplicate pairs*. On the same sample:

#align(center)[
  smallest six eigenvalues $= (#text(luma(120))[0.000, 0.000], #text(blue)[0.360, 0.360], #text(purple)[0.515, 0.515], dots)$
]

Every eigenvalue has multiplicity 2. That means the *entire* eigenbasis lives in 2-D degenerate subspaces, each spanned by one $G_1$-mode and one $G_2$-mode, and `eigh` hands back an *arbitrary rotation* within each. So even setting aside the usual sign ambiguity ($plus.minus$ per eigenvector), the eigenvector rows of $G_1$ and $G_2$ are related by an unknown, per-eigenspace rotation that *mixes the two components together*. The "approximately permutation-equivariant" hope in the config comment collapses: there is no fixed transform the classifier can learn to undo.

= Side by side: what each leaks, what each hides

#align(center)[
#table(
  columns: (auto, 1fr, 1fr),
  align: (left, left, left),
  stroke: 0.5pt + luma(180),
  inset: 7pt,
  table.header([], [*Adjacency rows* (in=30)], [*Laplacian eigvecs* (in=16)]),

  [Degree (the separating signal)],
  [#good[Leaked] — row sum #sym.eq degree],
  [#bad[Erased] — normalization divides it out (corr 0.09)],

  [Cross-graph node matching],
  [#bad[Hidden] — $G_1$/$G_2$ in disjoint column blocks],
  [#bad[Hidden] — sign + rotation ambiguity per eigenspace],

  [Stability of the token],
  [#good[Deterministic] given node ids],
  [#bad[Arbitrary] basis in degenerate / null spaces],

  [Effect of disconnected pair],
  [#good[None] — rows are local],
  [#bad[Severe] — multiplicity-2 null space → indicator vectors],

  [Effect of iso pair being cospectral],
  [#good[None]],
  [#bad[Severe] — every eigenvalue doubled → full mixing],

  [What it spends capacity describing],
  [Local neighbourhoods (incl. degree)],
  [Relative position — #emph[which the task never asks about]],

  [Result],
  [#good[\~0.85] (degree-test ceiling)],
  [#bad[\~0.60] (barely above chance)],
)
]

= So was the intuition wrong?

Not in general — it was *mismatched to the benchmark*. Laplacian eigenvectors are the stronger encoding for tasks that genuinely require global structure and where the discriminating graphs *share* coarse statistics like degree. This benchmark is the opposite: its negatives are pre-separated by degree, and it packs two graphs into one disconnected, cospectral blob — the single worst input for Laplacian PE, because it maximizes eigenspace degeneracy. Adjacency rows win not by being more expressive but by *leaking the one statistic that happens to be a perfect label*.

#note[
  *Two ways to make the comparison fair — and actually test the original hypothesis.*
  + #good[*Compute LPE per component, not on the union.*] Run the eigendecomposition on $G_1$ and $G_2$ separately (no shared null space, no cross-component rotation), and add sign-invariant post-processing (e.g. SignNet, or feed $abs(v)$ / $v v^top$). This removes Breaks 2–3, leaving a real test of whether *position* helps.
  + #good[*Make the task need structure, not degree.*] Generate non-iso negatives that are *degree-preserving* (degree-sequence-matched, or even cospectral). Then adjacency-row degree-reading collapses to chance, and a structural encoding has something to prove. This is the experiment that would support the thesis claim.
]
