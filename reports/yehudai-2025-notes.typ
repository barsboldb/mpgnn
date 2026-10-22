// ============================================================
//  Reading Companion — Yehudai, Sanford, Bechler-Speicher, Fischer,
//  Gilad-Bachrach, Globerson 2025
//  "Depth-Width Tradeoffs for Transformers on Graph Tasks"
//  NeurIPS 2025 · arXiv 2503.01805v3 (papers/yehudai-2025-...pdf)
//  Section and theorem numbers follow v3.
// ============================================================

// ---- palette ----
#let c-algo   = rgb("#1b6ca8")   // theory / transformer side  (blue)
#let c-heur   = rgb("#c0392b")   // limits / negative results  (red)
#let c-accent = rgb("#7d3c98")   // theorems                   (purple)
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
        align(left)[Reading Companion],
        align(right)[Yehudai et al. 2025 · Depth-Width Tradeoffs on Graph Tasks])
      #line(length: 100%, stroke: 0.4pt + luma(200))
    ]
  },
)

#set text(font: ("New Computer Modern", "Linux Libertine"), size: 10pt, fill: c-ink)
#set par(justify: true, leading: 0.62em)
#set heading(numbering: none)
#show heading.where(level: 1): it => {
  v(0.4em)
  block(width: 100%, inset: (y: 6pt), stroke: (bottom: 1.2pt + c-algo), sticky: true,
    text(15pt, weight: "bold", fill: c-algo, it.body))
  v(0.2em)
}
#show heading.where(level: 2): it => {
  v(0.3em); block(sticky: true, text(11.5pt, weight: "bold", fill: c-ink, it.body)); v(0.1em)
}
#show link: it => text(fill: c-algo, it)
#show table: set par(justify: false)

// ---- callout boxes ----
#let callout(title, body, col: c-algo, sym: "") = block(
  width: 100%, fill: col.lighten(91%), stroke: (left: 2.5pt + col),
  inset: (x: 9pt, y: 7pt), radius: 2pt, breakable: true,
  [#text(weight: "bold", fill: col, [#sym #title]) #v(-0.3em) #body],
)
#let defn(t, b)    = callout("Definition — " + t, b, col: c-ink, sym: "▸")
#let thm(t, b)     = callout("Theorem " + t, b, col: c-accent, sym: "◆")
#let note(b)       = callout("My note", b, col: c-good, sym: "✎")
#let idea(b)       = callout("Thesis hook", b, col: rgb("#b9770e"), sym: "💡")
#let ask(b)        = callout("Open question", b, col: c-heur, sym: "?")
#let fix(b)        = callout("Check against our thesis", b, col: c-heur, sym: "!")
#let kbd(b) = box(fill: luma(235), inset: (x: 3pt, y: 1pt), radius: 2pt,
  text(font: "DejaVu Sans Mono", size: 8.5pt, b))
#let tbl(..args) = table(stroke: 0.4pt + luma(170), inset: 5pt, ..args)

// ============================================================
#align(center)[
  #text(17pt, weight: "bold")[Depth-Width Tradeoffs for Transformers\ on Graph Tasks]
  #v(-0.5em)
  #text(11pt, style: "italic", fill: luma(90))[Linear width can replace logarithmic depth — for some tasks]
  #v(0.3em)
  #text(9.5pt, fill: luma(110))[Yehudai, Sanford, Bechler-Speicher, Fischer, Gilad-Bachrach, Globerson
  (NYU · Google Research · Meta AI · Bar-Ilan · Tel Aviv) · NeurIPS 2025 · arXiv 2503.01805v3]
  #v(0.2em)
  #text(8.5pt, fill: luma(140))[Reading companion · the supervisor-assigned paper · checked against the thesis 2026-10-22]
]
#v(0.6em)

#callout("TL;DR", [
Sanford et al. 2024a sorted graph tasks by the *depth* a narrow transformer needs
(connectivity: $Theta(log N)$ layers at width $N^epsilon$). This paper asks the opposite
question: *fix the depth and let the width grow — how wide must it be?* With each node's
*adjacency row* as its token, they build a *width hierarchy* (Figure 1): node degree
needs $O(log n)$ width; 1-vs-2 cycles and $A^L$ need $O(n)$; counting a $k$-node subgraph
needs at most $O(n^(2-1/k))$; Eulerian-cycle verification needs nearly $n^2$; and $n^2$
suffices for *anything*, because the whole graph then fits in one token. Upper bounds are
explicit constructions that lean on *arbitrarily powerful MLPs*; lower bounds come from
communication complexity (unconditional for 2-cycle detection, conditional on the
1-vs-2-cycle conjecture elsewhere). The experiments show that at a fixed parameter
budget, shallow-wide models train and infer faster with the same accuracy, and that the
width needed to *fit* 4-cycle counts grows roughly linearly in $n$.
], col: c-accent, sym: "★")

#grid(columns: (1fr, 1fr), gutter: 8pt,
  callout("The big question", [
    "What are the algorithmic capabilities obtained by increasing the width of a
    transformer?" — measured as the width needed, at constant depth, as a function of
    the number of nodes $n$.
  ], col: c-algo, sym: "✦"),
  callout("The one-line answer", [
    It depends on the task: some need only $O(n)$ width at constant depth, some need
    super-linear width, one needs nearly $n^2$. Width is a real substitute for depth —
    but only up to the information in the input.
  ], col: c-good, sym: "✦"),
)

= 1 · Setup

== The transformer they analyse (§3.1)

#defn("Model")[
Input tokens $x_1, dots, x_N in RR^(d_"in")$. Each layer: multi-head self-attention with
$H$ heads, then a residual connection, then a per-token MLP. *Each head has its own
$m times m$ key, query and value matrices* ($K_h, Q_h, V_h in RR^(m times m)$), and the
head outputs are summed. Width $m$ = embedding dimension. Bit precision $p = O(log n)$.
]

Three assumptions do most of the work in the proofs:
+ *The MLP is arbitrary.* It may compute any function of a single token (and of its
  position). This is standard in this line of theory (Sanford et al. 2024b) and isolates
  what *attention* can and cannot move between tokens. The MLPs in their constructions
  are polynomial in $n$, but some are huge (Remark 5.3: exponential in $k$ for subgraph
  counting).
+ *Positional information is free.* The MLP sees the token index $i$, so a token knows
  "I am node $i$".
+ *Logarithmic precision.* Every number carries $O(log n)$ bits. The lower bounds count
  bits, so this assumption decides how strong they are (see §4).

#note[
*Mapping to our model.* Their $H$ heads each have the full width $m$, so a layer has
$m H$ "channels". Ours split the width: head size 8, $H = m\/8$, so all heads together
span exactly $m$. Our $m$ therefore plays the role of their $m$ (one head), and of their
$m H$ in Sanford's bounds. Our MLP is $4m$ wide with ReLU — far from arbitrary — and we
train in 32-bit floats, not $O(log n)$ bits.
]

== The tokenization: adjacency rows (§3.1, "Graph inputs")

#defn("Node-adjacency tokenization")[
One token per node: token $i$ is row $i$ of the adjacency matrix $A in {0,1}^(n times n)$.
So $N = n$ tokens, each $n$ numbers wide.
]

*Why not Sanford's edge list?* One token per edge gives $N = n + |E|$ tokens; for a dense
graph that is $Theta(n^2)$ tokens, and attention's quadratic cost becomes $Omega(n^4)$.
Node-wise tokens keep the sequence at $n$. Like the edge list, adjacency rows are *not*
permutation invariant (a token's coordinates are tied to node numbering).

#note[
This is *our* tokenization (we add self-loops: $A + I$). So their theorems are the
relevant theory for our models, not Sanford's — Sanford's $N$ and $sqrt(N)$ are stated
for edge tokens. Consequence: an adjacency-row token already carries $n$ bits (its whole
neighbourhood). Every bound below has to be read with that in mind: a lot of the "work"
is done by the input before any attention happens.
]

= 2 · The hierarchy at a glance (Figure 1)

#align(center, tbl(columns: (auto, auto, auto, auto), align: (left, center, center, left),
  [*Task*], [*Width that suffices*], [*Width that is too small*], [*Source*],
  [node degree], [$O(log n)$ (even $m = 1$, App. A)], [—], [Sanford 2024a; App. A],
  [2-cycle detection, max degree $d$], [$O(d log n)$, 1 layer], [$m p H L = o(d)$], [Thm 4.4, B.3],
  [2-cycle detection (directed)], [$O(n)$, 2 layers], [$m p H L = o(n)$ — *unconditional*], [Thm 4.3, 4.2],
  [$A^L$ (walk counts)], [$O(n)$, $O(L)$ layers], [—], [Thm 4.3],
  [1 vs 2 cycles], [$O(n)$, 2 layers], [$O(n^(1-epsilon))$ at depth $o(log n)$ — conditional], [Thm 4.1; Sanford 2024a],
  [connectivity (general)], [$O(n log^3 n)$, constant depth — *sketch only*, randomized], [depth $Omega(log n)$ or width $Omega(n)$ (App. A)], [§4.1, Ahn et al. 2012],
  [count / detect a $k$-node subgraph], [$O(n^(2-1/k))$, $O(1)$ layers], [—], [Thm 5.2],
  [Eulerian cycle verification], [$n^2$], [$O(n^(2-epsilon))$ at depth $o(log n)$ — conditional], [Thm 5.1],
  [any function $f(A)$], [$O(n^2)$: the whole graph fits in one token], [—], [§5],
))

#note[
Read the table as "width needed at *constant* depth". The two conditional lower bounds rest
on the *1 vs 2 cycles conjecture* (App. B.1): an MPC protocol with $n^(1-epsilon)$ memory
per machine cannot tell one $n$-cycle from two $n\/2$-cycles in $o(log n)$ rounds. It is
widely believed, unproven, and the backbone of Sanford et al. 2024a's depth lower bounds
too.
]

= 3 · The positive results — how width replaces depth

== Theorem 4.1 — one vs two cycles with two layers

#thm("4.1")[
There exists a transformer with 2 layers of self-attention and embedding dimension $O(n)$
that solves the 1 vs 2 cycle problem.
]

*How (App. B.2).* The trick is to put the *whole graph into one token* and let the MLP
solve it.
+ Layer 1: attention does nothing ($V = 0$, the residual keeps the input). The MLP turns
  node $i$'s adjacency row into the IDs of its two neighbours $(j, k)$ — possible because
  $j + k$ and $j^2 + k^2$ determine ${j, k}$ (Lemma B.2) — and writes $(i, j, k)$ into its
  own 3 slots of a $3n$-dimensional vector.
+ Layer 2: attention with $K = Q = 0$ is uniform, i.e. every token receives the *average*
  of all tokens; $V = n I$ turns it into the *sum*. Now every token holds the full edge
  list ($3n$ numbers).
+ The MLP runs any connectivity algorithm on that edge list (universal approximation).

#note[
This is less a "transformer algorithm" than a *compression argument*: a cycle has only $n$
edges, so the graph fits in $O(n)$ numbers. The authors say it themselves — the result
"demonstrates the brittleness of communication-based transformer lower bounds, since even
a slight increase in the embedding dimension breaks them", and "is not surprising, since
the graph in this task is very sparse". It needs one layer of *attention* that is
uniform — no routing along paths at all.
]

=== General connectivity: only a sketch

A dense graph has up to $n^2\/2$ edges, which do not fit in $O(n)$ numbers. For general
connectivity they *sketch* (not a theorem): compress each adjacency row with the linear
sketches of Ahn, Guha and McGregor (2012) to $"poly"(log n)$ numbers, gather all sketches
into one token ($O(n log^3 n)$ total), and let the MLP run Ahn et al.'s algorithm. It
works only with high probability (random projections), "thus, we don't present it here
formally".

#fix[
Our thesis says that with adjacency rows, constant depth and $O(n)$ width suffice for
connectivity (abstract l. 17–19; ch. 1 summary; ch. 3 §Хэлэлцүүлэг "бүтээц $O(n)$ өргөн
ашигладаг"). *Strictly:* the formal constant-depth, $O(n)$-width theorem is for 1-vs-2
cycles (and 2-cycles); for general connectivity it is a randomized sketch with
$O(n log^3 n)$ width. Appendix A states the adjacency tokenization needs "either depth
$Omega(log n)$ or width $Omega(n)$" for connectivity.
]

== Theorem 4.3 — powers of $A$, one hop per layer

#thm("4.3")[
There exists an $O(L)$-layer transformer with embedding dimension $m = O(n)$ such that,
for any graph embedded as rows of $A$, the output at token $i$ is row $i$ of $A^L$.
]

*How (App. B.4).* Each token carries $(A_i, e_i, A^ell_i)$ — its adjacency row, a one-hot
position, and the current power. Keys and queries are chosen so that the attention score
between $i$ and $j$ is $c dot A_(i j)$; with a large temperature $c$ the softmax becomes a
hard *average over $i$'s neighbours*. The value vectors carry $A^ell$, so the attention
output is $A dot A^ell$ divided by the degree; the MLP multiplies the degree back in. Each
layer multiplies by $A$ *once*: $A^ell -> A^(ell+1)$.

#fix[
Our ch. 1 (ch1-theory.tex, "Гүн ба өргөний солилцоо", first bullet) says this construction
is "the transformer form of repeated squaring" ($M -> M^2$). *It is not:* it multiplies
by $A$ once per layer, so $L$ layers give $L$ hops. Repeated squaring would give $2^L$
hops; deciding connectivity through $A^L$ this way needs $L approx$ diameter (the paper
says "checking whether $A^n$ doesn't contain any zero entries" — $n$ layers).
]

#idea[
This is the "one hop per layer" hypothesis our probes tested, and they found neither it
nor doubling: 3-hop information appears after layer 1 and then stays flat, while the
"connected" signal grows a little in every layer (progress report, Probing). Our
6-layer models solve graphs of diameter 10–13, so they cannot be running Theorem 4.3's
algorithm.
]

== Theorem 4.4 — sparse graphs need much less width

#thm("4.4")[
For any $n$ and $d <= n$, there exists a single-layer transformer with embedding dimension
$O(d log n)$ that detects 2-cycles in any graph with maximum degree $d$. This is optimal up
to log factors.
]

*How (App. B.5).* Give each node a random code $y_i in RR^p$, $p = O(d log n)$, and encode
a neighbour set $x$ (at most $d$ ones) as $phi(x)$ with $chevron.l phi(x), y_i chevron.r = 1$
if $i in x$ and $<= 1\/2$ otherwise (restricted isometry, Candès–Tao). Queries carry
$(phi(x_i), y_i)$, keys $(y_j, phi(x_j))$, so the score is large exactly when $j in x_i$
*and* $i in x_j$, i.e. when $i -> j -> i$ is a 2-cycle. A "dummy node" absorbs the
attention when no 2-cycle exists.

#note[
Width scales with the *degree*, not with $n$: the adjacency row is mostly zeros and can be
compressed. Our `swap` graphs have degree ≈ 2.5, so by this logic a narrow model could
represent local tasks. Our Q6 test (fixed random $n times 96$ projection of the adjacency
row) is the same idea — random node codes — and showed that the input compression is not
what drives the growth of $m^*$.
]

== Theorem 5.2 — subgraph counting between linear and quadratic

#thm("5.2")[
For a fixed $k$-node pattern $G'$, there is a transformer with $O(1)$ layers and width
$O(n^(2-1/k))$ that counts the occurrences of $G'$ in any $n$-node graph.
]

*How (App. C.1).* The "tri, tri again" algorithm (Dolev et al. 2012). Split the nodes into
$n^(1\/k)$ groups of $n^(1-1\/k)$. Layer 2 gathers each group's internal edges into one
token ($<= n^(2-2\/k)$ edges). Layer 3 lets each token collect one *combination of $k$
groups*: there are $binom(n^(1\/k), k) <= n$ combinations, so they fit in the $n$ tokens.
Every copy of $G'$ lies inside some combination; the MLP counts there.

#note[
The MLP is exponential in $k$ (Remark 5.3), and some copies are counted in several
combinations — fine for *detection*, needs care for exact counts. Compare Sanford et al.
2024a Thm 23: depth $O(log log n)$ but $O(n^(k-1))$ blank tokens. Width and extra tokens
are two currencies for the same work.
]

= 4 · The negative results — when width cannot be small

== Theorem 4.2 — 2-cycle detection needs linear width (unconditional)

#thm("4.2")[
Let $T$ have width $m$, depth $L$, precision $p$ and $H$ heads per layer, with adjacency-row
inputs. If $T$ detects 2-cycles in directed graphs then $m p H L = Omega(n)$ with residual
connections, and $m p H = Omega(n)$ without.
]

*How (App. B.3): set disjointness.* Alice holds $a in {0,1}^(n^2)$, Bob holds $b$; they must
decide whether some $a_i = b_i = 1$, which needs $n^2$ bits of communication. Build a graph
on $2n$ nodes: $a$ = edges from side 1 to side 2, $b$ = edges back. A directed 2-cycle
exists iff the sets intersect. Alice owns side 1's tokens, Bob side 2's. To simulate one
layer, Bob sends Alice his keys, queries and values: $O(n m p H)$ bits. Over $L$ layers,
$O(n m p H L) >= n^2$, hence $m p H L = Omega(n)$.

#idea[
*Plug in our numbers.* In our model the per-layer message is about $3 m$ numbers per token
(the heads split $m$), at $p = 32$ bits. The bound becomes $3 dot 32 dot m dot L >= n$; at
$L = 6$ that is $m >= n\/576$ — trivially satisfied. Lower bounds of this kind count
*bits*, and a 32-bit float carries far more than the $O(log n)$ bits the theory assumes.
So *no communication bound explains our measured $m^* approx 0.7n$–$1.0n$*: the binding
constraint is what gradient descent can learn to use, not what the channel can carry.
This sharpens the "representation vs learning" argument in our ch. 3.
]

== Theorem 5.1 — Eulerian cycles need nearly quadratic width (conditional)

#thm("5.1")[
Under the 1 vs 2 cycles conjecture, Eulerian cycle verification on multigraphs (with
adjacency inputs) cannot be solved with $m = O(n^(2-epsilon))$ unless $L = Omega(log n)$.
]

*How (App. C.2).* Reduce 1-vs-2 cycles on $N = n^2$ nodes to Eulerian verification on an
$n$-node multigraph: map each original node to one of $n$ super-nodes, keep all $n^2$
edges, and encode the cycle's successor structure as path fragments. One cycle ↔ the
fragments form one Eulerian cycle. A cheap transformer does the conversion, so a
sub-quadratic-width solver would break the conjecture.

#note[
The only task shown to need (nearly) $n^2$ width — and it is contrived. It does establish
that the trivial "copy the whole graph" upper bound can be tight.
]

= 5 · Appendix A — tokenization trade-offs

- *Adjacency rows are good at local tasks*: degree with $m = 1$ ($chevron.l bold(1), x_i chevron.r$).
  Laplacian eigenvectors (the $m$ smallest) cannot even give degrees without $m$ growing
  with $n$ (footnote 6: $n\/3$ disjoint paths make the eigenvectors component indicators).
- *Laplacian eigenvectors are good at global tasks*: the multiplicity of eigenvalue 0 is
  the number of components, so connectivity is *free* given the eigenvalues — while
  adjacency rows need "either depth $Omega(log n)$ or width $Omega(n)$".
- *At $m = Omega(n^2)$ both are universal* — one token holds everything.

#idea[
This is the theory behind our encoding chapter: an encoding that precomputes global
structure can give the answer away. Our RWSE-16 leak and the $k$-hop rows are the same
phenomenon in degrees: the more of the global computation the input already contains, the
less width the model needs.
]

= 6 · Experiments — what they measured, and how it compares to ours

== §6.1 — same parameter count, different shape

- Tasks: connectivity (their own generator), triangle count and 4-cycle count (Chen et
  al. 2020, Erdős–Rényi $p = 0.1$). 5000 graphs each; $n = 50$ and 100.
- About 100k parameters split as $(L, m) in {(1,125), (2,89), (4,63), (8,45), (10,40)}$;
  100 epochs; 3 seeds. Learning rate tuned in ${10^(-4), 5 dot 10^(-5)}$, batch size in
  ${32, 64}$, dropout 0.1.
- Result: loss and accuracy are about the same for every shape; shallow-wide is much
  faster to train and run (GPUs parallelize within a layer, not across layers).

#note[
*Their connectivity data is easy.* Classes come from mixtures of generators (ER, random
geometric, Barabási–Albert, SBM) with connected graphs drawn at *higher density* than
disconnected ones ($p >= ln n\/n$ for connected ER, "a lower $p$" for disconnected). Our
reproduction (#kbd("docs/yehudai-empirical.md")) found that a *1-layer, width-64*
adjacency-row model reaches 100% test accuracy at $n = 50$ — with 100 training graphs in
their own code, by epoch 4 in ours. So "every depth gives the same accuracy" says little
about depth on connectivity: one layer already suffices on this data. (The density
correlation is our reading of their generator description, not their claim.)
]

== §6.2 — the critical width

- *Definition:* the width at which the *training loss* plateaus above 0.05, i.e. the
  smallest width that can *fit* the training set. Grid search over hyperparameters, best
  fit kept.
- Setup: *one layer, two heads*; $n = 50$–400 (step 50), $m = 100$–800 (step 100);
  triangle and 4-cycle counting.
- Result: "the critical width increases roughly linearly with the graph size" (Fig. 3;
  triangles in Fig. 9). The text also says "sub-quadratically … as argued in Section 5.2".

#note[
Four differences from our measurement, each of which we found matters:
+ *Fit, not generalization.* On 5000 graphs a wide model can memorize; our Q1c and the
  32k-graph fine grid show the two critical widths separating when data is scarce.
+ *A two-value learning-rate grid* ($10^(-4)$, $5 dot 10^(-5)$), with widths from 100 to 800:
  our Q1 shows the best rate moves with width, by more than a factor of two over such a
  range.
+ *Coarse width grid* (steps of 100), so each critical width is known to within ±50.
+ *A local task* (counting small patterns), at depth 1. Connectivity is global.
Their "linear" and our $n^(1.74)$ are therefore not in conflict — different task, depth and
criterion — but theirs is the only prior critical-width measurement for graph
transformers, and ours is its stricter extension.
]

== §6.3 — tokenizations on real molecules

ROC-AUC on three OGB molecule datasets (3 seeds):

#align(center, tbl(columns: 4, align: (left, center, center, center),
  [*Tokenization*], [*MOLHIV*], [*MOLBBBP*], [*MOLBACE*],
  [edge list], [54.0 ± 1.4], [64.7 ± 1.7], [66.1 ± 3.9],
  [adjacency rows], [61.9 ± 1.1], [*67.6 ± 2.6*], [*68.6 ± 2.3*],
  [Laplacian eigenvectors], [*68.1 ± 1.5*], [55.3 ± 4.8], [63.6 ± 2.3],
))

Adjacency rows beat the edge list everywhere and the Laplacian on two of three. Their
argument for the tokenization: it works on real data, not only in proofs.

= 7 · Checking our thesis against the paper

#align(center, tbl(columns: (3.3cm, 1fr, 1fr), align: (left, left, left),
  [*Where*], [*What we wrote*], [*Verdict*],
  [bibliography], [title "…in Algorithmic Reasoning of Graph Tasks with Transformers", ICML 2025],
    [*wrong*: published title "Depth-Width Tradeoffs for Transformers on Graph Tasks", NeurIPS 2025 (the arXiv listing keeps the old title)],
  [ch. 1, "Гүн ба өргөний солилцоо"], [$O(L)$ layers, $O(n)$ width compute $A^L$ — "the transformer form of repeated squaring"],
    [*wrong*: one hop per layer (§3 above)],
  [ch. 1, same], [1 vs 2 cycles: 2 layers, $O(n)$ width; sub-linear width at constant depth fails only conditionally], [correct (Thm 4.1, §4.1)],
  [ch. 1, same], [critical width for 4-cycle counting grows linearly in $n$], [correct; add: *training-loss* criterion, 1 layer],
  [ch. 1, "Өргөнийг туршилтаар…"], [they used *only one* tokenization and kept the learning rate *fixed* across widths],
    [*imprecise*: they compare three tokenizations on OGB (§6.3), though the critical-width study uses only adjacency rows; the rate was tuned, but only over ${10^(-4), 5 dot 10^(-5)}$],
  [ch. 2, §Сургалтын хурд], [fixing the rate across widths is common, citing them], [same nuance: two-value grid],
  [abstract; ch. 1 summary; ch. 3 discussion], [with adjacency rows, constant depth and $O(n)$ width suffice (for connectivity)],
    [*overstated*: formal only for 1-vs-2 cycles / 2-cycles; general connectivity is a randomized $O(n log^3 n)$ sketch],
  [ch. 1, Merrill comparison], [their construction gives each token $n$ bits and allows arbitrary MLPs], [correct (§3.1)],
  [progress report, conclusions], [critical width "measures *fitting* on counting tasks at depth 1 with 5000 graphs and a fixed learning-rate grid"], [correct],
  [progress report, conclusions], ["quadratic width suffices for any task by copying the whole graph"], [correct (§5)],
))

= 8 · Hooks into our work

#idea[
- *Representation vs learning, made concrete.* Their theory gives $O(n)$ (and Thm 4.2's
  $m p H L = Omega(n)$, which is trivially met in floating point). We measure the width
  gradient descent needs: $n^(1.74)$ at depth 6, ≈ $n^3$ at depth 2 (Q7c). Their §7 lists
  "optimization and generalization" as future work — that is exactly our gap.
- *Depth 2 is their regime.* Thm 4.1 is a 2-layer, $O(n)$-width construction. Our Q7a
  found 2 layers learn connectivity at $m approx 1.1 n$ at $n = 40$ — in line — but Q7c
  found the 2-layer critical width growing ≈ $n^3$ across sizes. Gradient descent does not
  find the "gather everything into one token" solution as $n$ grows.
- *Their mechanism vs ours.* Theorems 4.1 and 4.3 imply two concrete internal pictures:
  "the whole graph in one token" and "one more hop per layer". Our probes rule out the
  second and find no per-node reachable set (the first would show up as nodes storing
  global information). The model compares pairs of vectors instead.
]

#ask[
Does the $n^(2-1/k)$ bound for counting predict a *super-linear* fitted critical width at
larger $n$ that their coarse grid could not resolve? Their Fig. 3 points span
$n = 50$–400 with width steps of 100.
]

= 9 · One-paragraph summary

Yehudai et al. turn Sanford et al.'s depth question around: at *constant* depth, how wide
must a transformer be for a graph task, when each node's adjacency row is its token?
Their hierarchy runs from $O(log n)$ (degrees) through $O(n)$ (1 vs 2 cycles, 2-cycles, $A^L$
with $O(L)$ layers) and $O(n^(2-1/k))$ (subgraph counting) to nearly $n^2$ (Eulerian cycle
verification), with $n^2$ always enough. The constructions rely on arbitrary MLPs and
often on packing the graph into a single token; the lower bounds count bits through the
attention layer, unconditionally for 2-cycles and conditionally elsewhere. For general
connectivity there is only a randomized sketch. Experimentally, shallow-wide models match
deep-narrow ones at equal parameters and run faster, and the width needed to fit
small-pattern counts grows about linearly. For our thesis it is the theory of our exact
tokenization, the source of the "critical width" method, and the baseline our
connectivity measurements ($n^(1.74)$, learned, generalizing) extend and qualify.
