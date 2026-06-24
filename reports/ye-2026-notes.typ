#import "@preview/cetz:0.4.2"

// ============================================================
//  Reading Companion — Ye et al. 2026
//  "Transformers Provably Learn Algorithmic Solutions for
//   Graph Connectivity, But Only with the Right Data"
// ============================================================

// ---- palette ----
#let c-algo   = rgb("#1b6ca8")   // algorithmic / I-channel  (blue)
#let c-heur   = rgb("#c0392b")   // heuristic   / J-channel  (red)
#let c-accent = rgb("#7d3c98")   // purple accent
#let c-good   = rgb("#1e8449")   // green
#let c-ink    = rgb("#222222")
#let c-faint  = rgb("#f4f6f8")

#set page(
  paper: "a4",
  margin: (x: 1.9cm, top: 2.2cm, bottom: 1.9cm),
  numbering: "1",
  header: context {
    if counter(page).get().first() > 1 [
      #set text(8pt, fill: luma(120))
      #grid(columns: (1fr, 1fr),
        align(left)[Reading Companion],
        align(right)[Ye et al. 2026 · Transformers & Graph Connectivity])
      #line(length: 100%, stroke: 0.4pt + luma(200))
    ]
  },
)

#set text(font: ("New Computer Modern", "Linux Libertine"), size: 10pt, fill: c-ink)
#set par(justify: true, leading: 0.62em)
#set heading(numbering: none)
#show heading.where(level: 1): it => {
  v(0.4em)
  block(width: 100%, inset: (y: 6pt), stroke: (bottom: 1.2pt + c-algo),
    text(15pt, weight: "bold", fill: c-algo, it.body))
  v(0.2em)
}
#show heading.where(level: 2): it => {
  v(0.3em); text(11.5pt, weight: "bold", fill: c-ink, it.body); v(0.1em)
}
#show link: it => text(fill: c-algo, it)

// ---- callout boxes ----
#let callout(title, body, col: c-algo, sym: "") = block(
  width: 100%, fill: col.lighten(91%), stroke: (left: 2.5pt + col),
  inset: (x: 9pt, y: 7pt), radius: 2pt, breakable: true,
  [#text(weight: "bold", fill: col, [#sym #title]) #v(-0.3em) #body],
)
#let defn(t, b)    = callout("Definition — " + t, b, col: c-ink, sym: "▸")
#let thm(t, b)     = callout("Theorem — " + t, b, col: c-accent, sym: "◆")
#let note(b)       = callout("My note", b, col: c-good, sym: "✎")
#let idea(b)       = callout("Thesis hook", b, col: rgb("#b9770e"), sym: "💡")
#let ask(b)        = callout("Open question", b, col: c-heur, sym: "?")
#let kbd(b) = box(fill: luma(235), inset: (x: 3pt, y: 1pt), radius: 2pt, text(font: "DejaVu Sans Mono", size: 8.5pt, b))

// ============================================================
#align(center)[
  #text(17pt, weight: "bold")[Transformers Provably Learn Algorithmic\ Solutions for Graph Connectivity]
  #v(-0.5em)
  #text(11pt, style: "italic", fill: luma(90))[…But Only with the Right Data]
  #v(0.3em)
  #text(9.5pt, fill: luma(110))[Ye, Fu, Jia, Sharan · Preprint Feb 2026 (arXiv 2510.19753v2)]
  #v(0.2em)
  #text(8.5pt, fill: luma(140))[Reading companion · GNN / Graph-Transformer thesis exploration]
]
#v(0.6em)

// ---------- TL;DR ----------
#callout("TL;DR", [
A depth-$L$ Transformer can *provably* solve graph connectivity by computing
powers of the adjacency matrix — but *only on graphs whose diameter is*
$lt.eq 3^L$ (the model's *capacity*). Expressivity is not the bottleneck;
the *training distribution* is. If too many training graphs exceed capacity
($"diam" > 3^L$), gradient descent abandons the matrix-powering algorithm and
settles for a brittle *degree-counting heuristic*. The fix is a *data lever*:
restrict training to within-capacity graphs and the exact algorithm emerges —
and this transfers from the toy *Disentangled Transformer* to standard ones.
], col: c-accent, sym: "★")

#grid(columns: (1fr, 1fr), gutter: 8pt,
  callout("The big question", [
    Why do Transformers prefer brittle *heuristics* over *verifiably correct
    algorithms*, even when the task admits an algorithmic solution and the
    architecture is expressive enough to represent it?
  ], col: c-algo, sym: "✦"),
  callout("The one-line answer", [
    Because the *training data* contains too many instances *harder than the
    model's exact capacity* ($3^L$), and those instances reward a global
    shortcut. Curate the data $arrow.r$ recover the algorithm.
  ], col: c-good, sym: "✦"),
)

= 1 · Setup — what is actually being learned

#defn("Task")[
Input = *self-loop-augmented adjacency matrix* $A in {0,1}^(n times n)$
(i.e. $A + I$). Target = *connectivity matrix* $R$, where $R_(i j)=1$ iff a
path exists between $v_i, v_j$ — equivalently iff $[A^n]_(i j) > 0$. Models map
$cal(M): {0,1}^(n times n) -> RR^(n times n)$; "perfect" means the sign pattern
of $cal(M)(A)$ matches $R$ everywhere. Metric: *Exact-Match Accuracy* (whole
matrix correct), averaged over Erdős–Rényi $"ER"(n,p)$ graphs.
]

#note[
The self-loop trick is the whole game: $[A^k]_(i j)$ counts walks of length
*exactly* $k$ but with self-loops a walk can "wait", so $[A^k]_(i j)>0$ iff
distance $lt.eq k$. Connectivity therefore = "is some power of $A$ positive
here" = *transitive closure via matrix powering* (Warshall/Floyd). This is the
algorithmic target the whole paper is chasing.
]

== The architecture: Disentangled Transformer (DT)
A linear read-in $X W_"in"$, $L$ attention blocks, linear read-out. The twist:
each block *appends* its output as new coordinates of the residual stream
(instead of summing), so read/write paths stay traceable — making the dynamics
analyzable. Attention is $"Attn"(h; W) = 1/n "ReLU"(h W h^top) h$. The hidden
width grows like $d_ell = 2^(ell+1) n$. Standard attention-only Transformers can
be *re-expressed* as DTs, so results are claimed to transfer (verified
empirically, Fig 7 & 10).

#note[
The DT is a *mechanistic proxy*, à la Friedman/Nichani. The bilinear
$h W h^top$ form under non-negative weights is what makes the algebra of the
two-channel decomposition (§3) go through. Worth keeping in mind how much of the
theory leans on the *non-negative weight* assumption — they note training does
*not* enforce it, yet predictions still hold.
]

= 2 · Capacity — the $3^L$ reach

#thm("4.3 Expressivity (upper reach)")[
There exists an $L$-layer DT that is *perfect on every graph with*
$"diam"(G) lt.eq 3^L$, by implementing
$sum_(j=0)^(3^L) alpha_j A^j$ with $alpha_j > 0$ — the matrix-powering
algorithm. (Sketch: set every $W_ell = I$.)
]
#thm("4.5 Capacity (tight upper bound)")[
For *any* non-negative weights there is a graph with $"diam" = 3^L + 1$ on which
the DT fails. So $3^L$ is exactly the model's *capacity* — it cannot master
connectivity beyond path length $3^L$, and needs $n = Omega(3^L)$ nodes to even
pose the hard case.
]

Why *triple* per layer? One attention layer composes a node's features with its
neighbors' — reaching $1$ hop of new info — but stacked through the appended
stream each layer can *cube* the previous reach: $1 -> 3 -> 9 -> dots -> 3^L$.

#figure(
  cetz.canvas({
    import cetz.draw: *
    let y = 0
    // chain of nodes
    let N = 10
    for i in range(N) {
      let x = i * 1.05
      let col = if i == 0 { c-accent } else { luma(70) }
      circle((x, y), radius: 0.16, fill: col, stroke: none)
    }
    for i in range(N - 1) {
      line((i*1.05 + 0.16, y), ((i+1)*1.05 - 0.16, y), stroke: 0.8pt + luma(120))
    }
    // source label
    content((0, -0.55), text(8pt, fill: c-accent)[source])
    // reach brackets
    let bracket(x0, x1, yy, lbl, col) = {
      line((x0, yy), (x0, yy+0.12), stroke: 1pt + col)
      line((x0, yy+0.12), (x1, yy+0.12), stroke: 1pt + col)
      line((x1, yy), (x1, yy+0.12), stroke: 1pt + col)
      content(((x0+x1)/2, yy+0.42), text(8pt, fill: col, lbl))
    }
    bracket(0, 1.05, 0.45, [$L=1:$ reach $3^1=3$], c-good)
    // L=2 conceptual
    content((5.2, -1.15), text(8.5pt, fill: c-algo)[
      each added layer *cubes* the reach: $#h(2pt) 1 arrow.r 3 arrow.r 9 arrow.r dots.c arrow.r 3^L$
    ])
    // boundary marker at 3^L+1
    content((9.45, 0.55), text(8pt, fill: c-heur)[$3^L+1$: fails])
    line((9.45, 0.2), (9.45, -0.2), stroke: (dash: "dashed", paint: c-heur))
  }),
  caption: [Reach grows as $3^L$ in graph *diameter*, not in node count $n$. The decisive
  depth law is $L gt.eq log_3 "diam"(G)$ — non-asymptotic and exact, unlike the usual $Theta(log n)$.],
)

#note[
This *exact, non-asymptotic* $3^L$ is the paper's lever over prior work
(Merrill–Sabharwal give $cal(O)(exp(L))$ asymptotics). Because the bound is
exact, they get a *clean dichotomy* (within- vs beyond-capacity) that drives
every later result. The sharpness is the contribution, not the existence of a
bound.
]

= 3 · The two channels — where the heuristic hides

#thm("4.7 Layerwise permutation-equivariant form")[
Under symmetry of the data ($A -> R$ commutes with relabeling $P A P^top ->
P R P^top$) and non-negative weights, every trained layer weight decomposes as
$ W_ell = A_ell times.circle I_n + B_ell times.circle J_n, quad A_ell, B_ell in RR^(2 times 2), $
where $J_n = bb(1) bb(1)^top$ is all-ones. Two functionally distinct channels.
]

#grid(columns: (1fr, 1fr), gutter: 8pt,
  callout([$I_n$-channel — ALGORITHM], [
    *Local.* Combines features only between *graph neighbors*. Across layers it
    composes multi-hop info $arrow.r$ powers of $A$ $arrow.r$
    $sum_j alpha_j A^j$ = exact matrix-powering. *Correct everywhere within
    capacity.*
  ], col: c-algo, sym: "⚙"),
  callout([$J_n$-channel — HEURISTIC], [
    *Global, rank-one.* $A J_n = d bb(1)^top$ broadcasts the *degree vector*.
    Predicts "connected" from *node degrees* and their products. Correlates with
    truth on dense ER graphs; *fails adversarially* (two cliques: both
    high-degree, yet disconnected).
  ], col: c-heur, sym: "⚡"),
)

#figure(
  cetz.canvas({
    import cetz.draw: *
    // LEFT: algorithmic channel — local composition on a small graph
    let gx = 0
    let nodes = ((0,0), (1,0.4), (0.8,-0.6), (2,0))
    for (i,p) in nodes.enumerate() {
      circle((gx + p.at(0), p.at(1)), radius: 0.13, fill: c-algo, stroke: none)
    }
    line((gx+0,0),(gx+1,0.4), stroke:0.8pt+luma(120))
    line((gx+0,0),(gx+0.8,-0.6), stroke:0.8pt+luma(120))
    line((gx+1,0.4),(gx+2,0), stroke:0.8pt+luma(120))
    content((gx+1, -1.25), text(8.5pt, fill: c-algo)[*local* — neighbor mixing $arrow.r A^j$])
    content((gx+1, 1.25), text(9pt, weight:"bold", fill: c-algo)[$A times.circle I_n$])

    // RIGHT: heuristic channel — global broadcast (star to all)
    let hx = 5.2
    let hub = (hx+1, 0)
    let peri = ((hx, 0.7), (hx + 2, 0.7), (hx - 0.1, -0.6), (hx + 2.1, -0.6), (hx + 1, 1))
    circle(hub, radius: 0.16, fill: c-heur, stroke: none)
    for p in peri {
      circle(p, radius: 0.1, fill: c-heur.lighten(30%), stroke: none)
      line(hub, p, stroke: (dash:"dotted", paint: c-heur))
    }
    content((hx+1, -1.25), text(8.5pt, fill: c-heur)[*global* — degree broadcast $A J = d bb(1)^top$])
    content((hx+1, 1.45), text(9pt, weight:"bold", fill: c-heur)[$B times.circle J_n$])
    // separator
    line((4.4,-1.4),(4.4,1.6), stroke:(dash:"dashed", paint: luma(170)))
  }),
  caption: [The trained weight is a *superposition* $A_ell times.circle I_n + B_ell times.circle J_n$.
  Learning connectivity = a tug-of-war over how much energy lives in each channel.],
)

= 4 · Training dynamics — who wins the tug-of-war

#thm("4.9 / C.5 / C.9 (dynamics, informal)")[
Projected gradient descent on the regularized objective converges to KKT points
at the standard $cal(O)(1/epsilon)$ rate. *Two phases:*\
*Phase 1* — both channels ramp up on easy within-component pairs (fast,
transient, $tilde.op 2 dot 10^2$ of $10^4$ steps).\
*Phase 2* — the data decides. Disconnected/beyond-capacity graphs generate
*false positives* in the $J$-channel and push $B_ell -> 0$; connected graphs
*reward* it. Net sign of the gradient depends on the *fraction of
beyond-capacity graphs*.
]

#thm("4.10 (Learning the Algorithm)")[
If gradient *penalty* from disconnected graphs outweighs the *reward* from
connected ones, the only KKT-compliant value is $B_ell^* = 0$ — the model
converges to the *pure matrix-powering algorithm*.
]

#figure(
  cetz.canvas({
    import cetz.draw: *
    // timeline axis
    line((0,0),(11,0), stroke: 1pt+luma(120), mark:(end:"stealth"))
    content((11, -0.4), text(8pt)[train step])
    // phase 1 region
    rect((0,-0.05),(2.6,2.4), fill: luma(150).lighten(70%), stroke:none)
    content((1.3, 2.65), text(8.5pt, weight:"bold")[Phase 1])
    content((1.3, -0.45), text(7.5pt, fill:luma(110))[both channels ramp])
    content((6.8, 2.65), text(8.5pt, weight:"bold")[Phase 2 — data decides])
    // both rise
    line((0,0.1),(2.6,1.7), stroke: 2pt + c-algo)
    line((0,0.1),(2.6,1.55), stroke: 2pt + c-heur)
    // fork: I keeps rising, two J outcomes
    line((2.6,1.7),(10.5,2.25), stroke: 2pt + c-algo)
    content((10.8,2.25), text(8pt, fill:c-algo)[$A$ (algo)])
    // J suppressed (within-capacity data)
    line((2.6,1.55),(10.5,0.15), stroke: 2pt + c-good)
    content((9.4,0.45), text(7.5pt, fill:c-good)[within-cap data: $B arrow.r 0$ ✓])
    // J promoted (beyond-capacity data)
    line((2.6,1.55),(10.5,1.95), stroke: (dash:"dashed", paint:c-heur, thickness:2pt))
    content((9.0,2.05), text(7.5pt, fill:c-heur)[beyond-cap data: $B$ stays ✗])
    content((-0.3, 3.0), anchor: "west", text(8pt, fill: luma(110))[channel energy →])
  }),
  caption: [Phase 1 is architecture-driven and identical everywhere. Phase 2 is
  *data-driven*: the same model forks into the exact algorithm ($B arrow.r 0$) or a
  heuristic mixture ($B$ persists) depending on the training distribution.],
)

= 5 · The data lever — the actionable result

#defn("4.6 Within- vs beyond-capacity")[
A node pair $(i,j)$ is *within capacity* if $[A^(3^L)]_(i j) > 0$ (i.e.
$d_G(i,j) lt.eq 3^L$), *beyond capacity* otherwise. A graph is within-capacity
if $"diam"(G) lt.eq 3^L$. Define $rho(cal(G))$ = fraction of beyond-capacity
node pairs in the distribution.
]

#callout("The lever", [
*Don't* train on the raw ER distribution. *Up-weight* (or restrict to)
within-capacity graphs $cal(G)_(lt.eq) = {G : "diam"(G) lt.eq 3^L}$. Then the
$A times.circle I_n$ channel is promoted to $tilde.op 100%$ of the weight energy,
the $B times.circle J_n$ heuristic is suppressed, and the model generalizes OOD
(e.g. to 2-Chain / 2-Clique graphs it never saw). *Only a small $rho^* > 0$ is
tolerable* — a little beyond-capacity data is fine, a lot is fatal.
], col: rgb("#b9770e"), sym: "🔧")

#figure(
  cetz.canvas({
    import cetz.draw: *
    // Two training-distribution recipes -> outcomes
    // recipe A (bad): mostly beyond-capacity
    let bx = 0
    rect((bx,0),(bx+2.6,1.4), stroke: 1pt+c-heur, fill: c-heur.lighten(90%), radius:3pt)
    content((bx+1.3,1.05), text(8.5pt, weight:"bold")[raw ER train])
    content((bx+1.3,0.55), text(7.5pt)[many $"diam">3^L$])
    content((bx+1.3,0.2), text(7pt, fill:c-heur)[high $rho$])
    // arrow
    line((bx+2.7,0.7),(bx+3.9,0.7), stroke:1.2pt+luma(120), mark:(end:"stealth"))
    rect((bx+4,0),(bx+6.6,1.4), stroke:1pt+c-heur, fill:c-heur.lighten(95%), radius:3pt)
    content((bx+5.3,0.95), text(8.5pt, weight:"bold", fill:c-heur)[HEURISTIC])
    content((bx+5.3,0.45), text(7.5pt)[degree-counting])
    content((bx+5.3,0.13), text(7pt, fill:c-heur)[fails OOD ✗])

    // recipe B (good)
    let cy = -2.1
    rect((bx,cy),(bx+2.6,cy+1.4), stroke:1pt+c-good, fill:c-good.lighten(90%), radius:3pt)
    content((bx+1.3,cy+1.05), text(8.5pt, weight:"bold")[restricted train])
    content((bx+1.3,cy+0.55), text(7.5pt)[$"diam" lt.eq 3^L$])
    content((bx+1.3,cy+0.2), text(7pt, fill:c-good)[small $rho lt.eq rho^*$])
    line((bx+2.7,cy+0.7),(bx+3.9,cy+0.7), stroke:1.2pt+luma(120), mark:(end:"stealth"))
    rect((bx+4,cy),(bx+6.6,cy+1.4), stroke:1pt+c-good, fill:c-good.lighten(95%), radius:3pt)
    content((bx+5.3,cy+0.95), text(8.5pt, weight:"bold", fill:c-good)[ALGORITHM])
    content((bx+5.3,cy+0.45), text(7.5pt)[matrix powering])
    content((bx+5.3,cy+0.13), text(7pt, fill:c-good)[generalizes ✓])

    // OOD test illustration on the right: two chains (the failure case)
    let tx = 8.2
    content((tx+1, 1.55), text(8pt, weight:"bold")[OOD test: 2-Chain])
    for i in range(4) { circle((tx+i*0.55, 0.9), radius:0.08, fill:luma(70), stroke:none) }
    for i in range(3) { line((tx+i*0.55+0.08,0.9),(tx+(i+1)*0.55-0.08,0.9), stroke:0.7pt+luma(120)) }
    for i in range(4) { circle((tx+i*0.55, 0.2), radius:0.08, fill:luma(70), stroke:none) }
    for i in range(3) { line((tx+i*0.55+0.08,0.2),(tx+(i+1)*0.55-0.08,0.2), stroke:0.7pt+luma(120)) }
    content((tx+1, -0.35), text(7pt, fill:c-heur)[heuristic: "connected" ✗])
    content((tx+1, -0.7), text(7pt, fill:c-good)[algorithm: "disjoint" ✓])
  }),
  caption: [The data lever in one picture. Same architecture, same OOD test (two
  isolated chains) — outcome flips entirely on the training distribution. The
  benefit transfers to *standard* Transformers (Fig 7).],
)

= 6 · Why this matters for a GNN / Graph-Transformer thesis

#idea[
*Capacity as a curriculum signal.* The $3^L$ bound is exact, so you can
*measure* whether each training graph is within/beyond capacity and design a
principled *difficulty-aware curriculum* or *rejection-sampling* scheme. Thesis
angle: does an analogous exact-capacity law exist for *message-passing GNNs*
(reach $= L$ hops, linear not $3^L$) and for *sparse / windowed* graph
Transformers? The contrast $L$ vs $3^L$ vs full attention is a clean axis.
]
#idea[
*Channel decomposition as a diagnostic.* The $A times.circle I + B times.circle J$
split gives a *measurable "heuristic energy" $norm(B)/norm(W)$*. Could become a
general *probe* for shortcut-learning in graph models — track it during training
across tasks (counting, shortest-path, bipartiteness) and architectures. A
lightweight, interpretable thesis contribution.
]
#idea[
*Generalize the failure taxonomy.* Here the heuristic = degree counting and the
adversarial case = two cliques. For other graph tasks (cycle detection,
coloring, matching) what is the natural heuristic channel and its adversarial
distribution? A thesis could map *task $arrow.r$ heuristic $arrow.r$ data-lever*
systematically.
]
#idea[
*Beyond ER.* All theory is on Erdős–Rényi. Real graphs are heavy-tailed /
community-structured, where degree heuristics are *stronger*. Does the data
lever still recover the algorithm on SBM / power-law / molecular graphs? Strong
empirical thesis with a clear hypothesis from this paper.
]

= 7 · Open questions & critique

#ask[
*Non-negativity gap.* Theory assumes $W_ell gt.eq 0$; training does not enforce
it yet predictions hold (their words). *Why?* A rigorous account of the
non-negative case as an attractor would tighten the story — possible thesis
sub-problem.
]
#ask[
*Does $3^L$ survive multi-head / softmax / causal masking?* The DT uses
single-head ReLU attention, no softmax, no causal mask. How much of the exact
capacity is an artifact of these simplifications vs. fundamental?
]
#ask[
*What is $rho^*$ quantitatively?* They show a small tolerable fraction of
beyond-capacity data exists, but the threshold depends on graph distribution.
Can it be predicted a-priori from $(n, p, L)$? That would make the lever a
practical recipe rather than a knob to tune.
]
#ask[
*Cost of the lever.* Restricting to within-capacity graphs shrinks the usable
data and may hurt *length generalization* (Fig 5: $d=2$ fails to length-
generalize). Is there a tension between "learns the algorithm" and "extrapolates
to larger graphs"? The sweet spot ($d = "Cap"$) seems narrow.
]

= 8 · Connections to my reading list
- *Sanford et al. 2024 / 2023* (depth–width tradeoffs, graph tasks): complementary
  expressivity lower bounds — cross-check the $3^L$ reach against their
  communication-complexity arguments.
- *Loukas 2020* (what GNNs cannot learn, depth×width): the GNN analogue of the
  capacity story — message passing reaches $L$ hops, not $3^L$.
- *NeurIPS 2025 depth–width for Transformers on graphs*: directly adjacent; read
  next, contrast capacity definitions.

#v(0.4em)
#line(length: 100%, stroke: 0.4pt + luma(200))
#text(8pt, fill: luma(130))[
*Reading status:* main body (§1–6, pp. 1–9) read · appendix proofs (B, C) skimmed
· next: re-derive Thm 4.7 Kronecker decomposition by hand; pull Fig 6 ($rho$ sweep)
into the $rho^*$ question. · _Companion generated #datetime.today().display()._
]
