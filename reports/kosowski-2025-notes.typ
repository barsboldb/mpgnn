// ============================================================
//  Reading Companion — Kosowski et al. 2025
//  "The Dragon Hatchling: The Missing Link between the
//   Transformer and Models of the Brain"  (arXiv 2509.26507v1)
//  Uses the shared paper-companion.typ template.
// ============================================================

#import "paper-companion.typ": *

// ---- role tints (mapped onto the shared palette) ----
#let c-tf   = c-algo     // Transformer / tensor side (blue)
#let c-bdh  = c-good     // BDH graph side            (green)
#let c-brain= c-accent   // brain models              (purple)
#let c-syn  = c-idea     // synapse / Hebbian state   (amber)

#let tag(name, col) = box(fill: col.lighten(78%), inset: (x: 4pt, y: 1pt),
  radius: 2pt, outset: (y: 1pt))[#text(weight: "bold", size: 8.5pt, fill: col.darken(8%))[#name]]

#show: companion.with(
  title: "The Dragon Hatchling",
  subtitle: "The missing link between the Transformer and models of the brain",
  authors: "Kosowski, Uznański, Chorowski, Stamirowska, Bartoszkiewicz (Pathway)",
  venue: "Preprint Sep 2025 (arXiv 2509.26507v1)",
  tagline: "Reading companion · GNN / Graph-Transformer thesis exploration",
  running: "Kosowski et al. 2025 · The Dragon Hatchling (BDH)",
)

#tldr[
A Transformer and a brain look *structurally incompatible* — dense tensors vs. a
uniform, scale-free graph of local neurons. *BDH (Dragon Hatchling)* is the
bridge: a language model defined as *local graph dynamics* on $n$ neuron particles,
where *attention is edge-reweighting* of synapses under a *Hebbian* rule ("neurons
that fire together wire together"). Its tensor-friendly twin *BDH-GPU* trains on GPUs
with *Transformer-like scaling laws* (10M–1B params, rivals GPT-2). The architecture
fuses two micro-foundations: *modus-ponens reasoning* (fixed weights $G$) and
*Hebbian fast-weights* (evolving synapse state $sigma$). *Punchline:* BDH-GPU's
activations are *positive and sparse* ($approx 5%$), its trained weight graphs are
*scale-free + modular* (emergent, not designed), and individual *synapses are
monosemantic* — a single edge fires on "currency" across English *and* French. It
opens a path to an *axiomatic / thermodynamic-limit* theory of reasoning models.
]

#qa(
  [Can we build a language model that *performs* like a Transformer but admits a
   *brain-like* micro-interpretation — uniform, scale-free, local graph dynamics —
   so that its inference is *interpretable* and its scaling behavior *foreseeable*?],
  [Yes: define the model as an *edge-reweighting kernel* on a neuron graph
   (attention $=$ Hebbian synapse update). Restrict it to a *low-rank, mean-field*
   form → BDH-GPU, a state-space model that trains by backprop, scales in *one*
   neuron dimension $n$, and *empirically* develops scale-free modular structure
   and monosemantic synapses.],
)

= 1 · The problem — why brains and Transformers seem unrelated

#note[
*The motivating gap.* The brain is a uniform, scale-free, *distributed* graph
($n approx 8 dot 10^10$ neurons, $m > 10^14$ synapses) with *local* dynamics; the
Transformer is a *centralized* tensor pipeline. Directly simulating one step of brain
reasoning by a Transformer with chain-of-thought would need *billions* of CoT tokens.
The paper wants a *tighter, more direct* correspondence — an architecture that is
*simultaneously* a performant LM and a local graph system. This is the same
"architecture-as-distributed-computing" lens as Sanford's MPC↔Transformer, pushed all
the way to neuron-level particle dynamics.
]

#idea[
*Towards "Axiomatic AI" and scale-free foreseeability.* The deeper agenda: a model
that is *scale-free in size and time* admits a *thermodynamic limit* $cal(P)_cal(A) =
lim_(n arrow oo) cal(P)_cal(A)(n)$, so that small tests *extrapolate* to large
deployments — a route to *PAC-like bounds for generalization of reasoning over time*
(avoiding "paperclip-factory" failures of autonomous agents). This is the
*safety/uncertainty* payoff and ties straight to the Hüllermeier epistemic-uncertainty
reading. _(see `huellermeier-2020-notes`.)_
]

= 2 · The bridge — four architectures, one function

#figure(
  cetz.canvas({
    import cetz.draw: *
    let col(x, title, sub, c) = {
      rect((x - 0.95, -2.4), (x + 0.95, 2.4), radius: 4pt, fill: c.lighten(90%), stroke: c + 0.6pt)
      content((x, 2.1), text(8pt, weight: "bold", fill: c)[#title])
      content((x, 1.75), text(6pt, style: "italic", fill: luma(110))[#sub])
    }
    // shaded halves
    content((-2.6, 2.95), text(7.5pt, weight: "bold", fill: c-tf)[tensor · centralized])
    content((3.2, 2.95), text(7.5pt, weight: "bold", fill: c-bdh)[graph · distributed])
    col(-3.4, "Transformer", "KV-cache", c-tf)
    col(-1.1, "BDH-GPU", "state in " + $RR^(n times d)$, c-tf)
    col(1.2, "BDH", "synapse " + $sigma$, c-bdh)
    col(3.5, "Brain", "spiking + Hebbian", c-brain)
    // row labels
    for (y, lbl) in ((1.1, "attention"), (-0.1, "local dynamics"), (-1.4, "feed-forward")) {
      content((-5.0, y), text(6.5pt, fill: luma(120))[#lbl])
    }
    // bridge arcs (BDH-GPU is the keystone)
    line((-2.45, 0), (-2.05, 0), mark: (end: "stealth", start: "stealth"), stroke: luma(140) + 0.6pt)
    content((-2.25, 0.3), text(6pt, fill: luma(120))[linear])
    content((-2.25, -0.3), text(6pt, fill: luma(120))[algebra])
    line((-0.15, 0), (0.25, 0), mark: (end: "stealth", start: "stealth"), stroke: c-syn + 1pt)
    content((0.05, 0.35), text(6.5pt, weight: "bold", fill: c-syn)[$rho = E sigma$])
    line((2.15, 0), (2.55, 0), mark: (end: "stealth", start: "stealth"), stroke: luma(140) + 0.6pt)
    content((2.35, 0.3), text(6pt, fill: luma(120))[emulate])
  }),
  caption: [The central claim (paper's Fig 1). BDH-GPU is the *keystone* of a
  four-way bridge: it is a *tensor* model (trains like the Transformer) yet is
  *formally equivalent* to BDH, a *local graph* model that a spiking Hebbian brain
  circuit can emulate. The same three mechanisms — attention, local dynamics,
  feed-forward — appear on every pier.],
)

#defn("BDH — a language model as local graph dynamics")[
$n$ neurons (graph nodes) connected by synapses (edges). *Fixed* parameters live in
neuron-neuron graphs $G_x, G_y$ (the *ruleset* / "program", learned by backprop);
*evolving* state lives on synapse edges $sigma(i,j)$ (*fast weights*, the working
memory). Inference runs a distributed protocol of *compute + communicate* rounds —
"communication by wire". The system fuses two micro-foundations: *modus-ponens*
inference $X(i), sigma(i,j) arrow A(j)$ and *Hebbian* reweighting
$Y(i), X(j) arrow sigma(i,j)$.
]

#note[
*The 1:1 parameter↔state ratio is the design key.* A fast-weights system over $n$
facts has $m = O(n^2)$ synapse state entries but comparable trainable parameters; a
classic RNN (LSTM) keeps only $O(n)$ state. The Transformer's success, the authors
argue, comes from this *balanced* ratio — parameters and state both $approx m$. BDH
makes it explicit by choosing sparsity $n << m << n^2$, giving a *graph* on $n$ nodes
and $m$ edges, where edges *triple-task* as: carrying state, carrying parameters, and
*mediating communication*.
]

= 3 · The equations of reasoning

#figure(
  cetz.canvas({
    import cetz.draw: *
    let r = 1.9
    let node(ang, lbl, sub, c) = {
      let p = (r * calc.cos(ang * 1deg), r * calc.sin(ang * 1deg))
      circle(p, radius: 0.72, fill: c.lighten(85%), stroke: c + 0.7pt)
      content((p.at(0), p.at(1) + 0.18), text(7pt, weight: "bold", fill: c)[#lbl])
      content((p.at(0), p.at(1) - 0.22), text(5.5pt, fill: luma(110))[#sub])
    }
    node(135, [Round $4l$], [infer from state], c-bdh)
    node(45,  [$4l{+}1$], [Hebbian reweight], c-syn)
    node(-45, [$4l{+}2$], [replicator], c-heur)
    node(-135,[$4l{+}3$], [infer from params], c-tf)
    // cyclic arrows
    let arc(a0, a1) = {
      let m = (a0 + a1) / 2
      let p0 = ((r) * calc.cos(a0 * 1deg), (r) * calc.sin(a0 * 1deg))
      let p1 = ((r) * calc.cos(a1 * 1deg), (r) * calc.sin(a1 * 1deg))
      line(p0, p1, mark: (end: "stealth"), stroke: luma(120) + 0.7pt)
    }
    arc(120, 60); arc(30, -30); arc(-60, -120); arc(-150, -210)
    content((0, 0.35), text(7pt, weight: "bold")[round-robin])
    content((0, 0.0), text(6pt, fill: luma(110))[mod $4L$])
    content((0, -0.35), text(6pt, fill: luma(110))[$L approx 8$])
    // side annotations
    content((4.3, 1.3), text(6.5pt, fill: c-bdh)[$X, sigma arrow A$])
    content((4.3, -1.3), text(6.5pt, fill: c-syn)[$Y, X arrow.long^(G_s) sigma$])
    content((-4.3, 1.3), text(6.5pt, fill: c-tf)[$Y arrow.long^(G_x) X$])
    content((-4.3, -1.3), text(6.5pt, fill: c-heur)[$A arrow.long^(G_y) Y$])
  }),
  caption: [The "equations of reasoning" (paper's Table 1). One BDH layer $l$ is a
  cycle of four local-graph rounds: read activation from synapse *state*; *Hebbian*
  reweight of synapses (attention!); neuron *replicator* dynamics (evolutionary
  selection of active neurons); read activation from fixed *parameters*. New tokens
  enter every $4L$ rounds — $L$ plays the role of Transformer depth.],
)

#defn("Edge-reweighting kernel")[
A restriction of a general chemical-reaction / replicator *interaction kernel*
$q'_k := (1 - d_k) q_k + sum_(i,j) r_(i j k) q_i q_j$ to *node-edge-node* triples
$k in {i, j}$ on a graph: local rules use only state on a node $i$, or on an edge
$(i,j)$ and its endpoints. This is *exactly* the class a spiking, Hebbian neuron graph
can implement — the formal sense in which BDH is "brain-runnable".
]

#idea[
*Attention as a micro-inductive bias of reasoning.* Read $sigma(i,j) > 0$ as a
*positive bias toward the implication $i arrow j$* accumulated from past context.
The local rule mirrors the logical axiom $(X arrow (i arrow j)) arrow ((X arrow i)
arrow (X arrow j))$ — *modus ponens with a learned utility* on which implications to
explore next. For my thesis this reframes graph attention as *programmable message
passing over a learned implication graph* — a clean conceptual cousin of GNN message
passing where edge weights are *dynamic, context-dependent* fast weights.
]

= 4 · BDH-GPU — the tensor-friendly twin

#defn("BDH-GPU (n, d) state-space system, Eq. 8")[
Three parameter matrices: encoder $E in RR^(d times n)$ and decoders
$D_x, D_y in RR^(n times d)$, shared across all $L$ layers (Universal-Transformer
style). For each token $t$, layer $l$:
$ x_(t,l) = x_(t,l-1) + (D_x "LN"(E y_(t,l-1)))^+, $
$ y_(t,l) = (D_y "LN"(rho_(t,l) x_(t,l)))^+ dot.o x_(t,l), $
with attention state $rho_(t,l) in RR^(n times d)$ updated by a *rank-1
key$times$value* rule (linear attention). Total params $= (3 + o(1)) n d$, *all* in
$E, D_x, D_y$. Scales in *one* dimension $n$ (neurons); $d << n$ is the low-rank
*synaptic* dimension ($d = 256$, $n$ up to $65536$).
]

#figure(
  cetz.canvas({
    import cetz.draw: *
    // neuron layer (wide n) -> encoder -> bottleneck d -> decoders -> neuron layer
    let nx(y, c, lbl) = {
      rect((-4.2, y), (-1.2, y + 0.5), fill: c.lighten(82%), stroke: c + 0.5pt)
      content((-2.7, y + 0.25), text(7pt)[#lbl])
    }
    // input
    content((-2.7, 2.5), text(7pt, weight: "bold")[neuron layer $x_(l-1) in RR^n$ (sparse, $>= 0$)])
    nx(1.9, c-tf, [wide: $n approx 32768$])
    // encoder bottleneck
    line((-2.7, 1.9), (-2.7, 1.35), mark: (end: "stealth"), stroke: c-ink + 0.5pt)
    content((-1.6, 1.62), text(6pt, fill: luma(110))[$E$ ($n arrow d$)])
    rect((-3.4, 0.85), (-2.0, 1.35), fill: c-syn.lighten(70%), stroke: c-syn + 0.6pt)
    content((-2.7, 1.1), text(6.5pt, weight: "bold", fill: c-syn.darken(8%))[synaptic $d=256$])
    // two paths
    // FF path (ReLU-lowrank) left
    line((-3.0, 0.85), (-3.6, 0.35), mark: (end: "stealth"), stroke: c-bdh + 0.7pt)
    rect((-4.6, -0.4), (-3.0, 0.2), fill: c-bdh.lighten(85%), stroke: c-bdh + 0.6pt)
    content((-3.8, -0.1), text(6.5pt, weight: "bold", fill: c-bdh)[ReLU-lowrank])
    content((-3.8, -0.55), text(5.5pt, fill: luma(110))[$(D_x dot)^+$ → $x_l$])
    // attention path right
    line((-2.4, 0.85), (-1.0, 0.35), mark: (end: "stealth"), stroke: c-accent + 0.7pt)
    rect((-1.4, -0.4), (0.6, 0.2), fill: c-accent.lighten(88%), stroke: c-accent + 0.6pt)
    content((-0.4, -0.1), text(6.5pt, weight: "bold", fill: c-accent)[linear attention])
    content((-0.4, -0.55), text(5.5pt, fill: luma(110))[state $rho_l in RR^(n times d)$, rank-1 update])
    // output
    nx(-1.6, c-tf, [$y_l = (D_y "LN"(rho x))^+ dot.o x_l$])
    content((-2.7, -1.85), text(6pt, fill: luma(120))[positive · sparse ($approx 5%$ non-zero)])
    line((-3.8, -0.4), (-3.0, -1.1), mark: (end: "stealth"), stroke: c-bdh + 0.6pt)
    line((-0.4, -0.4), (-2.4, -1.1), mark: (end: "stealth"), stroke: c-accent + 0.6pt)

    // legend / contrast box
    content((3.0, 1.8), text(7pt, weight: "bold")[vs. Transformer])
    let bullet(y, t) = content((3.0, y), text(6pt, fill: luma(90))[#t])
    bullet(1.4, [• attention in *big* dim $n$, not $d$])
    bullet(1.05, [• keys/queries = activations, $>= 0$])
    bullet(0.7, [• state $rho$ sized like params])
    bullet(0.35, [• *no* context-length bound])
    bullet(0.0, [• 3 matrices, all layers shared])
  }),
  caption: [One BDH-GPU layer (paper's Fig 6). Wide positive neuron activations
  $x in RR^n$ are squeezed through the *synaptic* bottleneck $d << n$ by encoder $E$,
  then split into a *ReLU-lowrank* feed-forward path ($D_x$) and a *linear-attention*
  path carrying the persistent state $rho$. Attention happens in the *large neuron
  dimension* with *positive, sparse* keys — the opposite of the Transformer's small
  dense head space.],
)

#thm("BDH ⊇ BDH-GPU expressiveness (Claims 3–4, Obs. 4)")[
For the same asymptotic $O(n d)$ parameters, the *graph* feed-forward mechanism of BDH
is *strictly more expressive* than BDH-GPU's tensor ReLU-lowrank block (an arbitrary
sparse graph $G in cal(G)^2(n,m)$ need not admit an exact low-rank factorization
$G = D E$). BDH and BDH-GPU are *formally equivalent* up to LayerNorm placement, via
$G_x^e - G_x^i = D_x E$, $G_y^e - G_y^i = D_y E$, $G_s = bb(1)^(n times n)$.
]

#note[
*Scaling-law verdict (§4.2, Fig 7).* On translation, BDH-GPU's validation-loss-vs-size
curve *tracks GPT-2/GPT-XL* across 10M–1B params, and it *learns faster per data
token* (better in the scarce-data regime). FLOPS per token $approx O(n d L)$. It is a
genuine *state-of-the-art-class* architecture, not a toy — which is what makes the
brain-bridge claim weighty rather than cute.
]

= 5 · Emergent structure — scale-free, modular, monosemantic

== ReLU-lowrank as denoising signal propagation (§5.3)

#thm("ReLU as a noise threshold (Claim 5)")[
A low-rank linear map $G = D E$ approximates a target $G'$ only with $O(sqrt(log n \/ d))$
error in the $infinity$-norm (Johnson–Lindenstrauss) — *useless* in $L_2 \/ L_1$. But
adding a *negative bias + ReLU gate*, $f_(D E)(z) = (D E z)^+$, *suppresses the noise*:
it can reproduce e.g. a random-walk (Markov) transition $z arrow (G' z)^+$ to $L_1$-error
$O(epsilon)$. The ReLU turns a lossy low-rank channel into a faithful *sparse-positive*
propagator.
]

#note[
*Why modularity emerges (Claim 6, Obs. 6).* A neuron $j$ activates only when its
local "F-score" $rho = sqrt((|C_j| \/ |A|)(|C_j| \/ |B_j|))$ clears a threshold —
i.e. when the incoming signal is *concentrated in $j$'s own cluster*. This is
*precisely* communication on a graph with positive *Newman modularity*: the
ReLU-lowrank net can represent in-cluster spreading on a $k$-block stochastic-block-model
with arbitrarily small positive modularity. *Modular, scale-free structure is forced by
the function*, not designed in.
]

#figure(
  cetz.canvas({
    import cetz.draw: *
    // (a) core-periphery blob
    content((-2.6, 2.2), text(7.5pt, weight: "bold")[(a) trained graph $G = D_x E$])
    // dense core
    for i in range(22) {
      let a = i * 16.36 * 1deg
      let rr = 0.18 + 0.22 * calc.rem(i, 3)
      circle((-2.6 + rr * calc.cos(a), 0.6 + rr * calc.sin(a)), radius: 0.035, fill: c-tf, stroke: none)
    }
    circle((-2.6, 0.6), radius: 0.55, fill: c-tf.lighten(70%), stroke: none)
    // periphery
    for i in range(16) {
      let a = i * 22.5 * 1deg
      let rr = 1.05 + 0.35 * calc.rem(i, 4)
      let p = (-2.6 + rr * calc.cos(a), 0.6 + rr * calc.sin(a))
      circle(p, radius: 0.03, fill: luma(120), stroke: none)
      line((-2.6 + 0.5 * calc.cos(a), 0.6 + 0.5 * calc.sin(a)), p, stroke: luma(190) + 0.3pt)
    }
    content((-2.6, -1.05), text(6pt, fill: luma(110))[core-periphery · high modularity])

    // (b) power-law degree, log-log
    let ox = 1.0; let oy = -1.0; let w = 3.0; let h = 2.6
    line((ox, oy), (ox + w, oy), mark: (end: "stealth"), stroke: c-ink + 0.6pt)
    line((ox, oy), (ox, oy + h), mark: (end: "stealth"), stroke: c-ink + 0.6pt)
    content((ox + w/2, oy - 0.4), text(6.5pt)[log degree])
    content((ox - 0.4, oy + h/2), text(6.5pt)[#std.rotate(-90deg, reflow: true)[log freq]])
    content((ox + w/2, oy + h + 0.05), text(7.5pt, weight: "bold")[(b) power-law degrees])
    // scatter following a line of negative slope
    let pts = ((0.2,2.4),(0.5,2.05),(0.8,1.75),(1.1,1.5),(1.4,1.2),(1.7,1.05),(2.0,0.8),(2.3,0.55),(2.6,0.35),(2.85,0.2))
    for p in pts {
      circle((ox + p.at(0), oy + p.at(1)), radius: 0.05, fill: c-bdh, stroke: none)
    }
    line((ox + 0.2, oy + 2.45), (ox + 2.85, oy + 0.15), stroke: (dash: "dashed", paint: c-heur, thickness: 0.8pt))
    content((ox + 2.1, oy + 1.7), text(6pt, fill: c-heur)[slope $approx -gamma$])
  }),
  caption: [Emergent network structure (paper's Figs 9–11). *(a)* The trained
  encoder-decoder product $G = D_x E$, read as a neuron-neuron graph, shows a
  *core-periphery, high-modularity* layout. *(b)* Its degree distribution is
  *heavy-tailed / power-law* — the lithmus test of a *scale-free* system. Both arise
  from training alone, with no structural prior (L1-regularization was *off*).],
)

== Monosemantic synapses & sparse activation (§6.2–6.4)

#figure(
  cetz.canvas({
    import cetz.draw: *
    // neuron i -- synapse -- neuron j
    circle((-4.0, 0.9), radius: 0.22, fill: c-tf.lighten(60%), stroke: c-tf + 0.6pt)
    content((-4.0, 0.9), text(6pt)[$i$])
    circle((-4.0, -0.9), radius: 0.22, fill: c-tf.lighten(60%), stroke: c-tf + 0.6pt)
    content((-4.0, -0.9), text(6pt)[$j$])
    line((-4.0, 0.68), (-4.0, -0.68), stroke: c-syn + 2pt)
    content((-3.45, 0), text(6.5pt, fill: c-syn.darken(8%))[$sigma(i,j)$])
    content((-4.0, 1.4), text(6.5pt, fill: luma(110))[Hebbian: $y_i$ then $x_j$ → ↑$sigma$])

    // token strip with activation highlight
    let toks = ("the", "US", "Dollar", "rose", "vs", "the", "British", "Pound")
    let hot = (false, false, true, false, false, false, false, true)
    for (k, w) in toks.enumerate() {
      let x = -2.0 + k * 0.95
      let c = if hot.at(k) { c-heur.lighten(55%) } else { luma(235) }
      rect((x, -0.3), (x + 0.85, 0.3), fill: c, stroke: luma(170) + 0.3pt)
      content((x + 0.42, 0), text(6pt)[#w])
    }
    content((1.4, 0.75), text(6.5pt, weight: "bold", fill: c-heur)[«currency synapse» fires])
    content((1.4, -0.7), text(6pt, fill: luma(110))[same edge fires on "livre sterling" (fr)])
  }),
  caption: [A monosemantic synapse (paper's Figs 12–13). One state entry $sigma(i,j)$
  — named the *"currency synapse"* — strengthens *exactly* when a currency concept is
  in context, *across languages* ("Dollar"/"Pound" ↔ "livre sterling"). A one-sided
  Mann–Whitney test confirmed the selectivity ($p < 10^(-14)$). State is readable at
  the level of *individual edges* — interpretability of *state*, not just weights.],
)

#note[
*Sparse, surprisal-driven activation.* Only $approx 5%$ of neurons fire per token;
higher layers go *quiet* once input becomes predictable (Fig 14). This is *native*
adaptive computation — no separate halting network — and the reason synapse updates
(and thus interpretable concept-tracking) are rare and clean. Sparsity also enables
the §7.2 experiments: *training without backprop-through-time* (only remember *which*
synapse changed), and §7.1 *model merging by concatenation* in the neuron dimension.
]

= 6 · Why this matters for my thesis

#idea[
*A graph-native language model — the direct GNN/Transformer synthesis.* BDH is
*literally* a graph model whose attention is *dynamic edge-reweighting* and whose
feed-forward is *local message propagation* with a ReLU threshold. This is the cleanest
bridge yet between my two poles: it is a Transformer-class LM *and* a message-passing
graph system. Thesis angle: position BDH as the *limit point* of the
"GNN ↔ Graph-Transformer" axis and ask which BDH mechanisms transfer to standard GNNs
(e.g. positive-sparse activations, Hebbian fast-weight edges on a fixed graph).
]
#idea[
*BDH is a distributed-computing model — reuse the MPC lens.* BDH-Normfree's dynamics
are a *mean-field broadcast* among $n$ particles ($O(d L)$ scalars per pairwise
interaction): rounds ↔ layers, neurons ↔ machines, $d$ ↔ local-memory/bandwidth.
This is *the same dictionary* as Sanford's MPC↔Transformer. Thesis: place BDH on
Sanford's hierarchy — is its $O(n d L)$-per-token, log-depth-shareable computation a
*parallelizable*-class system? _(see `sanford-2024-notes`, `sanford-2023-notes`.)_
]
#idea[
*Capacity = neurons × synaptic-rank, an emergent ruler.* BDH's expressivity is set by
$(n, d)$ with state sized like parameters; emergent modularity means *effective*
capacity concentrates in a scale-free core. Loukas measures GNN capacity as
depth × width with a $d w = tilde(Omega)(n^delta)$ floor. Thesis: compare Loukas's
*designed* depth×width capacity to BDH's *emergent* $n times d$ + modularity — does
scale-free structure *beat* the Loukas lower bounds in practice? _(see `loukas-2020-notes`.)_
]
#idea[
*Interpretability-of-state as an epistemic-uncertainty handle.* Monosemantic synapses
make the *in-context state* readable per edge. Combined with the paper's
"axiomatic AI / thermodynamic-limit" goal (PAC-like bounds for reasoning over time),
this is a concrete substrate for *epistemic-uncertainty* estimation on a graph model —
read which concept-synapses are (un)confident. _(see `huellermeier-2020-notes`.)_
]

= 7 · Open questions and critique

#ask[
*Equivalence vs. practice.* BDH and BDH-GPU are *formally* equivalent only up to
LayerNorm placement, and BDH's sparse trainable $G_s$ is *dropped* in BDH-GPU (set to
all-ones). How much expressivity / interpretability is lost by the GPU restriction —
and would a *sparse, trainable-$sigma$* BDH (the "more expressive" version) actually
train and scale? Untested.
]
#ask[
*Emergent structure — claim strength.* Scale-free / modularity is shown for $3$ of $4$
heads on $5$ seeds via *Louvain* community detection and power-law-*looking* degree
plots. Power-law fits are notoriously fragile. Is the structure *robustly* scale-free,
or merely heavy-tailed? And is modularity *causal* for performance or a byproduct?
]
#ask[
*The axiomatic / thermodynamic-limit theory is aspirational.* The PAC-like bounds for
"generalization of reasoning over time", the existence of a limit object
$cal(P)_cal(A)$, and length-generalization guarantees are *motivated* but *not proved*.
Right now BDH offers a *plausible substrate* for such a theory, not the theory itself.
]
#ask[
*Brain claim — analogy or mechanism?* The bridge to spiking Hebbian circuits is at the
level of *function* (attention as synapse plasticity), and explicitly *not* about
longer-term learning / backprop in the brain. How much is a genuine mechanistic claim
vs. a structural analogy? The §8.2 hypotheses are carefully hedged.
]

= 8 · Connections to my reading list

BDH sits at the *center* of my cluster — it is simultaneously a graph model, a
Transformer-class LM, and a distributed-computing system:

- *Sanford 2024 / 2023* (Transformer ↔ MPC, communication complexity): BDH-Normfree's
  *mean-field broadcast* over $n$ particles is a distributed-computing model with the
  *same* rounds↔depth, memory↔width dictionary. Natural to place BDH on the
  parallelizable/search hierarchy. _(see `sanford-2024-notes`, `sanford-2023-notes`)_
- *Loukas 2020* (what GNNs can't learn, depth×width capacity): BDH is local graph
  dynamics with capacity $approx n times d$; contrast Loukas's *designed* capacity
  floor with BDH's *emergent* scale-free modular capacity. _(see `loukas-2020-notes`)_
- *Hüllermeier & Waegeman 2021* (aleatoric/epistemic uncertainty): BDH's
  *axiomatic-AI / PAC-over-time* goal and *state interpretability* are an
  epistemic-uncertainty and safety angle on graph reasoning models.
  _(see `huellermeier-2020-notes`)_
- *Ye 2026* (provable learning of graph algorithms): the *learnability* counterpart —
  BDH *claims* SOTA learning empirically; cross-check against provable-learning results
  and the $log_3 "diam"$ / $Theta(log N)$ depth laws. _(see `ye-2026-notes`)_

#v(0.4em)
#line(length: 100%, stroke: 0.4pt + luma(200))
#text(8pt, fill: luma(130))[
*Reading status:* §1–4 (motivation, the bridge, equations of reasoning, BDH-GPU
state-space) read closely · §5 (ReLU-lowrank denoising, emergent modularity/scale-free)
read · §6 (linear attention, monosemanticity, sparse activation) read · §7 (model
merging, no-BPTT) + §8 conclusions read · Appendices A–E skimmed · next: place BDH on
Sanford's MPC hierarchy and sketch a sparse-trainable-$sigma$ GNN variant. ·
_Companion generated #datetime.today().display()._
]
