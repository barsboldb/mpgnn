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
#let lesson(body) = block(fill: rgb("#e8f0fe"), inset: 8pt, radius: 3pt, width: 100%)[*Lesson.* #body]

#let blue   = rgb("#1a73e8")
#let purple = rgb("#9334e6")
#let orange = rgb("#e8710a")
#let green  = rgb("#137333")
#let red    = rgb("#c5221f")

// Color discipline for the whole document:
//   blue   = the working configuration / the trace model
//   red    = the failing configuration / the baseline
//   purple = the secondary measure (trace exact-match)

#align(center)[
  #text(size: 19pt, weight: "bold")[The Chain-of-Thought Journey]
  #v(0.3em)
  #text(size: 11pt)[Five ways a transformer refused to run BFS, and what each one taught us]
  #v(0.2em)
  #text(size: 9.5pt, fill: luma(100))[2026-07-03 #sym.arrow 07-05 · `src/cot.py`, `src/cot_tokens.py` · #datetime.today().display()]
]

#v(0.6em)

#note[
  *In one sentence.* We wanted a shallow transformer to decide whether a graph is
  connected by _writing out_ a breadth-first search, token by token — and getting there
  required five separate fixes, because five separate things silently sabotage a model
  that is supposed to learn an _algorithm_ rather than a _statistic_.
]

= The map

Every stage below produced a wrong result that _looked_ like a different problem than it
was. The journey, compressed:

#figure(
  cetz.canvas(length: 0.72cm, {
    import cetz.draw: *
    let stage(x, y, w, title, sub, col) = {
      rect((x, y), (x + w, y + 2.0), fill: col.lighten(88%), stroke: col + 0.8pt, radius: 0.15)
      content((x + w/2, y + 1.42), text(size: 7.5pt, weight: "bold", fill: col)[#title])
      content((x + w/2, y + 0.62), text(size: 6.8pt, fill: luma(60))[#sub])
    }
    let arrow(x1, x2, y) = line((x1, y), (x2, y), stroke: luma(120) + 0.8pt, mark: (end: ">", scale: 0.6))

    stage(0.0, 4.2, 4.0,  [scratchpad CoT],       [stuck at chance],        red)
    stage(4.9, 4.2, 4.0,  [real CoT, v1],         [1.0 instantly (!)],      red)
    stage(9.8, 4.2, 4.0,  [leak fixed],           [chance again],           red)
    stage(14.7, 4.2, 4.0, [local trace],          [still chance],           red)
    arrow(4.05, 4.85, 5.2)
    arrow(8.95, 9.75, 5.2)
    arrow(13.85, 14.65, 5.2)
    line((16.7, 4.15), (16.7, 3.35), (2.0, 3.35), (2.0, 2.55), stroke: luma(120) + 0.8pt, mark: (end: ">", scale: 0.6))

    stage(0.0, 0.5, 4.0,  [zero regularization],  [lookup: 0.15 #sym.arrow 0.999], blue)
    stage(4.9, 0.5, 4.0,  [4#sym.times data],     [grokking at ep. 60],   blue)
    stage(9.8, 0.5, 4.6,  [*BFS executed*],       [0.96 / 0.99, flat in diameter], green)
    arrow(4.05, 4.85, 1.5)
    arrow(8.95, 9.75, 1.5)

    content((8.35, 3.0), text(size: 7pt, fill: luma(100))[
      each failure diagnosed by its own isolating experiment
    ])
  }),
  caption: [The route. Four configurations that failed for four different reasons, then
    two fixes that were about the _optimizer_ and the _data_, not the architecture.],
) <fig-map>

= Background, from zero

== The task, and why depth is the enemy

The task is binary *graph connectivity*: given all the edges, is there a path between
every pair of nodes? The catch is _reachability distance_. Information in a transformer
(or GNN) layer can only combine things that layer can see together: a message-passing
layer moves information *one hop* along edges; even a global-attention layer needs on the
order of $log n$ layers to _compute_ reachability (Sanford et al. 2024). So a model with a
*fixed* number of layers has a fixed reasoning horizon — and a graph with a large diameter
(longest shortest path) sits beyond it. Our datasets make this bite on purpose: caterpillar
graphs with exact diameters 2–18, and an adversarial two-blob family where the connected
and disconnected classes differ by *one single edge* with all simple statistics
(degrees, edge counts) matched exactly.

== What chain-of-thought is supposed to buy

A transformer that _generates text_ is not depth-limited in the same way: after emitting a
token, that token becomes *input* for the next step. Each generated token is one more
sequential computation step — so a depth-2 model that emits $d$ tokens has effectively
bought $d$ extra rounds (Merrill & Sabharwal 2024). The plan: make the model write out a
breadth-first search —

#align(center)[
  `prompt: N 0 1 2 .. E 0 3 1 4 ..  TRACE` #h(0.8em) `completion: <the BFS> ANS YES EOS`
]

— training it with *teacher forcing* and evaluating with *greedy decoding*.

#note[
  *Two words worth defining.* _Teacher forcing_: during training the model always sees the
  *correct* trace so far and only predicts the next token — like doing a proof with the
  textbook open. _Greedy decoding_: at test time it sees only its *own* previous tokens —
  the textbook is closed. A model can be excellent at the first and useless at the second;
  telling those apart is half of this document.
]

= Act I — the scratchpad that was never chain-of-thought

*What we built.* Before true CoT, the repo's "CoT" was $K$ _learnable scratchpad tokens_
spliced into a single encoder pass, with an attention mask sketching "rounds":
$["vertices"] + ["edges"] + [c_1 .. c_K] + ["task"]$.

*What we saw.* Never better than the no-scratchpad baseline; both ≈ chance on the hard
dataset (0.565 vs 0.63).

*What was actually wrong.* Four things, all structural — no amount of tuning could help:

+ *No supervision.* Only the task token gets a loss. Nothing ever tells $c_k$ to hold
  "the nodes reachable in $k$ hops"; that reading was a hope, not an objective.
+ *No content.* The $c_i$ are the _same learned constants for every graph_ — empty
  registers, not computed steps.
+ *Bypassable.* The task token attends to the graph directly, so the optimizer can — and
  measurably did — ignore the scratchpad entirely.
+ *Inert at depth 1.* Inside a single forward pass, $c_2$ can only see what $c_1$
  _was_, not what $c_1$ _computed_, unless there is one layer per hop. The "sequential"
  chain silently requires the very depth it was meant to replace.

#figure(
  cetz.canvas(length: 0.62cm, {
    import cetz.draw: *
    let tok(x, label, col) = {
      rect((x, 0), (x + 1.5, 1.1), fill: col.lighten(85%), stroke: col + 0.8pt, radius: 0.12)
      content((x + 0.75, 0.55), text(size: 7.5pt, fill: col)[#label])
    }
    tok(0, [graph], luma(80))
    tok(1.7, [graph], luma(80))
    tok(3.9, [$c_1$], purple)
    tok(5.6, [$c_2$], purple)
    tok(7.3, [$c_3$], purple)
    tok(9.5, [task], blue)

    // bypass arc: graph -> task directly
    bezier((1.6, 1.25), (10.1, 1.25), (5.8, 3.1), stroke: red + 1pt, mark: (end: ">", scale: 0.7))
    content((5.8, 2.75), text(size: 7.5pt, fill: red)[the bypass: task reads the graph directly — scratchpad optional])

    // c1 -> c2 within one pass
    bezier((4.65, -0.15), (6.35, -0.15), (5.5, -1.0), stroke: purple + 0.9pt, mark: (end: ">", scale: 0.7))
    content((5.5, -1.5), text(size: 7.5pt, fill: purple)[$c_2$ sees $c_1$'s _initial_ value — one layer per hop or nothing])
    content((4.7, -2.4), text(size: 7.5pt, fill: luma(100))[and no loss ever touches any $c_i$])
  }),
  caption: [Why the scratchpad could not work: unsupervised, content-free, bypassable,
    and sequential only if depth pays for it anyway.],
) <fig-scratchpad>

*The fix.* Replace it wholesale with a real autoregressive decoder
(`cot_mode: autoregressive`): a supervised BFS trace as the target, tokens generated one
at a time. *Why that works:* generation makes the steps _real_ — each token is computed,
written, and becomes input; supervision makes them _correct_ — the loss grades every
intermediate step, not just the verdict.

#lesson[CoT gains come from supervised intermediate steps. Unsupervised "thinking slots"
in a single pass are just extra width wearing a costume.]

= Act II — the perfect score that was a leak

*What we saw.* The very first real CoT run: decoded accuracy *1.0 by epoch 5*. Champagne?
No: trace exact-match was 0.000 and the out-of-distribution probe scored 0.37. The model
answered perfectly _without ever producing a correct trace_.

*What was actually wrong.* In the caterpillar generator, a connected graph is one spanning
tree ($n-1$ edges) and a disconnected one is two trees ($n-2$ edges). *Edge count was the
label.* No GNN ever noticed — edge count isn't in their features. But a token model's
prompt lists every edge, so the connected prompt is exactly *2 tokens longer*, and the
learned *positional embedding* of the `TRACE` token differs between classes. The model
read the answer off _where in the sequence it was standing_.

#figure(
  cetz.canvas(length: 0.62cm, {
    import cetz.draw: *
    let strip(y, n-cells, label, tracecol, col) = {
      for i in range(n-cells) {
        rect((i * 0.62, y), (i * 0.62 + 0.54, y + 0.8), fill: col.lighten(88%), stroke: col + 0.5pt, radius: 0.06)
      }
      rect((n-cells * 0.62, y), (n-cells * 0.62 + 1.7, y + 0.8), fill: tracecol.lighten(75%), stroke: tracecol + 0.9pt, radius: 0.06)
      content((n-cells * 0.62 + 0.85, y + 0.4), text(size: 7pt, weight: "bold", fill: tracecol)[TRACE])
      content((-2.6, y + 0.4), text(size: 8pt)[#label])
    }
    strip(2.0, 14, [connected], red, luma(120))
    strip(0.6, 12, [disconnected], red, luma(120))
    line((14 * 0.62 + 0.85, 3.1), (14 * 0.62 + 0.85, 2.9), stroke: red + 1pt)
    line((12 * 0.62 + 0.85, 3.1), (12 * 0.62 + 0.85, 2.9), stroke: red + 1pt)
    content((10.2, 3.5), text(size: 7.5pt, fill: red)[`TRACE` sits at a different _absolute position_ per class
      #sym.arrow its positional embedding is the label])
  }),
  caption: [The sequence-length leak: n−1 vs n−2 edges means the prompt length differs by
    2 tokens, and a learned position table can read that directly.],
) <fig-leak>

*The fix.* Pad every graph to $n-1+k$ edges, $k tilde U{1..3}$ drawn _independently of the
label_, using chords that provably cannot change the graph's exact diameter (a leaf
connected to a backbone neighbour of its own attachment; property-tested on 500 random
caterpillars). *Why that works:* it removes the observable — after the fix, edge count
(and hence sequence length) carries zero information about the class, so the only path to
the answer runs through actual reachability.

#lesson[Every model family needs its own leak audit. Changing the representation
(graph #sym.arrow token sequence) creates observables that never existed before —
sequence length first among them. A perfect score that arrives too early is a symptom,
not a success.]

= Act III — format learns, computation doesn't

*What we saw.* Leak fixed, chance again — but a _strange_ chance. The loss froze at ~1.87
(depths 1 and 2, full data, 20k steps). Teacher-forced answer accuracy was 1.0, decoded
accuracy 0.5, trace exact-match 0. A per-position diagnostic (bucket every trace position
by what it is, measure teacher-forced accuracy per bucket) localized it brutally: the
model had learned _everything about how a trace looks_ — separators 0.88, answer-reading
1.00, stopping 1.00 — and _nothing about what goes in it_: retrieving the neighbours of
node 0 from the edge list sat at *5.6%*, with the whole correct prefix handed to it.

*What was actually wrong.* Two stacked problems. First, the known _next-token pitfall_
(Bachmann & Nagarajan 2024): teacher forcing lets the model earn almost all its loss
reduction from easy, statistical conditionals, and the one hard computation gets a weak
gradient with no partial credit. Second, our compact trace made the hard part _maximally_
hard: each BFS level was written as a *sorted set*, so its very first token is the
*minimum of the whole frontier* — to be graded correct you must already have done the
entire level's computation. All-or-nothing gradients on a global set operation.

*The fix.* A verbose trace format (`bfs_expand`) in which every next token is computable
from _local_ context:

#figure(
  cetz.canvas(length: 0.62cm, {
    import cetz.draw: *
    content((0.2, 5.0), anchor: "west", text(size: 8.5pt)[*compact* (`bfs_levels`): #h(0.4em) #raw("0 | 3 7 | 5 12 14 | ...")])
    content((0.9, 3.9), anchor: "west", text(size: 7.5pt, fill: red)[first token of a level = *min of the whole frontier* —
      global computation, graded all-or-nothing])

    content((0.2, 1.7), anchor: "west", text(size: 8.5pt)[*verbose* (`bfs_expand`): #h(0.4em) #raw("0 | EXP 0 3 7 | EXP 3 12 EXP 7 5 14 | ...")])
    content((7.2, 0.1), text(size: 7.5pt, fill: green)[parent after `EXP` = _copy_ of previous level (induction);
      children = _lookup_ keyed by the parent token right before them])
  }),
  caption: [The same BFS, two gradings. The verbose format costs ~2#sym.times the tokens
    and buys a learnable curriculum: each next token is one small, locally-checkable step.],
) <fig-formats>

*Why that works:* gradient descent learns what the loss makes _locally_ checkable.
"Copy the previous level in order" is a textbook two-layer circuit (an induction head);
"emit the neighbours of the token right before you" is one content lookup. The compact
format demanded their composition _plus_ a sort _plus_ a set-minimum before the first
token of a level could ever be right.

#lesson[A CoT trace is a curriculum, not a log. Design every next token to be computable
from local context, or the gradient never finds the algorithm — it will learn the
formatting and stop.]

= Act IV — the optimizer was the saboteur

*What we saw.* The verbose format _still_ plateaued. Even pure copying sat at 0.119 after
20k steps at depth 2 — and copying is the easiest relational skill a transformer has. At
this point the suspects were exotic (distractor tokens? embedding tying? attention
numerics?). The scientific move was to shrink the task to one atom: a probe
(`bfs_l1`) whose entire completion is _the sorted neighbours of node 0_ — one lookup,
nothing else — and then to turn knobs one at a time.

*What was actually wrong.* The first, cheapest knob was the answer.
With the config's `weight_decay: 0.01` and `dropout: 0.1` the probe plateaued at
trace-EM ≈ 0.15. Setting *both to zero* — no other change — took it to *0.95 by epoch 10*
and 0.999 by 20:

#figure(
  cetz.canvas(length: 0.62cm, {
    import cetz.draw: *
    let X = 12.0
    let Y = 6.0
    let px(e) = e / 160 * X
    let py(a) = a * Y

    // axes + recessive grid
    for a in (0.0, 0.25, 0.5, 0.75, 1.0) {
      line((0, py(a)), (X, py(a)), stroke: luma(225) + 0.4pt)
      content((-0.55, py(a)), text(size: 7pt, fill: luma(120))[#a])
    }
    for e in (0, 40, 80, 120, 160) {
      content((px(e), -0.5), text(size: 7pt, fill: luma(120))[#e])
    }
    line((0, 0), (0, Y), stroke: luma(120) + 0.7pt)
    line((0, 0), (X, 0), stroke: luma(120) + 0.7pt)
    content((X / 2, -1.15), text(size: 7.5pt, fill: luma(100))[epoch])
    content((-1.5, Y / 2), angle: 90deg, text(size: 7.5pt, fill: luma(100))[trace exact-match])

    // with regularization (red): plateau at ~0.15
    let reg = ((10, 0.052), (20, 0.057), (30, 0.101), (40, 0.111), (50, 0.121), (60, 0.124),
               (70, 0.141), (80, 0.134), (90, 0.143), (100, 0.144), (120, 0.153), (140, 0.139), (160, 0.143))
    for i in range(reg.len() - 1) {
      line((px(reg.at(i).at(0)), py(reg.at(i).at(1))),
           (px(reg.at(i + 1).at(0)), py(reg.at(i + 1).at(1))), stroke: red + 1.3pt)
    }
    content((px(160) - 0.2, py(0.143) + 0.55), anchor: "east", text(size: 7.5pt, fill: red)[wd 0.01 + dropout 0.1])

    // zero reg (blue): 0.95 by epoch 10
    let noreg = ((0, 0.0), (10, 0.954), (20, 0.998), (30, 0.999), (60, 0.999), (100, 0.996), (120, 0.998))
    for i in range(noreg.len() - 1) {
      line((px(noreg.at(i).at(0)), py(noreg.at(i).at(1))),
           (px(noreg.at(i + 1).at(0)), py(noreg.at(i + 1).at(1))), stroke: blue + 1.3pt)
    }
    content((px(62), py(0.999) - 0.55), text(size: 7.5pt, fill: blue)[weight decay 0, dropout 0])
  }),
  caption: [The atomic-lookup probe (`bfs_l1`, depth 2, same data, same everything).
    Zero regularization is the only difference between the two curves.],
) <fig-probe>

*Why that happens.* A retrieval circuit is a small set of _precise_ weights: a key that
must match a query exactly, an attention pattern that must be sharp. Weight decay (as
Adam's L2 pull) shrinks all weights toward zero every step — for a big model a nuisance,
for a 430k-parameter model with tied embeddings on a 42-token vocabulary it is a constant
tax on exactly the weights the circuit needs, with no redundancy to pay it from. Dropout,
meanwhile, injects noise into the very attention pattern that is trying to become sharp.
Both settings are habits imported from _statistical_ fitting, where they fight
overfitting; here they fought _learning itself_. Notably this is the mirror image of the
famous grokking result (Power et al. 2022), where weight decay *enables* delayed
generalization — at our scale and task it *prevented* the circuit from ever forming.

#lesson[Regularizers tuned for statistical learning can be _the_ blocker for algorithmic
circuit formation at small scale. They are one `-o weight_decay=0 -o dropout=0` away from
being ruled out — sweep them to zero before redesigning anything.]

= Act V — circuits versus memorization, an economics problem

*What we saw.* Zero regularization, full trace, depth 2, 8k graphs: the loss finally
_collapsed_ — and decoding still failed. The diagnostic showed a partial success with a
clean edge: copying 0.989 and first-level lookup 0.995 (on unseen graphs!), but deeper
levels — "neighbours of this parent _minus everything already visited_" — stuck at ~0.6.
Under exact-match, 0.6 per token compounds to ~0 per trace.

The intuitive move, adding depth, backfired instructively: at depth 4 the training loss
went to _literally 0.0000_ while decoded trace-EM *declined*. The extra capacity was spent
memorizing 6,400 training sequences, not building the missing circuit. (Its by-diameter
accuracy even inverted — the model faithfully read its own derailed traces as
"disconnected", scoring 0.25 exactly where connected graphs live.)

*What was actually wrong — and the fix.* Nothing was "wrong": it was a price comparison.
A memorizing solution costs weights proportional to the training set; a circuit costs a
fixed amount. Because node ids are freshly permuted in every graph, data is
memorization-hostile: quadruple it and memorization's price quadruples while the circuit's
stays flat. At *32k graphs, depth 2* the balance tipped, with a textbook grokking curve —
flat for 55 epochs, then a phase transition:

#figure(
  cetz.canvas(length: 0.62cm, {
    import cetz.draw: *
    let X = 12.0
    let Y = 6.0
    let px(e) = e / 100 * X
    let py(a) = a * Y

    for a in (0.0, 0.25, 0.5, 0.75, 1.0) {
      line((0, py(a)), (X, py(a)), stroke: luma(225) + 0.4pt)
      content((-0.55, py(a)), text(size: 7pt, fill: luma(120))[#a])
    }
    for e in (0, 25, 50, 75, 100) {
      content((px(e), -0.5), text(size: 7pt, fill: luma(120))[#e])
    }
    line((0, 0), (0, Y), stroke: luma(120) + 0.7pt)
    line((0, 0), (X, 0), stroke: luma(120) + 0.7pt)
    content((X / 2, -1.15), text(size: 7.5pt, fill: luma(100))[epoch])
    content((-1.5, Y / 2), angle: 90deg, text(size: 7.5pt, fill: luma(100))[held-out performance])

    let dec = ((10, 0.551), (20, 0.537), (30, 0.560), (40, 0.528), (50, 0.532),
               (60, 0.605), (70, 0.809), (80, 0.884), (90, 0.915), (100, 0.964))
    let em  = ((10, 0.028), (20, 0.032), (30, 0.033), (40, 0.034), (50, 0.033),
               (60, 0.057), (70, 0.410), (80, 0.551), (90, 0.640), (100, 0.796))
    for i in range(dec.len() - 1) {
      line((px(dec.at(i).at(0)), py(dec.at(i).at(1))),
           (px(dec.at(i + 1).at(0)), py(dec.at(i + 1).at(1))), stroke: blue + 1.3pt)
    }
    for i in range(em.len() - 1) {
      line((px(em.at(i).at(0)), py(em.at(i).at(1))),
           (px(em.at(i + 1).at(0)), py(em.at(i + 1).at(1))), stroke: purple + 1.3pt)
    }
    content((px(97), py(0.964) + 0.45), anchor: "east", text(size: 7.5pt, fill: blue)[decoded answer accuracy])
    content((px(99), py(0.70) - 0.1), anchor: "east", text(size: 7.5pt, fill: purple)[trace exact-match])
    line((px(57), 0.1), (px(57), Y - 0.1), stroke: (paint: luma(150), thickness: 0.6pt, dash: "dashed"))
    content((px(57) + 0.2, Y - 0.45), anchor: "west", text(size: 7.5pt, fill: luma(100))[the phase transition])
  }),
  caption: [The grokking run: depth 2, 32k permuted graphs, zero regularization
    (`diameter_controlled`, run `20260705_005620`). Fifty flat epochs, then the circuit
    outcompetes memorization.],
) <fig-grok>

#lesson[Circuits vs memorization is an economics problem. Permuted-id data makes
memorization expensive; capacity makes it cheap. When generalization stalls, add data
before depth — extra layers may finance exactly the wrong solution.]

= The finish line

The same recipe then solved the _adversarial_ dataset (`connectedness_hard_diam` — the
one where the classes differ by a single edge and every single-pass architecture we ever
tried sits at chance): decoded *0.9925*, trace exact-match *0.90*. The two summary
pictures:

#figure(
  cetz.canvas(length: 0.62cm, {
    import cetz.draw: *
    let X = 12.0
    let Y = 5.2
    let pxd(d) = (d - 2) / 16 * X
    let py(a) = a * Y

    for a in (0.0, 0.25, 0.5, 0.75, 1.0) {
      line((0, py(a)), (X, py(a)), stroke: luma(225) + 0.4pt)
      content((-0.55, py(a)), text(size: 7pt, fill: luma(120))[#a])
    }
    for d in (2, 6, 10, 14, 18) {
      content((pxd(d), -0.5), text(size: 7pt, fill: luma(120))[#d])
    }
    line((0, 0), (0, Y), stroke: luma(120) + 0.7pt)
    line((0, 0), (X, 0), stroke: luma(120) + 0.7pt)
    content((X / 2, -1.15), text(size: 7.5pt, fill: luma(100))[graph diameter])
    content((-1.5, Y / 2), angle: 90deg, text(size: 7.5pt, fill: luma(100))[decoded accuracy])

    let tr = ((2, 0.983), (3, 0.944), (4, 0.933), (5, 0.956), (6, 0.955), (7, 0.964), (8, 0.971),
              (9, 0.987), (10, 0.981), (11, 0.920), (12, 0.969), (13, 0.938), (14, 0.958),
              (15, 0.929), (16, 0.982), (17, 0.995), (18, 0.951))
    let ao = ((2, 0.604), (3, 0.575), (4, 0.552), (5, 0.490), (6, 0.496), (7, 0.486), (8, 0.471),
              (9, 0.476), (10, 0.475), (11, 0.517), (12, 0.536), (13, 0.583), (14, 0.508),
              (15, 0.561), (16, 0.524), (17, 0.591), (18, 0.571))
    for i in range(tr.len() - 1) {
      line((pxd(tr.at(i).at(0)), py(tr.at(i).at(1))),
           (pxd(tr.at(i + 1).at(0)), py(tr.at(i + 1).at(1))), stroke: blue + 1.3pt)
    }
    for i in range(ao.len() - 1) {
      line((pxd(ao.at(i).at(0)), py(ao.at(i).at(1))),
           (pxd(ao.at(i + 1).at(0)), py(ao.at(i + 1).at(1))), stroke: red + 1.3pt)
    }
    content((pxd(10), py(0.99) + 0.42), text(size: 7.5pt, fill: blue)[with BFS trace — *flat in diameter*])
    content((pxd(10), py(0.44) - 0.45), text(size: 7.5pt, fill: red)[answer-only — chance (the wiggle is class mix per bucket)])
  }),
  caption: [The thesis figure (`diameter_controlled`, 32k, depth 2): same model, same
    data, same optimizer — the trace is the only difference. Diameter, the quantity that
    defeats fixed depth, stops mattering when it is converted into tokens.],
) <fig-diam>

#table(
  columns: (auto, auto, auto, auto),
  align: (left, right, right, left),
  stroke: 0.4pt + luma(180),
  table.header([*sub-circuit* (teacher-forced, held-out)], [*before*], [*after*], [*what it is*]),
  [parent after `EXP`], [0.119], [0.998], [induction copy of the previous level],
  [level-1 children],  [0.056], [1.000], [edge-list lookup],
  [level-2+ children], [#sym.tilde 0.6], [0.985–0.996], [lookup minus the visited set],
  [SEP / ANS / EOS],   [0.88–1.0], [0.998–1.0], [trace format & stopping],
)

The mechanism receipts matter as much as the headline: the accuracy is carried by
verified sub-circuits, not by an unexplained blob. And the honest caveat: all of this is
*in-distribution* execution. The grokked caterpillar model fails on dense ER graphs
(0.32); the blob-trained model does better (0.54) — a hint that _distribution diversity_,
not the mechanism, is the remaining barrier. That is the open question this journey
hands to the next one.

= The checklist

For a fixed-depth transformer to execute a graph algorithm:

+ *Supervise the steps*, not just the answer — else: chance. (Act I)
+ *Audit the serialization for leaks*; permute everything permutable — else: a fake 1.0. (Act II)
+ *Make every next token locally computable* — else: format without content. (Act III)
+ *Zero regularization at small scale* — else: circuits never form. (Act IV)
+ *Enough distinct data that memorization costs more than the circuit* — else: perfect
  training loss, chance decoding. (Act V)
+ *Depth #sym.gt.eq the per-step circuit* (two layers, for copy + lookup) — and no more
  than the data can discipline.

#v(0.5em)
#text(size: 8.5pt, fill: luma(100))[
  Every number is reproducible: configs `configs/cot_ar*.yaml`, run JSONs in `results/`
  (git provenance embedded), per-position diagnostics via `diag_cot_levels.py`, day-by-day
  narrative in `docs/CHANGELOG.md` (2026-07-03 #sym.arrow 07-05). Companion prose report:
  `reports/executing-graph-algorithms.md`.
]
