# Three candidate thesis questions

Drafted 2026-09-17 from `docs/RESEARCH-QUESTIONS.md` Part C, re-framed against the
literature sweep in `docs/RESEARCH-QUESTIONS-LITERATURE.md`. Each is a whole diploma, not
a chapter: a thesis statement, the chapter spine, what is still unrun, and the risk.

**Current lean: A.** Not settled — B and C are both viable and are kept here in full
because the deciding experiments (`wl_gather` for B, the GraphQA min-degree measurement
for C) are cheap enough that the choice can be revisited.

---

## A — The price of a learnable algorithm *(leading candidate)*

> *Fixed-depth transformers can **represent** graph connectivity but do not **learn** it
> from leak-audited data at any depth or width trainable at this scale. Depth does not buy
> it; data volume buys only memorization; supervising the algorithm's intermediate steps
> buys it at depth 2, with accuracy flat in diameter. The question is not "can gradient
> descent find it" but **"what does it cost to make it findable"**.*

Spine: Q2 → Q1 → Q5 → Q6.

| # | chapter | source | status |
|---|---|---|---|
| 1 | Leak audit and the difficulty ladder | Q2 | four leak classes, all measured |
| 2 | The grid that fails — depth × width × encoding, plus the overfit-N control | Q1 | ~20 runs exist, must be re-run pre-registered |
| 3 | Trace supervision makes it learnable at depth 2 | Q5 | 0.964 vs 0.510 answer-only, flat across diam 2–18 |
| 4 | Which steps must be tokenized (the silent-op law) | Q6 | upside chapter, not load-bearing |

**The novelty, stated precisely.** Not "GD fails" (Abbe 2024 explains when it must) and
not "data matters" (Ye 2026 owns that): it is the **overfit-N control**
(`overfit10` 1.00 → `overfit100` 0.76 → full data 0.545) separating capacity from
generalization, run alongside a pre-registered grid on data with a measured leak audit.
Neither neighbouring paper has that separator.

**The reframe that matters.** Do not write it as "does GD find it?" — that is the question
Ye et al. 2026's title answers positively, and the thesis would be arguing scope against a
recent paper. Written as *what does it cost*, Ye becomes a cited neighbour (data is one
more lever on the same axis) rather than a competitor.

**Remaining work**
1. Close the local bridge-detection shortcut in `connectedness_hard` (GIN-degree 0.975 at
   depth < diameter). **Gating** — every negative result rests on this dataset.
2. Re-run the grid as one pre-registered sweep: depth {1,2,4,8} × width {64,128,256} ×
   encoding {edge_list, adj_rows}, seeds, trained to saturation.
3. Report **samples-to-learn curves**, not final numbers, so "train longer" is not a free
   rebuttal.
4. The answer-only control on `connectedness_hard` (the other two datasets have it).
5. Optional hardening: neural compilation (arXiv 2505.18623) — compile the correct circuit
   into weights, measure the distance from what GD found. Turns "GD didn't find it" from
   an absence into a measurement.
6. Optional upside: `wl_gather`, which promotes chapter 4 to the headline.

**Risk: medium-low.** Fails gracefully — a null `wl_gather` costs a chapter, not the
thesis. Must be positioned against Ye 2026 from page one. Data already in hand: ~80%.

---

## B — What must be written down (the silent-op law)

> *Given a fixed task, the **decomposition** it is trained on decides whether it is
> learnable. An operation whose output is not emitted as a token is learned only if it is
> computable from the emitted context within the model's per-token budget; operations
> requiring an unbounded set-valued intermediate (multiset hash, set-minus) fail, and the
> branching factor is the knob that determines when.*

Spine: Q5 (setup) → Q6 → Q9 → Q7 (optional).

| # | chapter | source | status |
|---|---|---|---|
| 1 | Trace supervision makes algorithms learnable | Q5 | done |
| 2 | The silent-op law — three clean instances | Q6 | instance 3 *predicted then fixed*: 0.5342 → 0.9972 |
| 3 | Out-of-sample: the same prediction on a second algorithm | Q9 | `wl_gather` — **decides the thesis** |
| 4 | Regularizers vs circuit formation | Q7 | optional; sign contested at scale |

**Why it is the strongest claim available.** It generalizes past graphs into a statement
about which *reasoning styles* are trainable, and instance 3 is predicted-then-fixed rather
than post-hoc: the diagnostic flagged visited-set subtraction as silent (0.27/0.24 at
levels 2–3), `bfs_check` emitted each membership test as a token, and the formerly silent
op became the model's most reliable circuit (1.000 over 342,755 positions).

**The distinction to lead with.** Abbe et al. 2024's globality barrier is about the
*target distribution* — which tasks are efficiently weakly learnable. This law is about
the *trace format* — which decomposition of a fixed task is trainable. Same task, two
formats, different outcomes (`bfs_levels` 0.056 vs `bfs_expand`; 0.5342 vs 0.9972 on
identical graphs) is evidence their framework cannot produce, because the task's globality
never changed.

**Remaining work**
1. Run `wl_gather`. **Load-bearing.**
2. Unstall `iso_wl` first — it currently sits at teacher-forced ≈1.0, decoded 0.52,
   trace-EM 0.003 (the textbook Bachmann & Nagarajan teacher-forcing split).
3. Decide instance 2's status: the 8k-sparse stall was fixed by **4× data**, not by
   tokenizing the op — off-mechanism inside a tokenization law. Bring it under the
   branching-factor formulation or demote it to a footnote.
4. Position against Abbe 2024 and Zhou et al.'s RASP-L in the introduction, not the
   discussion.

**Risk: high.** Two unknowns stacked — `wl_gather` runs inside a task that is not yet
working. Largest payoff if it lands. Data already in hand: ~55%.

---

## C — What graph-reasoning benchmarks actually measure

> *A large share of graph-reasoning evaluation measures encoding- and generator-induced
> shortcuts rather than reasoning. Here is a taxonomy of four leak classes, a falsifiable
> audit rung for each, and what survives when the audit is applied to three reference
> setups.*

Spine: Q2 → Q4 → Q3.

| # | chapter | source | status |
|---|---|---|---|
| 1 | Four leak classes: generator / encoding / serialization / residual construction | Q2 | all four measured; ER near threshold ⇒ "min degree ≥ 1" solves 98.2% |
| 2 | Encodings as precomputation oracles — the exchange rate in layers | Q4 | `adj_rows` 1.00 at fixed n vs 0.59 at variable n; mostly re-analysis |
| 3 | Auditing the reference setups: Yehudai / Sanford–GraphQA / Ye | Q3 | clique probe: **0.000** disconnected, **1.000** connected, in *both* regimes |

**The asset.** The clique probe has no counterpart anywhere in the sweep — nothing
third-party replicates or scopes Ye's capacity lever. Stated as an OOD probe result (not
as "their data leaked"), it is the sharpest single object in the project.

**Remaining work**
1. Measure the min-degree rule on GraphQA connectivity — half a day, against the field's
   reference benchmark, and the paper itself concedes its graphs "do not resemble the
   large-diameter worst-case instance" its negative results describe.
2. Re-test Ye's capacity lever on `connectedness_hard_diam` with `local: true`, so the
   3^L wall can bind. Logged as future work in June, never run, and it decides Q3.
3. Quantify the encoding exchange rate (layers saved per encoding at fixed accuracy).

**Risk: low technically, medium rhetorically** — a committee may ask "where is the new
science". The answer the literature supports: falsifiable audit rungs with measured
baselines, not a checklist. Data already in hand: ~70%.

---

## Comparison

| | A — price of learnability | B — silent-op law | C — benchmark integrity |
|---|---|---|---|
| strongest claim | trace supervision beats depth at matched cost | trace *format* decides trainability | benchmarks measure shortcuts |
| data in hand | ~80% | ~55% | ~70% |
| load-bearing unrun experiment | none (shortcut fix + re-run) | `wl_gather` | none (two cheap measurements) |
| nearest threat | Ye 2026 | Abbe 2024, RASP-L | "where is the new science" |
| failure mode | loses a chapter | loses the thesis | reads as methods-only |

**Shared prerequisite.** All three need the `connectedness_hard` bridge shortcut closed
(Q2, item 2 in Part D). That work is not wasted under any choice — do it first.
