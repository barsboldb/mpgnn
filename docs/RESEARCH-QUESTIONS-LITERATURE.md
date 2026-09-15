# Literature coverage for the Part B candidate questions

Companion to `docs/RESEARCH-QUESTIONS.md` (2026-08-26): for each of the ten candidate
questions, how well the field already covers it, the notable papers to cite or defend
against, and where the defensible novelty sits. Same sweep and same caveat as
`docs/QUESTIONS-LITERATURE.md` (which covers the older `docs/QUESTIONS.md` framing and
remains valid as a paper index): links are verbatim from the sweep, arXiv IDs for
well-known papers were filled from memory — verify before they enter a bibliography.

Verdict scale: **crowded** (well covered; novelty must be narrow) / **partial** (pieces
exist, the combination is open) / **open** (little or nothing directly on it).

---

## Q1 — representability vs trainability for connectivity

**Verdict: partial — and the framing now has a literature to stand in, which is good news
for the chapter and bad news for claiming the question is untouched.** The
expressivity/learnability gap is named and being worked on; what is missing is a
*negative learnability result on leak-audited data with matched controls*.

- Sanford, Hsu, Telgarsky, **Understanding Transformer Reasoning Capabilities via Graph
  Algorithms**, NeurIPS 2024 — supplies the representability half (LD regime, Thm 1/2/3/6).
  https://arxiv.org/abs/2405.18512
- Ye, Fu, Jia, Sharan, **Transformers Provably Learn Algorithmic Solutions for Graph
  Connectivity, But Only with the Right Data** — the field's current answer to *this exact
  question*, and therefore the paper Q1 must be positioned against, not merely cited.
  Their answer: GD does find the algorithm, conditional on the data distribution.
  https://arxiv.org/abs/2510.19753
- Back de Luca & Fountoulakis, **Simulation of Graph Algorithms with Looped
  Transformers**, ICML 2024 — explicit statement that backprop cannot find their
  hand-built weights. The cleanest citable admission of the gap.
  https://arxiv.org/abs/2402.01107
- **Mind The Gap: Quantifying Mechanistic Gaps in Algorithmic Reasoning via Neural
  Compilation**, 2025 — compiles the correct algorithm into weights, then measures how far
  trained weights land from it. A methodology Q1 could borrow wholesale: it converts "GD
  didn't find it" from an absence into a measurement. https://arxiv.org/abs/2505.18623
- Yehudai, Sanford, Bechler-Speicher, Fischer, Gilad-Bachrach, Globerson, **Depth-Width
  Tradeoffs for Transformers on Graph Tasks**, NeurIPS 2025 — the assigned paper; the
  width axis (A8 follow-on: never tested past 256). https://arxiv.org/abs/2503.01805
- Abbe, Bengio, Lotfi, Sandon, Saremi, **How Far Can Transformers Reason? The Globality
  Barrier and Inductive Scratchpad**, NeurIPS 2024 — *the* theoretical account of why a
  representable target can still be unlearnable: globality degree bounds efficient weak
  learning, and it is an NC^0-flavoured notion, deliberately distinct from the TC^0/TC^1
  expressivity results. This is the theory Q1's negative result should be stated in.
  https://arxiv.org/abs/2406.06467

**Novelty to claim:** not "GD fails" (Abbe explains when it must) and not "data matters"
(Ye owns that), but *a pre-registered depth × width × encoding grid on leak-audited data,
with an overfit control separating capacity from generalization*. The overfit-N control
(1.00 / 0.76 / 0.545) is the part of the existing evidence the literature does not have.

---

## Q2 — leak taxonomy and audit protocol

**Verdict: partial, and the best risk-adjusted question in the file.** Shortcut learning
and benchmark leakage are established concerns with no graph-reasoning-specific audit
protocol. The four-leak taxonomy (generator / encoding / serialization / residual
construction) does not exist anywhere in this form.

- Geirhos et al., **Shortcut Learning in Deep Neural Networks**, Nat. Mach. Intell. 2020
  — the umbrella framing. https://arxiv.org/abs/2004.07780
- **Rethinking Graph Classification in Presence of Isomorphism**, Doklady Math. 2024 —
  isomorphic duplicates across splits change model rankings.
  https://link.springer.com/article/10.1134/S1064562424602385
- Dwivedi et al., **Benchmarking Graph Neural Networks**, JMLR 2023 — benchmarks too weak
  to separate models. https://jmlr.org/papers/volume24/22-0567/22-0567.pdf
- Tönshoff, Ritzert, Rosenbluth, Grohe, **Where Did the Gap Go? Reassessing the Long-Range
  Graph Benchmark**, LoG 2023 — tuned baselines erase a headline architectural gap. The
  closest methodological precedent for "run your own model on the reference dataset".
  https://arxiv.org/abs/2309.00367
- **Implicit degree bias in the link prediction task**, 2024 — degree-separable negatives,
  the same failure one task over. https://arxiv.org/abs/2405.14985
- Palowitch et al., **GraphWorld**, KDD 2022 — controlled synthetic generators instead of
  inherited dataset bias; the constructive half of the protocol.
  https://arxiv.org/abs/2203.00112
- Hu et al., **Open Graph Benchmark**, NeurIPS 2020 — split design as a first-class
  concern. https://arxiv.org/abs/2005.00687
- Target of the audit: Fatemi, Halcrow, Perozzi, **Talk Like a Graph** (GraphQA)
  https://arxiv.org/abs/2310.04560 and Wang et al., **NLGraph**
  https://arxiv.org/abs/2305.10037 — the ER, 5–20-node benchmarks the min-degree rule
  should be measured against (A7 gap 1).

**On the committee's "where is the new science" risk:** the answer the literature
supports is *falsifiable audit rungs with measured baselines*, not a checklist. The
degree-0 rule at 98.2% on a published generator family is the kind of number that makes a
methods contribution land.

---

## Q3 — what is Ye et al.'s "right data" actually doing?

**Verdict: open, high stakes.** A direct engagement with one recent paper. No third-party
replication or scope analysis of their capacity lever appeared in the sweep — which means
both that the contribution is real and that there is no cover if the reproduction is off.

- Ye, Fu, Jia, Sharan, **Transformers Provably Learn Algorithmic Solutions for Graph
  Connectivity, But Only with the Right Data** — the subject.
  https://arxiv.org/abs/2510.19753
- Abbe et al., **Globality Barrier and Inductive Scratchpad**, NeurIPS 2024 — gives the
  vocabulary for "which distributional property is doing the work"; their agnostic vs
  educated vs inductive scratchpad distinction is the same axis one level up.
  https://arxiv.org/abs/2406.06467
- Bevilacqua, Zhou, Ribeiro, **Neural Algorithmic Reasoning with Causal Regularisation**
  (Hint-ReLIC), ICML 2023 — the other "make the distribution suppress the shortcut" lever,
  from the GNN side; the natural comparison. https://arxiv.org/abs/2302.10258
- Mahdavi et al., **Towards Better OOD Generalization of Neural Algorithmic Reasoning**,
  LoG 2022. https://arxiv.org/abs/2211.00692
- On the global-vs-local confound (their 3^L wall assumes local mixing): Alon & Yahav,
  **On the Bottleneck of GNNs and its Practical Implications**, ICLR 2021 — over-squashing
  is the same geometry seen from the other side. https://arxiv.org/abs/2006.05205

**The clique probe (0.0 disconnected / 1.0 connected, both regimes) has no counterpart in
the sweep.** That is the publishable object; state it as an OOD probe result, not as "their
data leaked".

---

## Q4 — encodings as precomputation oracles

**Verdict: crowded on the parts, open on the accounting.** Everyone knows LapPE and
distance biases inject structure; nobody prices it *in layers*.

- Dwivedi & Bresson, **A Generalization of Transformer Networks to Graphs** (LapPE)
  https://arxiv.org/abs/2012.09699
- Dwivedi et al., **Graph Neural Networks with Learnable Structural and Positional
  Representations** (RWSE), ICLR 2022. https://arxiv.org/abs/2110.07875
- Ying et al., **Do Transformers Really Perform Bad for Graph Representation?**
  (Graphormer) — spatial/shortest-path attention bias, i.e. all-pairs distances handed to
  the model. The direct precedent for the `spd_bias` rung.
  https://arxiv.org/abs/2106.05234
- Rampášek et al., **GraphGPS**, NeurIPS 2022 — the PE/SE zoo as a design space.
  https://arxiv.org/abs/2205.12454
- Airale et al., **Simple Path Structural Encoding for Graph Transformers**, ICML 2025 —
  richer encodings still being proposed; the moving target.
  https://arxiv.org/abs/2502.09365
- Lim et al., **Sign and Basis Invariant Networks**, ICLR 2023 — why eigenvector encodings
  are fragile (the "not canonical across components" observation in Q9).
  https://arxiv.org/abs/2202.13013
- Müller, Galkin, Morris, Rampášek, **Attending to Graph Transformers** (survey) — states
  plainly that no expressivity hierarchy over encodings exists. That absence is Q4's
  opening. https://arxiv.org/abs/2302.04181
- Theory for the exchange rate: Sanford et al. 2024 (depth↔MPC rounds) and Merrill &
  Sabharwal, **A Little Depth Goes a Long Way** (2025) https://arxiv.org/abs/2503.03961.

**Novelty to claim:** the *exchange rate* — layers saved per encoding at fixed accuracy —
and the fixed-vs-variable-n result (1.00 vs 0.59 on the same tokenization) as its cleanest
instance. Frame as "encodings are preprocessing oracles, and here is the price list".

---

## Q5 — does process supervision buy what depth and width cannot?

**Verdict: crowded on the headline, partial on the payload.** "CoT helps" is 2021-old;
"accuracy flat in diameter at fixed depth, with a matched answer-only control at chance"
is the contribution and is much harder to find.

- Nye et al., **Show Your Work: Scratchpads**, 2021. https://arxiv.org/abs/2112.00114
- Wei et al., **Chain-of-Thought Prompting Elicits Reasoning**, NeurIPS 2022.
  https://arxiv.org/abs/2201.11903
- Merrill & Sabharwal, **The Expressive Power of Transformers with Chain of Thought**,
  ICLR 2024 — the theory that permits the flatness. https://arxiv.org/abs/2310.07923
- Li, Liu, Zhou, Ma, **Chain of Thought Empowers Transformers to Solve Inherently Serial
  Problems**, ICLR 2024. https://arxiv.org/abs/2402.12875
- Lightman et al., **Let's Verify Step by Step**, ICLR 2024 — process vs outcome
  supervision at scale; this is the frame that makes the miniature experiment legible to a
  committee. https://arxiv.org/abs/2305.20050
- Uesato et al., **Solving Math Word Problems with Process- and Outcome-Based Feedback**,
  2022 — the original controlled comparison. https://arxiv.org/abs/2211.14275
- Veličković et al., **CLRS**, ICML 2022 — per-step "hints" are the same intervention in
  the GNN world. https://arxiv.org/abs/2205.15659
- Counterweight, mandatory: Rodionov & Prokhorenkova, **Neural Algorithmic Reasoning
  Without Intermediate Supervision**, NeurIPS 2023 — hints can *hurt*; no-hint training is
  competitive. https://arxiv.org/abs/2306.13411

**Framing advice the literature supports:** lead with the *control*, not the accuracy. The
0.964 vs 0.510 pair and the flatness across diameters 2–18 are the result; 0.9972 alone
reads as "CoT works, again".

---

## Q6 — which operations must be tokenized? (the silent-op law)

**Verdict: partial — the sharpest question in the file, and the one with the most
dangerous neighbours.** Two papers are close enough that the claim must be positioned
against them explicitly, as Part B already anticipates.

- Abbe et al., **Globality Barrier and Inductive Scratchpad**, NeurIPS 2024 — the nearest
  neighbour by far: globality degree says which steps are efficiently learnable, and their
  *educated vs agnostic scratchpad* distinction is nearly the silent-op law stated in
  their vocabulary. Read closely; the honest positioning is that the law is the empirical,
  per-token-diagnosable version of their barrier, with a constructive fix (`bfs_check`) and
  an out-of-sample prediction. https://arxiv.org/abs/2406.06467
- Bachmann & Nagarajan, **The Pitfalls of Next-Token Prediction**, ICML 2024 — teacher
  forcing + lookahead = Clever Hans; explains *why* teacher-forced ≈1.0 coexists with
  decoded 0.52 (the Q9 stall signature). https://arxiv.org/abs/2403.06963
- Pfau, Merrill, Bowman, **Let's Think Dot by Dot**, COLM 2024 — filler tokens work only
  under dense supervision, and the usable class is characterized by quantifier depth of a
  first-order formula. A multiset-hash or a set-minus is exactly a quantified op: this is
  the closest thing to a *theoretical* statement of the law.
  https://arxiv.org/abs/2404.15758
- Zhou et al., **What Algorithms can Transformers Learn? A Study in Length
  Generalization** (RASP-L), ICLR 2024 — "learnable iff short RASP-L program exists" is a
  rival predictive criterion; the silent-op law should be compared with it head-on.
  https://arxiv.org/abs/2310.16028
- Wang, Min, Deng et al., **Towards Understanding CoT Prompting: What Matters**, ACL 2023
  — invalid rationales keep 80–90% of the gain; the adversarial reading of "which tokens
  matter". https://arxiv.org/abs/2212.10001
- Lanham et al., **Measuring Faithfulness in Chain-of-Thought Reasoning**, 2023 — the
  ablation template (truncate / paraphrase / corrupt a step).
  https://arxiv.org/abs/2307.13702

**Novelty to claim:** predicted-then-fixed is the whole case. Instance 3 (diagnostic →
prediction → `bfs_check` → 0.5342 → 0.9972, the silent op becoming the most reliable one
at 1.000 over 342,755 positions) is a stronger empirical object than anything in the
neighbour papers. Run `wl_gather` and the out-of-sample confirmation closes it.

---

## Q7 — do standard regularizers prevent circuit formation at small scale?

**Verdict: crowded, with a live sign disagreement — which is exactly the contribution.**
Do not claim the mechanism; claim the *regime contrast*.

- Power et al., **Grokking**, 2022 — weight decay as the trigger.
  https://arxiv.org/abs/2201.02177
- Nanda et al., **Progress Measures for Grokking via Mechanistic Interpretability**, ICLR
  2023. https://arxiv.org/abs/2301.05217
- Varma et al., **Explaining Grokking through Circuit Efficiency**, 2023 — memorizing vs
  generalizing circuits under a norm penalty. https://arxiv.org/abs/2309.02390
- Kobayashi, Akram, von Oswald, **Weight Decay Induces Low-Rank Attention Layers**,
  NeurIPS 2024 — same-scale mechanism, and their own experiment is a 2-layer transformer on
  associative recall. The best-matched support available.
  https://arxiv.org/abs/2410.23819
- Lv et al., **Language Models "Grok" to Copy**, NAACL 2025 — the opposite sign at scale:
  weight decay and attention dropout *accelerate* induction-head formation (15B → 10B
  tokens). https://aclanthology.org/2025.naacl-short.61/
- Loshchilov & Hutter, **Decoupled Weight Decay Regularization**, ICLR 2019 — required for
  the AdamW-vs-Adam experiment to mean anything. https://arxiv.org/abs/1711.05101

**The publishable sentence:** *the same regularizer that accelerates copy-circuit
formation at billion-token scale prevents retrieval-circuit formation at 10^4-sample
scale* — with the dropout/weight-decay bisect as the evidence. Part B is right that it is a
chapter, not a thesis.

---

## Q8 — parallel depth vs sequential tokens (cost accounting)

**Verdict: partial; the theory is settled and the empirical exchange rate is missing.**

- Merrill & Sabharwal, ICLR 2024 (CoT buys serial depth)
  https://arxiv.org/abs/2310.07923 · Li et al., ICLR 2024 https://arxiv.org/abs/2402.12875
- Sanford, Hsu, Telgarsky, **Transformers, Parallel Computation, and Logarithmic Depth**,
  ICML 2024 — the parallelism side of the ledger. https://arxiv.org/abs/2402.09268
- Merrill & Sabharwal, **A Little Depth Goes a Long Way**, 2025.
  https://arxiv.org/abs/2503.03961
- Yehudai et al., NeurIPS 2025 — depth-width tradeoffs, the same ledger in the
  expressivity regime. https://arxiv.org/abs/2503.01805
- Snell et al., **Scaling LLM Test-Time Compute Optimally**, 2024 — the same
  depth-vs-tokens economics at the deployment end; the citation that makes the accounting
  feel current rather than parochial. https://arxiv.org/abs/2408.03314
- Engineering side, if the decode cost is reported: Pope et al., **Efficiently Scaling
  Transformer Inference**, MLSys 2023. https://arxiv.org/abs/2211.05102

**Novelty to claim:** FLOPs / wall-clock / *samples-to-learn* on matched ladders. The third
axis is the one nobody reports, and it is the one this repo can measure.

---

## Q9 — does the recipe generalize beyond BFS (1-WL colour refinement)?

**Verdict: open for the exact question; crowded around it.** See
`docs/QUESTIONS-LITERATURE.md` §Q1 for the full list; the essentials:

- Xu, Hu, Leskovec, Jegelka, **How Powerful are GNNs?**, ICLR 2019 — the ceiling.
  https://arxiv.org/abs/1810.00826
- Morris et al., **Weisfeiler and Leman go Neural**, AAAI 2019
  https://arxiv.org/abs/1810.02244 · **WL goes ML: The Story so far** (JMLR 2023 survey)
  https://www.jmlr.org/papers/volume24/22-0240/22-0240.pdf
- Müller, Morris et al., **Aligning Transformers with Weisfeiler–Leman**, ICML 2024 — the
  expressivity-side neighbour. https://arxiv.org/abs/2406.03148
- Dupty, Dong, Lee, **Graph Representation Learning with Individualization and
  Refinement** — if the chapter ever reaches past 1-WL.
  https://arxiv.org/abs/2203.09141
- Bachmann & Nagarajan 2024 — the explanation for the *current stall* (teacher-forced ≈1.0,
  decoded 0.52, trace-EM 0.003 is the textbook teacher-forcing/autoregression split).
  https://arxiv.org/abs/2403.06963

**Status note:** the sweep found nothing on *trace-supervised execution of colour
refinement*. The `iso_wl` leak audit (degree-matched pairs, WL-separable negatives via
degree-preserving double-edge swaps, 419.7 vs 420.6 mean tokens) is itself a contribution
under Q2 even if the run never lands.

---

## Q10 — is neural algorithm execution ever distribution-general?

**Verdict: crowded and unsolved field-wide — which makes the *localization* the
contribution, not the failure.**

- Bevilacqua, Zhou, Ribeiro, **Hint-ReLIC**, ICML 2023 — up to 3× OOD improvement on CLRS
  via trace-invariance augmentation. https://arxiv.org/abs/2302.10258
- Ibarz et al., **A Generalist Neural Algorithmic Learner**, LoG 2022 — multi-task /
  mixed-generator training, i.e. the proposed remedy, already tried at scale.
  https://arxiv.org/abs/2209.11142
- Mahdavi et al., **Towards Better OOD Generalization of Neural Algorithmic Reasoning**,
  LoG 2022. https://arxiv.org/abs/2211.00692
- Zhang et al., **Can LLM Graph Reasoning Generalize beyond Pattern Memorization?**, EMNLP
  Findings 2024 — the same collapse at LLM scale. https://arxiv.org/abs/2406.15992
- Abbe et al. 2024 — *educated scratchpads break globality but do not necessarily
  generalize OOD; inductive scratchpads do*. This is a direct, testable prescription for
  the trace format, and the most actionable citation in this section.
  https://arxiv.org/abs/2406.06467
- On the derailed-trace failure: Turpin et al., **LMs Don't Always Say What They Think**,
  NeurIPS 2023 https://arxiv.org/abs/2305.04388 · Chen et al. (Anthropic), **Reasoning
  Models Don't Always Say What They Think**, 2025 https://arxiv.org/abs/2505.05410 ·
  Lanham et al. 2023 https://arxiv.org/abs/2307.13702

**Novelty to claim:** *execution transfers, read-off does not* (ER trace-EM 0.188 with
answer accuracy 0.389, below the 0.755 always-NO marginal) localizes OOD failure to the
answer-readout circuit. The sweep found no comparable decomposition. The below-chance
caterpillar result (0.29, faithfully reading a derailed trace) is the miniature, per-token
diagnosable version of the faithfulness literature above.

---

## Part A — where the literature bears on the fact-check

- **A4/A5 (the leaks)** → Geirhos 2020; the isomorphism-duplicate and degree-bias papers
  in Q2. The four-leak set is the asset; keep the measured falsifications attached.
- **A6/A7 (Sanford)** → the paper itself (https://arxiv.org/abs/2405.18512). The
  "transformers beat GNNs" correction is supported field-wide by Tönshoff et al.'s
  gap-reassessment (https://arxiv.org/abs/2309.00367): tuned baselines close claimed gaps.
- **A8 (`ln 2` plateau)** → two distinct mechanisms, and the literature separates them:
  Sanford Thm 6 is a *width* bound; the learned-node-id gradient cancellation is a
  trainability failure closer to Abbe's globality barrier. Part A is right to warn against
  merging them.
- **A9 (Ye)** → see Q3.

---

## Summary table

| Q | Coverage | Where the novelty is | Nearest threat |
|---|---|---|---|
| Q1 representability/trainability | partial | pre-registered grid + overfit control on leak-audited data | Ye 2026 (answers it positively); Abbe 2024 (explains it away) |
| Q2 leak taxonomy | partial | the four-rung audit with measured falsifications | Tönshoff 2023 (same move, one benchmark) |
| Q3 Ye's "right data" | open | clique probe + scope of the capacity lever | reproduction fidelity, not a paper |
| Q4 encodings as oracles | crowded parts / open accounting | layers-per-encoding exchange rate | Graphormer, GraphGPS own the parts |
| Q5 process supervision | crowded headline / partial payload | matched control + flatness in diameter | Rodionov 2023 (hints unnecessary) |
| Q6 silent-op law | partial | predicted-then-fixed, per-token diagnosis | Abbe 2024 globality; Zhou 2024 RASP-L |
| Q7 regularizers | crowded, sign-disputed | the small-scale regime contrast | Lv 2025 (opposite sign at scale) |
| Q8 depth vs tokens | partial | samples-to-learn as the third cost axis | theory already settled |
| Q9 beyond BFS (1-WL) | open (traces) / crowded (expressivity) | trace-supervised WL execution | Müller 2024; the run must land |
| Q10 distribution-general | crowded, unsolved | execution-vs-readout localization | Hint-ReLIC, Generalist NAR |

### Reading against Part C
- **Option 1 (Q2→Q1→Q5→Q6)** — strongest literature position, *provided* Q1 is framed
  against Ye 2026 and Q6 against Abbe 2024 from the first draft. Both are unavoidable.
- **Option 2 (Q2→Q4→Q3)** — safest; its literature is thin in the right way (no audit
  protocol exists) and crowded in the harmless way (encodings are well documented).
- **Option 3 (Q6→Q9→Q7)** — sharpest, and the most exposed: Abbe 2024 sits on Q6, Lv 2025
  contests Q7's sign, and Q9 is unrun.

### Five papers to read before writing any of it
1. Abbe et al. 2024, **Globality Barrier and Inductive Scratchpad** — touches Q1, Q3, Q6, Q10.
2. Ye et al. 2026 — owns Q1/Q3; already in `papers/`, notes exist.
3. Bachmann & Nagarajan 2024 — explains the Q9 stall and threatens Q6.
4. Rodionov & Prokhorenkova 2023 — the counterweight to Q5's whole premise.
5. Pfau, Merrill, Bowman 2024 — the closest theory to the silent-op law.
