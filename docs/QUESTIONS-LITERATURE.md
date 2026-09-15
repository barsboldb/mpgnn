# Literature coverage for the thesis questions

Companion to `docs/QUESTIONS.md`: for each question asked there, how well the research
field already covers it, and the notable papers to cite (or to defend against).
Compiled 2026-09-15 from a web literature sweep. Links found in the sweep are verbatim;
arXiv IDs for well-known papers were filled in from memory — verify each before it goes
into a bibliography. Verdict scale:
**crowded** (well covered, novelty must be narrow) / **partial** (pieces exist, the
combination is open) / **open** (little or nothing directly on it).

---

## The spine — "what does it take for a fixed-depth transformer to execute a graph algorithm?"

**Verdict: partial.** The *representational* side is settled theory; the *trainability*
side (what supervision makes a trainable circuit, measured with leak audits and matched
controls) is where the thesis lives. Defend novelty as "trainability, not expressivity".

- Sanford, Hsu, Telgarsky, **Understanding Transformer Reasoning Capabilities via Graph
  Algorithms**, NeurIPS 2024 — the direct anchor: log-depth necessary and sufficient for
  connectivity; 9 graph tasks split by depth/width/extra-token regime.
  https://proceedings.neurips.cc/paper_files/paper/2024/hash/8f395480c04ac6dfb2c2326a639df88e-Abstract-Conference.html
- Back de Luca & Fountoulakis, **Simulation of Graph Algorithms with Looped
  Transformers**, ICML 2024 — BFS/DFS/Dijkstra/Kosaraju simulated exactly, width
  independent of graph size; explicitly flags that backprop cannot find these weights.
  That gap *is* this thesis's question. https://arxiv.org/abs/2402.01107
- Nye et al., **Show Your Work: Scratchpads for Intermediate Computation**, 2021 —
  the origin of "emit intermediate steps" as a training intervention.
  https://arxiv.org/abs/2112.00114
- Merrill & Sabharwal, **The Expressive Power of Transformers with Chain of Thought**,
  ICLR 2024 — CoT tokens buy serial depth beyond TC^0. https://arxiv.org/abs/2310.07923
- Li, Liu, Zhou, Ma, **Chain of Thought Empowers Transformers to Solve Inherently Serial
  Problems**, ICLR 2024 — same conclusion, constructive. https://arxiv.org/abs/2402.12875
- Veličković et al., **The CLRS Algorithmic Reasoning Benchmark**, ICML 2022 — the
  "hints" (per-step trace targets) idea in the GNN world; the closest prior art to trace
  supervision. https://arxiv.org/abs/2205.15659
- Lightman et al., **Let's Verify Step by Step**, ICLR 2024 — process vs outcome
  supervision at LLM scale; the framing the thesis reproduces at miniature scale with
  clean controls. https://arxiv.org/abs/2305.20050
- Xu, Hu, Leskovec, Jegelka, **How Powerful are Graph Neural Networks?**, ICLR 2019 —
  the 1-WL ceiling used as the theory anchor for the next chapter.
  https://arxiv.org/abs/1810.00826

**Gap to claim:** nobody runs the *ingredient ablation* (local trace targets, zero weight
decay, leak audit, data-over-depth) with matched controls at chance and per-sub-circuit
mechanism verification. Cite Back de Luca's trainability caveat as the open door.

---

## Q1 — recipe generality: fixed-depth transformer executing color refinement (1-WL)

**Verdict: open** for the exact question; **crowded** around it. WL-and-transformers
exists, but as *expressivity alignment*, not as *can a transformer be trained to execute
the rounds with accuracy flat in round count*.

- Müller, Morris et al., **Aligning Transformers with Weisfeiler–Leman**, ICML 2024 —
  graph transformers with PEs placed in the k-WL hierarchy; the expressivity-side
  neighbour of Q1. https://arxiv.org/abs/2406.03148
- Morris et al., **Weisfeiler and Leman go Machine Learning: The Story so far**, JMLR
  2023 — the survey to cite for everything WL; establishes what is already known.
  https://www.jmlr.org/papers/volume24/22-0240/22-0240.pdf
- Morris et al., **Weisfeiler and Leman Go Neural**, AAAI 2019 — k-GNN, the other half of
  the 1-WL ceiling result. https://arxiv.org/abs/1810.02244
- Pfau, Merrill, Bowman, **Let's Think Dot by Dot**, COLM 2024 — critical for the
  "globality pitfall" design: filler tokens help only with *dense supervision*, and the
  usable class is characterized by quantifier depth of a first-order formula. WL's
  "hash the multiset of neighbour colors" is exactly such a quantified op.
  https://arxiv.org/abs/2404.15758
- Sanford et al. 2024 (above) — supplies the "flat in rounds" experimental grammar.

**Gap to claim:** trace-supervised *execution* of color refinement, with round count as
the difficulty knob and answer-only as the failing control, appears unoccupied. The
local-decomposition trick (sorted neighbor-color list before the new color) is the novel
mechanism claim; Pfau et al. is the theory that predicts it is needed.

---

## Q2 — past the 1-WL ceiling with individualization–refinement traces

**Verdict: crowded on the architecture side, open on the trace-supervision side.**
Beating 1-WL is a whole subfield — but everyone does it by *changing the architecture*,
nobody by *training on IR search traces*. That is the wedge; it is also why reviewers
will demand comparison against these baselines.

- Dupty, Dong, Lee, **Graph Representation Learning with Individualization and
  Refinement** (GNN-IR) — exactly the nauty-style individualize/refine paradigm inside a
  GNN. The closest prior art; must be cited and compared.
  https://arxiv.org/abs/2203.09141
- Pellizzoni et al., **On the Expressivity and Sample Complexity of Node-Individualized
  GNNs**, NeurIPS 2024 — individualization raises expressivity but costs sample
  complexity; a direct predictor of how Q2 will fail if it fails.
  https://mlanthology.org/neurips/2024/pellizzoni2024neurips-expressivity/
- **Logical Expressiveness of GNNs with Hierarchical Node Individualization** (HE-GNN),
  NeurIPS 2025 — isomorphism-invariant IR-style model.
  https://arxiv.org/abs/2506.13911
- Bouritsas, Frasca, Zafeiriou, Bronstein, **Improving GNN Expressivity via Subgraph
  Isomorphism Counting** (GSN) — the standard beyond-1-WL baseline.
  https://arxiv.org/abs/2006.09252
- Bachmann & Nagarajan, **The Pitfalls of Next-Token Prediction**, ICML 2024 — the
  named risk: teacher forcing + lookahead structure = Clever Hans cheat. Backtracking
  traces are precisely that structure. https://arxiv.org/abs/2403.06963
- Cai–Fürer–Immerman gadgets and strongly regular graphs: standard WL-equivalent test
  families, covered in the Morris et al. JMLR survey above.

---

## Q3 — the shortcut ladder: leak-audited isomorphism datasets

**Verdict: partial, and sympathetic.** Dataset leakage in graph benchmarks is a live,
publishable concern — but the specific audit (negatives separable by degree histogram)
is the kind of finding the field keeps rediscovering per-dataset, so a *ladder* is a real
contribution.

- **Rethinking Graph Classification in Presence of Isomorphism**, Doklady Math. 2024 —
  common graph-classification datasets contain isomorphic duplicates; model rankings
  change once they are removed. https://link.springer.com/article/10.1134/S1064562424602385
- Dwivedi et al., **Benchmarking Graph Neural Networks**, JMLR 2023 — the standard
  citation for "benchmarks too weak to separate models".
  https://jmlr.org/papers/volume24/22-0567/22-0567.pdf
- Hu et al., **Open Graph Benchmark**, NeurIPS 2020 — split design as a first-class
  concern. https://arxiv.org/abs/2005.00687
- Palowitch et al., **GraphWorld**, KDD 2022 — synthetic generators to control graph
  statistics instead of inheriting a dataset's biases; methodological sibling of the
  `diameter_controlled` / `wl_round_controlled` generators.
  https://arxiv.org/abs/2203.00112
- **Implicit degree bias in the link prediction task**, 2024 — degree-based separability
  of negatives, the same failure mode one task over.
  https://arxiv.org/abs/2405.14985
- Geirhos et al., **Shortcut Learning in Deep Neural Networks**, Nat. Mach. Intell. 2020
  — the umbrella framing for the whole ladder. https://arxiv.org/abs/2004.07780

---

## Q4 — walk traces: the algorithm, or just grounded tokens?

**Verdict: partial — and the best-positioned question in the file.** Both endpoints are
well covered (filler tokens; walks-as-graph-representation), but they have never been put
on one axis. The ladder filler → walk → BFS is, as far as the sweep shows, unoccupied.

- Pfau, Merrill, Bowman, **Let's Think Dot by Dot**, 2024 — the filler rung, with theory
  for when content-free tokens suffice. https://arxiv.org/abs/2404.15758
- Wang, Min, Deng et al., **Towards Understanding Chain-of-Thought Prompting: An
  Empirical Study of What Matters**, ACL 2023 — *invalid* reasoning retains 80–90% of CoT
  gains; the strongest prior evidence for the "any grounded scaffold works" branch.
  https://arxiv.org/abs/2212.10001
- Tönshoff, Ritzert, Wolf, Grohe, **Walking Out of the Weisfeiler–Leman Hierarchy**
  (CRaWl) — walks as the record a network reads; expressivity bounded by window size.
  https://arxiv.org/abs/2102.08786
- Kim et al., **Revisiting Random Walks for Learning on Graphs**, ICLR 2025 — random-walk
  neural networks formalized (walker / recorder / reader / aggregator); shows walks +
  a transformer reader suffice for universality. The paper Q4 inverts by making the
  transformer itself the walker. https://arxiv.org/abs/2407.01214
- Perozzi et al., **DeepWalk** (KDD 2014) and Grover & Leskovec, **node2vec** (KDD 2016)
  — walks as external preprocessing, the line Q4 argues against.
  https://arxiv.org/abs/1403.6652 · https://arxiv.org/abs/1607.00653
- Lanham et al., **Measuring Faithfulness in Chain-of-Thought Reasoning**, 2023 —
  filler/paraphrase/truncation ablations; the methodological template for the three-rung
  control design. https://arxiv.org/abs/2307.13702
- Goyal et al., **Think Before You Speak: Training LMs with Pause Tokens**, ICLR 2024 —
  the pause-token variant of the filler rung. https://arxiv.org/abs/2310.02226

---

## Settled framing — graph algorithms as a model organism

**Verdict: the framing is defensible and has precedent**; cite the interpretability
lineage so "toy task" reads as method, not as limitation.

- Elhage et al., **Toy Models of Superposition**, Anthropic 2022 — the model-organism
  method in its purest form. https://transformer-circuits.pub/2022/toy_model/index.html
- Nanda et al., **Progress Measures for Grokking via Mechanistic Interpretability**,
  ICLR 2023 — miniature algorithmic task reverse-engineered end to end.
  https://arxiv.org/abs/2301.05217
- Power et al., **Grokking: Generalization Beyond Overfitting on Small Algorithmic
  Datasets**, 2022 — small algorithmic datasets as the standard laboratory; weight decay
  as the trigger (direct support for the weight-decay ingredient).
  https://arxiv.org/abs/2201.02177
- Varma et al., **Explaining Grokking through Circuit Efficiency**, 2023 — memorizing vs
  generalizing circuits under weight decay. https://arxiv.org/abs/2309.02390
- Turpin, Michael, Perez, Bowman, **Language Models Don't Always Say What They Think**,
  NeurIPS 2023 — CoT unfaithfulness at scale; the phenomenon the derailed-trace-read-
  faithfully result reproduces diagnosably. https://arxiv.org/abs/2305.04388
- Chen et al. (Anthropic), **Reasoning Models Don't Always Say What They Think**, 2025 —
  the reasoning-model follow-up. https://arxiv.org/abs/2505.05410
- On "call the classical algorithm as a tool": Schick et al., **Toolformer**, NeurIPS
  2023. https://arxiv.org/abs/2302.04761

---

## Future-work note — learned steering of combinatorial heuristics (compilers)

**Verdict: crowded and deployed** — which is the point: it is the practical anchor, not a
claimed contribution. Keep the dampers on record.

- Trofin et al., **MLGO: A Machine Learning Guided Compiler Optimizations Framework**,
  2021 — the deployed inlining/regalloc-eviction policy in LLVM.
  https://arxiv.org/abs/2101.04808
- **The Next 700 ML-Enabled Compiler Optimizations**, 2023 — the MLGO team's survey of
  where learned policies fit in a compiler. https://arxiv.org/abs/2311.10800
- VenkataKeerthy et al., **RL4ReAl: Reinforcement Learning for Register Allocation**,
  CC 2023 — RL over the regalloc decision points. https://arxiv.org/abs/2204.02013
- Chaitin, **Register Allocation and Spilling via Graph Coloring**, 1982 — the classical
  reduction. (ACM DL)
- Schuetz, Brubaker, Katzgraber, **Graph Coloring with Physics-Inspired GNNs**, Nat.
  Mach. Intell. 2022 — and the rebuttal, Angelini & Ricci-Tersenghi, **Modern graph
  neural networks do worse than classical greedy algorithms**, Nat. Mach. Intell. 2023.
  The matched-baseline cautionary tale. https://arxiv.org/abs/2202.01606 ·
  https://arxiv.org/abs/2206.13211

---

## Standing open problems

### OOD / distribution-general algorithm execution
**Verdict: crowded, unsolved** — the field's own hardest open problem, so "future work"
is the honest disposition.
- Bevilacqua, Zhou, Ribeiro, **Neural Algorithmic Reasoning with Causal Regularisation**
  (Hint-ReLIC), ICML 2023 — up to 3× OOD improvement on CLRS via trace-invariance
  augmentation; the most direct methodological neighbour of trace supervision.
  https://arxiv.org/abs/2302.10258
- Mahdavi et al., **Towards Better OOD Generalization of Neural Algorithmic Reasoning**,
  LoG 2022. https://arxiv.org/abs/2211.00692
- Rodionov & Prokhorenkova, **Neural Algorithmic Reasoning Without Intermediate
  Supervision**, NeurIPS 2023 — the counterweight: hints can *hurt*; no-hint training is
  competitive. Cite as the adversarial reading of the trace-supervision claim.
  https://arxiv.org/abs/2306.13411
- Zhang et al., **Can LLM Graph Reasoning Generalize beyond Pattern Memorization?**,
  EMNLP Findings 2024 — the LLM-scale version of the same collapse.
  https://arxiv.org/abs/2406.15992
- Ibarz et al., **A Generalist Neural Algorithmic Learner**, LoG 2022 — multi-task /
  mixed-generator training, which is the proposed remedy.
  https://arxiv.org/abs/2209.11142

### Length / size OOD (`cot_pos: learned` vs `none`)
**Verdict: crowded.** Do not claim novelty; adopt the known-best practice and cite.
- Kazemnejad et al., **The Impact of Positional Encoding on Length Generalization**,
  NeurIPS 2023 — NoPE beats explicit PEs on downstream length generalization. Directly
  predicts `cot_pos: none` wins. https://arxiv.org/abs/2305.19466
- Ruoss et al., **Randomized Positional Encodings Boost Length Generalization**, ACL
  2023. https://arxiv.org/abs/2305.16843
- McLeish et al., **Transformers Can Do Arithmetic with the Right Embeddings** (Abacus),
  NeurIPS 2024 — 6× length generalization. https://arxiv.org/abs/2405.17399
- Fan et al., **Looped Transformers for Length Generalization**, ICLR 2025 — recurrence
  as the alternative lever. https://arxiv.org/abs/2409.15647
- Zhou et al., **What Algorithms can Transformers Learn? A Study in Length
  Generalization** (RASP-L), ICLR 2024 — the predictive account of *which* tasks length-
  generalize at all. https://arxiv.org/abs/2310.16028

### AdamW (decoupled) vs Adam (L2) at the same λ
**Verdict: settled in the optimizer literature** — an internal control, not a claim.
- Loshchilov & Hutter, **Decoupled Weight Decay Regularization**, ICLR 2019.
  https://arxiv.org/abs/1711.05101
- Power et al. 2022 / Varma et al. 2023 (above) for the weight-decay→grokking mechanism.

### KV cache for `generate()`
**Verdict: engineering, not research.** Cite only if the cost chapter needs it.
- Pope et al., **Efficiently Scaling Transformer Inference**, MLSys 2023.
  https://arxiv.org/abs/2211.05102

---

## Earlier candidates

1. **Depth vs sequential tokens (cost accounting)** — *partial, and the strongest
   analytical chapter.* The theory says CoT buys serial depth (Merrill & Sabharwal 2024;
   Li et al. 2024) and log-depth suffices for connectivity (Sanford et al. 2024), but the
   *empirical* FLOPs / wall-clock / samples-to-grok exchange rate over matched ladders is
   not tabulated anywhere the sweep found. Add: Merrill & Sabharwal, **A Little Depth
   Goes a Long Way** (2025) https://arxiv.org/abs/2503.03961 and Sanford et al.,
   **Transformers, Parallel Computation, and Logarithmic Depth**, ICML 2024
   https://arxiv.org/abs/2402.09268.
2. **Cheap permutation-invariant embeddings** — *crowded; supporting chapter at most.*
   Dwivedi & Bresson LapPE https://arxiv.org/abs/2012.09699 · Dwivedi et al. **Graph
   Neural Networks with Learnable Structural and Positional Representations**, ICLR 2022
   (RWSE) https://arxiv.org/abs/2110.07875 · Rampášek et al. **GraphGPS**, NeurIPS 2022
   https://arxiv.org/abs/2205.12454 · Lim et al. **Sign and Basis Invariant Networks**,
   ICLR 2023 https://arxiv.org/abs/2202.13013 · Müller et al. **Attending to Graph
   Transformers** (survey) https://arxiv.org/abs/2302.04181. The completeness ceiling
   ("a cheap complete invariant would solve GI") is exactly the point these make.
3. **A better architecture, as fast** — *crowded and inconclusive field-wide*; the null
   result (architectures band together in-distribution, all collapse OOD) is consistent
   with Tönshoff et al., **Where Did the Gap Go? Reassessing the Long-Range Graph
   Benchmark**, LoG 2023 https://arxiv.org/abs/2309.00367 — tuned baselines close claimed
   architectural gaps. Fold into discussion as planned.

---

## Summary table

| Question | Coverage | Where the novelty is |
|---|---|---|
| Spine (BFS/connectivity) | partial | trainability + ingredient ablation, not expressivity |
| Q1 1-WL execution | open | trace-supervised execution flat in WL rounds; local decomposition |
| Q2 past 1-WL | crowded (architectures) / open (traces) | IR *traces*, not IR architecture |
| Q3 shortcut ladder | partial | the graded, leak-audited ladder as an artifact |
| Q4 walk traces | partial | filler→walk→BFS on one axis; model as native walker |
| Model-organism framing | precedented | rhetorical, not a claim |
| OOD execution | crowded, unsolved | future work; cite Hint-ReLIC and the no-hint counterweight |
| Length OOD | crowded | adopt NoPE, cite; no claim |
| AdamW vs Adam | settled | internal control |
| Depth vs tokens cost | partial | the empirical exchange rate table |
| Invariant embeddings | crowded | supporting chapter |
| Better architecture | crowded | discussion only |

### Three papers that most threaten the thesis (read closely, address head-on)
1. Rodionov & Prokhorenkova 2023 — intermediate supervision may be *unnecessary*.
2. Wang et al. ACL 2023 — *invalid* CoT keeps most of the gain (Q4's alternative branch).
3. Angelini & Ricci-Tersenghi 2023 — matched classical baselines beat the neural method.
