# Shortcut Learning in Graph Reasoning Tasks: Literature Review for "Analyzing the Effect of Network Width on Graph Transformers"

Shortcut learning on graph-reasoning benchmarks is well documented: there is direct theory and evidence that transformers trained on random-graph connectivity learn a degree-based heuristic rather than the algorithm. However, no paper has yet audited GraphQA connectivity with simple-rule or shallow-classifier baselines the way you did. No paper measures how the width needed to *learn* (as opposed to *represent*) a graph task scales once shortcuts are removed, and there is no accepted audit protocol in graph ML. Your 97.3% endpoint-edge rule and 99.5% 2-hop GBDT results therefore fill a real gap, and your "swap" generator sits closest in design to Saparov et al.'s balanced distribution and Ye et al.'s within-capacity data.

## TL;DR
- **Shortcuts are confirmed for connectivity-type tasks, but almost always through OOD failure rather than baseline audits.** Ye et al. (arXiv:2510.19753) prove that a degree-based "heuristic channel" is learned when training graphs exceed depth capacity. NLGraph (arXiv:2305.10037, Table 4, text-davinci-003) built chain/clique splits to invert the degree cue. Mean accuracy fell from 78.40 on the general split to 54.83 on CHAIN (−23.57) and 71.00 on CLIQUE (−7.40). Only CoT on CHAIN dropped by more than 40 points (85.33→40.83). Saparov et al. (arXiv:2412.04703, Fig. 2) show that naïve random-DAG training fails beyond small lookahead. Neither Sanford et al. (arXiv:2405.18512) nor Fatemi et al. (arXiv:2310.04560) report a majority or local-rule baseline for connectivity.
- **Whether a model learns the algorithm or the shortcut depends mainly on the data distribution, not on scale.** Restricting diameter to within capacity (Ye et al.) or balancing lookahead (Saparov et al.) produces algorithmic solutions. Saparov et al.'s abstract states: "This difficulty is not resolved even as the number of parameters is increased, suggesting that increasing model scale will not lead to robust search abilities." Roy & Saparov (arXiv:2509.22343, which carries an author-flagged error) report "increasingly better generalization to infer connectivity over grid graphs" with scale, but not on graphs with many disconnected components. No paper isolates width.
- **Representational theory is ahead of learning theory.** Sanford et al. prove that log depth is necessary and sufficient for connectivity at sub-linear width. Yehudai et al. (NeurIPS 2025, arXiv:2503.01805) show that linear width lets constant depth suffice, with Ω(n) lower bounds for 2-cycle detection. Abbe et al.'s globality degree (arXiv:2406.06467) is the only learnability-side locality measure, and nobody has connected it to width. That is the thesis's opening.

## 1. Evidence table

| Paper | Benchmark / task | Shortcut found or prevented | Evidence (where) | Relevance |
|---|---|---|---|---|
| Ye, Fu, Jia, Sharan, "Transformers Provably Learn Algorithmic Solutions for Graph Connectivity, But Only with the Right Data" (arXiv:2510.19753; v1 titled "When Do Transformers Learn Heuristics for Graph Connectivity?")\[1\] | Connectivity on Erdős–Rényi (ER) graphs, adjacency input | **Found and prevented.** Degree-based heuristic ("both nodes have high degree"). Prevented by restricting training to diameter ≤ 3^L.\[2\] | Thm 4.3/4.5: an L-layer Disentangled Transformer has capacity diameter exactly 3^L. Thm 4.7: weights decompose into an algorithmic I_n channel and a heuristic J_n channel. Thm C.5/C.9: beyond-capacity samples push toward the heuristic. Fig. 1: 2-layer models are perfect in-distribution but fail on two disjoint chains/cliques. Fig. 6: the result transfers to standard transformers.\[3\] | **High** |
| Sanford et al., "Understanding Transformer Reasoning Capabilities via Graph Algorithms" (arXiv:2405.18512, NeurIPS 2024) | GraphQA connectivity, edge-list tokens | **Not audited.** Reports accuracy only and gives no majority or local baseline. | Fig. 3a (v1): 60M transformer 92.9 (1K) / 98.0 (100K); GCN 83.8/83.8; MPNN 94.0/94.4; GIN 93.8/94.0; fine-tuned 11B T5 98.4 (1K). App. E.1: ER only, 5–20 nodes, test set averages 12.37 nodes and 39.79 edges. §4.1 claims connectivity cannot be solved "by only investigating the neighbors of the source and sink node."\[4\] Your 97.3% rule contradicts this empirically on this distribution. | **High** (your reproduction target) |
| Fatemi, Halcrow, Perozzi, "Talk like a Graph" (arXiv:2310.04560, ICLR 2024) | GraphQA (edge existence, cycle check, reachability, …) | **Partially audited.** Majority baselines are given for edge existence (53.96% "no") and cycle check (81.96% "yes").\[5\] None is given for reachability. | §3.1.1: LLMs fall below the majority baseline on these tasks. §3.4/Fig. 3: larger PaLM 2 helps, except that edge existence never beats the majority baseline.\[5\] Camera-ready App. Table 9: reachability ZERO-SHOT 84.9, COT 45.2. Generators: ER, BA, SFN, SBM, star, path, complete, all 5–20 nodes. ER mean degree is 5.70 (Table 5).\[5\]\[6\] | **High** |
| Wang et al., NLGraph (arXiv:2305.10037, NeurIPS 2023) | Connectivity (LLM prompting) | **Found.** Degree / mention-count correlation. | §4.4, Table 4 (text-davinci-003, 120 examples per split): the "chain" and "clique" cases, constructed as the "exact opposite of the spurious correlation," average 54.83 and 71.00 against 78.40 on the general split. Only CoT on CHAIN falls by more than 40 points (85.33→40.83), and 0-CoT on CHAIN rises to 92.50. The text's claim of ">40% across various settings" therefore overstates its own table. The authors hypothesise that LLMs are "just counting node mentions." | **High** (the closest precedent to your swap generator) |
| Saparov et al., "Transformers Struggle to Learn to Search" (arXiv:2412.04703, ICLR 2025) | DAG next-step search, lookahead L | **Prevented** via a "balanced" distribution that is uniform over lookahead. | §3, Fig. 2 (41-vertex graphs, 883M examples): naïve-distribution model ≈0.26–0.52 at L ≥ 5; balanced model ≈0.96–1.00 for all L; the model trained on L ≤ 12 generalises to 13–14 only. §4: mechanistic "exponential path-merging". Abstract: increasing parameters does not resolve difficulty on larger graphs.\[7\] | **High** |
| Roy & Saparov (arXiv:2509.22343) | Directed connectivity / transitivity | **Partly found.** Many disconnected components remain hard; grid-like graphs are easy.\[8\] | Abstract: scale improves generalisation on grid graphs. The arXiv comment says "This paper contains some assumption which is not correct".\[8\] A third-party index lists it as ICLR 2026, which I could not verify.\[9\] | Medium (cite with caution) |
| Bachmann & Nagarajan, "The Pitfalls of Next-Token Prediction" (arXiv:2403.06963, ICML 2024) | Path-star graph path-finding | **Found.** "Clever Hans cheat" under teacher forcing. | Path-star construction; with teacher forcing the model copies adjacency and leaves the first "indecipherable token" unlearned. Teacherless multi-token training helps in some settings.\[10\]\[11\] Frydenlund (arXiv:2503.10542) reports a spurious correlation in the original setup.\[12\] | Medium (output-side shortcut, not input-side) |
| Abbe et al., "How Far Can Transformers Reason? The Globality Barrier and Inductive Scratchpad" (arXiv:2406.06467, NeurIPS 2024) | Cycle task (one vs two cycles), syllogism chains | **Prevented in theory.** High-globality targets cannot be weakly learned efficiently. | Globality degree relates to correlation with NC⁰.\[13\] Secondary summary (not verified in the paper text): glob(cycle task(n)) ≥ n. The inductive scratchpad breaks the barrier and improves OOD generalisation.\[14\] | **High** (theory for Q4) |
| Abbe et al., "Generalization on the Unseen, Logic Reasoning and Degree Curriculum" (arXiv:2301.13105, ICML 2023) | Boolean targets, unseen-domain holdout | **Mechanism.** Min-degree implicit bias picks the simplest interpolator. | Theorems for random-feature models and diagonal linear networks; experiments include transformers. Degree-Curriculum.\[15\] | Medium-High (explains why low-order local cues win) |
| Zhang et al., "On the Paradox of Learning to Reason from Data" (arXiv:2205.11502, IJCAI 2023) | SimpleLogic (forward-chaining, i.e. reachability in rule graphs) | **Found.** The statistical feature #rule predicts the label. | §3: a BERT parameterisation solves the task 100%, yet trained BERT fails off-distribution. §4: removing one statistical feature helps, but jointly removing all of them is infeasible (Table 4).\[16\] | **High** (the "representable but not learned" template) |
| Mahdavi et al. (arXiv:2211.00692, TMLR 2023) | CLRS-30 | **Found (benchmark flaws).** Input distribution shift: node degree grows with graph size; positional scalars leak. | §3: challenges include non-representative generation and uninformative ID validation.\[17\] Random scalar indexing reportedly improves mean OOD from 60.47 to 66.18 (secondary summary, not verified).\[18\] | Medium |
| Minder et al., SALSA-CLRS (arXiv:2309.12253, LoG 2023 ext. abstract) | Sparse CLRS graph algorithms | **Prevention-by-OOD.** Tests at 5×–100× the training size with multiple generators. | Abstract and §: test sets at 5×, 10×, 50× and 100× size.\[19\] | Medium |
| Zhang, Wang, Feng et al., NLGIFT (arXiv:2406.15992, EMNLP Findings 2024) | Instruction-tuned graph LLMs (NLGraph tasks) | **Found.** Pattern memorisation. | Abstract: generalisation is acceptable on semantic/numeric patterns but fails on reasoning and real-world patterns.\[20\] §5: mixing tasks drops the majority task by up to 43%.\[20\] | Medium |
| Hu et al., "Rethinking and Benchmarking LLMs for Graph Reasoning" (arXiv:2509.24260) | GraphArena, GraphQA, GraphWiz/GraphInstruct, NLGraph | **Found (template overfitting).** | §3.1.2: GraphWiz "overfits to the nine graph reasoning tasks" and misapplies templates. §4.1: train/test task overlap. Simple-RTC is near-perfect on existing benchmarks.\[21\]\[22\] | Medium-Low (LLM side) |
| Bechler-Speicher et al., "GNNs Use Graphs When They Shouldn't" (arXiv:2309.04332, ICML 2024) | Synthetic and real graph classification | **Inverse shortcut.** GNNs overfit uninformative structure. | Table 1: GNN vs GNN_∅ (empty-graph ablation). Regular graphs reduce overfitting.\[23\] | Low-Medium (contributes the graph-less ablation to the audit toolkit) |
| Yehudai, Sanford et al., "Depth-Width Tradeoffs for Transformers on Graph Tasks" (arXiv:2503.01805, NeurIPS 2025) | Node-adjacency tokenisation: connectivity, cycles, subgraph counting, Eulerian | **No shortcut analysis.** Representational width thresholds only. | Thm 4.1: 2 layers with O(n) width solve 1-vs-2 cycles. Thm 4.2: 2-cycle detection needs mpHL = Ω(n). Thm 4.3: O(L) layers at O(n) width compute A^L. Thm 4.4: O(d log n) width for degree-d graphs. §5: near-quadratic width for Eulerian verification. §6 empirics match parameter counts.\[24\] | **High** (width theory) |
| Loukas, "What GNNs cannot learn: depth vs width" (arXiv:1907.03199, ICLR 2020) | MPNNs on cycle detection, subgraph verification and similar tasks | **Theory.** depth×width (communication capacity) lower bounds via CONGEST. | Main theorems: decision problems are impossible unless depth·width exceeds a graph-size-dependent bound.\[25\] | Medium (GNN analogue of width bounds) |

## 2. Synthesis by research question

### Q1. Which benchmarks admit shortcuts?
- **GraphQA connectivity (reachability).** Fatemi et al. publish majority baselines only for edge existence and cycle check (§3.1.1), and Sanford et al. publish none (Fig. 3a).\[4\]\[5\] The ER generator in GraphQA has mean degree 5.70 on 12.37 nodes (Fatemi Table 5; Sanford App. E.1).\[4\]\[6\] Such dense, small graphs are almost always connected and have tiny diameter, so short-path and degree cues are expected to dominate. Your audit numbers (97.3% rule, 99.5% 2-hop GBDT, 94% of positives within 2 hops) are, as far as I can find, the first explicit published-style audit of this split. The likely label skew toward "connected" is supported only by inference and by an unreviewed student report, so you should report the label ratio yourself.
- **Cycle check (GraphQA).** 81.96% of ER instances contain a cycle (Fatemi §3.1.1).\[5\] Sanford et al. (§4.2, App. E.1) note the shortest cycle is "almost always of length three, and only 3.2% of instances are larger". In their words the task is "inherently local", so the benchmark's own authors effectively concede a shortcut.\[4\]
- **NLGraph connectivity.** The benchmark authors identified the shortcut themselves (§4.4): "since nodes with higher degrees are more frequently mentioned and they are more likely to be connected, LLMs might just be counting node mentions instead of actually finding paths." They tested it on 120-example chain and clique splits (Table 4).
- **Random-graph connectivity for trained transformers.** Ye et al. (Fig. 1) and Saparov et al. (Fig. 2) both show that ER-style or naïve generators yield perfect in-distribution accuracy alongside OOD collapse.\[3\]\[7\]
- **CLRS-30.** Mahdavi et al. find degree-distribution shift with size and a positional-index leak, and that ID validation does not predict OOD performance.\[17\]\[26\]
- **GraphWiz/GraphInstruct and NLGraph-tuned LLMs.** Hu et al. and NLGIFT report template and pattern memorisation.\[20\]\[21\] This is a task-level shortcut rather than a structural one.
- **GraphArena.** I found no explicit shortcut audit. Its reliance on feasibility checks of the solution path partly protects it.\[27\]
- **Path-star.** Bachmann & Nagarajan document an output-side Clever Hans shortcut.\[10\]

**Contradiction to flag:** Sanford §4.1 asserts that connectivity cannot be solved from source/sink neighbourhoods.\[4\] That is true in the worst case but false on the GraphQA distribution per your audit. Their 98.0% therefore does not discriminate between the algorithm and a 2-hop shortcut.

### Q2. Shortcut-resistant tasks and generators
- **Lookahead-balanced DAGs.** Saparov et al. §3 and App. A.1.2 make lookahead uniform so that no heuristic beats chance at a given depth. Certification is empirical: cross-distribution test matrices (Fig. 2) plus mechanistic circuit extraction (§4).\[7\]
- **Diameter-controlled data.** Ye et al. certify through a theorem (the within-capacity population gradient suppresses the J_n channel) and an empirical measurement of I_n/J_n "energy share" (App. D.1).\[3\] The authors report that the model performs perfectly when the I-channel share exceeds roughly 90%.
- **Adversarial inversions.** NLGraph's chain and clique splits are certified only informally, by construction.\[28\]
- **Locally indistinguishable classes.** The 1-vs-2-cycle problem (Sanford et al.; Yehudai et al. Thm 4.1; Abbe et al. cycle task) is the canonical case:\[24\] every node has degree 2 in both classes, so all r-hop balls with r < n/2 are identical. This is the cleanest certification available because indistinguishability holds by isomorphism of neighbourhoods. Your "swap" generator is in this family.
- **Statistical-feature removal.** Zhang et al. (IJCAI 2023) show that you can balance one feature, but jointly balancing all of them is infeasible.\[29\] This is the main theoretical warning against claiming that a generator is "shortcut-free".
- **No paper provides a formal certificate covering all bounded-complexity features.** Certification is always relative to a declared feature family.

### Q3. When does a model take the shortcut?
- **Data distribution is the dominant lever.** Ye et al. find that the share of beyond-capacity graphs decides which channel wins (Thm C.9).\[3\] Saparov et al. find that naïve vs balanced data separates failure from success (Fig. 2).\[7\] Zhang et al. find that removing a statistical feature improves generalisation.\[30\]
- **Depth.** In Ye et al., capacity is set by depth (3^L), so for fixed data a deeper model takes the shortcut less often.\[2\]
- **Model size.**
  - Saparov et al.: more parameters do not fix the growing difficulty on larger graphs, and chain-of-thought (DFS or selection-inference) does not either.\[7\]
  - Roy & Saparov: scale helps on grid graphs but not on many-component graphs (the paper carries an author-flagged error).\[8\]
  - Fatemi et al. §3.4: larger PaLM 2 is better except on edge existence.\[5\]
- **These scale results are not contradictory, but they are confounded.** None separates width from depth, and none removes shortcuts before scaling.
- **Training length and sample size.** Sanford App. E.4.1 reports that connectivity models perfectly fit training data at most sample sizes, with train and test accuracy converging only at 100K.\[4\] Saparov et al. use streaming data (883M examples).\[7\]
- **Curriculum.** Abbe et al. (2023) propose Degree-Curriculum.\[15\] Abbe et al. (2024) find that inductive scratchpads break globality.\[13\] Saparov et al. suggest curriculum and looped transformers as future work.\[7\]
- **Implicit bias.** The min-degree bias (Abbe et al. 2023) predicts that SGD prefers the lowest-order interpolator, i.e. local features, whenever one fits the training data.\[31\]
- **Width specifically: no result found.** This is the central gap.

### Q4. Formal notions of locality and globality
- **Globality degree** (Abbe et al. 2024): the minimal number of tokens k such that the target correlates non-negligibly with those k tokens plus histogram information. High globality implies no efficient weak learning.\[13\] This is the learnability-side measure, and it directly predicts that a low-globality surrogate (an endpoint-edge rule) will be learned first.
- **Communication complexity and MPC.**
  - Sanford et al. prove a log-depth requirement for connectivity at sub-linear width, conditional on the 1-vs-2-cycle conjecture.\[4\]\[24\]
  - Yehudai et al. Thm 4.2 gives an unconditional width×precision×heads×depth = Ω(n) bound for 2-cycle detection under adjacency tokenisation, via set disjointness.\[24\]
  - Loukas (2020) gives CONGEST-based depth×width bounds for MPNNs.\[32\]
- **WL limits.** Sanford App. D.2 notes that 1-WL cannot distinguish connected from disconnected graphs (e.g. a 2n-cycle vs two n-cycles). Without node IDs, MPNNs cannot solve connectivity at all.\[4\]
- **k-hop indistinguishability.** This is implicit in Linial-style locality (radius-r views) and in Ye et al.'s capacity bound (3^L hops). I found no paper that defines "k-hop indistinguishable class pairs" as a dataset-design criterion for transformers. That is a gap, and your swap generator could formalise it.
- **Reconciling the two families.** Expressivity bounds (TC⁰, MPC, communication) say what a model *can* compute. Globality and min-degree bias say what it *will* learn. Abbe et al. (2024) state the contrast explicitly: globality relates to NC⁰ correlations, not to TC⁰ expressivity.\[13\]

### Q5. Audit methodology
There is no accepted protocol in graph ML. The practices in use are:
1. **Majority/label-prior baselines** (Fatemi §3.1.1; GraphOmni, arXiv:2504.12764, claims to be the only LLM graph benchmark with a random baseline).\[5\]\[33\]
2. **Hand-built adversarial splits** that invert a suspected cue (NLGraph Table 4; Ye Fig. 1 disjoint chains/cliques).\[3\]\[28\]
3. **Cross-distribution test matrices** (Saparov Fig. 2).\[7\]
4. **Size or generator extrapolation** (SALSA-CLRS at 5–100×; Mahdavi et al.).\[19\]
5. **Structure ablations** (GNN vs GNN_∅, Bechler-Speicher Table 1).\[23\]
6. **Mechanistic checks**: channel energy share (Ye App. D.1), circuit reconstruction (Saparov §4), Clever-Hans cheat accuracy (Bachmann & Nagarajan).\[3\]\[7\]\[34\]
7. **Shallow classifiers on surface features**: done systematically in NLP (logistic regression on n-grams, arXiv:2410.11672) but, as far as I found, not on graph benchmarks.\[35\]

**Recommended protocol for the thesis**, combining the above:
- (a) label prior;
- (b) single-rule baselines (endpoint degree, endpoint-has-edge, common neighbour);
- (c) GBDT on k-hop features for k = 1, 2, 3, reported as a curve in k;
- (d) distance-stratified accuracy;
- (e) an indistinguishable-class split (swap);
- (f) a graph-less or permuted-ID ablation.

Report the gap between model accuracy and the best k-hop baseline as the "algorithmic margin".

### Q6. Representation vs learning gap
- **Measured directly for graph tasks.**
  - Ye et al. prove that the algorithm is representable within capacity and show that SGD finds it only with the right data.\[2\] This is the closest to a quantified gap, since channel energy share correlates linearly with accuracy (Fig. 4 area).
  - Saparov et al. show the target is representable (log-depth path-merging is found), yet learning gets harder with graph size at fixed parameters.\[7\]
  - Zhang et al. construct a perfect BERT parameterisation and show that trained BERT does not reach it.\[16\]
- **Not measured.** Yehudai et al. §6 test depth versus width at fixed parameter count on accuracy and runtime.\[24\] They do not check whether trained wide-shallow models implement the Thm 4.3 construction or a shortcut. Sanford et al. compare against theory only through task-class accuracy differences.\[4\]
- **Bottom line:** the width at which a task becomes *representable* (Yehudai: O(n) for 2-cycle and adjacency powers; O(d log n) for bounded degree) has never been compared with the width at which SGD *learns* it on shortcut-free data. That comparison is your thesis contribution.

## 3. Must-read papers (priority order)
1. **Ye et al. 2025/26 (arXiv:2510.19753).** Provides the algorithm vs heuristic channel decomposition and the data lever. Check its related work for Nichani et al. 2024 and Merrill & Sabharwal 2025, which supply its capacity proofs.
2. **Yehudai et al. NeurIPS 2025 (arXiv:2503.01805).** The only width-centric theory for graph transformers. Use Thm 4.2–4.4 to set width predictions under adjacency tokenisation.
3. **Saparov et al. ICLR 2025 (arXiv:2412.04703).** The balanced generator, the negative scale result and the mechanistic method you can reuse.
4. **Sanford et al. NeurIPS 2024 (arXiv:2405.18512).** The baseline you are auditing. Note the internal inconsistencies (11B vs "12B" T5; "1 million parameters" vs 60M in App. E.4).\[4\]
5. **Abbe et al. NeurIPS 2024 (arXiv:2406.06467).** Globality degree is the theoretical predictor of which shortcut is learned first.
6. **Wang et al. NeurIPS 2023, NLGraph (arXiv:2305.10037), §4.4.** The precedent for cue-inverting splits.
7. **Zhang et al. IJCAI 2023 (arXiv:2205.11502).** Argues that statistical features cannot all be removed. Frame the swap generator's guarantees relative to a feature family.
8. **Abbe et al. ICML 2023 (arXiv:2301.13105).** Min-degree bias, the implicit-bias explanation for preferring local rules.
9. **Fatemi et al. ICLR 2024 (arXiv:2310.04560).** The GraphQA generator details and the only published majority baselines.\[6\]
10. **Loukas ICLR 2020 (arXiv:1907.03199).** The GNN depth×width precedent for the communication-capacity framing.

## 4. Open gaps
- **Width-to-learn scaling on shortcut-free data.** Compare the minimum width m*(n) at which SGD reaches ≥ 99% on swap-type data with Yehudai's representational O(n) / O(d log n), at fixed depth.
- **Width vs depth as a shortcut lever.** Ye et al. treat capacity as depth-only (3^L). Whether extra width shifts the I_n/J_n balance is untested.
- **Published local-baseline audits of GraphQA and NLGraph.** NLGraph balances connectivity labels by construction ("half of the node pairs are connected and the other half are not connected," §2.2) and reports a 50.00 random baseline (Table 2). GraphQA reports no label prior for connectivity. Neither benchmark reports k-hop GBDT or single-rule baselines.
- **A formal "k-hop indistinguishability" certificate** for generators, linking globality degree to concrete graph constructions beyond the cycle task.
- **The interaction between shortcut availability and the width threshold.** Does the presence of a 97% local rule raise the width needed to later find the algorithm (a gradient-starvation analogue of the Clever Hans "indecipherable token")?
- **Tokenisation dependence.** Edge-list (Sanford) and adjacency-row (Yehudai) inputs carry different leakage (node indices, degree via row sums).\[24\] Your first benchmark's index/degree leakage has no published counterpart.

## Caveats
- Sanford et al. numbers come from arXiv v1 (Fig. 3a); the NeurIPS camera-ready may differ.\[4\]
- Fatemi et al.'s reachability task and Table 9 appear only in the ICLR camera-ready.\[6\]
- The globality bound glob(cycle) ≥ n and the Mahdavi 60.47→66.18 figure are taken from secondary summaries.
- Roy & Saparov carries an author note that an assumption is incorrect.\[36\]
- Ye et al.'s venue is unconfirmed. The arXiv listing (v2, 17 Feb 2026) names no venue, though its HTML carries ICML keywords.
- I did not verify GraphArena- or GraphWiz-specific structural shortcut audits beyond Hu et al.'s template-overfitting critique.

## Sources

1. [When Do Transformers Learn Heuristics for Graph Connectivity?](https://arxiv.org/html/2510.19753v1)
2. [Transformers Provably Learn Algorithmic Solutions for Graph Connectivity, But Only with the Right Data](https://arxiv.org/pdf/2510.19753)
3. [Transformers Provably Learn Algorithmic Solutions for Graph Connectivity, But Only with the Right Data](https://arxiv.org/html/2510.19753v2)
4. <https://arxiv.org/pdf/2405.18512>
5. [Talk like a Graph: Encoding Graphs for Large Language Models](https://arxiv.org/pdf/2310.04560)
6. <https://proceedings.iclr.cc/paper_files/paper/2024/file/bf72f65f30eedf5d48da6980ee02b589-Paper-Conference.pdf>
7. [Transformers Struggle to Learn to Search](https://arxiv.org/pdf/2412.04703)
8. [\[2509.22343\] Transformers Can Learn Connectivity in Some Graphs but Not Others](https://arxiv.org/abs/2509.22343)
9. [Transformers Can Learn Connectivity in Some Graphs but Not Others — Lacuna](https://lacuna.tiptreesystems.com/work/transformers-can-learn-connectivity-in-some-graphs-but-not-others/wrk_99ad094b67825c0380dcc8d6b560bd9f)
10. [fly51fly on X: "\[CL\] The pitfalls of next-token prediction G Bachmann, V Nagarajan \[ETH Zürich & Google Research\] (2024) https://t.co/4xpr4bu2ws - Next-token prediction has become a core part of modern language models, but there is a growing belief that it cannot truly model human thought and https://t.co/ErUhVm3auR" / X](https://x.com/fly51fly/status/1767555854918885567)
11. [Looking beyond the next token](https://arxiv.org/pdf/2504.11336)
12. [Language Models, Graph Searching, and Supervision Adulteration: When More Supervision is Less and How to Make More More](https://arxiv.org/pdf/2503.10542)
13. [How Far Can Transformers Reason? The Globality Barrier and Inductive Scratchpad](https://arxiv.org/abs/2406.06467)
14. [\[Quick Review\] How Far Can Transformers Reason? The Globality Barrier and Inductive Scratchpad](https://liner.com/review/how-far-can-transformers-reason-the-globality-barrier-and-inductive)
15. [\[2301.13105\] Generalization on the Unseen, Logic Reasoning and Degree Curriculum](https://arxiv.org/abs/2301.13105)
16. [On the Paradox of Learning to Reason from Data \[Quick Review\]](https://liner.com/review/on-paradox-learning-to-reason-from-data)
17. [arxiv.org](https://arxiv.org/abs/2211.00692v1)
18. [Towards Better Out-of-Distribution Generalization of Neural Algorithmic Reasoning Tasks — Lacuna](https://lacuna.tiptreesystems.com/work/towards-better-out-of-distribution-generalization-of-neural-algorithmic/wrk_088dcd9abbeb16eebfe53db88535c43b)
19. [SALSA-CLRS: A Sparse and Scalable Benchmark for Algorithmic Reasoning](https://tik-db.ee.ethz.ch/file/84ab60cc12b0da83441aceafd78a875b/)
20. [Can LLM Graph Reasoning Generalize beyond Pattern Memorization?](https://arxiv.org/pdf/2406.15992)
21. <https://arxiv.org/html/2509.24260v1>
22. [Rethinking and Benchmarking Large Language Models for Graph Reasoning](https://arxiv.org/pdf/2509.24260)
23. [Graph Neural Networks Use Graphs When They Shouldn't](https://arxiv.org/pdf/2309.04332)
24. <https://proceedings.neurips.cc/paper_files/paper/2025/file/2130b8a44e2e28e25dc7d0ee4eb6d9cf-Paper-Conference.pdf>
25. [What graph neural networks cannot learn: depth vs width](https://www.researchgate.net/publication/343713980_What_graph_neural_networks_cannot_learn_depth_vs_width)
26. [(PDF) Towards Better Out-of-Distribution Generalization of Neural Algorithmic Reasoning Tasks](https://www.researchgate.net/publication/365014227_Towards_Better_Out-of-Distribution_Generalization_of_Neural_Algorithmic_Reasoning_Tasks)
27. [GRAPHARENA: EVALUATING AND EXPLORING LARGE ...](https://proceedings.iclr.cc/paper_files/paper/2025/file/77fa8253adfc8b33209639f3e9985741-Paper-Conference.pdf)
28. [Can Language Models Solve Graph Problems in Natural Language?](https://arxiv.org/pdf/2305.10037)
29. [On the Paradox of Learning to Reason from Data](http://web.cs.ucla.edu/~kwchang/bibliography/zhang2023on/)
30. [On the Paradox of Learning to Reason from Data](http://starai.cs.ucla.edu/papers/ZhangIJCAI23.pdf)
31. [Generalization on the Unseen, Logic Reasoning and Degree Curriculum — Lacuna](https://lacuna.tiptreesystems.com/work/generalization-on-the-unseen-logic-reasoning-and-degree-curriculum/wrk_2a234c53bb5b30109e037f50592d9fb8)
32. [How hard is to distinguish graphs with graph neural networks? Andreas Loukas](https://proceedings.neurips.cc/paper_files/paper/2020/file/23685a2431acad7789c1e3d43ea1522c-Paper.pdf)
33. [www.arxiv.org](https://www.arxiv.org/pdf/2504.12764v1)
34. [The Pitfalls of Next-Token Prediction](https://openreview.net/attachment?id=23d4bQkTGI&name=pdf)
35. [Leaving the barn door open for Clever Hans: Simple features predict LLM benchmark answers](https://arxiv.org/pdf/2410.11672)
36. [Computer Science Sep 2025](https://arxiv.org/list/cs/2025-09?skip=10400&show=100)
