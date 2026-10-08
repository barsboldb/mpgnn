<!--
Novelty check produced by Claude Research mode on 2026-10-08 from the prompt in the
2026-10-08 session (findings 1–7 of docs/WIDTH-QUESTIONS.md / reports/width-progress.pdf).
Saved verbatim below this header.

Verified against local PDFs on 2026-10-08 (papers/):
- Yehudai et al. 2503.01805: §6.1 fixed-100k-parameter (depth, width) pairs (1,125) (2,89)
  (4,63) (8,45) (10,40); LR tuned in {1e-4, 5e-5}; datasets of 5000 graphs; critical width
  = training-loss plateau > 0.05, 1 layer, 2 heads, "increases roughly linearly with the
  graph size"; "quadratic width should suffice for solving any task, since it can be used to
  record the entire graph". All as stated below.
- Ye, Fu, Jia, Sharan 2510.19753: 3^L diameter capacity (Disentangled Transformer),
  within/beyond-capacity data effect, 1 billion ER(n=20) training graphs, AdamW lr 1e-4,
  weight decay 1e-4, cosine. As stated below. NOTE: the report did not know that our
  connectivity-matrix task, adjacency-row tokens and bilinear read-out were adopted FROM
  this paper in June 2026 (docs/CHANGELOG.md 2026-06-25) — the similarity is by design.
Not verified by us (report opened them, or search snippet only): Abbe et al. 2024, Mahdavi
et al. 2023, NLGraph, DeZoort & Hanin 2026, Saparov et al. contradiction quote, Sanford–Hsu–
Telgarsky App. G, Barak et al., Edelman et al., Abboud et al., Sato et al. Verify before citing.
Venue conflict: the report gives Yehudai et al. as NeurIPS 2025 (spotlight); our
WIDTH-LITERATURE.md says ICML 2025 from the arXiv v3 header — check the proceedings.
-->

# Novelty Check: "Analyzing the Effect of Network Width on Graph Transformers" (Findings 1–7)

The main result (Finding 5) looks new: I found no paper that measures how the width SGD needs to *learn* connectivity scales with graph size n, and none that reports a polynomial (≈n²) learnable-width law sitting against sublinear/linear representational bounds. Most other findings are partial replications that should be cited. Your setup is almost identical to that of Ye, Fu, Jia & Sharan (arXiv 2510.19753): same (A+I) adjacency-row tokens, no positional encodings, pre-norm encoder, n×n connectivity-matrix read-out, exact-match metric.\[1\] Your "one cycle vs two cycles" training distribution was already used as a *training* benchmark by Abbe et al. (NeurIPS 2024).\[2\] Neither paper varies width.

## TL;DR
- **Likely new:** Finding 5 (learnable critical width ∝ n^≈1.9 at log depth, robust to data/steps/LR/input encoding), Finding 4 in its width-specific form, the per-width-LR-tuned critical-width methodology for graph transformers (Finding 1 as applied), and Finding 7 (≈20–25% width overhead for random node-ID projection, same slope). No paper I opened reports these.
- **Replications / must-cite:** Ye et al. 2510.19753 uses the same tokenization and read-out and finds that "Within-capacity graphs (diameter ≤ 3^L) drive the learning of the algorithmic solution while beyond-capacity graphs drive the learning of a simple heuristic based on node degrees". Abbe et al. NeurIPS 2024 introduces the 1-vs-2-cycle "cycle task" for training, shows globality ≥ n, and its abstract states that "distributions with high globality cannot be learned efficiently". Also cite Yehudai et al. NeurIPS 2025 (2503.01805). Yehudai et al. contain things you may have missed: a "critical width" experiment (roughly linear in n for 4-cycle counting), a 2-layer/linear-width construction for 1-vs-2 cycles that matches your Finding 6, and a connectivity experiment with only 5,000 graphs and an LR grid of {1e-4, 5e-5}.
- **Tension/contradictions to address:** Yehudai et al. report connectivity accuracy roughly flat across depth 1–10 at ~100k parameters for n=100 (but on a leaky, mixed-generator dataset). Merrill & Sabharwal (arXiv:2503.03961) show depth need only grow as Θ(log n) while "width must be scaled superpolynomially" for regular-language recognition, and Sanford–Hsu–Telgarsky report a similar depth advantage on k-hop. Your ≈n² scaling also sits above Yehudai et al.'s representational result that connectivity and fixed-length cycle detection are tasks "for which linear width is necessary and sufficient for dense graph input".

## Key Findings (per finding)

### Finding 1 — Learning-rate confound in width sweeps
**Verdict: PARTIALLY SHOWN (general / GNN level); NOT FOUND for graph transformers or algorithmic graph tasks.**
- DeZoort & Hanin, "Hyperparameter Transfer in Graph Neural Networks," arXiv:2607.05017 (2026) — https://arxiv.org/html/2607.05017v1. Proposes a µP/CompleteP-style parameterization for GNNs that "yields stable feature updates, learning rate transfer, and improved performance as width and depth increase". The Adam rule is η = η0/√D, so the optimal LR shrinks with width. Datasets are MNIST-superpixels, PascalVOC-SP, QM9 and Cora-type node classification, not reasoning tasks or transformers.\[3\]
- The general SP effect ("optimal learning rate decreases as the width increases… test accuracy at width = 16384 is lower than width = 512") appears in arXiv:2312.12226 (MLP/CNN/ResNet, K-FAC/SGD)\[4\] — https://arxiv.org/pdf/2312.12226 (search-result excerpt only, not fully opened).
- Why your warning matters: the closest graph-transformer width studies used near-fixed LRs. Yehudai et al. tuned LR only over {1e-4, 5e-5} for their depth–width comparison.\[5\] Ye et al. trained standard transformers with AdamW at a single LR of 1e-4.\[6\] Their width/depth conclusions could therefore carry the confound you document.
- **Framing:** cite Tensor Programs V and DeZoort & Hanin. Present your result as the first demonstration (as far as I found) of the confound in a graph-transformer/algorithmic width sweep, showing how it can flip conclusions.

### Finding 2 — Width floor and data-dependent ceiling (memorization); more data removes the ceiling
**Verdict: RELATED BUT DIFFERENT (classic model-wise double descent; not shown for graph algorithms).**
- The pattern matches model-wise double descent / benign-overfitting results (Belkin et al.; Nakkiran et al.), surveyed in "A Farewell to the Bias-Variance Tradeoff?" arXiv:2109.02355 — https://arxiv.org/pdf/2109.02355 (search excerpt). The survey reports Nakkiran et al.'s finding that width-wise double descent appears when training is long enough and is amplified by label noise.\[7\]
- No graph-algorithm paper I found shows a test-accuracy ceiling in width that more data removes at equal steps. Note that Yehudai et al.'s connectivity and counting datasets contain only 5,000 graphs each, so their critical-width curves may sit in your "memorization" regime.\[5\] Ye et al. and Saparov et al. sidestep the issue with effectively unlimited data (Ye et al. use ~1 billion ER graphs).\[6\]
- **Framing:** replication of a known phenomenon in a new setting; useful as a methodological caveat for small-data graph-transformer width studies.

### Finding 3 — Connectivity benchmarks leak (node indices; degree/short-cycle statistics; bounded-hop locality)
**Verdict: PARTIALLY SHOWN for (a) and (b); (c) and locally-indistinguishable *training* benchmarks were ALREADY SHOWN by Abbe et al.; your three-way audit as a package was NOT FOUND.**
- (b) Degree shortcuts:
  - **Ye et al.**, arXiv:2510.19753 (v1 title "When Do Transformers Learn Heuristics for Graph Connectivity?"; v2 "Transformers Provably Learn Algorithmic Solutions for Graph Connectivity, But Only with the Right Data") — https://arxiv.org/abs/2510.19753. They prove that trained weights split into an algorithmic channel (A⊗I, matrix powering) and a heuristic channel (J-channel) that "computes global statistics, specifically products of node degrees".\[8\]\[9\] In their Fig. 1/§3.3, 2-layer transformers trained on ER(n=20) "achieve perfect in-distribution accuracy on random graphs yet fail catastrophically on simple OOD instances such as two disjoint chains", with exact match falling to nearly zero on 2Chain/2Clique. Their mechanism is degree products on ER graphs, not your bridge-closes-no-cycle / short-cycle-count argument.
  - **Wang et al., "Can Language Models Solve Graph Problems in Natural Language?" (NLGraph), NeurIPS 2023**, arXiv:2305.10037 — https://arxiv.org/pdf/2305.10037. LLMs "use node mention frequency to determine connectivity". Chain and clique adversarial cases expose this.\[10\]\[11\] This is for prompted LLMs, not trained-from-scratch models.
  - **Abbe et al.** (below) state that a random-graph connectivity version is solved via degree shortcuts (their App. B.1).\[2\]
- (a) Node-index leakage:
  - **Mahdavi et al., "Towards Better OOD Generalization of Neural Algorithmic Reasoning Tasks," TMLR 2023**, arXiv:2211.00692 — https://arxiv.org/abs/2211.00692. In CLRS, "node index serves as a unique flag", models "might rely on spurious correlations" between indices, and a Bridges test set allows "cheating solutions".\[12\] This shows index fragility and spurious correlation, not IDs revealing component membership.
  - **Dwivedi et al., "Benchmarking Graph Neural Networks," JMLR 24 (2023)** — https://jmlr.org/papers/volume24/22-0567/22-0567.pdf. Randomly permuting node ordering for index PEs "improves significantly the performances over keeping fixed the original node ordering" (search excerpt).\[13\]
  - I found no paper documenting that generator-assigned node indices reveal component membership in connectivity data. Your (a) looks new as a documented failure.
- (c) Locality and locally-indistinguishable training distributions:
  - **Abbe, Bengio, Lotfi, Sandon, Saremi, "How Far Can Transformers Reason? The Globality Barrier and Inductive Scratchpad," NeurIPS 2024**, arXiv:2406.06467 — https://proceedings.neurips.cc/paper_files/paper/2024/file/3107e4bdb658c79053d7ef59cbc804dd-Paper-Conference.pdf. Their "cycle task" (two disjoint n-cycles vs one 2n-cycle, query pair) is built to preclude spurious correlations: "No simple statistics based on degrees, edge counts, or finite motif counts" help, and "any set of n − 1 edges have the same distribution", so globality ≥ n. It is used as a training benchmark and "put forward… as a simple benchmark to test the global reasoning capabilities of models".\[2\]
  - Differences from your setup: edge-list tokens with a single query pair, pure cycles without chords, and very small n.
  - **Framing:** your dense-target, chorded "swap two far edges" construction plus explicit audits (degree/cycle statistics and ≤4-hop checks at chance) is a stronger, adjacency-row version of Abbe's cycle task. Cite Abbe as the origin of the idea.
- Roy & Saparov, arXiv:2509.22343 (2025) — https://arxiv.org/abs/2509.22343. Find that transformers learn connectivity on grid graphs via "low-dimensional vertex-embedding heuristic" but not on disconnected chains.\[14\]\[15\] **The arXiv listing carries the author comment "This paper contains some assumption which is not correct"**, so treat it with caution.\[16\]

### Finding 4 — On shortcut-free data, width decides whether learning starts at all
**Verdict: PARTIALLY SHOWN (learning onset fails with n); the width-specific threshold was NOT FOUND, and one result partly contradicts it.**
- Abbe et al. (NeurIPS 2024): iterations to reach ≥95% on the cycle task grow "exponentially" with n for GPT-2-style models of 10M/25M/85M parameters, and "the 10M model fails to learn for n ≥ 7 in 100k iterations".\[2\] They vary total parameter count, not width at fixed depth.
- Saparov et al., "Transformers Struggle to Learn to Search," ICLR 2025, arXiv:2412.04703 — https://arxiv.org/pdf/2412.04703 (search excerpt). At 8 layers and hidden size 16, the fraction of 14 seeds that learn drops to near zero as graph size grows. With graph size fixed at 31, "there is no discernible pattern between the size of the model and the amount of training needed to find the global minimum".\[17\] **This partly contradicts "width decides onset"**, but it concerns edge-list next-token search with very small widths.
- Barak et al., "Hidden Progress in Deep Learning," NeurIPS 2022, arXiv:2207.08799 (search excerpt) is the canonical source for plateau-then-onset learning, with an appendix on convergence time vs width.\[18\] Edelman et al., "Pareto Frontiers in Deep Feature Learning: Data, Compute, Width, and Luck," NeurIPS 2023 (search excerpt) shows width trading against data and time for sparse parity.\[19\]\[20\] Cite both as mechanism analogues: width as parallel search over features.

### Finding 5 (MAIN) — Learnable critical width ∝ n^≈1.9 at depth ⌈log₂ n⌉
**Verdict: NOT FOUND. Closely related representation-vs-learning gap results exist, plus one partial contradiction.**
- No paper I found measures the width (or size) SGD needs to learn connectivity as a function of n, or fits a scaling exponent with robustness controls.
- Closest empirical analogue: **Yehudai et al.** (arXiv:2503.01805; NeurIPS 2025 spotlight as "Depth-Width Tradeoffs for Transformers on Graph Tasks") — https://arxiv.org/abs/2503.01805.
  - Their Sec. 6.2 defines a "critical width" for 4-cycle and triangle counting (1-layer, adjacency rows, n=50–400) and finds it "increases roughly linearly with the graph size".\[5\]
  - It differs from yours: it is a fitting criterion (training-loss plateau > 0.05, not test accuracy), uses counting tasks rather than connectivity, depth 1, and 5,000 graphs.\[5\]
  - Their theory also states that connectivity at constant depth needs linear width for dense adjacency input, and that 1-vs-2 cycles is solvable with 2 layers and O(n) width (Thm 4.1).\[5\] So your ≈n² *learnable* width is ~n above their representational width. That is a clean representation–learning gap to report.
- Representation-vs-learning gap literature to cite:
  - **Abbe et al.** contrast TC⁰/TC¹ expressivity with learnability via globality and prove a negative result for a cycle-task variant.\[2\]
  - **Ye et al.** show the architecture is "provably expressive enough", yet training picks the degree heuristic unless the data lie within capacity.\[1\]\[6\]\[8\]
  - **Merrill & Sabharwal, "A Little Depth Goes a Long Way," NeurIPS 2025**, arXiv:2503.03961 — https://ar5iv.labs.arxiv.org/html/2503.03961. Log depth can express connectivity (theory only for connectivity).\[21\] Their width/depth experiments are on A₅ state tracking, not graphs.\[22\]
- Interpretation you should test: n² equals the number of adjacency bits. Yehudai et al. note that "quadratic width should suffice for solving any task, since it can be used to record the entire graph".\[5\] SGD may be finding a "copy the whole graph into each token, then compute" solution rather than log-depth squaring. Your (A+I)P random-projection control (96-wide input) argues against a pure input-width artefact, but probing intermediate layers for A^(2^ℓ) structure (as in Ye et al.) would settle this.
- Caveat on the claim itself: n = 32–56 spans less than a factor of 2, so the exponent is weakly identified; your bootstrap range 1.68–2.03 reflects this. State it as "≈ n²" with that range, not as a law.

### Finding 6 — Depth–width trade-off at n = 40 (2 layers: m ≈ 1.1n; 6 layers: 45 → 30; 8 ≈ 6)
**Verdict: PARTIALLY SHOWN (empirical depth–width trade-offs on graph/algorithmic tasks exist); critical width as a function of depth for connectivity was NOT FOUND.**
- Yehudai et al. Sec. 6.1: at ~100k parameters for n=100 connectivity, (depth, width) ∈ {(1,125),(2,89),(4,63),(8,45),(10,40)} give consistent loss/accuracy.\[5\] This is consistent with your "depth helps little", but their connectivity data mix ER/RGG/BA/SBM generators with a graph-level label, which your audits suggest is leaky.\[5\]
- Your 2-layer m ≈ 1.1n matches Yehudai's Thm 4.1 (2 layers, O(n) width for 1-vs-2 cycles), a nice theory–experiment agreement to report.
- Sanford, Hsu, Telgarsky, "Transformers, Parallel Computation, and Logarithmic Depth," arXiv:2402.09268 (ICML 2024), App. G (search excerpt of the arXiv PDF): on k-hop induction heads, "doubling the width is roughly equivalent in performance to incrementing the depth by one" (empirical claim L·log m = Ω(log k)).\[23\] You know this paper, but this appendix result is the closest empirical depth–width exchange rate.
- Ye et al.: capacity is 3^L in *diameter*, not n.\[1\]\[8\]\[9\] With diameter ≈ 10 at n = 40, a 2-layer model (3² = 9) sits just at capacity, which may explain why depth beyond ~3 adds little. Note that their bound is for Disentangled Transformers with nonnegative weights.
- **Contradiction:** Merrill & Sabharwal (arXiv:2503.03961) show that for regular-language recognition depth need only grow as Θ(log n) while "width must be scaled superpolynomially… Thus scaling depth more efficiently allows solving these reasoning problems compared to scaling width or using CoT". Your connectivity result (depth 2→6 cuts width by only ~1/3) points the other way for this task.

### Finding 7 — Random ±1/√k node-ID projection costs ~20–25% more width, same slope
**Verdict: RELATED BUT DIFFERENT; the specific measurement was NOT FOUND.**
- Random node features: Abboud et al., "The Surprising Power of GNNs with Random Node Initialization," IJCAI 2021 — https://www.ijcai.org/proceedings/2021/0291.pdf (search excerpt). Sato, Yamada, Kashima, "Random Features Strengthen GNNs," SDM 2021 (Semantic Scholar entry). Both give expressivity results for MPNNs, not learnable-width costs.\[24\]\[25\]
- Random low-dimensional node codes in transformer constructions: Yehudai et al. Thm 4.4 uses RIP-style random vectors (O(d log n) width), and their connectivity sketch uses linear sketching (Ahn et al. 2012).\[5\] Bechler-Speicher et al., "Lost in Tokenization," arXiv:2605.22471 (2026) — https://arxiv.org/abs/2605.22471 (search excerpt) prove tokenization changes depth regimes.\[26\]
- **Framing:** a minor new empirical observation; cite the above as context.

## Contradictions to discuss explicitly
- **Linear vs quadratic width:** Yehudai et al.: linear width suffices to represent 1-vs-2 cycles/connectivity at constant depth, and empirical critical width is ~linear for counting.\[5\] Your ≈n² is for *learning* and generalization, which is consistent if framed as a gap. But reviewers will ask why the linear construction is not found.
- **Depth efficiency:** Merrill & Sabharwal (arXiv:2503.03961) show depth need only grow as Θ(log n) while "width must be scaled superpolynomially" for regular-language recognition, and Sanford et al. (k-hop) also find depth very efficient; you find a modest depth benefit.
- **Model scale helping:** Saparov et al. (ICLR 2025) report that "as the input graph size increases, the transformer has greater difficulty in learning the task. This difficulty is not resolved even as the number of parameters is increased"; Roy & Saparov find scale helps on grids but not chains (flagged paper). Your result (width ≥ threshold is necessary and decisive) is sharper and task-specific.
- **Exponential vs polynomial:** Abbe et al. report "exponential" growth in learning cost with n on the cycle task. Your polynomial critical width (with adjacency-row tokens that act as implicit node IDs) is not contradictory, since Abbe et al. note that positional information can make high-globality targets easier.\[2\] Discuss it.

## Recommendations
- **Top 5 papers to add to related work:**
  1. Ye, Fu, Jia, Sharan, arXiv:2510.19753 — closest setup; degree shortcut; 3^L capacity.
  2. Abbe et al., NeurIPS 2024 — cycle task, globality, learnability vs expressivity.
  3. Yehudai et al., NeurIPS 2025 / arXiv:2503.01805 — critical-width methodology, linear-width 1-vs-2-cycle construction, LR grid, small data.
  4. Mahdavi et al., TMLR 2023 — node-index spurious correlations in CLRS.
  5. Wang et al., NLGraph, NeurIPS 2023 — degree/mention-frequency shortcuts in connectivity.
- Also cite: Saparov et al. ICLR 2025; Merrill & Sabharwal NeurIPS 2025; Sanford–Hsu–Telgarsky App. G; DeZoort & Hanin 2026; Abboud et al. / Sato et al.; Barak et al. 2022.
- **Positioning:** present Finding 5 as "the first empirical scaling law for learnable width on a shortcut-free connectivity task, quantifying a representation–learning gap". Present Findings 1–3 as methodological contributions that explain why prior width studies (fixed LR, 5k graphs, leaky generators) may not be reliable.
- **Cheap experiments that would strengthen novelty:**
  - Measure the 2-layer critical width vs n to test Yehudai's linear prediction directly.
  - Probe layers for A^(2^ℓ) structure versus whole-graph copying.
  - Extend the n range (even at reduced seeds) to tighten the exponent.

## Caveats
- Fully opened by me: Ye et al. (arXiv PDF), Abbe et al. (NeurIPS PDF), Yehudai et al. (arXiv PDF), Merrill & Sabharwal (ar5iv, introduction section). Opened by my sub-search: Mahdavi et al., DeZoort & Hanin.
- Seen only via search-result excerpts of the actual paper pages (verify before citing specifics): Saparov et al., Roy & Saparov (abstract page opened), NLGraph, Dwivedi et al., Sanford–Hsu–Telgarsky App. G, Barak et al., Edelman et al., Abboud et al., Sato et al., Bechler-Speicher et al., arXiv:2312.12226, arXiv:2109.02355, Di Giovanni et al. (ICML 2023, arXiv:2302.02941; width mitigates over-squashing in MPNNs, theory only, relevant only as background).\[27\]
- Not searched due to budget: GraphWiz, GraphQA leakage audits, CLRS-text, LoG/TMLR-specific width studies. Absence of evidence there is not evidence of absence.

## Sources

1. <https://arxiv.org/pdf/2510.19753>
2. <https://proceedings.neurips.cc/paper_files/paper/2024/file/3107e4bdb658c79053d7ef59cbc804dd-Paper-Conference.pdf>
3. <https://arxiv.org/html/2607.05017v1>
4. [On the Parameterization of Second-Order Optimization Effective Towards the Infinite Width](https://arxiv.org/pdf/2312.12226)
5. [Depth-Width tradeoffs in Algorithmic Reasoning of Graph Tasks with Transformers](https://arxiv.org/pdf/2503.01805)
6. [Transformers Provably Learn Algorithmic Solutions for Graph Connectivity, But Only with the Right Data](https://arxiv.org/html/2510.19753)
7. [A Farewell to the Bias-Variance Tradeoff? An Overview of the Theory of Overparameterized Machine Learning](https://arxiv.org/pdf/2109.02355)
8. [Transformers Provably Learn Algorithmic Solutions for Graph Connectivity, But Only with the Right Data](https://arxiv.org/html/2510.19753v2)
9. [Transformers Provably Learn Algorithmic Solutions for Graph Connectivity, But Only with the Right Data — Lacuna](https://lacuna.tiptreesystems.com/work/transformers-provably-learn-algorithmic-solutions-for-graph-connectivity-but/wrk_d8b9fb12aac0418ac57b33162e05cd54)
10. [Can Language Models Solve Graph Problems in Natural Language?](https://proceedings.neurips.cc/paper_files/paper/2023/file/622afc4edf2824a1b6aaf5afe153fa93-Paper-Conference.pdf)
11. [Can Language Models Solve Graph Problems in Natural Language?](https://arxiv.org/pdf/2305.10037)
12. <https://arxiv.org/pdf/2211.00692>
13. [Benchmarking Graph Neural Networks](https://jmlr.org/papers/volume24/22-0567/22-0567.pdf)
14. [Transformers Can Learn Connectivity in Some Graphs but Not Others](https://arxiv.org/pdf/2509.22343)
15. [Transformers Can Learn Connectivity in Some Graphs but Not Others](https://arxiv.org/abs/2509.22343)
16. [Computer Science Sep 2025](https://arxiv.org/list/cs/2025-09?skip=10400&show=100)
17. [Transformers Struggle to Learn to Search](https://arxiv.org/pdf/2412.04703)
18. [Hidden Progress in Deep Learning:](https://proceedings.neurips.cc/paper_files/paper/2022/file/884baf65392170763b27c914087bde01-Paper-Conference.pdf)
19. [\[2207.08799\] Hidden Progress in Deep Learning: SGD Learns Parities Near the Computational Limit](https://ar5iv.labs.arxiv.org/html/2207.08799)
20. [Pareto Frontiers in Deep Feature Learning: Data, Compute, Width, and Luck](https://proceedings.neurips.cc/paper_files/paper/2023/file/960573a3b797441aec39caa9f74bc793-Paper-Conference.pdf)
21. [A Little Depth Goes a Long Way: The Expressive Power of Log-Depth Transformers](https://ar5iv.labs.arxiv.org/html/2503.03961)
22. [A Little Depth Goes a Long Way: The Expressive Power of Log-Depth Transformers](https://www.alphaxiv.org/abs/2503.03961)
23. [Transformers, parallel computation, and logarithmic depth](https://arxiv.org/pdf/2402.09268)
24. [The Surprising Power of Graph Neural Networks with Random Node Initialization](https://www.ijcai.org/proceedings/2021/0291.pdf)
25. [\[PDF\] Random Features Strengthen Graph Neural Networks](https://www.semanticscholar.org/paper/Random-Features-Strengthen-Graph-Neural-Networks-Sato-Yamada/33e31195ab6853dfb8b1d90b07da5755f9bf5de0)
26. [\[2605.22471\] Lost in Tokenization: Fundamental Trade-offs in Graph Tokenization for Transformers](https://arxiv.org/abs/2605.22471)
27. [\[2302.02941\] On Over-Squashing in Message Passing Neural Networks: The Impact of Width, Depth, and Topology](https://arxiv.org/abs/2302.02941)
