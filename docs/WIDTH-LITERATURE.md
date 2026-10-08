# Width in Graph Transformers — Literature Sweep

Thesis: *Analyzing the Effect of Network Width on Graph Transformers* (settled 2026-09-24).
Sweep done 2026-09-24 with web search + arXiv abstract/HTML fetches.

**Verification legend.** "Verified" = arXiv abstract page (and, where a claim is
quantitative, the arXiv HTML full text) was fetched in this sweep. Venues marked
*(venue unverified)* were not confirmed from a proceedings/venue page. Theorem-level
statements were read from the arXiv HTML via a summarizing fetch — re-check the exact
constants against the PDF before quoting them in the thesis.

**Notation used below.** `n` = number of graph vertices, `N` = sequence length (tokens),
`m` = embedding dimension (width), `H` = number of heads, `p` = bits of precision,
`L` = depth (layers). "Width" in the transformer-theory literature almost always
means `m` (sometimes the product `mHp`); FFN width is usually abstracted away by
assuming the MLPs compute arbitrary token-wise functions.

**Taxonomy (framing).** Müller, Galkin, Morris, Rampášek, *Attending to Graph
Transformers*, TMLR 2024 — [arXiv:2302.04181](https://arxiv.org/abs/2302.04181)
(verified). Structure enters via tokens / positional encodings / attention bias /
interleaved message passing. Our codebase: tokens (vertex+edge+task, adjacency rows)
and attention bias (Graphormer-style SPD bias).

---

## 1. Theory: width for transformers on graphs

### 1.1 Sanford, Fatemi, Hall, Tsitsulin, Kazemi, Halcrow, Perozzi, Mirrokni (2024a)
*Understanding Transformer Reasoning Capabilities via Graph Algorithms.* NeurIPS 2024.
[arXiv:2405.18512](https://arxiv.org/abs/2405.18512) ·
[NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/file/8f395480c04ac6dfb2c2326a639df88e-Paper-Conference.pdf) — **verified**.

Width claims (precise):
- **MPC simulation (Thm 1):** any `R`-round MPC protocol with `N` machines and
  `O(N^δ)` bits of local memory each is simulated by a transformer of depth
  `L = O(R)` and embedding dimension `m = O(N^{δ+ε})`. So *width ↔ per-machine memory*,
  *depth ↔ rounds*.
- **Retrieval tasks** (node count, edge count, edge existence, node degree) — class
  `Depth1`: one layer, `m = O(log N)`.
- **Parallelizable tasks** — `LogDepth`: connectivity at `L = O(log N)`, `m = O(N^ε)`.
  Cycle check and bipartiteness sit in `LogDepthPause ∩ LogDepthWide` (need either
  pause tokens or extra width).
- **Search tasks** (shortest path, diameter) — `LogDepthWide`: `L = O(log N)` with
  `m = O(N^{1/2+ε})`.
- **Triangle counting:** `m = O(N^ε)`, `L = O(log log N)` (with pause tokens depending on
  arboricity).
- **Single-layer lower bound:** any 1-layer transformer solving connectivity, shortest
  path, or cycle detection needs `mH = Ω̃(N)` — i.e. *heads and width trade off
  multiplicatively*.
- **Depth lower bound** `L = Ω(log N)` for parallelizable tasks is **conditional** on the
  one-cycle-vs-two-cycle MPC conjecture (their Conjecture 13).
- Tokenization: vertex tokens + edge tokens + task token, `N = O(|V|+|E|)`.
- Empirics: GraphQA; 60M from-scratch transformer and fine-tuned T5-11B vs GNNs. Width
  is **not** swept systematically.

*Bearing:* the **framework**. Gives the width regimes `O(log N)` / `O(N^ε)` /
`O(N^{1/2+ε})` / `Ω̃(N)` that a width sweep can try to locate empirically.

### 1.2 Yehudai, Sanford, Bechler-Speicher, Fischer, Gilad-Bachrach, Globerson (2025)
*Depth-Width Tradeoffs in Algorithmic Reasoning of Graph Tasks with Transformers.*
ICML 2025 (per arXiv v3 header). [arXiv:2503.01805](https://arxiv.org/abs/2503.01805) —
**verified**.

Width claims (precise; tokenization = **adjacency rows**, `n` tokens of dim `n`,
precision `O(log n)`, MLPs arbitrary):
- **1-vs-2-cycle:** 2 layers, `m = O(n)` suffice (Thm 4.1). Sub-linear width
  `O(n^{1−ε})` at constant depth is ruled out **only conditionally** (Conjecture 13 of
  Sanford 2024a).
- **2-cycle detection (directed):** unconditional lower bound `mpHL = Ω(n)` with
  residuals, `mpH = Ω(n)` without (Thm 4.2, via set disjointness). Upper bound: `O(L)`
  layers of width `O(n)` compute `A^L` (Thm 4.3).
- **Bounded degree `d`:** 1 layer with `m = O(d log n)`; matching `mpHL = Ω(d)`.
- **k-subgraph counting:** `O(1)` layers, `m = O(n^{2−1/k})` (Thm 5.2).
- **Eulerian cycle verification:** conditionally needs near-quadratic width unless
  `L = Ω(log n)` (Thm 5.1).
- **Connectivity:** in the main-text theorems read in this sweep there is **no explicit
  "O(1) depth, O(n) width" theorem for connectivity** (the headline says constant
  depth suffices for "a host of" graph problems; connectivity appears in their
  experiments). Check the appendix before claiming it; state this carefully in the thesis.
- Empirics: at fixed ~100k params, (L, m) ∈ {(1,125),(2,89),(4,63),(8,45),(10,40)} —
  shallow-wide trains/infers faster with equal accuracy (connectivity, triangle,
  4-cycle count). **Critical width** (width at which training loss plateaus > 0.05) for
  4-cycle counting grows ~linearly with `n` (n = 50–400).

*Bearing:* the closest **competitor** and the source of the headline prediction
"linear width ⇒ constant depth". Their critical-width protocol is the template to
replicate/extend. Our own reproduction notes: `docs/yehudai-empirical.md`.

### 1.3 Sanford, Hsu, Telgarsky (2023)
*Representational Strengths and Limitations of Transformers.* NeurIPS 2023.
[arXiv:2306.02896](https://arxiv.org/abs/2306.02896) — **verified**.

Communication-complexity lower bounds on `m`. q-sparse averaging (qSA): one attention
layer with `m = Ω(d' + q log N)` (finite precision) approximates it, while `mp ≤ c·q`
makes it impossible (Thm 4) — **minimum width scales linearly in the sparsity q, only
logarithmically in N**. Match3 (triple detection): `mpH ≤ c·N / log log N` ⇒ one
multi-head layer cannot compute it (Thm 7), while pairwise Match2 needs `m = 3`.
Precision and heads enter multiplicatively with `m`.

*Bearing:* **framework** — the "width × precision × heads" currency; qSA is the
prototype for retrieval-style graph queries (look up q neighbours).

### 1.4 Sanford, Hsu, Telgarsky (2024b)
*Transformers, Parallel Computation, and Logarithmic Depth.* ICML 2024 (PMLR v235).
[arXiv:2402.09268](https://arxiv.org/abs/2402.09268) — **verified**.

Constant layers ↔ constant MPC rounds (both directions); k-hop pointer chasing in
`O(log k)` depth. The MPC local memory is what the width pays for.

*Bearing:* **framework** underlying 1.1.

### 1.5 Sanford, Hsu, Telgarsky (2024c)
*One-layer transformers fail to solve the induction heads task.* arXiv 2024
*(venue unverified)*. [arXiv:2408.14332](https://arxiv.org/abs/2408.14332) — **verified**.

A 1-layer transformer needs size exponentially larger than a 2-layer one for induction
heads (communication complexity). *Bearing:* a clean example of "width cannot cheaply
replace one layer" — the counter-pole to Yehudai's "width replaces many layers".

### 1.6 Peng, Narayanan, Papadimitriou (2024)
*On Limitations of the Transformer Architecture.* arXiv 2024 *(venue unverified)*.
[arXiv:2402.08164](https://arxiv.org/abs/2402.08164) — **verified**.

One attention layer cannot compute function composition (e.g. grandparent lookup)
whenever `n log n > H(m+1)p` (domain size `n`). *Bearing:* **framework** — two-hop
lookup = 2-step path query on a graph; gives a concrete 1-layer width threshold
`H·m·p ≈ n log n` testable on a "neighbour-of-neighbour" task.

### 1.7 Chen, Peng, Wu (2024/25)
*Theoretical Limitations of Multi-layer Transformer.* FOCS 2025 (per search result).
[arXiv:2412.02975](https://arxiv.org/abs/2412.02975) — **verified**.

First **unconditional** multi-layer lower bound: an `L`-layer *decoder-only*
transformer needs model dimension `n^{Ω(1)}` to compose `L` functions; `L` vs `L+1`
layers is an exponential gap. *Bearing:* **framework** for depth–width in decoder
(AR-CoT) settings; relevant to our autoregressive mode.

### 1.8 Merrill, Sabharwal (2025)
*A Little Depth Goes a Long Way: The Expressive Power of Log-Depth Transformers.*
NeurIPS 2025 (per arXiv comment). [arXiv:2503.03961](https://arxiv.org/abs/2503.03961) —
**verified**.

Graph connectivity by a *fixed-width* universal transformer unrolled `⌈log₂ n⌉` times
(`p = c log n`). Fixed-depth transformers with polynomial width stay in TC⁰, so for
problems believed outside TC⁰ (undirected connectivity is L-complete, directed
reachability NL-complete) **width must grow super-polynomially** at fixed depth,
under standard conjectures (the paper cites TC⁰ ≠ NC¹). "Depth scaling is more
efficient than scaling width or CoT."

*Bearing:* **tension with 1.2** — resolved by the input model: Yehudai's adjacency-row
tokens hand each token `n` bits, and their constant-depth constructions assume
arbitrary MLPs; Merrill–Sabharwal count uniform circuits of polynomial size. The thesis
should state which regime its models live in.

### 1.9 Merrill, Sabharwal (2023)
*The Parallelism Tradeoff: Limitations of Log-Precision Transformers.* TACL 2023.
[arXiv:2207.00729](https://arxiv.org/abs/2207.00729) — **verified**.

Log-precision transformers ⊆ uniform TC⁰. *Bearing:* precision–width interplay: all
width bounds above are stated at `p = O(log n)`; with float32 and small `n` precision
is not the binding constraint, but in theory `m·p` is the resource.

### 1.10 Bechler-Speicher, Yehudai, Harari, Sanford, Globerson, Bruna (2026)
*Lost in Tokenization: Fundamental Trade-offs in Graph Tokenization for Transformers.*
arXiv 2026. [arXiv:2605.22471](https://arxiv.org/abs/2605.22471) — **verified (abstract only)**.

Random-walk tokenization is lossy; spectral is complete but ill-conditioned for local
tasks; adjacency is the baseline. Tokenizations induce different **depth regimes**, and
shallow transformers cannot cheaply convert between them. *Bearing:* **competitor /
framework** — width results are tokenization-dependent; our vertex+edge tokens vs
adjacency rows are different regimes.

### 1.11 Other graph-transformer theory touched by width
- Kim et al. *Pure Transformers are Powerful Graph Learners (TokenGT).* NeurIPS 2022.
  [arXiv:2207.02505](https://arxiv.org/abs/2207.02505) — verified. 2-IGN expressivity via
  node+edge tokens with **orthonormal node identifiers** `P ∈ ℝ^{n×d_p}`; orthonormal rows
  imply `d_p ≥ n` (our inference, not a stated theorem), and the construction uses
  15 = bell(4) heads. *Bearing:* tokens-only design pays for expressivity in width
  (identifier dim) and head count.
- Zhang, Luo, Wang, He. *Rethinking the Expressive Power of GNNs via Graph
  Biconnectivity.* ICLR 2023 (outstanding paper). [arXiv:2301.09505](https://arxiv.org/abs/2301.09505)
  — verified. Distance-based attention bias (GD-WL, Graphormer-style) decides
  biconnectivity. *Bearing:* for the SPD-bias variant, much of the global structure is
  precomputed, so the width needed for connectivity may collapse — a confound to
  report, not a result about width.

---

## 2. Width results for GNNs / message passing that transfer

### 2.1 Loukas (2020a)
*What Graph Neural Networks Cannot Learn: Depth vs Width.* ICLR 2020.
[arXiv:1907.03199](https://arxiv.org/abs/1907.03199) ·
[OpenReview](https://openreview.net/forum?id=B112bp4YwS) — **verified**.

Via CONGEST-model reductions: MPNNs are Turing-universal given enough depth and width,
but several decision/optimization/estimation problems are impossible unless
**depth × width exceeds a polynomial in n**, even for tasks that look simple
(abstract-level, verified). The per-problem bounds (which problems, which exponents)
were **not** checked in this sweep — read the paper's table before citing specifics. *Bearing:* the
MPNN analogue of Sanford's `mH`/`mpHL` bounds; the obvious baseline for a GNN-vs-GT
width comparison.

### 2.2 Loukas (2020b)
*How hard is to distinguish graphs with graph neural networks?* NeurIPS 2020.
[arXiv:2005.06649](https://arxiv.org/abs/2005.06649) — **verified**.

"Communication capacity" (depth, message size, global state, width) must grow
linearly in n to distinguish trees and quadratically for general graphs; **empirics
show accuracy tracks capacity**. *Bearing:* **method** — a single scalar capacity that
predicted trained-model failure; we can define the transformer analogue (`m·p·H·L`) and
test whether accuracy collapses onto one curve.

### 2.3 Aamand et al. (2022)
*Exponentially Improving the Complexity of Simulating the Weisfeiler-Lehman Test with
GNNs.* NeurIPS 2022. [arXiv:2211.03232](https://arxiv.org/abs/2211.03232) — **verified**.

WL simulation needs only `O(log n)`-bit messages and polylog(n)-parameter combine
functions, with matching log lower bounds. *Bearing:* **framework** — WL-level tasks
(node degree, local counts) should need only logarithmic width; separates "local" from
"global" width demand.

### 2.4 Di Giovanni, Giusti, Barbero, Luise, Liò, Bronstein (2023)
*On Over-squashing in Message Passing Neural Networks: The Impact of Width, Depth, and
Topology.* ICML 2023. [arXiv:2302.02941](https://arxiv.org/abs/2302.02941) — **verified**.

Width can mitigate over-squashing but makes the network more sensitive overall; depth
does not help (vanishing gradients); topology (commute time) dominates. *Bearing:*
**framework** for MPNN baselines and the interleaved-MP (GPS) class; predicts
diminishing/unstable returns from width.

---

## 3. Empirical studies that vary width in graph transformers

### 3.1 Yehudai et al. 2025 — see 1.2
Only paper found with a **systematic width sweep on graph algorithmic tasks**
(critical width vs n; iso-parameter depth/width grid). Tasks: connectivity, triangle
and 4-cycle counting; adjacency-row tokenization only; single architecture.

### 3.2 Sypetkowski et al. (2024)
*On the Scalability of GNNs for Molecular Graphs (MolGPS).* NeurIPS 2024.
[arXiv:2404.11568](https://arxiv.org/abs/2404.11568) — **verified**.

Scales width and depth of MPNN++, a pure Transformer and GPS++ to 1–3B params on
molecular pretraining. **Width scaling works** (strong downstream Spearman correlations,
Transformer strongest at large scale; MPNN++ more parameter-efficient at small scale);
**depth scaling does not** (plateau at 8–16 layers). Width handled with **μP**, depth
with depth-μP. *Bearing:* **method** precedent — μP is already used for graph
transformers; also a competitor datapoint, but on property prediction, not algorithms.

### 3.3 Liu et al. (2024)
*Towards Neural Scaling Laws on Graphs.* LoG 2024 (PMLR v269).
[arXiv:2402.02054](https://arxiv.org/abs/2402.02054) — **verified**.

Models up to 100M params: beyond parameter count, **depth** materially changes model
scaling (unlike NLP/CV). *Bearing:* warns that "params" is not a sufficient axis —
report width and depth separately.

### 3.4 Rampášek et al. (2022) GraphGPS; Kim et al. (2022) TokenGT; Ying et al. (2021) Graphormer
- GraphGPS, NeurIPS 2022, [arXiv:2205.12454](https://arxiv.org/abs/2205.12454) —
  verified: **no width/head ablation**; fixed budgets (~500k params for ZINC/LRGB, ~100k
  for MNIST/CIFAR10), width chosen by line search.
- TokenGT (above): no hidden-dim/head ablation; uses Graphormer's 12 layers / 768 / 32 heads.
- Graphormer, *Do Transformers Really Perform Bad for Graph Representation?*, NeurIPS
  2021, [arXiv:2106.05234](https://arxiv.org/abs/2106.05234) — verified existence; model
  sizes reported in paper, no width ablation found in this sweep (not checked in full text).

*Bearing:* confirms the **gap** — the standard graph-transformer papers fix width by
budget convention and never isolate it.

### 3.5 Tönshoff, Ritzert, Rosenbluth, Grohe
*Where Did the Gap Go? Reassessing the Long-Range Graph Benchmark.* TMLR 2024 (also
LoG 2023). [arXiv:2309.00367](https://arxiv.org/abs/2309.00367) — **verified**.

The reported GT-over-MPNN gap on LRGB vanishes after basic hyperparameter tuning.
*Bearing:* **method/pitfall** — any width effect must survive per-width tuning.

### 3.6 Zhao et al. (2023)
*Are More Layers Beneficial to Graph Transformers?* ICLR 2023.
[arXiv:2303.00579](https://arxiv.org/abs/2303.00579) — **verified**.

Deep GTs hit "vanishing capacity of global attention"; DeepGraph fixes it with
substructure tokens. *Bearing:* depth counterpart to the thesis; motivates width as
the more reliable scaling axis for GTs.

### 3.7 Saparov et al. (2025)
*Transformers Struggle to Learn to Search.* ICLR 2025.
[arXiv:2412.04703](https://arxiv.org/abs/2412.04703) — **verified**.

Graph search gets harder to learn as graphs grow; "increasing model scale will not lead
to robust search abilities"; mechanism: per-vertex reachable sets expanded layer by
layer (exactly the pointer-doubling picture). *Bearing:* **competitor** — directly
relevant to search-style tasks; a width sweep can test whether width (not total scale)
changes this.

### 3.8 Roy, Saparov (2025) — **withdrawn**
*Transformers Can Learn Connectivity in Some Graphs but Not Others.*
[arXiv:2509.22343](https://arxiv.org/abs/2509.22343) — verified, **withdrawn v2 (2026-04)**.
Reported that larger models help on grid graphs but data scale beats model scale.
Cite only as withdrawn, if at all.

### 3.9 Bhalla, Fan, Chen, Yu (2025)
*Higher Embedding Dimension Creates a Stronger World Model for a Simple Sorting Task.*
arXiv. [arXiv:2510.18315](https://arxiv.org/abs/2510.18315) — **verified**.

Small `m` already reaches high accuracy, but larger `m` yields more faithful/robust
internal representations. *Bearing:* **method** — measure width effects beyond
accuracy (probe quality, robustness, OOD/size generalization).

---

## 4. General (non-graph) width methodology

- **Yang et al. Tensor Programs V (μTransfer).** NeurIPS 2021.
  [arXiv:2203.03466](https://arxiv.org/abs/2203.03466) — verified. Under μP the optimal LR
  (and init/multipliers) are stable across width; tune small, transfer zero-shot.
  *Method:* the principled way to sweep width without retuning each point.
- **Yang, Yu, Zhu, Hayou. Tensor Programs VI (Depth-μP).** ICLR 2024.
  [arXiv:2310.02244](https://arxiv.org/abs/2310.02244) — verified. Depthwise transfer for
  resnets; limitations for multi-layer blocks (as in transformers). *Method:* only
  needed if depth is co-varied.
- **Everett et al. Scaling Exponents Across Parameterizations and Optimizers.** ICML 2024.
  [arXiv:2407.05872](https://arxiv.org/abs/2407.05872) — verified. Several
  parameterizations (not only μP) transfer; per-layer LR in standard param. can beat μP;
  **Adam ε must scale with width** (or use Adam-atan2). *Method:* a cheap alternative to
  full μP.
- **Lingle. An Empirical Study of μP Learning Rate Transfer.** arXiv 2024.
  [arXiv:2404.05728](https://arxiv.org/abs/2404.05728) — verified. μ-Transfer works in
  most transformer settings; documents failure cases. *Method:* sanity-check transfer.
- **Wortsman et al. Small-scale Proxies for Large-scale Transformer Training
  Instabilities.** ICLR 2024 (oral). [arXiv:2309.14322](https://arxiv.org/abs/2309.14322) —
  verified. Attention-logit growth and output-logit divergence reproduce in small
  models at high LR; qk-layernorm etc. fix them; "LR sensitivity" metric.
  *Method:* report LR sensitivity per width.
- **Kaplan et al. Scaling Laws for Neural Language Models.** arXiv 2020 (OpenAI).
  [arXiv:2001.08361](https://arxiv.org/abs/2001.08361) — verified. Width/depth have
  "minimal effects within a wide range" at fixed params (for LM loss). *Competitor
  null hypothesis* for algorithmic tasks.
- **Levine et al. The Depth-to-Width Interplay in Self-Attention.** NeurIPS 2020.
  [arXiv:2006.12467](https://arxiv.org/abs/2006.12467) — verified. Width-dependent
  transition between depth-efficiency and depth-inefficiency (validated at depths
  6–48); argues GPT-3 is "too deep for its size". *Framework* for iso-parameter grids.
- **Tay et al. Scale Efficiently.** ICLR 2022. [arXiv:2109.10686](https://arxiv.org/abs/2109.10686)
  — verified. Shape matters downstream even when upstream loss is shape-insensitive.
- **Petty et al. The Impact of Depth on Compositional Generalization.** NAACL 2024.
  [arXiv:2310.19956](https://arxiv.org/abs/2310.19956) — verified. Iso-parameter
  families (41M/134M/374M) varying depth vs width; depth helps with fast-diminishing
  returns. *Method template* for iso-parameter comparisons.
- **Nakkiran et al. Deep Double Descent.** arXiv 2019 (ICLR 2020; venue not re-checked).
  [arXiv:1912.02292](https://arxiv.org/abs/1912.02292) — verified. Model-wise (width)
  double descent — test error can be non-monotone in width, especially with label noise
  and small data. *Pitfall.*

---

## 5. Heads vs embedding size as distinct "width" notions

- **Bhojanapalli, Yun, Rawat, Reddi, Kumar. Low-Rank Bottleneck in Multi-head Attention
  Models.** ICML 2020. [arXiv:2002.07028](https://arxiv.org/abs/2002.07028) — verified.
  With head size `m/H`, adding heads at fixed `m` lowers each head's rank and limits
  expressivity; decouple head size from `H` (set it ≥ sequence length).
- **Amsel, Yehudai, Bruna. On the Benefits of Rank in Attention Layers.** arXiv 2024
  *(venue unverified)*. [arXiv:2407.16153](https://arxiv.org/abs/2407.16153) — verified.
  Functions representable by one full-rank head need exponentially many (in `m`)
  low-rank heads; depth partially compensates.
- **Yu, Jiang, Bao, Yu, Li. The Effect of Attention Head Count on Transformer
  Approximation.** ICLR 2026 (per arXiv). [arXiv:2510.06662](https://arxiv.org/abs/2510.06662)
  — verified. Few heads ⇒ parameters must scale like `O(1/ε^{cT})`; single head can
  memorize with `m = O(T)`, pushing work into the FFN.
- **Michel, Levy, Neubig. Are Sixteen Heads Really Better than One?** NeurIPS 2019.
  [arXiv:1905.10650](https://arxiv.org/abs/1905.10650) — verified. Many heads prunable at
  test time — trained head count ≠ used head count.
- **In the graph theory above,** heads appear as `mH` (Sanford 2024a single-layer),
  `mpH(L)` (Yehudai; Sanford 2023 Match3), and `H(m+1)p` (Peng et al.). The theory
  treats `m` and `H` as **interchangeable in the product**, whereas the practical
  literature (Bhojanapalli, Amsel) says they are **not** interchangeable at fixed `m`
  because head dim = `m/H`. This mismatch is directly testable.

---

## 6. Added from the 2026-10-08 novelty check

Source: `docs/novelty-check-2026-10-08.md` (Claude Research run on findings 1–7). Its verdict:
the main result — critical width to *learn* connectivity ∝ n^≈1.9 — was **not found** in
prior work. Status per entry: **verified** = checked against the PDF in `papers/`;
**report-opened** = the research run opened the paper, we did not; **snippet** = seen only
as a search excerpt. Verify the last two before citing specifics.

### 6.1 Ye, Fu, Jia, Sharan (2026)
*Transformers Provably Learn Algorithmic Solutions for Graph Connectivity, But Only with the
Right Data* (v1: *When Do Transformers Learn Heuristics for Graph Connectivity?*).
[arXiv:2510.19753](https://arxiv.org/abs/2510.19753) — **verified** (`papers/`, notes in
`reports/ye-2026-notes.typ`).
**Source of our task**: the connectivity-matrix target, (A + I) adjacency-row tokens and the
bilinear read-out were adopted from this paper in June 2026. An L-layer Disentangled
Transformer (non-negative weights) reaches diameter ≤ 3^L, tightly; within-capacity graphs
drive the algorithmic (matrix-powering) solution, beyond-capacity graphs a degree-product
heuristic. Trained on 10^9 ER(n=20) graphs, AdamW lr 1e-4 (single rate), d = 512.
*Bearing:* **must cite** as the task's origin; 3^L capacity explains Q7a (n=40, diameter
≈ 10: two layers give 9) — though their bound is for the restricted architecture and our
standard 2-layer model still reaches 0.99 pair accuracy.

### 6.2 Abbe, Bengio, Lotfi, Sandon, Saremi (2024)
*How Far Can Transformers Reason? The Globality Barrier and Inductive Scratchpad.* NeurIPS
2024. [arXiv:2406.06467](https://arxiv.org/abs/2406.06467) — **report-opened**.
"Cycle task": two disjoint n-cycles vs one 2n-cycle, built so no degree / edge-count / motif
statistic helps (globality ≥ n), proposed as a *training* benchmark; learning cost grows
exponentially with n at fixed model sizes (they vary total parameters, not width).
*Bearing:* **must cite** as the origin of the locally-indistinguishable benchmark idea
behind `swap` (ours adds chords, a dense pair target, and explicit audits).

### 6.3 Yehudai et al. 2025 — details we had not recorded (see 1.2)
**Verified** (`papers/`): §6.1 compares fixed-100k-parameter (depth, width) pairs (1,125),
(2,89), (4,63), (8,45), (10,40) — similar connectivity accuracy across depth, on a
mixed-generator graph-level dataset; LR tuned only in {1e-4, 5e-5}; datasets of 5000 graphs.
§6.2 critical width = training loss plateau > 0.05 (a *fitting* criterion), 1 layer, 2
heads, counting tasks, "increases roughly linearly with the graph size". Notes that
"quadratic width should suffice for solving any task, since it can be used to record the
entire graph". *Bearing:* their width study is exposed to our Q1 (fixed LR) and Q1c
(small-data) confounds; our ≈ n² sits ~n above their linear bounds; n² = adjacency bits
suggests a "copy the whole graph" solution — testable by probing (next steps).
Venue: the novelty check says NeurIPS 2025 (spotlight); 1.2 says ICML 2025 — check.

### 6.4 Mahdavi et al. (2023)
*Towards Better Out-of-Distribution Generalization of Neural Algorithmic Reasoning Tasks.*
TMLR 2023. [arXiv:2211.00692](https://arxiv.org/abs/2211.00692) — **report-opened**.
In CLRS, node indices act as unique flags and models exploit spurious index correlations.
*Bearing:* closest prior to our node-index leak (ours: generator indices reveal component
membership — not found elsewhere).

### 6.5 Wang et al. (2023) NLGraph
*Can Language Models Solve Graph Problems in Natural Language?* NeurIPS 2023.
[arXiv:2305.10037](https://arxiv.org/abs/2305.10037) — **snippet**. Prompted LLMs use node
mention frequency for connectivity. *Bearing:* degree-style shortcuts, LLM setting.

### 6.6 DeZoort, Hanin (2026)
*Hyperparameter Transfer in Graph Neural Networks.* [arXiv:2607.05017](https://arxiv.org/abs/2607.05017)
— **report-opened**. µP/CompleteP-style parameterisation for GNNs; Adam rate ∝ 1/√width.
*Bearing:* GNN-level support for Q1 (optimal LR shrinks with width); not transformers or
reasoning tasks.

### 6.7 Learning-onset analogues
Barak et al. 2022, *Hidden Progress in Deep Learning* (NeurIPS 2022,
[arXiv:2207.08799](https://arxiv.org/abs/2207.08799)) — plateau-then-onset learning;
Edelman et al. 2023, *Pareto Frontiers in Deep Feature Learning: Data, Compute, Width, and
Luck* (NeurIPS 2023) — width trades against data and time for sparse parity. Both
**snippet**. *Bearing:* mechanism analogues for Q3 trim ("width decides whether learning
starts").

### 6.8 Tensions to address in the thesis
- Saparov et al. 2025 (3.7): no clear link between model size and time-to-learn at fixed
  graph size (**snippet** quote) — vs our width-gated onset; different task (search,
  edge lists, widths ≤ 16).
- Merrill & Sabharwal 2025 (1.8): depth Θ(log n) suffices where width must grow
  superpolynomially (regular languages) — vs our mild depth benefit for connectivity.
- Sanford–Hsu–Telgarsky 2024 App. G (**snippet**): on k-hop, doubling width ≈ +1 layer.

## (a) Defensible gap

1. **No systematic empirical width study of graph transformers exists on algorithmic
   graph tasks across structure-injection types.** The only width sweep on graph
   algorithms (Yehudai 2025) uses one tokenization (adjacency rows, where each token
   already carries `n` bits), one architecture, and substructure counting for its
   critical-width curve; connectivity is only in the iso-parameter depth/width grid.
   GraphGPS/TokenGT/Graphormer never ablate width; MolGPS/Liu scale width but on
   molecular/property benchmarks with no link to the theory.
2. **Nobody has checked whether the theoretical width regimes are visible in trained
   models:** retrieval at `O(log N)`, connectivity at `O(N^ε)` with log depth, search at
   `O(N^{1/2+ε})`, single-layer `mH = Ω̃(N)`. The hierarchy of Sanford 2024a is
   representational; learnability of each width regime is open.
3. **Tokens-only vs attention-bias** (our two model classes) are not compared on width
   demand anywhere. The SPD bias precomputes global structure, so the width needed for
   connectivity should drop — quantifying that "width saved by structural bias" is new.
4. **`m` vs `H` vs FFN width** as separate axes on graph tasks: theory uses the product
   `mH`, practice says head dim matters — untested on graphs.

Suggested thesis claim: *"We measure the critical width of graph transformers as a
function of graph size, task class (retrieval / parallelizable / search), and
structure-injection mechanism (tokens vs attention bias), under width-consistent
hyperparameter transfer, and compare the measured scaling exponents to the
representational bounds of Sanford et al. 2024a and Yehudai et al. 2025."*

## (b) Testable predictions

1. **Task-class ordering of critical width** (Sanford 2024a): at fixed depth,
   `m*_retrieval < m*_connectivity < m*_shortest-path`. Retrieval critical width should
   grow at most ~log n; search the fastest.
2. **Single-layer threshold:** at `L = 1`, the critical `m·H` for connectivity / cycle
   check should grow ~linearly in `N` (≈ |V|+|E|); plot `m*H` vs N on log-log, expect
   slope ≈ 1 (up to log factors).
3. **Depth–width substitution:** for connectivity, the iso-accuracy frontier in (L, m)
   space should show width substituting for depth, with depth `~log n` at small width
   and `O(1)` depth once `m ≳ n` (adjacency-row tokens). Where the frontier bends is an
   empirical estimate of the MPC memory–rounds tradeoff.
4. **Tokenization dependence** (Lost in Tokenization; Yehudai): adjacency-row tokens
   should need less width than vertex+edge tokens at equal depth for global tasks.
5. **Attention bias lowers width demand:** Graphormer-style SPD bias should make
   connectivity's critical width ~flat in n (the bias already encodes reachability:
   finite SPD ⇔ same component), i.e. collapse connectivity into the retrieval class.
6. **Heads vs dimension:** at fixed `m·H` product, theory predicts equal capacity;
   Bhojanapalli predicts worse performance once head dim `m/H` drops below what the task
   needs. Sweep H at fixed m and m at fixed H.
7. **Capacity collapse** (Loukas 2020b analogue): accuracy vs `m·H·L` (or `m·p·H·L`)
   should collapse runs of different shapes onto one curve for 2-cycle-type tasks if
   the communication-complexity bound is the operative constraint.
8. **Width helps learnability beyond expressivity:** even above the representational
   threshold, larger m should reduce epochs-to-solve and seed variance (Yehudai's
   training-speed observation; Bhalla et al. representation quality).

## (c) Methodological pitfalls for width sweeps

1. **Learning rate must be re-tuned or transferred.** Under standard parameterization
   the optimal Adam LR shrinks with width; a fixed LR confounds width with
   under/over-stepping. Use μP (Yang TP V; precedent in MolGPS) or per-width LR sweeps
   (Everett: per-layer LR, scale Adam ε). Report LR-sensitivity curves (Wortsman).
2. **Iso-parameter vs iso-depth sweeps answer different questions.** State which one;
   report width, depth, heads, FFN ratio, and param count separately (Liu; Petty;
   Levine).
3. **Head dim confound.** Growing `m` at fixed `H` grows head dim; growing `H` at fixed
   `m` shrinks it. Fix one and report both (Bhojanapalli).
4. **Input-dimension bottleneck.** With adjacency-row tokens the input already has
   dimension `n`; the first projection to `m < n` is itself a width bottleneck, so
   "width" effects may reflect compression of the input, not attention capacity.
5. **Shortcut leakage.** Connectivity can be solved by local degree statistics on naive
   generators (see our dataset-leak note); a width curve on a leaky dataset measures
   the shortcut, not global reasoning. Use degree-matched positives/negatives.
6. **Tuning parity.** Width effects must survive per-configuration tuning of the
   baseline (Tönshoff et al.); report tuning budget per width point.
7. **Seeds and non-monotonicity.** Critical width is a threshold phenomenon; use ≥3–5
   seeds, report success probability vs width, not a single run. Model-wise double
   descent (Nakkiran) can make test error non-monotone in width with small data.
8. **Critical-width definition sensitivity.** Yehudai's "train loss plateaus > 0.05"
   is arbitrary; show robustness to the threshold and to training budget (a wider
   model may merely train faster within a fixed epoch budget).
9. **Size generalization vs in-distribution.** Theory is about fixed n; test both
   in-distribution and larger-n accuracy, since extra width may memorise size-specific
   solutions (Saparov: scale does not buy robust search).
10. **Precision is part of width in theory** (`m·p`); keep dtype fixed (fp32) across
    the sweep and note that mixed precision changes the effective resource.
11. **Attention-logit growth at large width/LR** (Wortsman) — use qk-norm or warm-up
    consistently across all widths, or instabilities will masquerade as a width effect.
