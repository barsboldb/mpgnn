# Papers

Reading PDFs backing the diploma. One paper per file, named `<first-author>-<year>-<short-slug>.pdf`
so it lines up with the notes convention in `reports/` (`kosowski-2025-notes.typ`, `ye-2026-notes.typ`).

The PDFs themselves are **gitignored** (see `.gitignore`) — they are large binaries and mostly
third-party publisher files. This index is tracked, so the bibliography is version-controlled and
anyone can re-fetch the set from the links below.

## Index

| File | Paper | Venue | Notes | Why it's here |
|---|---|---|---|---|
| `airale-2025-simple-path-structural-encoding.pdf` | Airale, Longa, Rigon, Passerini, Passerone — *Simple Path Structural Encoding for Graph Transformers* | ICML 2025 (PMLR 267); [arXiv:2502.09365v2](https://arxiv.org/abs/2502.09365) | [`reports/airale-2025-notes.typ`](../reports/airale-2025-notes.typ) | Edge structural encoding for graph transformers: simple-path counts (SPSE) as a replacement for RWSE. Directly relevant to `GlobalAttnConv`'s `spd_bias` — SPSE is the richer alternative to our shortest-path-distance bucket bias. |
| `abbe-2023-generalization-unseen-degree-curriculum.pdf` | Abbe, Bengio, Lotfi, Rizk — *Generalization on the Unseen, Logic Reasoning and Degree Curriculum* | ICML 2023 (outstanding paper); extended in JMLR 25 (2024); [arXiv:2301.13105v3](https://arxiv.org/abs/2301.13105) (the JMLR version, 58 pp.) | none | Boolean-function learning: on unseen inputs, GD-trained models drift to a min-degree interpolator, and a degree curriculum helps learn high-degree targets. Background for the locality/globality line that the `swap` shortcut audit draws on (the cycle task itself is Abbe et al. NeurIPS 2024). |
| `ye-2026-transformers-graph-connectivity.pdf` | Ye, Fu, Jia, Sharan — *Transformers Provably Learn Algorithmic Solutions for Graph Connectivity, But Only with the Right Data* | Preprint Feb 2026; [arXiv:2510.19753v2](https://arxiv.org/abs/2510.19753) | [`reports/ye-2026-notes.typ`](../reports/ye-2026-notes.typ) | The depth↔diameter capacity bound ($3^L$) and the **data lever**: training on beyond-capacity graphs makes GD abandon matrix-powering for a degree heuristic. The governing reference for our `connectedness_hard` / `diameter_controlled` experiments. |

| `sanford-2023-representational-strengths-limitations.pdf` | Sanford, Hsu, Telgarsky — *Representational Strengths and Limitations of Transformers* | NeurIPS 2023; [arXiv:2306.02896](https://arxiv.org/abs/2306.02896) | `reports/qsa-sparse-averaging.typ` (q-sparse averaging explainer) | The width lower-bound toolkit: q-sparse averaging needs one-layer width m ≈ q (m·p ≥ cq via set disjointness), Match3 brings in heads (m·p·H). The "width × precision × heads" currency the thesis measures. |
| `sanford-2024-transformer-reasoning-graph-algorithms.pdf` | Sanford, Fatemi, Hall, Tsitsulin, Kazemi, Halcrow, Perozzi, Mirrokni — *Understanding Transformer Reasoning Capabilities via Graph Algorithms* | NeurIPS 2024; [arXiv:2405.18512v1](https://arxiv.org/abs/2405.18512) | **none yet** — biggest literature hole (`docs/RESEARCH-QUESTIONS.md` A7) | The foundation the thesis builds on: the D1/LD/LDW/LDP hierarchy, depth↔MPC-rounds (Thm 1), connectivity as *parallelizable* at $L=O(log N)$, and the vertex+edge+task tokenization our `node_edge` mode copies. |

| `yehudai-2025-depth-width-tradeoffs-graph-tasks.pdf` | Yehudai, Sanford, Bechler-Speicher, Fischer, Gilad-Bachrach, Globerson — *Depth-Width Tradeoffs for Transformers on Graph Tasks* | NeurIPS 2025; [arXiv:2503.01805v3](https://arxiv.org/abs/2503.01805) (arXiv title differs: *…Tradeoffs in Algorithmic Reasoning of Graph Tasks with Transformers*) | `reports/yehudai-2025-notes.typ` (reading companion, checked against the thesis); `docs/yehudai-empirical.md` (reproduction) | The assigned paper, and the direct sequel to [[sanford-2024]] — Sanford is a co-author. Answers the width question Sanford left open: with **linear** width, **constant** depth suffices; some tasks need quadratic width. Their supplementary code is vendored in `yehudai/`. |

| `wang-2023-nlgraph.pdf` | Wang, Feng, He, Tan, Han, Tsvetkov — *Can Language Models Solve Graph Problems in Natural Language?* (NLGraph) | NeurIPS 2023; [arXiv:2305.10037v3](https://arxiv.org/abs/2305.10037) | `docs/shortcut-literature-2026-10-13.md` | The closest precedent to our shortcut audit: §4.4 / Table 4 build "chain" and "clique" connectivity splits that invert the degree cue, and accuracy drops (78.4 → 54.8 / 71.0 mean). They suspect LLMs "just count node mentions". |
| `zhang-2023-paradox-learning-reason.pdf` | Zhang, Li, Meng, Chang, Van den Broeck — *On the Paradox of Learning to Reason from Data* | IJCAI 2023; [arXiv:2205.11502](https://arxiv.org/abs/2205.11502) | `docs/shortcut-literature-2026-10-13.md` | The "representable but not learned" template: BERT can provably solve SimpleLogic (rule-graph reachability) but training finds statistical features instead; removing all such features at once is infeasible — so `swap` is shortcut-free only relative to a declared feature family. |
| `bechler-speicher-2024-gnns-use-graphs.pdf` | Bechler-Speicher, Amos, Gilad-Bachrach, Globerson — *Graph Neural Networks Use Graphs When They Shouldn't* | ICML 2024; [arXiv:2309.04332v2](https://arxiv.org/abs/2309.04332) | `docs/shortcut-literature-2026-10-13.md` | The inverse shortcut (overfitting uninformative structure) and the graph-less ablation (GNN vs GNN∅) — one more check for the audit protocol. |
| `saparov-2025-transformers-struggle-search.pdf` | Saparov, Pawar, Pimpalgaonkar, Joshi, Pang, Padmakumar, Kazemi, Kim, He — *Transformers Struggle to Learn to Search* | ICLR 2025; [arXiv:2412.04703](https://arxiv.org/abs/2412.04703) | `docs/shortcut-literature-2026-10-13.md` | Graph search is learnable only from a lookahead-balanced distribution, and more parameters don't fix it on larger graphs — the nearest empirical result to our representation-vs-learning gap (but it doesn't separate width from depth). |

## Status

- **Reading now:** `airale-2025-simple-path-structural-encoding` (started 2026-08-26)
- **Read, notes written:** `ye-2026-transformers-graph-connectivity`
- **Read, notes written:** `sanford-2024-transformer-reasoning-graph-algorithms`
- **Read, notes written:** Merrill & Sabharwal 2023 (arXiv 2207.00729) and 2025 (arXiv 2503.03961)
  (2026-10-22; `reports/merrill-sabharwal-notes.typ`; PDFs not stored here)
- **Read, notes written:** `yehudai-2025-depth-width-tradeoffs-graph-tasks` (2026-10-22;
  `reports/yehudai-2025-notes.typ`, with a check of every thesis claim about it)

## Adding a paper

```sh
curl -sL -o papers/<first-author>-<year>-<slug>.pdf <pdf-url>
```

Then add a row above. If the paper earns a full write-up, put it in `reports/<first-author>-<year>-notes.typ`
and link it from the Notes column.
