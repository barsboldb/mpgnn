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
| `ye-2026-transformers-graph-connectivity.pdf` | Ye, Fu, Jia, Sharan — *Transformers Provably Learn Algorithmic Solutions for Graph Connectivity, But Only with the Right Data* | Preprint Feb 2026; [arXiv:2510.19753v2](https://arxiv.org/abs/2510.19753) | [`reports/ye-2026-notes.typ`](../reports/ye-2026-notes.typ) | The depth↔diameter capacity bound ($3^L$) and the **data lever**: training on beyond-capacity graphs makes GD abandon matrix-powering for a degree heuristic. The governing reference for our `connectedness_hard` / `diameter_controlled` experiments. |

| `sanford-2024-transformer-reasoning-graph-algorithms.pdf` | Sanford, Fatemi, Hall, Tsitsulin, Kazemi, Halcrow, Perozzi, Mirrokni — *Understanding Transformer Reasoning Capabilities via Graph Algorithms* | NeurIPS 2024; [arXiv:2405.18512v1](https://arxiv.org/abs/2405.18512) | **none yet** — biggest literature hole (`docs/RESEARCH-QUESTIONS.md` A7) | The foundation the thesis builds on: the D1/LD/LDW/LDP hierarchy, depth↔MPC-rounds (Thm 1), connectivity as *parallelizable* at $L=O(log N)$, and the vertex+edge+task tokenization our `node_edge` mode copies. |

| `yehudai-2025-depth-width-tradeoffs-graph-tasks.pdf` | Yehudai, Sanford, Bechler-Speicher, Fischer, Gilad-Bachrach, Globerson — *Depth-Width Tradeoffs for Transformers on Graph Tasks* | NeurIPS 2025; [arXiv:2503.01805v3](https://arxiv.org/abs/2503.01805) (arXiv title differs: *…Tradeoffs in Algorithmic Reasoning of Graph Tasks with Transformers*) | `docs/yehudai-empirical.md` (reproduction, not a reading companion) | The assigned paper, and the direct sequel to [[sanford-2024]] — Sanford is a co-author. Answers the width question Sanford left open: with **linear** width, **constant** depth suffices; some tasks need quadratic width. Their supplementary code is vendored in `yehudai/`. |

## Status

- **Reading now:** `airale-2025-simple-path-structural-encoding` (started 2026-08-26)
- **Read, notes written:** `ye-2026-transformers-graph-connectivity`
- **Read, notes written:** `sanford-2024-transformer-reasoning-graph-algorithms`
- **Downloaded, reading companion outstanding:** `yehudai-2025-depth-width-tradeoffs-graph-tasks`
  (`docs/yehudai-empirical.md` covers the *reproduction*; the paper's theory is unwritten)

## Adding a paper

```sh
curl -sL -o papers/<first-author>-<year>-<slug>.pdf <pdf-url>
```

Then add a row above. If the paper earns a full write-up, put it in `reports/<first-author>-<year>-notes.typ`
and link it from the Notes column.
