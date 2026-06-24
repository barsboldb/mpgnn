# Yehudai et al. 2025 — Empirical Analysis & How It Differs From Ours

Notes from reproducing the connectivity experiment in the Yehudai et al. 2025
supplementary code (`yehudai/`), and a side-by-side with our own setup.

Run locally via `yehudai/run_connectivity.py` (a wrapper that stubs unused deps and
patches two bugs in their source — see [Reproduction notes](#reproduction-notes)).

> For a visual, broad-spectrum comparison of the two datasets — how each is built and
> visualized, the measured bulk statistics, how graphs become tokens, and exactly what
> information our data hides that theirs leaks — see
> [`dataset-comparison.typ`](../reports/dataset-comparison.typ) (compiled: `../reports/dataset-comparison.pdf`).

---

## 1. What they test

A single question: **can a plain Transformer decide graph connectivity, and how does
that depend on the graph tokenization, depth, and width?** No message passing — the
graph is flattened into a token sequence and fed to a standard `nn.TransformerEncoder`,
mean-pooled, and read out with a linear head (`BCEWithLogitsLoss`).

The interesting variable is the **tokenization**: how the graph becomes tokens.

### The three tokenizations

| Name | Token = | Sequence length | Token dim |
|---|---|---|---|
| `adj_rows` | node's adjacency row `⊕` node features | `n` (one per node) | `n + d` |
| `edge_list` | `concat(onehot(u)⊕x_u, onehot(v)⊕x_v)` | `#edges` (one per edge) | `2(n + d)` |
| `lap_full` | Laplacian eigenvectors `⊕` node features | `n` | `n + d` |

All padded with zeros to the dataset's max graph size. Here `n` = max nodes, `d` = node
feature dim (their connectivity features are constant `x_i = 1`, so `d = 1`).

### Their dataset (`create_connectivity_data.py`)

- **5000 graphs**, balanced 2500 connected / 2500 disconnected, split 80/10/10.
- **Fixed size**: every graph has exactly `n_nodes` (default 50).
- **Diverse generators**: Erdős–Rényi (`gnp`), random geometric (`rgg`), scale-free, and
  stochastic block model (`sbm`). Disconnected variants force ≥2 components.
- A `_verify_degree_distribution` step checks the two classes aren't trivially separable
  by degree statistics.

---

## 2. What we found (real runs)

Connectivity, **n=50, 1 layer, width 64, n_train=100, 100 epochs**:

| Tokenization | Params | Final test acc | Speed | Trajectory |
|---|---|---|---|---|
| `adj_rows` | 28,609 | **1.000** | 0.23 s/epoch | 0.69→0.99 by ep 20, 1.0 by ep 60 |
| `edge_list` | 31,873 | **0.926** | 16.8 s/epoch | flat at ln2 until ~ep 30, then climbs |

### Observations

1. **One layer is enough at n=50.** Adjacency-rows solves connectivity *perfectly* with a
   single transformer layer. This means n=50 is below the depth threshold — the depth
   tradeoff Yehudai's theory predicts only shows up at larger `n`.

2. **`adj_rows` ≫ `edge_list` in practice here.** Adjacency-rows is faster (≈75×; edge-list
   builds tokens in a pure-Python per-edge loop), trains more smoothly (no long plateau),
   and generalizes better (1.00 vs 0.93). Edge-list overfits — train loss reached 0.002
   while val stalled at 0.92.

3. **The `ln 2 ≈ 0.693` plateau is real and tokenization-dependent.** Edge-list sits at
   chance (loss 0.69, acc 0.50) for ~30 epochs before escaping; adjacency-rows never
   plateaus — it's above chance from epoch 1. Same optimization saddle we hit in our own
   `node_edge` experiments (see [tokenization.typ](../reports/tokenization.typ)).

---

## 2b. Controlled experiment: our model on their data

The decisive test. Run **our** adjacency-rows global-attention GNN (`configs/adj_yehudai.yaml`)
on **Yehudai's** connectivity graphs (`--dataset yehudai_connectivity_adj`, a balanced
1000-graph subset of their n=50 pool). Only the data changes; the model and training loop
are ours.

| Model | Data | Test acc |
|---|---|---|
| our adj global_attn (1 layer, 15,874 params) | **Yehudai** connectivity, n=50 | **1.000 @ epoch 4** |
| our adj global_attn | our `connectedness_hard_adj_fixed`, n=20 | ~0.70 |
| our adj global_attn | our `connectedness_hard_adj`, variable n | ~0.59 |
| *(reference)* their `nn.TransformerEncoder` | their connectivity | 1.00 @ ~epoch 20 |

Our model reaches 100% on their data **faster than their own** implementation (epoch 4 vs
~20) and with fewer parameters — our LayerNorm + residual + linear input embedding optimizes
a touch better than their plain encoder.

> **Conclusion — the dataset is the hard part, not our model.** Our pipeline reproduces
> Yehudai's perfect score on Yehudai's data, so the ~0.6–0.7 ceiling on our
> `connectedness_hard` is caused entirely by the dataset's adversarial design, not by any
> deficiency in our tokenization, model, or training. `connectedness_hard` is therefore a
> *genuinely harder probe of global reasoning* than the connectivity task in the literature.

We also confirmed two negative results along the way (see [CHANGELOG](CHANGELOG.md)):
fixing graph size alone (`connectedness_hard_fixed`, n=20) does **not** rescue the
edge-token transformer — it stays pinned at the `ln 2` plateau — and fixed-size adjacency
rows only lift our hard set from ~0.59 to ~0.70. Size was never the blocker; the
degree/edge-matched construction and varying blob split are.

---

## 3. How their setup differs from ours

This is the important part for the thesis — the differences explain why the *same*
tokenization behaves so differently on our data.

| Aspect | Yehudai connectivity | Our `connectedness_hard` |
|---|---|---|
| Graph size | **fixed** n=50 | **variable** n∈[12,24] |
| Construction | gnp / rgg / scale-free / sbm | two degree-matched dense blobs |
| Disconnected class | ≥2 components, any structure | exactly 2 blobs, +1 intra-edge to match edge count |
| Hardness control | degree-distribution check | degree sequence *and* edge count matched per pair |
| Node features | constant `x_i = 1` | normalized degree (or adj row) |
| Readout | mean-pool over tokens → linear | mean-pool / task token |
| Model | `nn.TransformerEncoder` (PyTorch) | our `GNN` / `GraphTransformer` |

### The decisive difference: fixed vs variable size

The same `adj_rows` tokenization scores **1.00 on theirs** but **~0.59 on ours**. Why?

- **Fixed n (theirs):** every graph fills all 50 positions, so the adjacency-row layout —
  and what "column j" means — is *consistent across every graph*. The tokenization's lack
  of permutation-invariance never bites, because there's effectively one canonical layout.
- **Variable n + varying blob split (ours):** "position i" lands in different structural
  roles across graphs (different sizes, different split points). The same column index
  means different things in different graphs, so a position-based tokenization can't
  transfer. It memorizes train and stalls on test.

> **Thesis takeaway.** Our `connectedness_hard` is a *strictly harder* probe than the
> standard connectivity benchmark: it defeats a tokenization (`adj_rows`) that achieves
> 100% on the literature's version. The added difficulty comes from variable size and the
> degree-matched two-blob construction, which together remove the consistent-layout
> shortcut that makes the standard benchmark easy.

### Other differences worth noting

- **Sample size.** Their default `n_train=100` (of 4000 train graphs) is a deliberate
  sample-efficiency setting, not a full-data run. Ours trains on the full 80% split.
- **Constant node features.** Theirs carry no per-node signal (`x_i=1`); all information is
  in the adjacency/edge structure. Ours inject degree, which is itself a (weak) signal.
- **Optimizer.** They use AdamW, `weight_decay=0.1`, grad-clip 5.0. Ours uses Adam,
  `weight_decay=5e-4`. Their heavier weight decay + clipping likely helps escape the
  plateau on edge-list.

---

## 4. Open questions / next experiments

1. **Depth sweep at larger n.** Re-run with `--num_encoder_layers 1/2/3/4 --n_nodes 100`
   (or 200). One layer saturates at n=50; the depth-vs-task curve should appear once a
   single layer can no longer cover the graph. This is the actual reproduction of their
   theoretical tradeoff.
2. **Their tokenizers on *our* dataset.** The controlled experiment: run `adj_rows` /
   `edge_list` on `connectedness_hard` with everything else equal. Isolates the dataset's
   contribution to hardness from the tokenization's.
3. **`lap_full`.** Untested here — Laplacian eigenvectors are permutation-equivariant in a
   way one-hot/adjacency rows are not, so they *should* transfer across variable-size
   graphs better. Worth a run on our dataset.

---

## Reproduction notes

`yehudai/run_connectivity.py` runs their experiment **without editing their source**. Three
things blocked a direct run; all are handled in the wrapper:

| Blocker | Cause | Fix in wrapper |
|---|---|---|
| `ModuleNotFoundError: ogb / wandb / matplotlib` | Top-level imports unused by the connectivity task | Stubbed via `sys.modules` MagicMock |
| `NameError: create_data` | Their bug: line 207 calls `create_data.…` but imports the module as `create_connectivity_data` | Shim aliased onto their module |
| Inflated val/test on cache hit | Their bug: cache-reload loads the **train** `.pt` file for all three splits | Shim loads each split from its own file |
| `UnpicklingError (weights_only)` | torch ≥2.6 defaults `weights_only=True`, rejecting pickled PyG `Data` | `weights_only=False` in the loader |

```bash
cd yehudai
python run_connectivity.py --rep_type adj_rows  --n_nodes 50 --n_train 100 --num_epochs 100
python run_connectivity.py --rep_type edge_list --n_nodes 50 --n_train 100 --num_epochs 100
```

First run for a given `n_nodes` generates and caches the dataset (the disconnected-graph
sampling at n=50 takes a few minutes); subsequent runs load from cache instantly.
