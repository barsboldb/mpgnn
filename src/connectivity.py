"""Connectivity-matrix task (Ye et al. 2026) as a reusable utility.

input  : self-loop-augmented adjacency  A + I   (each node = a row token)
target : connectivity matrix R, R_ij = 1 iff i, j in the same component
readout: a pairwise bilinear head  H W H^T -> [n, n] logits
metric : exact-match accuracy (the WHOLE n x n matrix must be correct)

The same encoder block (`src.transformer._EncoderBlock`) the main GraphTransformer
uses; `--local` masks attention to graph neighbours so reach is depth-bounded.
Driven either from the CLI (ye_connectivity.py) or via `main.py --config` with
`task: connectivity`. Logs through RunLogger, so runs plot alongside everything else.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import _EncoderBlock
from .logger import RunLogger
from .dataset import make_connectedness_hard_dataset

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")


# ── graphs and targets ────────────────────────────────────────────────────────

def er_adjacency(n, p, rng):
    """Erdos-Renyi adjacency (symmetric, no self-loops)."""
    upper = (rng.random((n, n)) < p).astype(np.float32)
    A = np.triu(upper, 1)
    return A + A.T


def _caterpillar(m, cap, rng):
    """Graph on m nodes with EXACT, controllable diameter (backbone path + interior
    leaves). Diameter sampled to straddle `cap`; degrees ~2 (no degree signal)."""
    A = np.zeros((m, m), dtype=np.float32)
    d = int(rng.integers(2, max(3, min(m - 1, round(2 * cap))) + 1))
    for i in range(d):
        A[i, i + 1] = A[i + 1, i] = 1.0
    interior = list(range(1, d)) or [0]
    for u in range(d + 1, m):
        v = int(rng.choice(interior))
        A[u, v] = A[v, u] = 1.0
    return A


def diameter_controlled(n, cap, rng):
    """Caterpillar with straddling diameter; half the time two disconnected pieces."""
    if rng.random() < 0.5:
        return _caterpillar(n, cap, rng)
    h = n // 2
    A = np.zeros((n, n), dtype=np.float32)
    A[:h, :h] = _caterpillar(h, cap, rng)
    A[h:, h:] = _caterpillar(n - h, cap, rng)
    return A


def two_cliques(n, bridge=False):
    """Two equal cliques; disconnected unless bridge=True (the adversarial OOD case)."""
    h = n // 2
    A = np.zeros((n, n), dtype=np.float32)
    A[:h, :h] = 1 - np.eye(h)
    A[h:, h:] = 1 - np.eye(n - h)
    if bridge:
        A[0, h] = A[h, 0] = 1.0
    return A


def component_labels(A):
    G = nx.from_numpy_array(A)
    lab = np.empty(A.shape[0], dtype=int)
    for c, nodes in enumerate(nx.connected_components(G)):
        for u in nodes:
            lab[u] = c
    return lab


def reachability(A):
    """R_ij = 1 iff i, j in the same connected component (transitive closure)."""
    lab = component_labels(A)
    return (lab[:, None] == lab[None, :]).astype(np.float32)


def graph_diameter(A):
    G = nx.from_numpy_array(A)
    d = 0
    for comp in nx.connected_components(G):
        if len(comp) > 1:
            d = max(d, nx.diameter(G.subgraph(comp)))
    return d


def _adj_from_aug(A_aug_row, n):
    return ((A_aug_row.cpu().numpy() - np.eye(n)) > 0.5).astype(np.float32)


# ── capacity diagnostics ──────────────────────────────────────────────────────

def capacity_stats(A_aug, cap, sample=400):
    """rho_hard = fraction of reachable pairs at distance > cap (provably unsolvable
    for a depth-bounded model), plus a diameter histogram."""
    n = A_aug.shape[1]
    m = min(sample, A_aug.shape[0])
    diam_hist, hard, reach, total = {}, 0, 0, 0
    for k in range(m):
        G = nx.from_numpy_array(_adj_from_aug(A_aug[k], n))
        spl = dict(nx.all_pairs_shortest_path_length(G))
        d = 0
        for i in range(n):
            row = spl.get(i, {})
            for j in range(i + 1, n):
                total += 1
                if j in row:
                    reach += 1
                    d = max(d, row[j])
                    if row[j] > cap:
                        hard += 1
        bucket = d if d <= cap else f">{cap}"
        diam_hist[bucket] = diam_hist.get(bucket, 0) + 1
    order = sorted(diam_hist.items(), key=lambda kv: (isinstance(kv[0], str), kv[0]))
    return {"rho_hard": hard / max(total, 1), "reach_frac": reach / max(total, 1),
            "n_sampled": m, "diam_hist": dict(order)}


def filter_within_capacity(A_aug, R, cap):
    n = A_aug.shape[1]
    keep = [k for k in range(A_aug.shape[0])
            if graph_diameter(_adj_from_aug(A_aug[k], n)) <= cap]
    return A_aug[keep], R[keep], len(keep)


def print_capacity(name, A_aug, cap):
    s = capacity_stats(A_aug, cap)
    print(f"  [{name}] rho_hard(dist>{cap})={s['rho_hard']:.4f}  "
          f"reach_frac={s['reach_frac']:.3f}  diam_hist={s['diam_hist']}  (n={s['n_sampled']})")


# ── datasets (graph -> A+I, R) ────────────────────────────────────────────────

def sample_graph(dist, n, p, cap, rng):
    if dist == "er":
        return er_adjacency(n, p, rng)
    if dist == "diam":
        return diameter_controlled(n, cap, rng)
    raise ValueError(f"unknown dist '{dist}'")


def make_set(num, n, p, cap, rng, dist="er", within_only=False, seed=0):
    if dist == "hard":
        return make_hard_set(num, n, cap, seed, within_only)
    As, Rs, tries = [], [], 0
    while len(As) < num and tries < num * 200:
        tries += 1
        A = sample_graph(dist, n, p, cap, rng)
        if within_only and graph_diameter(A) > cap:
            continue
        As.append(A + np.eye(n, dtype=np.float32))
        Rs.append(reachability(A))
    if len(As) < num:
        print(f"  [warn] only generated {len(As)}/{num} graphs (dist={dist}, within_only={within_only})")
    return torch.tensor(np.array(As)), torch.tensor(np.array(Rs))


def make_hard_set(num, n, cap, seed, within_only=False, pool_mult=4):
    """connectedness_hard graphs (two dense blobs +/- one bridge) at fixed n."""
    pool = make_connectedness_hard_dataset(num_graphs=num * pool_mult,
                                           min_nodes=n, max_nodes=n, seed=seed)
    As, Rs = [], []
    for g in pool:
        A = np.zeros((n, n), dtype=np.float32)
        ei = g.edge_index.numpy()
        A[ei[0], ei[1]] = 1.0
        A = ((A + A.T) > 0).astype(np.float32)
        if within_only and graph_diameter(A) > cap:
            continue
        As.append(A + np.eye(n, dtype=np.float32))
        Rs.append(reachability(A))
        if len(As) >= num:
            break
    if len(As) < num:
        print(f"  [warn] only got {len(As)}/{num} hard graphs within cap={cap}")
    return torch.tensor(np.array(As)), torch.tensor(np.array(Rs))


# ── model ─────────────────────────────────────────────────────────────────────

class ConnectivityTransformer(nn.Module):
    def __init__(self, n, d=64, depth=2, heads=4, dropout=0.0, local=False):
        super().__init__()
        self.local = local
        self.readin = nn.Linear(n, d)
        self.blocks = nn.ModuleList([_EncoderBlock(d, heads, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.W = nn.Linear(d, d, bias=False)

    def forward(self, Aaug):                          # [B, n, n] -> [B, n, n] logits
        h = self.readin(Aaug)
        attn_mask = (Aaug == 0) if self.local else None   # local: attend to neighbours only
        for blk in self.blocks:
            h = blk(h, None, attn_mask)
        h = self.norm(h)
        logits = torch.bmm(self.W(h), h.transpose(1, 2))
        return 0.5 * (logits + logits.transpose(1, 2))    # symmetric, like R

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(model, A, R, bs=128):
    model.eval()
    if A.size(0) == 0:
        return float("nan")
    em = 0.0
    for i in range(0, A.size(0), bs):
        lo = model(A[i:i+bs].to(DEVICE))
        em += ((lo > 0).float() == R[i:i+bs].to(DEVICE)).all(dim=(1, 2)).float().sum().item()
    return em / A.size(0)


@torch.no_grad()
def predicted_density(model, A):
    model.eval()
    return (model(A.to(DEVICE)) > 0).float().mean().item()


# ── orchestration (used by the CLI and by main.py) ────────────────────────────

def run_connectivity(cfg: dict, dataset: str = "connectivity"):
    """cfg keys: regime, dist, local, capacity, depth, n, p, train, test, hidden,
    heads, epochs, lr, batch, seed. Trains the matrix model and logs via RunLogger."""
    g = cfg.get
    regime, dist, local = g("regime", "within"), g("dist", "diam"), bool(g("local", False))
    depth, n, p = g("depth", 2), g("n", 24), g("p", 0.12)
    cap = g("capacity", 0) or (depth if local else 3 ** depth)
    rng = np.random.default_rng(g("seed", 0))
    attn = "local" if local else "global"
    print(f"Device {DEVICE} | depth={depth} attn={attn} capacity={cap} | "
          f"dist={dist} n={n} p={p} | regime={regime}")

    print("Generating data...")
    Atr, Rtr = make_set(g("train", 2000), n, p, cap, rng, dist=dist,
                        within_only=(regime == "within"), seed=g("seed", 0))
    Ate, Rte = make_set(g("test", 400), n, p, cap, rng, dist=dist, seed=g("seed", 0) + 9999)
    Ate_w, Rte_w, n_w = filter_within_capacity(Ate, Rte, cap)
    Acl, Rcl = _two_clique_set(g("test", 400), n, bridge=False)
    Abr, Rbr = _two_clique_set(g("test", 400), n, bridge=True)
    print(f"  train={Atr.size(0)}  test={Ate.size(0)} (within-cap {n_w})  "
          f"2clique(disc/bridge)={Acl.size(0)}")
    print("Capacity diagnostics:")
    print_capacity("train", Atr, cap)
    print_capacity("test", Ate, cap)

    model = ConnectivityTransformer(n, g("hidden", 64), depth, g("heads", 4), local=local).to(DEVICE)
    print(f"  model params: {model.num_parameters():,}")
    opt = torch.optim.Adam(model.parameters(), lr=g("lr", 1e-3))

    config = {"model": "connectivity_transformer", "task": "connectivity", "tokenization": "adj_matrix",
              "regime": regime, "dist": dist, "attn": attn, "depth": depth, "capacity": cap,
              "n": n, "p": p, "hidden_channels": g("hidden", 64), "heads": g("heads", 4),
              "lr": g("lr", 1e-3), "epochs": g("epochs", 200), "batch_size": g("batch", 128),
              "seed": g("seed", 0), "device": str(DEVICE),
              "layers": [{"type": "global_attn", "heads": g("heads", 4)} for _ in range(depth)]}
    logger = RunLogger(dataset, config, tag=f"{regime}_")

    Atr_d, Rtr_d = Atr.to(DEVICE), Rtr.to(DEVICE)
    N, bs, epochs = Atr.size(0), g("batch", 128), g("epochs", 200)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(N, device=DEVICE)
        tot = 0.0
        for i in range(0, N, bs):
            idx = perm[i:i+bs]
            loss = F.binary_cross_entropy_with_logits(model(Atr_d[idx]), Rtr_d[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * idx.size(0)
        entry = {"loss": round(tot / N, 4)}
        if epoch % 20 == 0 or epoch in (1, epochs):
            entry["train"] = round(evaluate(model, Atr, Rtr, bs), 4)
            entry["test"] = round(evaluate(model, Ate, Rte, bs), 4)
            entry["test_within"] = round(evaluate(model, Ate_w, Rte_w, bs), 4)
            entry["clique_disc"] = round(evaluate(model, Acl, Rcl, bs), 4)
            entry["clique_bridge"] = round(evaluate(model, Abr, Rbr, bs), 4)
            entry["clique_disc_dens"] = round(predicted_density(model, Acl), 4)
            print(f"  ep {epoch:03d}  loss={entry['loss']:.4f}  train={entry['train']:.3f}  "
                  f"test={entry['test']:.3f}  test_wc={entry['test_within']:.3f}  "
                  f"clique_disc={entry['clique_disc']:.3f}  bridge={entry['clique_bridge']:.3f}")
        logger.log(epoch, **entry)

    s = logger._summary()
    print("\n=== Exact-match (best test epoch) ===")
    for k in ("train", "test", "test_within", "clique_disc", "clique_bridge", "clique_disc_dens"):
        print(f"  {k:18s} {s.get(k, '-')}")
    logger.save()
    return s


def _two_clique_set(num, n, bridge=False):
    A = two_cliques(n, bridge=bridge)
    Aaug = torch.tensor(A + np.eye(n, dtype=np.float32)).unsqueeze(0).repeat(num, 1, 1)
    R = torch.tensor(reachability(A)).unsqueeze(0).repeat(num, 1, 1)
    return Aaug, R
