"""Layer probes for the connectivity transformer: what does each layer know about paths?

For a model saved by a width_sweep.py sweep with `save_models` (the `probe` sweep:
adjacency rows, depth 6, q3lr protocol), read the residual stream after the read-in
(layer 0) and after every block, and fit linear probes on fresh `swap` graphs:

  pair  a bilinear probe z_ij = h_i^T W h_j + b — the form of the model's own read-out —
        for "d(i, j) <= k". What the pair (h_i, h_j) knows about the distance.
  set   a linear map h_i -> R^n for node i's whole k-hop set, [d(i, j) <= k] over j.
        Whether node i's own vector stores its reachable set: n bits in m numbers.

k in {1, 2, 3, 4, 6, 8, inf} (inf = connected). Score: ROC-AUC on held-out graphs over
off-diagonal pairs. Every probe is also fitted on the same architecture untrained (same
seed), which calibrates what the probe itself can compute from adjacency-row features: a
bilinear form on adjacency rows already contains 2-hop counts, a_i . a_j = (A+I)^2_ij.

Reading: if layer l doubles the reach (path doubling, as in repeated squaring
M <- M^2), the decodable k grows ~2^l; if it adds one hop per layer, ~l + 1; a narrow
model that fails should stall at some layer.

Usage:
    python probe_layers.py                     # every model in results/width/models/probe
    python probe_layers.py --models 'n40_*'    # glob within that folder
    python probe_layers.py --summary           # tables from results/width/probe/*.json
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse.csgraph import shortest_path
from sklearn.metrics import roc_auc_score

from src.connectivity import ConnectivityTransformer
from width_sweep import relabelled, set_seed

MODEL_DIR = os.path.join("results", "width", "models", "probe")
OUT_DIR = os.path.join("results", "width", "probe")
KS = (1, 2, 3, 4, 6, 8, float("inf"))
# CUDA when there is one; else the CPU (MPS is ~18x slower than the CPU on these small fits)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NAME = re.compile(r"n(\d+)_L(\d+)_m(\d+)_lr([\d.e-]+)_s(\d+)\.pt$")


def probe_data(n, num, seed):
    """(A + I, distances) for `num` relabelled swap graphs (the q3 generator settings)."""
    A, _ = relabelled("swap", num, n, seed, chord_frac=0.25, dmin=6)
    D = np.stack([shortest_path(a.numpy() - np.eye(n, dtype=np.float32), unweighted=True,
                                directed=False) for a in A])
    return A.float(), torch.tensor(D, dtype=torch.float32)


def within(D, k):
    """0/1 targets d(i, j) <= k; k = inf means connected (d finite: inf <= inf is True)."""
    return (torch.isfinite(D) if k == float("inf") else D <= k).float()


@torch.no_grad()
def residual_states(model, A, bs=200):
    """[layer 0..L, graphs, n, m]: after the read-in and after each block, layer-normed
    (no affine) so probes see every layer on one scale."""
    model.eval()
    out = []
    for i in range(0, A.size(0), bs):
        h = model.readin(A[i:i + bs].to(DEVICE))
        hs = [h]
        for blk in model.blocks:
            h = blk(h, None, None)
            hs.append(h)
        out.append(torch.stack([F.layer_norm(x, x.shape[-1:]) for x in hs]).cpu())
    return torch.cat(out, dim=1)


def fit(kind, Htr, Ytr, Hte, Yte, steps=300, lr=1e-2):
    """Fit one probe on [graphs, n, m] states and [graphs, n, n] 0/1 targets; return the
    held-out AUC over off-diagonal pairs (nan if the test labels are all one class)."""
    n, m = Htr.shape[1], Htr.shape[2]
    off = ~torch.eye(n, dtype=torch.bool)
    if kind == "pair":
        W = torch.zeros(m, m, requires_grad=True, device=DEVICE)
        score = lambda H: H @ W @ H.transpose(1, 2)
    else:
        W = torch.zeros(m, n, requires_grad=True, device=DEVICE)
        score = lambda H: H @ W
    b = torch.zeros(1, requires_grad=True, device=DEVICE)
    Htr, Ytr = Htr.to(DEVICE), Ytr.to(DEVICE)
    pos = Ytr[:, off].mean().clamp(1e-4, 1 - 1e-4)
    opt = torch.optim.Adam([W, b], lr=lr)
    for _ in range(steps):
        z = score(Htr) + b
        loss = F.binary_cross_entropy_with_logits(z[:, off], Ytr[:, off],
                                                  pos_weight=(1 - pos) / pos)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        z = (score(Hte.to(DEVICE)) + b)[:, off].flatten().cpu().numpy()
    y = Yte[:, off].flatten().numpy()
    return float(roc_auc_score(y, z)) if 0 < y.mean() < 1 else float("nan")


def probe_model(path, n_train=500, n_test=300, data_cache={}):
    n, L, m, lr, seed = NAME.search(path).groups()
    n, L, m, seed = int(n), int(L), int(m), int(seed)
    if n not in data_cache:
        data_cache.clear()
        data_cache[n] = (probe_data(n, n_train, 9001), probe_data(n, n_test, 9002))
    (Atr, Dtr), (Ate, Dte) = data_cache[n]
    heads = max(1, m // 8)
    trained = ConnectivityTransformer(n, m, L, heads).to(DEVICE)
    trained.load_state_dict(torch.load(path, map_location=DEVICE))
    set_seed(seed)   # the untrained twin: same init as training started from
    untrained = ConnectivityTransformer(n, m, L, heads).to(DEVICE)
    res = {"model": os.path.basename(path), "n": n, "depth": L, "width": m, "seed": seed,
           "ks": [k if k != float("inf") else "inf" for k in KS], "probes": {}}
    for tag, model in (("trained", trained), ("untrained", untrained)):
        Str, Ste = residual_states(model, Atr), residual_states(model, Ate)
        for kind in ("pair", "set"):
            table = []
            for layer in range(L + 1):
                table.append([fit(kind, Str[layer], within(Dtr, k),
                                  Ste[layer], within(Dte, k)) for k in KS])
            res["probes"][f"{tag}/{kind}"] = table
            print(f"  {tag:9s} {kind:4s} " + " | ".join(
                f"L{l}: " + " ".join(f"{a:.2f}" for a in row) for l, row in enumerate(table)),
                flush=True)
    # what the model itself gets right at the end, as a reference for the probes
    with torch.no_grad():
        z = trained(Ate.to(DEVICE)).cpu()
    off = ~torch.eye(n, dtype=torch.bool)
    res["readout_pair_acc"] = float(((z > 0) == (Dte < float("inf")))[:, off].float().mean())
    return res


def first_layer(table, j, th):
    """First layer whose AUC for the j-th k reaches th ('-' if none)."""
    return next((str(l) for l, row in enumerate(table) if row[j] >= th), "-")


def summary(th=0.95):
    files = sorted(glob.glob(os.path.join(OUT_DIR, "*.json")))
    if not files:
        print(f"no results in {OUT_DIR}")
        return
    rs = sorted((json.load(open(f)) for f in files), key=lambda r: (r["n"], r["width"], r["seed"]))
    ks = rs[0]["ks"]
    for kind in ("pair", "set"):
        # Not "largest k decodable": trained models trade local detail for global reach in
        # the late layers (k = 1 AUC falls while k = inf rises), so read each k on its own.
        print(f"\n{kind} probe: first layer with AUC >= {th} for each k (trained / untrained)")
        print(f"  {'':34s}" + "".join(f"{'k=' + str(k):>8s}" for k in ks))
        for r in rs:
            t, u = r["probes"][f"trained/{kind}"], r["probes"][f"untrained/{kind}"]
            cells = [f"{first_layer(t, j, th)}/{first_layer(u, j, th)}" for j in range(len(ks))]
            print(f"  n={r['n']:<3d} m={r['width']:<3d} s={r['seed']}  pair acc {r['readout_pair_acc']:.3f}"
                  + "".join(f"{c:>8s}" for c in cells))
        print(f"\n{kind} probe AUC after the last layer, trained (untrained)")
        for r in rs:
            t, u = r["probes"][f"trained/{kind}"][-1], r["probes"][f"untrained/{kind}"][-1]
            print(f"  n={r['n']:<3d} m={r['width']:<3d} s={r['seed']}  "
                  + "  ".join(f"{a:.2f}({b:.2f})" for a, b in zip(t, u)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="*.pt", help="glob inside " + MODEL_DIR)
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.95)
    a = ap.parse_args()
    if a.summary:
        summary(a.threshold)
    else:
        os.makedirs(OUT_DIR, exist_ok=True)
        paths = sorted(glob.glob(os.path.join(a.model_dir, a.models)),
                       key=lambda p: tuple(int(x) for x in NAME.search(p).group(1, 3, 5)))
        for p in paths:
            out = os.path.join(OUT_DIR, os.path.basename(p).replace(".pt", ".json"))
            if os.path.exists(out):
                continue
            print(f"{os.path.basename(p)}", flush=True)
            json.dump(probe_model(p), open(out, "w"))
        summary(a.threshold)
