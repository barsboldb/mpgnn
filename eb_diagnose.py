"""Why do edge-bias models stall past n ~ 96? Diagnostics on saved eb_* checkpoints (2026-11-05).

Research report hypotheses: H1 the fixed scalar bias b * (A + I) is diluted as n grows (most
softmax mass lands on non-neighbours); H5 6 one-hop layers reach ~12 hops and far pairs are
the ones left wrong. Per model, on its own 400 test graphs (same seed as the sweep):

  b          learned edge bias per layer (mean / max over heads)
  nbr        attention mass on N(i) + i per layer (mean over heads, nodes, graphs) - H1
  ent        attention entropy per layer, in nats (uniform over n = log n)
  err_by_d   pair error rate of connected pairs by hop distance (1-4, 5-8, 9-12, 13+) - H5
  err_neg    error rate on disconnected pairs (label-0 graphs, across the two blobs)
  sep        per layer: mean cosine of h_i to its own component mean minus to the other
             component's (label-0 graphs) - how far the component fingerprint has formed
  cos        mean pairwise cosine of final h_i (representation collapse)

Usage:
    python eb_diagnose.py                          # every model in results/width/models/eb_adj_big
    python eb_diagnose.py --models 'n128_*' --sweep eb_adj_big
"""
import argparse
import glob
import json
import math
import os
import re

import numpy as np
import torch
from scipy.sparse.csgraph import connected_components, shortest_path

import width_sweep as ws

BINS = ((1, 4), (5, 8), (9, 12), (13, 10**6))


@torch.no_grad()
def diagnose(model, A, R):
    model.eval()
    B, n = A.shape[:2]
    m = model.inner
    K = len(m.blocks) * model.loops
    nbr = [[] for _ in range(K)]; ent = [[] for _ in range(K)]; sep = [[] for _ in range(K)]
    err_d = {b: [0, 0] for b in BINS}; neg = [0, 0]; cos = []
    off = ~torch.eye(n, dtype=torch.bool)
    for i in range(0, B, 50):
        Ab = A[i:i+50].float(); Rb = R[i:i+50] > 0.5; b = Ab.size(0)
        lab, comps = [], []
        for g in Ab.numpy():
            k, c = connected_components(g, directed=False); lab.append(k == 1); comps.append(c)
        h = m.readin(Ab)
        for i_l, (L, blk) in enumerate([lb for _ in range(model.loops) for lb in enumerate(m.blocks)]):
            bias = (Ab.unsqueeze(1) * (model.scale * model.edge_bias[L]).view(1, -1, 1, 1)).reshape(b * model.heads, n, n)
            z = blk.norm1(h)
            a, w = blk.attn(z, z, z, attn_mask=bias, need_weights=True, average_attn_weights=False)
            nbr[i_l].append((w * Ab.unsqueeze(1)).sum(-1).mean().item())
            ent[i_l].append(-(w.clamp_min(1e-12).log() * w).sum(-1).mean().item())
            h = h + a
            h = h + blk.ffn(blk.norm2(h))
            hn = torch.nn.functional.normalize(h, dim=-1)
            for g in range(b):
                if lab[g]:
                    continue
                c = torch.as_tensor(comps[g])
                mu = [torch.nn.functional.normalize(hn[g][c == k].mean(0), dim=0) for k in (0, 1)]
                own = torch.stack([hn[g][j] @ mu[c[j]] for j in range(n)])
                oth = torch.stack([hn[g][j] @ mu[1 - c[j]] for j in range(n)])
                sep[i_l].append((own - oth).mean().item())
        hf = torch.nn.functional.normalize(m.norm(h), dim=-1)
        cos.append((hf @ hf.transpose(1, 2))[:, off].mean().item())
        logits = model(Ab)
        wrong = (logits > 0) != Rb
        for g in range(b):
            D = shortest_path(Ab[g].numpy(), unweighted=True)
            for lo, hi in BINS:
                sel = torch.as_tensor((D >= lo) & (D <= hi) & np.isfinite(D))
                err_d[(lo, hi)][0] += wrong[g][sel].sum().item(); err_d[(lo, hi)][1] += sel.sum().item()
            sel = torch.as_tensor(~np.isfinite(D))
            neg[0] += wrong[g][sel].sum().item(); neg[1] += sel.sum().item()
    eb = model.edge_bias.detach()
    return {"b_mean": eb.mean(1).tolist(), "b_max": eb.max(1).values.tolist(),
            "nbr": [float(np.mean(x)) for x in nbr], "ent": [float(np.mean(x)) for x in ent],
            "sep": [float(np.mean(x)) for x in sep], "cos": float(np.mean(cos)),
            "err_by_d": {f"{lo}-{hi if hi < 10**6 else ''}": e / max(1, t) for (lo, hi), (e, t) in err_d.items()},
            "frac_by_d": {f"{lo}-{hi if hi < 10**6 else ''}": t for (lo, hi), (e, t) in err_d.items()},
            "err_neg": neg[0] / max(1, neg[1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="eb_adj_big")
    ap.add_argument("--models", default="*")
    a = ap.parse_args()
    out = os.path.join(ws.OUT_DIR, "eb_diag"); os.makedirs(out, exist_ok=True)
    for p in sorted(glob.glob(os.path.join(ws.OUT_DIR, "models", a.sweep, a.models + ".pt"))):
        n, L, mw, s = map(int, re.match(r"n(\d+)_L(\d+)_m(\d+)_lr[\d.e-]+_s(\d+)", os.path.basename(p)).groups())
        cfg = dict(ws.SWEEPS[a.sweep], n=n, depth=L, train=16); cfg.pop("train_from", None)
        _, _, (Ate, Rte) = ws.make_splits(cfg, s)
        heads = max(1, mw // cfg["head_dim"])
        model = ws.build_model(cfg, mw, heads, s)
        model.load_state_dict(torch.load(p, map_location="cpu"))
        r = diagnose(model, Ate, Rte)
        r.update(n=n, depth=L, width=mw, seed=s, sweep=a.sweep, test_pair=ws.metrics(model.to(ws.DEVICE), Ate, Rte)[1])
        with open(os.path.join(out, f"{a.sweep}_{os.path.basename(p)[:-3]}.json"), "w") as f:
            json.dump(r, f)
        f2 = lambda xs: " ".join(f"{x:.2f}" for x in xs)
        print(f"n={n:<3d} m={mw:<3d} s={s} pair={r['test_pair']:.3f} cos={r['cos']:.2f} neg_err={r['err_neg']:.3f}\n"
              f"   b    {f2(r['b_mean'])}   (max {f2(r['b_max'])})\n   nbr  {f2(r['nbr'])}\n"
              f"   ent  {f2(r['ent'])}  (log n {math.log(n):.2f})\n   sep  {f2(r['sep'])}\n"
              f"   err_by_d {' '.join(f'{k}:{v:.3f}' for k, v in r['err_by_d'].items())}", flush=True)


if __name__ == "__main__":
    main()
