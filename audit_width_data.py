"""Shortcut audit for the width-sweep datasets (connectivity-matrix target).

Before a generator is used in a sweep, measure how far cheap heuristics get:

  all-ones     predict "everything connected" — the trivial floor (exact 0.5).
  k-hop        R_hat_ij = [dist(i, j) <= k]: what a model that only sees k hops can
               reach. If exact-match is high at small k, the task is local and any
               model learns it; it should stay low until k approaches the diameter.
  stats -> y   gradient boosting on graph statistics (edge count, degree histogram,
               closed-walk counts tr(A^3..A^6) ~ triangles/short cycles) predicting
               the connected bit. Should sit at ~0.5; anything higher is a leak.
  stats + 4hop the combined heuristic: stats decide connected (-> all ones),
               otherwise 4-hop reachability. An upper bound on what a shallow,
               statistics-driven model gets on exact-match.

Usage:
    python audit_width_data.py                       # default candidates, n = 16/32/64
    python audit_width_data.py --gens sparse:0.5 swap:0.25:6 --ns 32 64   # gen:chord_frac[:dmin]
"""
import argparse

import numpy as np
from scipy.sparse.csgraph import shortest_path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

from src.connectivity import reachability
from width_sweep import raw_graphs

KS = (1, 2, 3, 4, 6, 8, 12, 16)


def graph_stats(A):
    deg = A.sum(1)
    hist = [np.sum(deg == d) for d in range(1, 6)] + [np.sum(deg >= 6)]
    walks, P = [], A.copy()
    for _ in range(2, 6):          # tr(A^3) .. tr(A^6)
        P = P @ A
        walks.append(np.trace(P @ A))
    return [A.sum() / 2, *hist, *walks]


def audit(gen, n, num, chord_frac, dmin=6, seed=123):
    rng = np.random.default_rng(seed)
    X, y, exact_k, diam, combo_ok = [], [], {k: [] for k in KS}, {0: [], 1: []}, []
    for A, label in raw_graphs(gen, num, n, seed, chord_frac, dmin):
        perm = rng.permutation(n)
        A = A[np.ix_(perm, perm)]
        R = reachability(A).astype(bool)
        D = shortest_path(A, unweighted=True, directed=False)
        finite = D[np.isfinite(D)]
        diam[label].append(int(finite.max()))
        for k in KS:
            exact_k[k].append(np.array_equal(D <= k, R))
        X.append(graph_stats(A)); y.append(label)
        combo_ok.append((D <= 4, R))
    X, y = np.array(X), np.array(y)
    clf = HistGradientBoostingClassifier(max_iter=200)
    stats_acc = cross_val_score(clf, X, y, cv=5).mean()
    # combined heuristic on held-out predictions
    pred = np.empty_like(y)
    for tr, te in _folds(len(y)):
        pred[te] = HistGradientBoostingClassifier(max_iter=200).fit(X[tr], y[tr]).predict(X[te])
    combo = np.mean([R.all() if p == 1 else np.array_equal(H, R)
                     for p, (H, R) in zip(pred, combo_ok)])
    return dict(diam0=diam[0], diam1=diam[1], stats=stats_acc, combo=combo,
                khop={k: np.mean(v) for k, v in exact_k.items()})


def _folds(N, k=5):
    idx = np.arange(N)
    for f in range(k):
        te = idx[f::k]
        yield np.setdiff1d(idx, te), te


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", nargs="+",
                    default=["hard", "hard_diam", "sparse:0.25", "sparse:0.5", "sparse:1.0"])
    ap.add_argument("--ns", nargs="+", type=int, default=[16, 32, 64])
    ap.add_argument("--num", type=int, default=2000)
    a = ap.parse_args()
    print(f"{'generator':<12} {'n':>3} | {'diam disc':>12} {'diam conn':>12} | "
          f"{'stats->y':>8} {'stats+4hop':>10} | k-hop exact-match: " + " ".join(f"{k:>5}" for k in KS))
    for spec in a.gens:
        gen, *rest = spec.split(":")
        cf = float(rest[0]) if rest else 0.5
        dmin = int(rest[1]) if len(rest) > 1 else 6
        for n in a.ns:
            try:
                r = audit(gen, n, a.num, cf, dmin)
            except ValueError as e:
                print(f"{spec:<12} {n:>3} | {e}")
                continue
            d0, d1 = r["diam0"], r["diam1"]
            dd = lambda d: f"{np.mean(d):4.1f} [{min(d)}-{max(d)}]"
            print(f"{spec:<12} {n:>3} | {dd(d0):>12} {dd(d1):>12} | {r['stats']:8.3f} "
                  f"{r['combo']:10.3f} | {'':>18}" + " ".join(f"{r['khop'][k]:5.3f}" for k in KS),
                  flush=True)
