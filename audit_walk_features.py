"""Leak audit for random-walk structural encodings (RWSE) on the swap graphs.

RWSE gives each node its return probabilities diag(P^j), j = 1..K, P = D^-1 (A + I).
For large K these approach deg / vol(component), which depends on whether the graph is
split — a connectivity leak. Two cheap probes, gradient boosting, 5-fold CV:
  graph connected?  per-j quantiles of the node features -> the graph's connected bit
                    (chance 0.5);
  far pair same-component?  |RWSE_u - RWSE_v|, RWSE_u + RWSE_v, deg_u + deg_v -> same
                    component, on pairs more than 4 hops apart (base = majority class).
Result (2026-10-14, 2000 graphs, swap cf 0.25 d_min 6): K <= 8 clean; K = 16 (the
GraphGPS default) 0.90 / 0.72 at n = 40 / 56; K = 32 0.997 / 0.982.

Usage: python audit_walk_features.py   (~5 min)
"""
import sys, numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from scipy.sparse.csgraph import shortest_path
from width_sweep import raw_graphs

def rwse(A, K):
    Aa = A + np.eye(len(A)); P = Aa / Aa.sum(1, keepdims=True)
    out, M = [], np.eye(len(A))
    for _ in range(K):
        M = M @ P; out.append(np.diag(M).copy())
    return np.stack(out, 1)                     # [n, K]

for n in (40, 56):
    graphs = list(raw_graphs("swap", 2000, n, 123, 0.25, 6))
    y = np.array([l for _, l in graphs])
    for K in (4, 8, 16, 32):
        # graph-level: per-j quantiles of the return probabilities -> connected?
        X = np.array([np.concatenate([np.quantile(rwse(A, K), q, axis=0) for q in (0, .25, .5, .75, 1)]) for A, _ in graphs])
        g = cross_val_score(HistGradientBoostingClassifier(max_iter=200), X, y, cv=5).mean()
        # pair-level: from |rwse_u - rwse_v| + both degrees, predict same-component, on pairs > 4 hops apart
        F, Y = [], []
        rng = np.random.default_rng(0)
        for A, _ in graphs[:600]:
            D = shortest_path(A, unweighted=True, directed=False); Rw = rwse(A, K); deg = A.sum(1)
            iu = np.argwhere(np.triu(D > 4, 1))
            iu = iu[rng.choice(len(iu), min(20, len(iu)), replace=False)]
            for u, v in iu:
                F.append(np.concatenate([np.abs(Rw[u] - Rw[v]), Rw[u] + Rw[v], [deg[u] + deg[v]]])); Y.append(np.isfinite(D[u, v]))
        F, Y = np.array(F), np.array(Y)
        p = cross_val_score(HistGradientBoostingClassifier(max_iter=200), F, Y, cv=5).mean()
        print(f"n={n} K={K:2d}  graph connected? {g:.3f}   far pair same-component? {p:.3f} (base {max(Y.mean(), 1 - Y.mean()):.3f})", flush=True)
