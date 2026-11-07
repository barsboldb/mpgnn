"""Figure of one swap graph (width_sweep.swap_blob_graphs) under both labels: the graph,
its input A + I and its target R, before and after the node relabelling.

Same steps and random calls as swap_blob_graphs / _cycle_blob_edges, but keeps the cycle
order (for the layout) and builds both labels on one base graph. Views follow the "Swap
generator explorer" widget.

    .venv/bin/python reports/figures/swap_generator_plot.py   # -> reports/figures/swap-generator.png
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from width_sweep import _far_pair  # noqa: E402

N, CHORD_FRAC, DMIN, SEED = 32, 0.25, 6, 3
C = dict(A="#0072B2", B="#CC79A7", lab0="#D55E00", lab1="#009E73", far="#E69F00",
         ink="#17202C", muted="#5A6577", soft="#E9EDF2")


def blob(nodes, n_chords, rng):
    """_cycle_blob_edges, also returning the cycle order: (order, cycle edges, chords)."""
    perm = [int(x) for x in rng.permutation(nodes)]
    m = len(perm)
    cycle = {tuple(sorted((perm[i], perm[(i + 1) % m]))) for i in range(m)}
    chords = set()
    n_chords = min(n_chords, m * (m - 1) // 2 - m)
    while len(chords) < n_chords:
        e = tuple(sorted(int(x) for x in rng.choice(nodes, size=2, replace=False)))
        if e not in cycle and e not in chords:
            chords.add(e)
    return perm, cycle, chords


def build(n, chord_frac, dmin, seed):
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        na = int(rng.integers(n // 4, n - n // 4 + 1))
        blobs = (list(range(na)), list(range(na, n)))
        parts = [blob(b, round(chord_frac * len(b)), rng) for b in blobs]
        A = np.zeros((n, n), dtype=np.float32)
        for _, cyc, ch in parts:
            for u, v in cyc | ch:
                A[u, v] = A[v, u] = 1.0
        a1, a2, da = _far_pair(A, blobs[0], rng)
        b1, b2, db = _far_pair(A, blobs[1], rng)
        if min(da, db) >= dmin:
            break
    new = {0: [(a1, a2), (b1, b2)], 1: [(a1, b1), (a2, b2)]}
    return dict(na=na, parts=parts, A=A, far=(a1, a2, b1, b2), d=(da, db), new=new,
                perm=np.random.default_rng(seed + 1).permutation(n))


def components(A):
    n, comp = len(A), -np.ones(len(A), int)
    for s in range(n):
        if comp[s] < 0:
            comp[s], stack = s, [s]
            while stack:
                u = stack.pop()
                for v in np.flatnonzero(A[u]):
                    if comp[v] < 0:
                        comp[v] = s
                        stack.append(v)
    return comp


def layout(g):
    pos = {}
    for (order, _, _), cx in zip(g["parts"], (-1.25, 1.25)):
        k = len(order)
        for i, u in enumerate(order):
            t = np.pi / 2 - 2 * np.pi * i / k
            pos[u] = (cx + np.cos(t), np.sin(t))
    return pos


def draw_graph(ax, g, pos, label):
    for order, cyc, ch in g["parts"]:
        for edges, ls in ((cyc, "-"), (ch, (0, (3, 2)))):
            for u, v in edges:
                ax.plot(*zip(pos[u], pos[v]), color=C["muted"], lw=0.9, ls=ls, zorder=1)
    for u, v in g["new"][label]:
        (x0, y0), (x1, y1) = pos[u], pos[v]
        if (u < g["na"]) != (v < g["na"]):         # cross edge: bend it
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2 + (0.35 if (y0 + y1) > 0 else -0.35)
            t = np.linspace(0, 1, 30)
            xs = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * mx + t ** 2 * x1
            ys = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * my + t ** 2 * y1
            ax.plot(xs, ys, color=C[f"lab{label}"], lw=2.6, zorder=2)
        else:
            ax.plot([x0, x1], [y0, y1], color=C[f"lab{label}"], lw=2.6, zorder=2)
    for u, (x, y) in pos.items():
        col = C["A"] if u < g["na"] else C["B"]
        ax.scatter(x, y, s=34, color="white", edgecolor=col, linewidth=1.4, zorder=3)
    for u, name in zip(g["far"], ("a₁", "a₂", "b₁", "b₂")):
        x, y = pos[u]
        ax.scatter(x, y, s=95, facecolor="none", edgecolor=C["far"], linewidth=2, zorder=4)
        r = np.hypot(x - (-1.25 if u < g["na"] else 1.25), y)
        dx, dy = (x - (-1.25 if u < g["na"] else 1.25)) / r, y / r
        ax.text(x + 0.22 * dx, y + 0.22 * dy, name, ha="center", va="center", fontsize=10,
                color=C["ink"])
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")


def matrix(ax, M, cmap, title, split=None):
    ax.imshow(M, cmap=cmap, vmin=0, vmax=cmap.N - 1, interpolation="nearest")
    if split is not None:
        for f in (ax.axhline, ax.axvline):
            f(split - 0.5, color=C["muted"], lw=0.8, ls=(0, (2, 2)))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#D7DDE5")
    ax.set_title(title, fontsize=10, color=C["ink"])


def main():
    g = build(N, CHORD_FRAC, DMIN, SEED)
    pos, n = layout(g), N
    P = g["perm"]                                   # as in relabelled(): row i = node P[i]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.6),
                             gridspec_kw=dict(width_ratios=[2.1, 1, 1, 1]))
    for label, row in zip((0, 1), axes):
        A = g["A"].copy()
        for u, v in g["new"][label]:
            A[u, v] = A[v, u] = 1.0
        comp = components(A)
        # input A + I: 0 empty, 1 base edge, 2 new edge, 3 diagonal
        M = A.copy()
        for u, v in g["new"][label]:
            M[u, v] = M[v, u] = 2
        np.fill_diagonal(M, 3)
        cin = ListedColormap([C["soft"], C["ink"], C[f"lab{label}"], C["muted"]])
        # target R: 0 unreachable, 1 same component as a1 (blob A), 2 the other component
        R = np.where(comp[:, None] == comp[None, :],
                     np.where(comp[:, None] == comp[g["far"][0]], 1, 2), 0)
        cR = ListedColormap([C["soft"], C["A"], C["B"]])
        draw_graph(row[0], g, pos, label)
        k = len(set(comp))
        row[0].set_title(f"label {label}: {'two components' if k == 2 else 'connected'}",
                         fontsize=12, color=C[f"lab{label}"], fontweight="semibold")
        matrix(row[1], M, cin, "input A + I", split=g["na"])
        matrix(row[2], R, cR, "target R", split=g["na"])
        matrix(row[3], R[np.ix_(P, P)], cR, "R after relabelling")
    fig.text(0.215, 0.015, f"n = {n}, chord_frac = {CHORD_FRAC}, d(a₁, a₂) = {g['d'][0]}, "
             f"d(b₁, b₂) = {g['d'][1]} (≥ dmin = {DMIN})", ha="center", fontsize=9,
             color=C["muted"])
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = os.path.join(os.path.dirname(__file__), "swap-generator.png")
    fig.savefig(out, dpi=200)
    print(out, g["na"], g["d"])


if __name__ == "__main__":
    main()
