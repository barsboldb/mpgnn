"""Edge-bias figures for the run report (reports/width-runs-2026-11-04.typ, 2026-11-06 update).
(a) m* vs n for eb_adj: head_dim 2 at n = 32-56, head_dim 8 at n = 72-128 (per-seed test-pair
crossings from `width_sweep.py <sweep> --scaling`), plain adjacency rows for reference.
Depth 12 (ebd12_*, 2026-11-07; n = 128 with 4 seeds), depth 12 at head_dim 2 and depth 16 at
n = 128 (2026-11-08) added to (a); m* <= 8 (passes at the smallest width) drawn at 8 as
an open triangle.
(b) plateau ablation, seeds 2-5: n = 96, m = 16 (ebx96_*) and n = 128, m = 24 (ebx128_*)."""
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.8))

adj = ([32, 40, 48, 56], [[20.7, 23.1, 23.4], [30.8, 29.4, 39.1], [41.2, 43.6, 49.4], [59.2, 56.7, 63.1]])
a.plot(adj[0], [np.mean(p) for p in adj[1]], marker="o", color="black", label="adjacency rows, depth 6, $n^{1.74}$")
hd2 = ([32, 40, 48, 56], [[5.6, 5.5], [5.8, 5.6], [3.7, 4.9], [4.4, 7.4]])
hd8 = ([72, 96], [[13.0, 14.9], [28.0, 15.7]])
for (ns, per), mk, lab in ((hd2, "s", "edge bias, head_dim 2, depth 6"), (hd8, "D", "edge bias, head_dim 8, depth 6")):
    a.plot(ns, [np.mean(p) for p in per], marker=mk, label=lab)
    for n, p in zip(ns, per):
        a.scatter([n] * len(p), p, s=8, color="grey", zorder=3)
a.scatter([128], [22.8], s=8, color="grey", zorder=3)
a.annotate("", xy=(128, 40), xytext=(128, 32), arrowprops=dict(arrowstyle="->", color="grey"))
a.text(118, 44, "n = 128: 22.8 / > 32", fontsize=7, color="grey", ha="center")
d12 = ([72, 96, 128], [[8, 8], [8.3, 8], [14.9, 14.4, 8, 14.9]])   # 8 = "<= 8" except n = 96 seed 0
a.plot(d12[0], [np.mean(p) for p in d12[1]], marker="^", color="C2", label="edge bias, head_dim 8, depth 12")
a.scatter([72, 96], [8, 8], marker="v", s=40, facecolors="none", edgecolors="C2", zorder=4)
a.text(84, 8.6, "<= 8", fontsize=7, color="C2", ha="center")
# 2026-11-08: depth 12 with head_dim 2 (ebd12hd2) at n = 72 / 96, depth 16 at n = 128 (ebd16_128;
# seed 2 passes at the smallest width, drawn at 8)
h12 = ([72, 96], [[7.0, 6.9], [4.8, 7.3]])
a.plot(h12[0], [np.mean(p) for p in h12[1]], marker="s", color="C4", label="edge bias, head_dim 2, depth 12")
for n, p in zip(*h12):
    a.scatter([n] * len(p), p, s=8, color="grey", zorder=3)
d16 = [11.2, 15.0, 8, 10.9]
a.scatter([128], [np.mean(d16)], marker="*", s=80, color="C3", zorder=4, label="edge bias, head_dim 8, depth 16")
a.set(xscale="log", yscale="log", xlabel="graph size n", ylabel="critical width m* (test pair 0.95)",
      title="(a) edge bias on adjacency rows")
a.set_xticks([32, 48, 72, 96, 128]); a.set_xticklabels([32, 48, 72, 96, 128])
a.set_yticks([4, 8, 16, 32, 64]); a.set_yticklabels([4, 8, 16, 32, 64])
a.minorticks_off(); a.grid(alpha=0.3); a.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)

conds = ["base", "logn", "ssmax", "clip", "wsd", "all", "d12", "loop2"]
labels = ["base", "bias\nlog n", "SSMax", "clip", "WSD", "all 3", "depth\n12", "loop\nx2"]
for n, dx, col in ((96, -0.15, "C0"), (128, 0.15, "C3")):
    for i, c in enumerate(conds):
        v = [json.loads(l)["final"]["test_pair"] for l in open(f"results/width/ebx{n}_{c}.jsonl")]
        b.scatter([i + dx] * len(v), v, color=col, s=14, zorder=3,
                  label=(f"n = {n}, m = {16 if n == 96 else 24}" if i == 0 else None))
b.axhline(0.95, color="grey", ls="--", lw=0.8)
b.set_xticks(range(len(conds))); b.set_xticklabels(labels, fontsize=8)
b.set(ylabel="final test pair accuracy", ylim=(0.76, 1.005),
      title="(b) one change each to the depth-6 model, seeds 2-5")
b.legend(fontsize=7, loc="lower right")
b.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig("reports/figures/width-runs-1106-edgebias.png", dpi=200)
