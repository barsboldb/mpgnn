"""Figures for reports/width-report-2026-11-19.typ (results 9-19 Nov 2026: every m* curve at one
depth, 12). Everything is read from results/width/*.jsonl; m* is recomputed here with the same
rule as `width_sweep.py --scaling` (crossing of 0.95 test pair accuracy, interpolated in log2 width,
on the final epoch). Style follows width_report_1110_plot.py.
Run from the repo root: .venv/bin/python reports/figures/width_report_1119_plot.py"""
import glob
import json
import math
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
OUT = "reports/figures/"
TH = 0.95
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2,
                     "lines.markersize": 6, "legend.frameon": False, "savefig.dpi": 200,
                     "savefig.bbox": "tight"})


def rows(*pats):
    return [json.loads(l) for p in pats for f in sorted(glob.glob("results/width/" + p))
            if f.endswith(".jsonl") for l in open(f) if l.strip()]


def crossing(ws, ys, th=TH):
    if ys[0] >= th:
        return ws[0]
    for i in range(1, len(ws)):
        if ys[i - 1] < th <= ys[i]:
            f = (th - ys[i - 1]) / (ys[i] - ys[i - 1])
            return 2 ** (math.log2(ws[i - 1]) + f * math.log2(ws[i] / ws[i - 1]))
    return float("nan")


ENC = {  # label, colour, marker, sweeps
    "base": ("adjacency rows, no bias (head_dim 8)", C[0], "o",
             ["q3lr_d12h8.*", "q3lr_d12h8_72.*"]),
    "eb": ("adjacency rows + edge bias (head_dim 2)", C[1], "^",
           ["ebd12hd2_s.*", "ebd12hd2_72.*", "ebd12hd2_96.*", "ebd12hd2_128.*", "ebd12hd2_L.*",
            "ebd12hd2_128w4.*"]),
    "rid": ("random IDs + edge bias (head_dim 2)", C[2], "s", ["ebd12rid.*", "ebd12rid_s23.*"]),
}
RUNS = {k: rows(*v[3]) for k, v in ENC.items()}
for rs in RUNS.values():
    for r in rs:
        r["peak"] = max(h["test_pair"] for h in r["history"])


def cells(rs):
    d = defaultdict(dict)  # n -> width -> {seed: final test pair}
    for r in rs:
        d[r["config"]["n"]].setdefault(r["width"], {})[r["seed"]] = r["final"]["test_pair"]
    return d


summary = {}
print("encoding  n   pooled m*   any-seed  all-seed   (final test pair >= 0.95)")
for k, rs in RUNS.items():
    summary[k] = {}
    for n, byw in sorted(cells(rs).items()):
        ws = sorted(byw)
        pooled = crossing(ws, [np.mean(list(byw[w].values())) for w in ws])
        anyw = next((w for w in ws if any(v >= TH for v in byw[w].values())), None)
        allw = next((w for w in ws if all(v >= TH for v in byw[w].values())), None)
        summary[k][n] = (pooled, anyw, allw, len(next(iter(byw.values()))))
        print(f"{k:<8} {n:>4}  {pooled:8.1f}  {anyw!s:>8}  {allw!s:>8}")

# 1. Headline at depth 12: m* against n for the three encodings.
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.plot([32, 40, 48, 56], [22.4, 33.1, 44.7, 59.6], color=INK2, lw=1, ls="--", marker=".",
        label="adjacency rows, depth 6 (earlier result)")
for k, (lab, col, mk, _) in ENC.items():
    pts = [(n, s) for n, s in sorted(summary[k].items()) if not math.isnan(s[0])]
    ns = [n for n, _ in pts]
    xoff = {"base": 1.0, "eb": 0.985, "rid": 1.015}[k]
    ax.plot([n * xoff for n in ns], [s[0] for _, s in pts], marker=mk, color=col, label=lab,
            markeredgecolor="white", markeredgewidth=0.8, zorder=3)
    for n, s in pts:  # bar: widest grid width where some seed passes .. where all seeds pass
        lo, hi = s[1] or s[0], s[2] or s[0]
        ax.plot([n * xoff] * 2, [min(lo, s[0]), max(hi, s[0])], color=col, lw=4, alpha=0.25,
                solid_capstyle="butt", zorder=2)
ax.annotate("", xy=(72, 230), xytext=(72, 150), arrowprops=dict(arrowstyle="->", color=C[0], lw=1.5))
ax.plot([72], [150], marker="o", color=C[0], markeredgecolor="white")
ax.text(74, 150, "> 144: no width passes\n(underfits in 24k steps)", fontsize=7, color=INK2, va="center")
ax.set_xscale("log"); ax.set_yscale("log"); ax.minorticks_off()
ax.set_xticks([32, 40, 48, 56, 72, 96, 128]); ax.set_xticklabels([32, 40, 48, 56, 72, 96, 128])
ax.set_yticks([4, 8, 16, 32, 64, 128]); ax.set_yticklabels([4, 8, 16, 32, 64, 128])
ax.set(xlabel="graph size n (nodes)", ylabel="critical width m* (test pair accuracy 0.95)", ylim=(3, 260))
ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
fig.savefig(OUT + "d12-mstar.png"); plt.close(fig)

# 2. Every run: final test pair accuracy against width, one panel per n, one dot per seed.
NS = [32, 40, 48, 56, 72, 96, 128]
fig, axs = plt.subplots(2, 4, figsize=(7.4, 4.0), sharey=True)
for ax, n in zip(axs.flat, NS):
    for k, (lab, col, mk, _) in ENC.items():
        byw = cells(RUNS[k]).get(n)
        if not byw:
            continue
        ws = sorted(byw)
        for w in ws:
            v = list(byw[w].values())
            jit = np.linspace(-0.06, 0.06, len(v)) if len(v) > 1 else [0]
            ax.scatter([w * 2 ** j for j in jit], v, s=12, color=col, marker=mk, alpha=0.75,
                       edgecolors="none", zorder=3)
        ax.plot(ws, [np.mean(list(byw[w].values())) for w in ws], color=col, lw=1.2, zorder=2)
    ax.axhline(TH, color=INK, ls="--", lw=0.8)
    ax.set_xscale("log", base=2); ax.minorticks_off()
    xt = [2, 4, 8, 16, 32, 64, 128]
    ax.set_xticks(xt); ax.set_xticklabels(xt, fontsize=7)
    ax.set_title(f"n = {n}", fontsize=8.5); ax.set_ylim(0.72, 1.01); ax.set_xlim(1.6, 190)
for ax in axs[:, 0]:
    ax.set_ylabel("final test pair acc.")
for ax in axs[1, :3]:
    ax.set_xlabel("width m")
leg = axs.flat[-1]; leg.axis("off")
for k, (lab, col, mk, _) in ENC.items():
    leg.scatter([], [], color=col, marker=mk, label=lab.replace(" (", "\n("))
leg.plot([], [], color=INK, ls="--", lw=0.8, label="0.95 threshold")
leg.legend(fontsize=6.5, loc="center", handletextpad=0.3)
fig.tight_layout(); fig.savefig(OUT + "d12-seeds.png"); plt.close(fig)

# 3. Training curves of random-ID edge-bias runs at m = 8: solved, stuck, collapsed.
fig, ax = plt.subplots(figsize=(5.6, 3.1))
pick = [((48, 8, 0), "n = 48, seed 0: solves", C[0], "-"),
        ((48, 8, 2), "n = 48, seed 2: solves, then collapses", C[1], "-"),
        ((32, 8, 2), "n = 32, seed 2: solves, then partly collapses", C[3], "-"),
        ((56, 8, 1), "n = 56, seed 1: reaches 0.97, falls back", C[4], "-"),
        ((48, 8, 1), "n = 48, seed 1: never leaves the trivial predictor", INK2, ":")]
by = {(r["config"]["n"], r["width"], r["seed"]): r for r in RUNS["rid"]}
for key, lab, col, ls in pick:
    h = by[key]["history"]
    ax.plot([e["epoch"] for e in h], [e["test_pair"] for e in h], color=col, ls=ls, lw=1.6, label=lab)
ax.axhline(TH, color=INK, ls="--", lw=0.8)
ax.set(xlabel="epoch (1000 steps each)", ylabel="test pair accuracy", ylim=(0.72, 1.01))
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
fig.savefig(OUT + "d12-collapse.png"); plt.close(fig)

# 4. Baseline at n = 72: depth 12 underfits; depth 6 (m = 96) for reference.
d6 = [r for r in rows("q3n72.shard*", "q3n72.jsonl") if r["width"] == 96]
d12 = defaultdict(list)
for r in RUNS["base"]:
    if r["config"]["n"] == 72:
        d12[r["width"]].append(r)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
for ax, key, title in ((axs[0], "train_pair", "(a) training set"), (axs[1], "test_pair", "(b) test set")):
    for (w, rs), col in zip(sorted(d12.items()), C):
        ep = [e["epoch"] for e in rs[0]["history"]]
        ax.plot(ep, np.mean([[e[key] for e in r["history"]] for r in rs], 0), color=col,
                label=f"depth 12, m = {w}")
    ep = [e["epoch"] for e in d6[0]["history"]]
    ax.plot(ep, np.mean([[e[key] for e in r["history"]] for r in d6], 0), color=INK2, ls="--", lw=1.4,
            label="depth 6, m = 96")
    ax.axhline(TH, color=INK, ls=":", lw=0.8)
    ax.set(title=title, xlabel="epoch (1000 steps each)", ylim=(0.72, 1.0))
axs[0].set_ylabel("pair accuracy (2-seed mean)")
axs[1].legend(fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(OUT + "d12-n72.png"); plt.close(fig)

# Pass counts for the tables: final >= 0.95 / peak (any epoch) >= 0.95.
for k in ("eb", "rid"):
    print(f"\n{k}: n width  pass@end/pass@peak/seeds")
    d = defaultdict(list)
    for r in RUNS[k]:
        d[(r["config"]["n"], r["width"])].append(r)
    for key in sorted(d):
        rs = d[key]
        print(f"  {key}  {sum(r['final']['test_pair'] >= TH for r in rs)}/"
              f"{sum(r['peak'] >= TH for r in rs)}/{len(rs)}")
