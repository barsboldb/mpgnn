"""Figures for reports/width-report-2026-11-10.typ (the full report since the width topic, 1 Oct -
10 Nov 2026). Curves are read from results/width/*.jsonl; critical widths m* are the per-seed
crossings printed by `width_sweep.py <sweep> --scaling` (seed means), as quoted in docs/CHANGELOG.md.
Palette: categorical slots 1-7 of the dataviz reference palette, in fixed order, each series also
with its own marker (three slots are below 3:1 contrast on white, so identity never rests on colour).
Run from the repo root: .venv/bin/python reports/figures/width_report_1110_plot.py"""
import glob
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
OUT = "reports/figures/"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2,
                     "lines.markersize": 6, "legend.frameon": False, "savefig.dpi": 200,
                     "savefig.bbox": "tight"})


def rows(*pats):
    return [json.loads(l) for p in pats for f in sorted(glob.glob("results/width/" + p)) for l in open(f)]


def logaxes(ax, xt, yt):
    ax.set_xscale("log"); ax.set_yscale("log"); ax.minorticks_off()
    ax.set_xticks(xt); ax.set_xticklabels(xt); ax.set_yticks(yt); ax.set_yticklabels(yt)


# 1. Headline: m* against n for every input / method (seed means of per-seed crossings).
fig, ax = plt.subplots(figsize=(6.6, 4.2))
series = [
    ("adjacency rows (baseline)", [32, 40, 48, 56], [22.4, 33.1, 44.7, 59.6], "o"),
    ("+ random-walk returns (RWSE, 8 steps)", [32, 40, 48, 56], [23.3, 31.9, 45.2, 57.9], "v"),
    ("+ auxiliary k-hop loss (training only)", [32, 40, 48, 56], [21.5, 26.8, 33.5, 45.1], "P"),
    ("+ 2-hop rows", [32, 40, 48, 56, 72], [9.5, 12.7, 16.9, 18.5, 27.1], "s"),
    ("+ 4-hop rows", [32, 40, 48, 56, 72], [3.5, 3.6, 3.7, 4.5, 6.5], "D"),
    ("edge bias in attention", [32, 40, 48, 56, 72, 96, 128], [4.6, 4.5, 4.5, 5.6, 6.9, 6.0, 6.6], "^"),
]
for (lab, ns, ms, mk), col in zip(series, C):
    ax.plot(ns, ms, marker=mk, color=col, label=lab, markeredgecolor="white", markeredgewidth=0.8)
ax.plot([32], [48.2], marker="X", color=C[6], ls="none", ms=8, label="edge tokens (Sanford), 4x steps")
ax.annotate("", xy=(40, 95), xytext=(40, 72), arrowprops=dict(arrowstyle="->", color=C[6], lw=1.5))
ax.plot([40], [72], marker="X", color=C[6], ms=8)
logaxes(ax, [32, 40, 48, 56, 72, 96, 128], [4, 8, 16, 32, 64])
ax.set(xlabel="graph size n (nodes)", ylabel="critical width m* (test pair accuracy 0.95)", ylim=(2.8, 110))
ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
fig.savefig(OUT + "rep-mstar.png"); plt.close(fig)

# 2. Q1c: test exact-match against width for three training-set sizes (LR picked on validation).
fig, ax = plt.subplots(figsize=(5.2, 3.0))
best = {}
for r in rows("q1c.shard*of8.jsonl"):
    k = (r["train_size"], r["width"], r["seed"])
    if k not in best or r["final"]["val"] > best[k]["final"]["val"]:
        best[k] = r
for (size, mk), col in zip(((500, "o"), (2000, "s"), (8000, "^")), C):
    ws = sorted({k[1] for k in best if k[0] == size})
    ax.plot(ws, [np.mean([best[k]["final"]["test"] for k in best if k[:2] == (size, w)]) for w in ws],
            marker=mk, color=col, label=f"{size} training graphs", markeredgecolor="white", markeredgewidth=0.8)
ax.axhline(0.5, color=INK2, ls=":", lw=1); ax.text(260, 0.51, "always-yes", fontsize=7, color=INK2, ha="right")
ax.set_xscale("log", base=2); ax.minorticks_off(); ax.set_xticks([8, 16, 32, 64, 128, 256])
ax.set_xticklabels([8, 16, 32, 64, 128, 256])
ax.set(xlabel="width m", ylabel="test accuracy (whole matrix right)", ylim=(0.3, 1.02))
ax.legend(fontsize=7.5, loc="lower left")
fig.savefig(OUT + "rep-q1c.png"); plt.close(fig)

# 3. Baseline pair-accuracy curves (adjacency rows, depth 6, 128k graphs, LR 1e-3).
fig, ax = plt.subplots(figsize=(5.6, 3.3))
acc = defaultdict(list)
for r in rows("q3lr*.jsonl", "q3n72.jsonl", "q3n72.shard*.jsonl"):
    c = r["config"]
    if c.get("depth") == 6 and c.get("train") == 128000 and c.get("steps") == 24000:
        acc[(c["n"], r["width"])].append(r["final"]["test_pair"])
for (n, mk), col in zip(((32, "o"), (40, "v"), (48, "s"), (56, "D"), (72, "^")), C):
    ws = sorted(w for (nn, w) in acc if nn == n)
    ax.plot(ws, [np.mean(acc[(n, w)]) for w in ws], marker=mk, color=col, label=f"n = {n}",
            markeredgecolor="white", markeredgewidth=0.8)
ax.axhline(0.95, color=INK, ls="--", lw=1); ax.text(13, 0.955, "0.95 threshold", fontsize=7, color=INK)
ax.set_xscale("log", base=2); ax.minorticks_off(); ax.set_xticks([12, 16, 24, 32, 48, 64, 96, 128])
ax.set_xticklabels([12, 16, 24, 32, 48, 64, 96, 128])
ax.set(xlabel="width m", ylabel="test pair accuracy (seed mean)", ylim=(0.74, 1.005))
ax.legend(fontsize=7.5, loc="lower right")
fig.savefig(OUT + "rep-baseline-curves.png"); plt.close(fig)

# 4. Depth against width for adjacency rows: (a) n = 40 over depth, (b) depth 2 vs 6 over n.
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.9))
a.plot([2, 3, 4, 6, 8], [45.3, 37.9, 34.5, 30.1, 32.3], marker="o", color=C[0])
a.set(xlabel="depth (layers)", ylabel="critical width m*", title="(a) n = 40", ylim=(0, 50))
a.set_xticks([2, 3, 4, 6, 8])
b.plot([32, 40, 48, 56], [22.4, 33.1, 44.7, 59.6], marker="o", color=C[0], label="depth 6")
b.plot([32, 40, 48], [28.2, 45.3, 92.7], marker="s", color=C[1], label="depth 2")
b.annotate("", xy=(56, 250), xytext=(56, 192), arrowprops=dict(arrowstyle="->", color=C[1], lw=1.5))
b.plot([56], [192], marker="s", color=C[1]); b.text(54, 205, "> 192", fontsize=7, color=INK2, ha="right")
logaxes(b, [32, 40, 48, 56], [16, 32, 64, 128, 256])
b.set(xlabel="graph size n", title="(b) depth 2 against depth 6"); b.legend(fontsize=7.5, loc="upper left")
fig.tight_layout(); fig.savefig(OUT + "rep-depth.png"); plt.close(fig)

# 5. Probing: can a bilinear probe read "connected" from each layer? (k = infinity, AUC, seed mean)
fig, ax = plt.subplots(figsize=(5.4, 3.1))
cfgs = [((40, 16), "o", "n = 40, m = 16 (below m*)"), ((40, 48), "s", "n = 40, m = 48 (above m*)"),
        ((56, 32), "v", "n = 56, m = 32 (below m*)"), ((56, 96), "D", "n = 56, m = 96 (above m*)")]
for ((n, m), mk, lab), col in zip(cfgs, C):
    ds = [json.load(open(f)) for f in glob.glob(f"results/width/probe/n{n}_L6_m{m}_*.json")]
    tr = np.mean([[layer[-1] for layer in d["probes"]["trained/pair"]] for d in ds], 0)
    un = np.mean([[layer[-1] for layer in d["probes"]["untrained/pair"]] for d in ds], 0)
    ax.plot(range(len(tr)), tr, marker=mk, color=col, label=lab, markeredgecolor="white", markeredgewidth=0.8)
    ax.plot(range(len(un)), un, color=col, lw=1, ls=":")
ax.text(6, 0.478, "dotted: same model untrained", fontsize=7, color=INK2, ha="right")
ax.set(xlabel="layer (0 = after read-in)", ylabel="probe AUC for 'connected'", ylim=(0.46, 1.02))
ax.legend(fontsize=7.5, loc="upper left")
fig.savefig(OUT + "rep-probe.png"); plt.close(fig)

# 6. Edge bias, depth 12, head_dim 2: test pair accuracy against width per n (small multiples).
eb = defaultdict(dict)
for r in rows("ebd12hd2_s.shard*.jsonl", "ebd12hd2_72.shard*.jsonl", "ebd12hd2_96.shard*.jsonl",
              "ebd12hd2_128.shard*.jsonl"):
    eb[r["config"]["n"]].setdefault(r["width"], []).append(r["final"]["test_pair"])
ns = sorted(eb)
fig, axs = plt.subplots(1, len(ns), figsize=(7.6, 2.2), sharey=True)
for ax, n in zip(axs, ns):
    ws = sorted(eb[n])
    ax.plot(ws, [np.mean(eb[n][w]) for w in ws], color=C[5], lw=1.5)
    for w in ws:
        ax.scatter([w] * len(eb[n][w]), eb[n][w], s=14, color=C[5], edgecolor="white", linewidth=0.6, zorder=3)
    ax.axhline(0.95, color=INK, ls="--", lw=0.8)
    ax.set_title(f"n = {n}", fontsize=8.5); ax.set_xticks(ws); ax.set_xticklabels(ws, fontsize=7)
    ax.set_ylim(0.72, 1.01)
axs[0].set_ylabel("test pair accuracy"); fig.supxlabel("width m (dots: seeds, line: mean)", fontsize=8.5, y=0.02)
fig.tight_layout(w_pad=0.4); fig.savefig(OUT + "rep-eb-curves.png"); plt.close(fig)

# 7. Plateau ablation: seeds out of 4 reaching 0.95, one change each to the depth-6 edge-bias model.
conds = ["base", "logn", "ssmax", "clip", "wsd", "all", "d12", "loop2"]
labels = ["none", "bias\nx log n", "scaled\nsoftmax", "grad\nclip", "WSD\nschedule", "all\nthree",
          "12\nlayers", "6 layers\nrun twice"]
fig, ax = plt.subplots(figsize=(6.4, 2.8))
x = np.arange(len(conds))
for i, (n, col) in enumerate(((96, C[0]), (128, C[1]))):
    k = [sum(json.loads(l)["final"]["test_pair"] >= 0.95 for l in open(f"results/width/ebx{n}_{c}.jsonl"))
         for c in conds]
    ax.bar(x + (i - 0.5) * 0.38, k, width=0.36, color=col, label=f"n = {n}, m = {16 if n == 96 else 24}")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5); ax.set_yticks(range(5))
ax.set(ylabel="seeds passing 0.95 (of 4)"); ax.grid(axis="x", visible=False)
ax.legend(fontsize=7.5, loc="upper left")
fig.savefig(OUT + "rep-plateau.png"); plt.close(fig)

# 8. n = 128: seeds passing per width, head_dim 8 (depth 12 and 16) against head_dim 2 (depth 12).
fig, ax = plt.subplots(figsize=(5.4, 2.8))
grp = [("ebd12_128.shard*.jsonl ebd12_128x.shard*.jsonl", "head dim 8, depth 12"),
       ("ebd16_128.shard*.jsonl", "head dim 8, depth 16"), ("ebd12hd2_128.shard*.jsonl", "head dim 2, depth 12")]
ws = [6, 8, 12, 16]; x = np.arange(len(ws))
for i, ((pats, lab), col) in enumerate(zip(grp, (C[0], C[1], C[5]))):
    cell = defaultdict(list)
    for r in rows(*pats.split()):
        cell[r["width"]].append(r["final"]["test_pair"] >= 0.95)
    k = [sum(cell[w]) if cell[w] else np.nan for w in ws]
    ax.bar(x + (i - 1) * 0.27, k, width=0.25, color=col, label=lab)
ax.set_xticks(x); ax.set_xticklabels([f"m = {w}" for w in ws]); ax.set_yticks(range(5))
for xx in (-0.27, 0.0):
    ax.text(xx, 0.08, "not run", fontsize=6.5, color=INK2, ha="center", va="bottom", rotation=90)
ax.set_xlim(-0.55, 3.55)
ax.set(ylabel="seeds passing 0.95 (of 4)", title="n = 128, edge bias"); ax.grid(axis="x", visible=False)
ax.legend(fontsize=7.5, loc="upper left")
fig.savefig(OUT + "rep-n128.png"); plt.close(fig)
print("done")
