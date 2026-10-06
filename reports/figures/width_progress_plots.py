"""Figures for reports/width-progress.typ, drawn from results/width/*.jsonl.

Run from the repo root:  .venv/bin/python reports/figures/width_progress_plots.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.getcwd())
from width_sweep import cell, crossing, load_runs  # noqa: E402

OUT = os.path.join("reports", "figures")
# Validated categorical slots 1-3 (dataviz reference palette, light mode, all-pairs).
# Validated categorical slots (dataviz reference palette, light): 1-3 all-pairs, 1-4 adjacent.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]               # secondary encoding: identity never by color alone
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["CMU Serif", "DejaVu Serif"],
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 2, "lines.markersize": 5.5, "savefig.dpi": 220,
    "savefig.bbox": "tight", "axes.unicode_minus": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
})


def tuned(runs, key="test"):
    """{width: [tuned value per seed]}: best-validation LR per (width, seed)."""
    out = {}
    for w in sorted({r["width"] for r in runs}):
        for s in sorted({r["seed"] for r in runs}):
            c = [r for r in runs if r["width"] == w and r["seed"] == s]
            if c:
                b = max(c, key=lambda r: r["best_val"]["val"])
                out.setdefault(w, []).append(b["best_val"][key])
    return out


def fixed(runs, lr, key="test", at="best_val"):
    out = {}
    for r in runs:
        if r["lr"] == lr:
            out.setdefault(r["width"], []).append(r[at][key])
    return dict(sorted(out.items()))


def legend_top(ax, ncol=4):
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=ncol,
              fontsize=8, handlelength=1.8, columnspacing=1.2, borderaxespad=0.2)


def line(ax, d, i, label, label_at=None, dy=0.0):
    ws = list(d)
    ys = [np.mean(d[w]) for w in ws]
    ax.plot(ws, ys, color=SERIES[i], marker=MARKERS[i], label=label,
            markeredgecolor="#fcfcfb", markeredgewidth=1.2, zorder=3)
    if label_at is not None:                       # selective direct label
        k = ws.index(label_at)
        ax.annotate(label, (ws[k], ys[k]), xytext=(6, 6 + dy), textcoords="offset points",
                    color=INK, fontsize=8.5)


def width_axis(ax, ws):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ws)
    ax.set_xticklabels([str(w) for w in ws])
    ax.minorticks_off()
    ax.set_xlabel("width m (embedding dimension)")


def fig_q1b():
    runs = load_runs("q1b")
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    line(ax, tuned(runs), 0, "LR tuned per width", label_at=128, dy=4)
    line(ax, fixed(runs, 1e-3), 1, "fixed LR 1e-3", label_at=32, dy=-4)
    ax.axhline(0.5, color=INK2, lw=1, ls=(0, (3, 3)), zorder=1)
    ax.text(5.6, 0.415, "dashed: all-connected predictor (0.5)", color=INK2, fontsize=7.5)
    width_axis(ax, [2, 4, 8, 16, 32, 64, 128, 256])
    ax.set_ylabel("test exact-match")
    ax.set_ylim(0.4, 1.02)
    legend_top(ax)
    fig.savefig(os.path.join(OUT, "width-q1b-lr.png"))


def fig_q1c():
    runs = load_runs("q1c")
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    for i, (n, at, dy) in enumerate(((500, 16, 0), (2000, 128, -18), (8000, 32, 2))):
        sub = [r for r in runs if cell(r)[2] == n]
        line(ax, tuned(sub), i, f"{n} graphs", label_at=at, dy=dy)
    ax.axhline(0.5, color=INK2, lw=1, ls=(0, (3, 3)), zorder=1)
    width_axis(ax, [8, 16, 32, 64, 128, 256])
    ax.set_ylabel("test exact-match (LR tuned)")
    ax.set_ylim(0.3, 1.04)
    legend_top(ax)
    fig.savefig(os.path.join(OUT, "width-q1c-data.png"))


def fig_q3trim():
    runs = load_runs("q3trim")
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    for i, (n, at, dy) in enumerate(((32, 32, -14), (64, 128, -16), (128, 128, 6))):
        sub = [r for r in runs if cell(r)[0] == n]
        # Final-epoch pair accuracy at the fixed LR 3e-3 (no selection on test). The
        # all-connected predictor's score is what every m=8 run sits at.
        pair = fixed(sub, 3e-3, "test_pair", at="final")
        trivial = np.mean(pair[8])
        gain = {w: [v - trivial for v in vs] for w, vs in pair.items()}
        line(ax, gain, i, f"n = {n}", label_at=at, dy=dy)
    ax.axhline(0, color=INK2, lw=1, ls=(0, (3, 3)), zorder=1)
    width_axis(ax, [8, 16, 32, 64, 128])
    ax.set_ylabel("final test pair acc. - trivial")
    legend_top(ax)
    fig.savefig(os.path.join(OUT, "width-q3trim-onset.png"))


def fig_q3fine_curves():
    runs = load_runs("q3fine")
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    for i, (n, at, dy) in enumerate(((32, 128, 2), (40, 128, -13), (48, 128, 3), (56, 128, -12))):
        sub = [r for r in runs if cell(r)[0] == n]
        d = {}
        for r in sub:
            # final epoch: the best-val checkpoint is picked on exact-match, which ties at
            # 0.5 when no graph is fully right and then returns epoch 1 (trivial pair acc).
            d.setdefault(r["width"], []).append(r["final"]["test_pair"])
        line(ax, dict(sorted(d.items())), i, f"n = {n}", label_at=at, dy=dy)
    ax.axhline(0.95, color=INK2, lw=1, ls=(0, (3, 3)), zorder=1)
    ax.text(16.5, 0.938, "threshold 0.95", color=INK2, fontsize=7.5)
    width_axis(ax, [16, 24, 32, 48, 64, 96, 128])
    ax.set_xlim(14, 190)
    ax.set_ylabel("test pair accuracy")
    ax.set_ylim(0.72, 1.01)
    legend_top(ax)
    fig.savefig(os.path.join(OUT, "width-q3fine-curves.png"))


def seed_crossings(runs, n, widths, key, at, th=0.95):
    """Per-seed critical widths (nan where a seed never reaches th)."""
    widths = widths or sorted({r["width"] for r in runs if cell(r)[0] == n})
    out = []
    for sd in sorted({r["seed"] for r in runs}):
        ys = [np.mean([r[at][key] for r in runs if cell(r)[0] == n and r["width"] == w
                       and r["seed"] == sd]) for w in widths]
        out.append(crossing(widths, ys, th))
    return out


def fig_q3fine_scaling():
    fine = [r for r in load_runs("q3fine") if r["width"] <= 96]   # 128: LR too hot
    big = [r for r in load_runs("q3data") + load_runs("q3big") if cell(r)[2] == 128000]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    series = (
        ("generalize, 32k graphs", fine, [32, 40, 48, 56], [16, 24, 32, 48, 64, 96], "test_pair"),
        ("fit, 32k graphs", fine, [32, 40, 48, 56], [16, 24, 32, 48, 64, 96], "train_pair"),
        ("learn, 128k graphs", big, [32, 40, 48, 56], None, "test_pair"),
    )
    for i, (label, runs, ns, widths, key) in enumerate(series):
        pts = []
        for n in ns:
            per = seed_crossings(runs, n, widths, key, "final")
            if not any(np.isnan(per)):
                pts.append((n, np.mean(per), min(per), max(per)))
        x, y, lo, hi = map(np.array, zip(*pts))
        ax.errorbar(x, y, yerr=[y - lo, hi - y], color=SERIES[i], marker=MARKERS[i],
                    label=label, markeredgecolor="#fcfcfb", markeredgewidth=1.2,
                    capsize=2.5, elinewidth=1, zorder=3)
        if len(x) >= 3:
            b = np.polyfit(np.log(x), np.log(y), 1)[0]
            ax.annotate(rf"$\propto n^{{{b:.2f}}}$", (x[-1], y[-1]),
                        xytext={0: (-46, 7), 1: (7, -3), 2: (7, -1)}[i],
                        textcoords="offset points", color=INK, fontsize=8)
    # 32k, n=56: no seed generalized at m <= 96
    ax.annotate("", xy=(56, 125), xytext=(56, 96),
                arrowprops=dict(arrowstyle="->", color=SERIES[0], lw=1.5))
    ax.plot([56], [96], marker="o", mfc="none", mec=SERIES[0], ms=6, zorder=3)
    ax.text(57.5, 104, "> 96", color=INK2, fontsize=7.5)
    ref = np.array([30, 62])
    ax.plot(ref, 18.6 * ref / 32, color=INK2, lw=1, ls=(0, (3, 3)), zorder=1)
    ax.text(62.5, 34.5, r"$\propto n$", color=INK2, fontsize=8)
    ns = [32, 40, 48, 56]
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
    ax.set_yticks([16, 24, 32, 48, 64, 96, 128])
    ax.set_yticklabels(["16", "24", "32", "48", "64", "96", "128"])
    ax.minorticks_off()
    ax.set_xlim(29, 70)
    ax.set_xlabel("graph size n")
    ax.set_ylabel("critical width m* (pair acc. 0.95)")
    legend_top(ax, ncol=2)
    fig.savefig(os.path.join(OUT, "width-q3fine-scaling.png"))


if __name__ == "__main__":
    fig_q1b()
    fig_q1c()
    fig_q3trim()
    fig_q3fine_curves()
    fig_q3fine_scaling()
    print("wrote", ", ".join(f for f in sorted(os.listdir(OUT)) if f.startswith("width-")))
