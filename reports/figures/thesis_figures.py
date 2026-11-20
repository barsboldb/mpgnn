"""Every figure of the thesis, in one style and with short labels. Reads results/width/*.jsonl and
writes thesis/figures/*.png plus a copy in reports/figures/thesis/ for the slides.

Same data and rules as the scripts each figure first came from (width_progress_plots.py for the
depth-6 figures, width_report_1120_plot.py for the depth-12 ones); only the styling is shared here.
Run from the repo root:  .venv/bin/python reports/figures/thesis_figures.py
"""
import glob
import json
import math
import os
import shutil
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join("reports", "figures"))
from width_sweep import cell, load_runs  # noqa: E402
from width_progress_plots import seed_crossings  # noqa: E402

OUTS = [os.path.join("thesis", "figures"), os.path.join("reports", "figures", "thesis")]
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MK = ["o", "s", "^", "D", "v"]
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
TH = 0.95
W1, H1 = 4.6, 3.0          # every single-panel figure
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["CMU Serif", "DejaVu Serif"], "mathtext.fontset": "cm",
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 2, "lines.markersize": 5.5, "savefig.dpi": 220, "savefig.bbox": "tight",
    "axes.unicode_minus": False, "figure.facecolor": BG, "axes.facecolor": BG,
    "legend.frameon": False, "legend.fontsize": 8,
})
EDGE = dict(markeredgecolor=BG, markeredgewidth=1.2)


def save(fig, name):
    for d in OUTS:
        os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(OUTS[0], name))
    shutil.copy(os.path.join(OUTS[0], name), os.path.join(OUTS[1], name))
    plt.close(fig)


def legend_top(ax, ncol):
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=ncol, handlelength=1.8,
              columnspacing=1.2, borderaxespad=0.2)


def log2_axes(ax, xt, yt=None):
    ax.set_xscale("log", base=2); ax.set_xticks(xt); ax.set_xticklabels([str(t) for t in xt])
    if yt:
        ax.set_yscale("log", base=2); ax.set_yticks(yt); ax.set_yticklabels([str(t) for t in yt])
    ax.minorticks_off()


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


def scaling_series(ax, i, label, runs, ns, key, off, fmt="{:.2f}"):
    pts = []
    for n in ns:
        per = seed_crossings(runs, n, None, key, "final")
        if len(per) >= 2 and not np.isnan(per).any():
            pts.append((n, np.mean(per), min(per), max(per)))
    x, y, lo, hi = map(np.array, zip(*pts))
    ax.errorbar(x, y, yerr=[y - lo, hi - y], color=C[i], marker=MK[i], label=label,
                capsize=2.5, elinewidth=1, zorder=3, **EDGE)
    b = np.polyfit(np.log(x), np.log(y), 1)[0]
    ax.annotate(rf"$n^{{{fmt.format(b)}}}$", (x[-1], y[-1]), xytext=off, textcoords="offset points",
                color=INK, fontsize=8)


# ---- Chapter 2: baseline -------------------------------------------------------------------
def fig_q3lr_curves():
    runs = load_runs("q3lr")
    fig, ax = plt.subplots(figsize=(W1, H1))
    for i, n in enumerate((32, 40, 48, 56)):
        d = defaultdict(list)
        for r in runs:
            if cell(r)[0] == n:
                d[r["width"]].append(r["final"]["test_pair"])
        ws = sorted(d)
        ax.plot(ws, [np.mean(d[w]) for w in ws], color=C[i], marker=MK[i], label=f"n = {n}", **EDGE)
    ax.axhline(TH, color=INK2, lw=1, ls=(0, (3, 3)), zorder=1)
    log2_axes(ax, [12, 16, 24, 32, 48, 64, 96])
    ax.set(xlim=(11, 105), ylim=(0.76, 1.005), xlabel="width m", ylabel="test pair acc.")
    legend_top(ax, 4)
    save(fig, "width-q3lr-curves.png")


def fig_q3_scaling():
    fine = [r for r in load_runs("q3fine") if r["width"] <= 96]
    final = load_runs("q3lr")
    ns = [32, 40, 48, 56]
    fig, ax = plt.subplots(figsize=(W1, H1))
    scaling_series(ax, 0, "test, 32k graphs", fine, ns, "test_pair", (-30, 7))
    scaling_series(ax, 1, "train, 32k graphs", fine, ns, "train_pair", (7, -3))
    scaling_series(ax, 2, "test, 128k graphs", final, ns, "test_pair", (7, -6))
    ax.annotate("", xy=(56, 125), xytext=(56, 96), arrowprops=dict(arrowstyle="->", color=C[0], lw=1.5))
    ax.plot([56], [96], marker="o", mfc="none", mec=C[0], ms=6, zorder=3)
    ax.text(57.5, 104, "> 96", color=INK2, fontsize=7.5)
    ref = np.array([30, 62])
    ax.plot(ref, 18.6 * ref / 32, color=INK2, lw=1, ls=(0, (3, 3)), zorder=1)
    ax.text(62.5, 34.5, r"$n^{1}$", color=INK2, fontsize=8)
    log2_axes(ax, ns, [16, 24, 32, 48, 64, 96, 128])
    ax.set(xlim=(29, 70), xlabel="graph size n", ylabel=r"width $m^*$")
    legend_top(ax, 3)
    save(fig, "width-q3-scaling.png")


# ---- Chapter 3: encodings at depth 6 ---------------------------------------------------------
def fig_encodings():
    fig, ax = plt.subplots(figsize=(W1, H1))
    series = (("adj. rows", "q3lr", (7, 3)), ("+ RWSE", "enc_rwse8", (7, -11)),
              ("+ 2-step", "enc_khop2", (7, -3)), ("+ 4-step", "enc_khop4", (7, -3)))
    for i, (label, sw, off) in enumerate(series):
        runs = load_runs(sw)
        scaling_series(ax, i, label, runs, sorted({r["config"]["n"] for r in runs}), "test_pair", off)
    edges = load_runs("enc_edges96k")
    m32 = np.mean(seed_crossings(edges, 32, None, "test_pair", "final"))
    m40 = np.nanmax(seed_crossings(edges, 40, None, "test_pair", "final"))
    ax.plot([32], [m32], marker="v", color=INK2, ms=6, ls="none", zorder=3, label="edge tokens")
    ax.plot([40], [m40], marker="v", mfc="none", mec=INK2, ms=6, zorder=3)
    ax.annotate("", xy=(40, m40 * 1.45), xytext=(40, m40 * 1.05),
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1.2))
    log2_axes(ax, [32, 40, 48, 56, 72], [2, 4, 8, 16, 32, 64, 128])
    ax.set(xlim=(29, 92), ylim=(2.5, 140), xlabel="graph size n", ylabel=r"width $m^*$")
    legend_top(ax, 3)
    save(fig, "width-encodings.png")


# ---- Chapter 3: depth 12, attention bias -----------------------------------------------------
ENC = {  # label, colour, marker, sweeps
    "base": ("adj. rows", C[0], "o", ["q3lr_d12h8.*", "q3lr_d12h8_72.*"]),
    "eb": ("adj. rows + bias", C[1], "^",
           ["ebd12hd2_s.*", "ebd12hd2_72.*", "ebd12hd2_96.*", "ebd12hd2_128.*", "ebd12hd2_L.*",
            "ebd12hd2_128w4.*"]),
    "rid": ("random ID + bias", C[2], "s", ["ebd12rid_b2c.*", "ebd12rid_b2.*", "ebd12rid_b2L.*"]),
}
RUNS = {k: rows(*v[3]) for k, v in ENC.items()}


def cells(rs):
    d = defaultdict(dict)  # n -> width -> {seed: final test pair}
    for r in rs:
        d[r["config"]["n"]].setdefault(r["width"], {})[r["seed"]] = r["final"]["test_pair"]
    return d


def fig_d12_mstar():
    fig, ax = plt.subplots(figsize=(W1, H1))
    ax.plot([32, 40, 48, 56], [22.4, 33.1, 44.7, 59.6], color=INK2, lw=1, ls="--", marker=".",
            label="adj. rows, depth 6")
    for k, (lab, col, mk, _) in ENC.items():
        pts = []
        for n, byw in sorted(cells(RUNS[k]).items()):
            ws = sorted(byw)
            pooled = crossing(ws, [np.mean(list(byw[w].values())) for w in ws])
            anyw = next((w for w in ws if any(v >= TH for v in byw[w].values())), None)
            allw = next((w for w in ws if all(v >= TH for v in byw[w].values())), None)
            if not math.isnan(pooled):
                pts.append((n, pooled, anyw or pooled, allw or pooled))
        xoff = {"base": 1.0, "eb": 0.985, "rid": 1.015}[k]
        ax.plot([n * xoff for n, *_ in pts], [p for _, p, *_ in pts], marker=mk, color=col,
                label=lab, zorder=3, **EDGE)
        for n, p, lo, hi in pts:  # band: first width where some run passes .. where all pass
            ax.plot([n * xoff] * 2, [min(lo, p), max(hi, p)], color=col, lw=4, alpha=0.25,
                    solid_capstyle="butt", zorder=2)
    ax.annotate("", xy=(72, 230), xytext=(72, 150), arrowprops=dict(arrowstyle="->", color=C[0], lw=1.5))
    ax.plot([72], [150], marker="o", color=C[0], **EDGE)
    ax.text(76, 150, "> 144", fontsize=7.5, color=INK2, va="center")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.minorticks_off()
    xt, yt = [32, 40, 48, 56, 72, 96, 128], [4, 8, 16, 32, 64, 128]
    ax.set_xticks(xt); ax.set_xticklabels(xt); ax.set_yticks(yt); ax.set_yticklabels(yt)
    ax.set(xlabel="graph size n", ylabel=r"width $m^*$", ylim=(3, 260))
    legend_top(ax, 2)
    save(fig, "d12-mstar.png")


def fig_d12_seeds():
    ns = [32, 40, 48, 56, 72, 96, 128]
    fig, axs = plt.subplots(2, 4, figsize=(7.0, 3.8), sharey=True)
    for ax, n in zip(axs.flat, ns):
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
        ax.axhline(TH, color=INK2, lw=1, ls=(0, (3, 3)), zorder=1)
        log2_axes(ax, [2, 8, 32, 128])
        ax.set_title(f"n = {n}", fontsize=9); ax.set_ylim(0.72, 1.01); ax.set_xlim(1.6, 190)
    for ax in axs[:, 0]:
        ax.set_ylabel("test pair acc.")
    for ax in axs[1, :3]:
        ax.set_xlabel("width m")
    leg = axs.flat[-1]; leg.axis("off")
    for k, (lab, col, mk, _) in ENC.items():
        leg.scatter([], [], color=col, marker=mk, label=lab)
    leg.legend(loc="center", handletextpad=0.3)
    fig.tight_layout()
    save(fig, "d12-seeds.png")


def fig_d12_collapse():
    """Random-ID runs at m = 8 with the bias started at 4: solved, collapsed, stuck."""
    by = {(r["config"]["n"], r["width"], r["seed"]): r for r in rows("ebd12rid.*", "ebd12rid_s23.*")}
    pick = [((48, 8, 0), "solves", "-"), ((48, 8, 2), "collapses", "-"),
            ((32, 8, 2), "partly collapses", "-"), ((56, 8, 1), "falls back", "-"),
            ((48, 8, 1), "stays at baseline", ":")]
    fig, ax = plt.subplots(figsize=(W1, H1))
    for i, (key, lab, ls) in enumerate(pick):
        h = by[key]["history"]
        ax.plot([e["epoch"] for e in h], [e["test_pair"] for e in h], color=INK2 if ls == ":" else C[i],
                ls=ls, lw=1.6, label=f"n = {key[0]}: {lab}")
    ax.axhline(TH, color=INK2, lw=1, ls=(0, (3, 3)), zorder=1)
    ax.set(xlabel="epoch", ylabel="test pair acc.", ylim=(0.72, 1.01))
    legend_top(ax, 2)
    save(fig, "d12-collapse.png")


if __name__ == "__main__":
    for f in (fig_q3lr_curves, fig_q3_scaling, fig_encodings, fig_d12_mstar, fig_d12_seeds,
              fig_d12_collapse):
        f()
        print("wrote", f.__name__)
