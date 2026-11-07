"""Critical width m* vs n for the 2026-11-04 run report (reports/width-runs-2026-11-04.typ).
Values: per-seed test-pair crossings from `width_sweep.py <sweep> --scaling`."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

series = {
    "adjacency rows (q3lr), n^1.74": ([32, 40, 48, 56], [[20.7, 23.1, 23.4], [30.8, 29.4, 39.1],
                                                          [41.2, 43.6, 49.4], [59.2, 56.7, 63.1]]),
    "adjacency rows + aux k-hop loss, n^1.30": ([32, 40, 48, 56], [[20.2, 22.7], [25.8, 27.8],
                                                                    [32.0, 35.0], [44.2, 46.0]]),
    "adjacency + 2-hop rows (enc_khop2), n^1.27": ([32, 40, 48, 56, 72],
                                                    [[10.4, 8.0, 10.1], [12.5, 13.5, 12.0], [16.7, 16.7, 17.4],
                                                     [19.7, 16.8, 19.0], [25.6, 27.7, 28.1]]),
}
fig, ax = plt.subplots(figsize=(6, 4))
for (label, (ns, per)), mk in zip(series.items(), "osD"):
    mean = [np.mean(p) for p in per]
    ax.plot(ns, mean, marker=mk, color="black" if mk == "o" else None, label=label)
    for n, p in zip(ns, per):
        ax.scatter([n] * len(p), p, s=8, color="grey", zorder=3)
ax.scatter([72], [80.9], marker="x", color="black", zorder=4,
           label="adjacency rows, n = 72: train-fit m* only (test never 0.95)")
ax.set(xscale="log", yscale="log", xlabel="graph size n", ylabel="critical width m* (test pair 0.95)")
ax.set_xticks([32, 40, 48, 56, 72]); ax.set_xticklabels([32, 40, 48, 56, 72])
ax.set_yticks([8, 16, 32, 64]); ax.set_yticklabels([8, 16, 32, 64])
ax.minorticks_off(); ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig("reports/figures/width-runs-1104-mstar.png", dpi=200)
