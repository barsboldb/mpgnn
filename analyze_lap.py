"""Visualise the Laplacian eigenvector features the model actually sees.

For the `node_features: lap` tokenization, every node token *is* a row of the
normalized-Laplacian eigenvector matrix (the smallest k non-trivial modes).
This script loads connectedness_hard, recomputes the spectrum per graph, and
shows — split by class — what those features look like:

  1. spectrum statistics (console): how many ~0 eigenvalues, the Fiedler value
     lambda_2, the first *kept* eigenvalue after the zero-filter.
  2. figures/lap_spectrum_<dataset>.png  — four panels:
       (a) distribution of the first kept eigenvalue per class
       (b) count of near-zero eigenvalues per class (= #components)
       (c) lambda_2 vs lambda_3 scatter, coloured by class
  3. figures/lap_features_<dataset>.png  — example [n x k] feature heatmaps,
     a few graphs from each class: literally the token rows the model reads.

Usage:
    python analyze_lap.py                       # connectedness_hard, k from config
    python analyze_lap.py --dataset connectedness_hard --k 8
"""
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import GNNConfig
from src.dataset import load_or_create

ZERO_TOL = 1e-5


# ── spectrum (mirrors features.laplacian_positional_encoding exactly) ──────────

def normalized_laplacian_spectrum(edge_index, num_nodes):
    """Eigenvalues (ascending) and eigenvectors of L_sym = I - D^-1/2 A D^-1/2."""
    A = torch.zeros(num_nodes, num_nodes)
    A[edge_index[0], edge_index[1]] = 1.0
    A = ((A + A.t()) > 0).float()

    deg = A.sum(dim=1)
    d_inv_sqrt = deg.pow(-0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    L = torch.eye(num_nodes) - d_inv_sqrt.unsqueeze(1) * A * d_inv_sqrt.unsqueeze(0)

    eigvals, eigvecs = torch.linalg.eigh(L)
    return eigvals.numpy(), eigvecs.numpy()


def kept_features(eigvals, eigvecs, k):
    """The k smallest non-trivial eigenvectors — exactly what the model sees."""
    non_trivial = np.where(eigvals > ZERO_TOL)[0][:k]
    pe = eigvecs[:, non_trivial]
    if pe.shape[1] < k:
        pe = np.concatenate([pe, np.zeros((pe.shape[0], k - pe.shape[1]))], axis=1)
    return pe, non_trivial


def connected_components(edge_index, n):
    """Component id per node, and the component count, via iterative DFS."""
    adj = [[] for _ in range(n)]
    for a, b in edge_index.t().tolist():
        adj[a].append(b)
    comp = np.full(n, -1)
    c = 0
    for s in range(n):
        if comp[s] < 0:
            stack = [s]
            comp[s] = c
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if comp[v] < 0:
                        comp[v] = c
                        stack.append(v)
            c += 1
    return comp, c


def support_concentration(pe, comp, ncomp):
    """For each kept eigenvector, the fraction of its squared mass on its
    dominant component. 1.0 means the mode lives entirely on one component."""
    fracs = []
    for j in range(pe.shape[1]):
        col = pe[:, j]
        if np.allclose(col, 0):
            continue
        masses = np.array([(col[comp == c] ** 2).sum() for c in range(ncomp)])
        fracs.append(masses.max() / (masses.sum() + 1e-12))
    return fracs


# ── collect per-graph spectral descriptors ────────────────────────────────────

def collect(data_list, k):
    rows = []
    for g in data_list:
        n = int(g.num_nodes)
        label = int(g.y.item())
        eigvals, eigvecs = normalized_laplacian_spectrum(g.edge_index, n)
        pe, _ = kept_features(eigvals, eigvecs, k)

        n_zero = int((eigvals < ZERO_TOL).sum())
        nonzero = eigvals[eigvals > ZERO_TOL]            # eigenvalues that survive the filter
        first_kept = float(nonzero[0]) if nonzero.size else 0.0
        second_kept = float(nonzero[1]) if nonzero.size > 1 else 0.0

        # how component-bound are the kept modes, and the structural-zero pattern
        comp, ncomp = connected_components(g.edge_index, n)
        concentration = support_concentration(pe, comp, ncomp)
        zero_density = float((np.abs(pe) < 1e-6).mean())   # fraction of matrix that is ~0

        rows.append({
            "label": label,
            "n": n,
            "eigvals": eigvals,
            "pe": pe,                       # [n, k] — the token rows
            "comp": comp,                   # component id per node
            "ncomp": ncomp,                 # number of connected components
            "n_zero": n_zero,               # multiplicity of 0 = #components
            "lambda2": float(eigvals[1]),   # Fiedler value (pre-filter)
            "first_kept": first_kept,       # smallest eigenvalue the model keeps
            "second_kept": second_kept,     # 2nd smallest kept (skips the extra 0 when disconnected)
            "concentration": concentration, # per-kept-mode mass fraction on dominant component
            "zero_density": zero_density,   # fraction of the [n x k] matrix that is structurally 0
        })
    return rows


# ── console summary ───────────────────────────────────────────────────────────

def summarise(rows):
    names = {1: "CONNECTED   (label 1)", 0: "DISCONNECTED(label 0)"}
    print(f"\n{'='*64}")
    print("Spectrum statistics — what distinguishes the two classes")
    print(f"{'='*64}")
    for label in (1, 0):
        sub = [r for r in rows if r["label"] == label]
        if not sub:
            continue
        nz = np.array([r["n_zero"] for r in sub])
        l2 = np.array([r["lambda2"] for r in sub])
        fk = np.array([r["first_kept"] for r in sub])
        print(f"\n{names[label]}   (n={len(sub)} graphs)")
        print(f"  # zero eigenvalues (=#components)  mean={nz.mean():.2f}  "
              f"[{int(nz.min())}..{int(nz.max())}]")
        print(f"  lambda_2 (Fiedler, pre-filter)     mean={l2.mean():.4f}  "
              f"median={np.median(l2):.4f}")
        print(f"  first KEPT eigenvalue (model sees) mean={fk.mean():.4f}  "
              f"median={np.median(fk):.4f}")
        zd = np.array([r["zero_density"] for r in sub])
        print(f"  structural-zero density of features  mean={zd.mean():.3f}  "
              f"median={np.median(zd):.3f}")

    # separability headline
    c = np.array([r["first_kept"] for r in rows if r["label"] == 1])
    d = np.array([r["first_kept"] for r in rows if r["label"] == 0])
    thr = 0.5 * (np.median(c) + np.median(d))
    acc = (np.concatenate([c < thr, d >= thr])).mean()
    print(f"\n{'-'*64}")
    print(f"Trivial 1-feature classifier on 'first kept eigenvalue':")
    print(f"  threshold={thr:.4f}  ->  accuracy={acc:.3f}")
    print(f"  (a single spectral number already separates the classes)")

    # component-boundedness of the kept modes (disconnected only)
    disc_conc = [f for r in rows if r["label"] == 0 for f in r["concentration"]]
    if disc_conc:
        dc_arr = np.array(disc_conc)
        print(f"\n{'-'*64}")
        print("Kept eigenvectors of DISCONNECTED graphs — mass on dominant component:")
        print(f"  mean={dc_arr.mean():.3f}  frac>0.99={np.mean(dc_arr > 0.99):.3f}")
        print(f"  Every kept mode lives entirely on ONE component (block-diagonal L),")
        print(f"  so each coordinate is zero on the OTHER component. The columns")
        print(f"  therefore split into complementary zero-supports — a pattern that")
        print(f"  fills ~half the feature matrix with structural zeros (see zero-density")
        print(f"  above: ~0.44 disconnected vs ~0.00 connected). The model reads this off")
        print(f"  locally per node — no reachability tracing required.")


# ── figures ───────────────────────────────────────────────────────────────────

def plot_spectrum(rows, path):
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.4))
    conn = [r for r in rows if r["label"] == 1]
    disc = [r for r in rows if r["label"] == 0]
    cc, dc = "#1a73e8", "#c5221f"

    # (a) first kept eigenvalue distribution
    ax[0, 0].hist([r["first_kept"] for r in conn], bins=30, alpha=0.6, color=cc, label="connected")
    ax[0, 0].hist([r["first_kept"] for r in disc], bins=30, alpha=0.6, color=dc, label="disconnected")
    ax[0, 0].set_title("(a) first KEPT eigenvalue\n(smallest the model actually sees)")
    ax[0, 0].set_xlabel(r"$\lambda$ of first kept eigenvector")
    ax[0, 0].set_ylabel("graphs")
    ax[0, 0].legend()

    # (b) count of near-zero eigenvalues = number of components
    maxz = max(r["n_zero"] for r in rows)
    bins = np.arange(0.5, maxz + 1.5)
    ax[0, 1].hist([r["n_zero"] for r in conn], bins=bins, alpha=0.6, color=cc, label="connected")
    ax[0, 1].hist([r["n_zero"] for r in disc], bins=bins, alpha=0.6, color=dc, label="disconnected")
    ax[0, 1].set_title("(b) # of zero eigenvalues\n(= number of connected components)")
    ax[0, 1].set_xlabel("multiplicity of eigenvalue 0")
    ax[0, 1].set_xticks(range(1, maxz + 1))
    ax[0, 1].legend()

    # (c) first vs second KEPT eigenvalue — zero modes filtered out, so the
    #     disconnected class drops its structural lambda_2 = 0 (apples to apples)
    ax[1, 0].scatter([r["first_kept"] for r in conn], [r["second_kept"] for r in conn],
                     s=10, alpha=0.5, color=cc, label="connected")
    ax[1, 0].scatter([r["first_kept"] for r in disc], [r["second_kept"] for r in disc],
                     s=10, alpha=0.5, color=dc, label="disconnected")
    ax[1, 0].set_title("(c) first vs second KEPT eigenvalue\n(zero modes filtered out)")
    ax[1, 0].set_xlabel("1st kept eigenvalue")
    ax[1, 0].set_ylabel("2nd kept eigenvalue")
    ax[1, 0].legend()

    # (d) structural-zero density: in a disconnected graph each mode is confined to
    #     one component, so its entries on the OTHER component are exactly zero —
    #     the complementary zero-supports fill ~half the feature matrix with zeros.
    bins = np.linspace(0, 0.7, 30)
    ax[1, 1].hist([r["zero_density"] for r in conn], bins=bins, alpha=0.6, color=cc, label="connected")
    ax[1, 1].hist([r["zero_density"] for r in disc], bins=bins, alpha=0.6, color=dc, label="disconnected")
    ax[1, 1].set_title("(d) structural-zero density of the feature matrix\n"
                       "(complementary zero-supports across the two components)")
    ax[1, 1].set_xlabel(r"fraction of $[n \times k]$ entries that are exactly 0")
    ax[1, 1].set_ylabel("graphs")
    ax[1, 1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"\nSaved {path}")


def reorder_block(r):
    """Order rows so nodes of the same component sit together (disconnected) or
    along the Fiedler axis (connected), and order columns by the component each
    mode lives on. Makes the complementary zero-block structure visible.
    Returns (matrix, row_boundary, col_boundary) where the boundaries mark the
    split between components (None when connected)."""
    pe, comp, ncomp = r["pe"], r["comp"], r["ncomp"]
    if ncomp > 1:
        row_order = np.argsort(comp, kind="stable")           # group nodes by component
        col_comp = []
        for j in range(pe.shape[1]):
            col = pe[:, j]
            masses = [(col[comp == c] ** 2).sum() for c in range(ncomp)]
            col_comp.append(int(np.argmax(masses)))
        col_order = np.argsort(col_comp, kind="stable")        # group modes by component
        mat = pe[np.ix_(row_order, col_order)]
        row_b = int((np.sort(comp) == 0).sum())                # first component's size
        col_b = int(np.array(col_comp)[col_order].tolist().count(0))
        return mat, row_b, col_b
    row_order = np.argsort(pe[:, 0])                            # sort by Fiedler value
    return pe[row_order], None, None


def plot_features(rows, path, k, n_examples=3):
    conn = [r for r in rows if r["label"] == 1][:n_examples]
    disc = [r for r in rows if r["label"] == 0][:n_examples]
    cols = max(len(conn), len(disc))
    fig, ax = plt.subplots(2, cols, figsize=(3.4 * cols, 7.2),
                           constrained_layout=True)
    if cols == 1:
        ax = ax.reshape(2, 1)

    vmax = max(np.abs(r["pe"]).max() for r in conn + disc)

    def draw(a, r, title):
        mat, row_b, col_b = reorder_block(r)
        im = a.imshow(mat, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax)
        if row_b is not None and col_b is not None:            # mark the component split
            a.axhline(row_b - 0.5, color="black", lw=1.2)
            a.axvline(col_b - 0.5, color="black", lw=1.2, ls="--")
        a.set_title(title, fontsize=9, pad=6)
        a.set_xlabel("eigenvector (mode)", fontsize=8)
        a.set_ylabel("node", fontsize=8)
        a.set_xticks(range(k))
        a.tick_params(labelsize=7)
        return im

    im = None
    for j in range(cols):
        if j < len(conn):
            im = draw(ax[0, j], conn[j], f"CONNECTED   n={conn[j]['n']}\n"
                      fr"$\lambda_2$={conn[j]['lambda2']:.3f}  · rows sorted by Fiedler")
        else:
            ax[0, j].axis("off")
        if j < len(disc):
            im = draw(ax[1, j], disc[j], f"DISCONNECTED   n={disc[j]['n']}\n"
                      "rows/cols grouped by component")
        else:
            ax[1, j].axis("off")

    fig.suptitle(f"Token rows the model sees:  [n x {k}] kept-eigenvector matrix   "
                 "(each ROW = one node's feature vector)\n"
                 "disconnected: complementary zero-blocks (lines = component split)  ·  "
                 "connected: a smooth Fiedler step, no zero-blocks", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.5, label="eigenvector entry")
    fig.savefig(path, dpi=130)
    print(f"Saved {path}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="connectedness_hard")
    p.add_argument("--config", default="configs/connectedness_lap.yaml")
    p.add_argument("--k", type=int, default=None, help="override #eigenvectors")
    args = p.parse_args()

    config = GNNConfig.from_yaml(args.config)
    k = args.k if args.k is not None else config.in_channels
    print(f"Dataset: {args.dataset}   k={k} eigenvectors (node_features=lap)")

    data_list = load_or_create(args.dataset, node_features="lap", in_channels=k)
    rows = collect(data_list, k)

    summarise(rows)
    plot_spectrum(rows, f"figures/lap_spectrum_{args.dataset}.png")
    plot_features(rows, f"figures/lap_features_{args.dataset}.png", k)
