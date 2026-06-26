"""Visualize a trained connectivity model's live activations on specific graphs.

Answers "which channels fire on which instances": for each inspected graph it
captures, via forward hooks, the per-layer node activations [nodes x channels]
and (for global_attn layers) the post-softmax attention map [nodes x nodes] —
literally who-attends-to-whom, i.e. how reachability propagates. It also shows the
adjacency, the true reachability matrix R, and the predicted P = sigmoid(logits).

Nodes are reordered by connected component so block structure is visible: if a
channel encodes component identity, its activation column shows the same blocks
as R. The colour strip on the left of each heatmap marks the component of each row.

Usage:
    python inspect_activations.py --ckpt checkpoints/<run_id>.pt \
        --dataset diameter_controlled --num 4

Saves figures/activations_<ckpt-stem>_g<k>.png (one per inspected graph).
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.config import GNNConfig
from src.model import build_model
from src.dataset import load_or_create
from src.layers import GlobalAttnConv
from main import _attach_components, DEVICE


def _component_perm(comp: torch.Tensor):
    """Order nodes so same-component nodes are contiguous (block-diagonal R)."""
    return torch.argsort(comp, stable=True)


def _adjacency(edge_index, n, device):
    A = torch.zeros(n, n, device=device)
    A[edge_index[0], edge_index[1]] = 1.0
    return A


def _imshow(ax, M, title, cmap="viridis", vmin=None, vmax=None):
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=6)
    return im


def _comp_lines(ax, comp_sorted, square: bool):
    """Draw lines at component boundaries (nodes are sorted by component, so each
    boundary is where comp_sorted changes). Horizontal lines split node-rows;
    for square matrices vertical lines split node-columns too."""
    bounds = [i - 0.5 for i in range(1, len(comp_sorted))
              if comp_sorted[i] != comp_sorted[i - 1]]
    for b in bounds:
        ax.axhline(b, color="cyan", lw=1.0, alpha=0.8)
        if square:
            ax.axvline(b, color="cyan", lw=1.0, alpha=0.8)


def inspect(ckpt_path: str, dataset_name: str, num: int, limit: int = 0):
    ckpt = torch.load(ckpt_path, weights_only=False)
    config = GNNConfig.from_dict(ckpt["config"])
    assert config.task == "connectivity", "inspection targets the connectivity task"
    model = build_model(config).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    data_list = load_or_create(dataset_name, node_features=config.node_features,
                               lpe_dim=config.lpe_dim, in_channels=config.in_channels)
    if limit > 0 and limit < len(data_list):
        data_list = data_list[:limit]
    _attach_components(data_list)

    # prefer a mix: alternate connected / disconnected graphs up to `num`
    conn, disc = [], []
    for d in data_list:
        (conn if int(d.comp.max()) == 0 else disc).append(d)
    picks, i = [], 0
    while len(picks) < num and (i < len(conn) or i < len(disc)):
        if i < len(disc):
            picks.append(disc[i])
        if len(picks) < num and i < len(conn):
            picks.append(conn[i])
        i += 1
    picks = picks[:num]

    convs = list(model.net.convs)                      # the embed stack
    attn_layers = [c for c in convs if isinstance(c, GlobalAttnConv)]

    stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    os.makedirs("figures", exist_ok=True)

    for k, data in enumerate(picks):
        data = data.to(DEVICE)

        # ---- capture activations (conv outputs) via forward hooks ----
        acts: list[torch.Tensor] = []
        handles = [c.register_forward_hook(
            lambda m, inp, out: acts.append(out.detach())) for c in convs]
        for c in attn_layers:
            c.store_attn = True
        with torch.no_grad():
            logits = model(data)[0]                    # [n, n]
        for h in handles:
            h.remove()
        for c in attn_layers:
            c.store_attn = False

        comp = data.comp
        n = comp.size(0)
        perm = _component_perm(comp)
        comp_sorted = comp[perm].cpu().numpy()

        A = _adjacency(data.edge_index, n, comp.device)[perm][:, perm].cpu().numpy()
        R = (comp[:, None] == comp[None, :]).float()[perm][:, perm].cpu().numpy()
        P = torch.sigmoid(logits)[perm][:, perm].cpu().numpy()

        # relu the conv outputs: this is what "fires" downstream (negatives die)
        acts_sorted = [torch.relu(a)[perm].cpu().numpy() for a in acts]
        attn_maps = [c.last_attn.mean(-1)[perm][:, perm].cpu().numpy()
                     for c in attn_layers if c.last_attn is not None]

        L = len(acts_sorted)
        ncols = max(3, L)
        nrows = 2 + (1 if attn_maps else 0)
        fig = plt.figure(figsize=(3.2 * ncols, 3.4 * nrows))
        gs = fig.add_gridspec(nrows, ncols)

        ncomp = int(comp.max()) + 1
        verdict = "connected" if ncomp == 1 else f"{ncomp} components"
        em = bool(((P > 0.5) == (R > 0.5)).all())
        fig.suptitle(f"{stem}  |  graph {k}: n={n}, {verdict}  |  "
                     f"exact-match={'YES' if em else 'no'}", fontsize=11)

        # row 0: structure
        for j, (M, title, cm) in enumerate([
            (A, "adjacency A", "Greys"),
            (R, "true R (same component)", "Greens"),
            (P, "pred P = sigmoid(logits)", "Greens"),
        ]):
            ax = fig.add_subplot(gs[0, j])
            _imshow(ax, M, title, cmap=cm, vmin=0, vmax=1)
            _comp_lines(ax, comp_sorted, square=True)
            ax.set_xlabel("node (sorted by comp)", fontsize=7)

        # row 1: per-layer node activations [nodes x channels]
        for j, a in enumerate(acts_sorted):
            ax = fig.add_subplot(gs[1, j])
            _imshow(ax, a, f"layer {j} act (relu)  [nodes x ch]")
            _comp_lines(ax, comp_sorted, square=False)
            ax.set_xlabel("channel", fontsize=7)
            ax.set_ylabel("node", fontsize=7)

        # row 2: attention maps [nodes x nodes]
        for j, am in enumerate(attn_maps):
            ax = fig.add_subplot(gs[2, j])
            _imshow(ax, am, f"layer {j} attention (mean over heads)", cmap="magma")
            _comp_lines(ax, comp_sorted, square=True)
            ax.set_xlabel("attended-to node j", fontsize=7)
            ax.set_ylabel("query node i", fontsize=7)

        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out = f"figures/activations_{stem}_g{k}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"  wrote {out}  (n={n}, {verdict}, EM={'yes' if em else 'no'})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="path to checkpoints/<run_id>.pt")
    p.add_argument("--dataset", required=True, help="dataset to draw graphs from")
    p.add_argument("--num", type=int, default=4, help="number of graphs to inspect")
    p.add_argument("--limit", type=int, default=0, help="cap dataset size before picking")
    args = p.parse_args()
    inspect(args.ckpt, args.dataset, args.num, args.limit)
