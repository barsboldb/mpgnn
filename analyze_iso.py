"""Analyse which isomorphism pairs the model gets wrong.

Trains the model once (same config as iso_adj.yaml), then evaluates on the
test set and computes structural statistics for correct vs misclassified pairs.

Usage:
    python analyze_iso.py
"""
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from torch_geometric.loader import DataLoader

from src.config import GNNConfig
from src.model import build_model
from src.dataset import load_or_create


# ── helpers ───────────────────────────────────────────────────────────────────

def degree_sequence(edge_index, n):
    deg = torch.zeros(n, dtype=torch.long)
    if edge_index.size(1) > 0:
        deg.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), dtype=torch.long))
    return tuple(sorted(deg.tolist()))


def graph_stats(data):
    """Structural statistics for one (G1, G2) pair."""
    n1 = int(data.n1.item())
    n2 = data.num_nodes - n1

    ei = data.edge_index
    # split edges into G1 (both endpoints < n1) and G2 (both >= n1)
    if ei.size(1) > 0:
        mask_g1 = ei[0] < n1
        mask_g2 = ei[0] >= n1
        ei_g1 = ei[:, mask_g1]
        ei_g2 = ei[:, mask_g2] - n1   # re-index G2 to 0-based
    else:
        ei_g1 = torch.zeros((2, 0), dtype=torch.long)
        ei_g2 = torch.zeros((2, 0), dtype=torch.long)

    # undirected edge counts (each edge stored twice)
    e1 = ei_g1.size(1) // 2
    e2 = ei_g2.size(1) // 2

    deg1 = degree_sequence(ei_g1, n1)
    deg2 = degree_sequence(ei_g2, n2)

    return {
        "n": n1,                              # nodes per component (G1 = G2 = n)
        "edges_g1": e1,
        "edges_g2": e2,
        "same_edge_count": e1 == e2,
        "mean_deg_g1": sum(deg1) / n1 if n1 else 0,
        "mean_deg_g2": sum(deg2) / n2 if n2 else 0,
        "deg_seq_g1": deg1,
        "deg_seq_g2": deg2,
        "same_deg_seq": deg1 == deg2,
        "deg_seq_diff": sum(abs(a - b) for a, b in zip(sorted(deg1), sorted(deg2))),
    }


# ── training ──────────────────────────────────────────────────────────────────

def train_and_collect(config, data_list, device, epochs=300):
    n = len(data_list)
    train_ds = data_list[:int(0.8 * n)]
    test_ds  = data_list[int(0.8 * n):]

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False)

    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    for epoch in range(1, epochs + 1):
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            F.cross_entropy(model(data), data.y).backward()
            optimizer.step()
        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                correct = sum(
                    (model(d.to(device)).argmax(1) == d.y.to(device)).sum().item()
                    for d in test_loader
                )
            print(f"  Epoch {epoch:03d}  test_acc={correct/len(test_ds):.4f}")

    # collect per-sample predictions on test set (no shuffle → stable order)
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            preds.append(model(data).argmax(1).cpu())
            labels.append(data.y.cpu())
    preds  = torch.cat(preds).tolist()
    labels = torch.cat(labels).tolist()
    return test_ds, preds, labels


# ── analysis ──────────────────────────────────────────────────────────────────

def analyse(test_ds, preds, labels):
    groups = defaultdict(list)   # "correct_iso", "correct_noniso", "wrong_iso", "wrong_noniso"
    for data, pred, label in zip(test_ds, preds, labels):
        correct = pred == label
        iso = label == 1
        key = ("correct" if correct else "wrong") + "_" + ("iso" if iso else "noniso")
        groups[key].append(graph_stats(data))

    def summarise(records, name):
        if not records:
            print(f"\n{name}: (none)")
            return
        print(f"\n{'─'*60}")
        print(f"{name}  (n={len(records)})")
        print(f"{'─'*60}")
        keys = ["n", "edges_g1", "edges_g2", "mean_deg_g1", "mean_deg_g2",
                "same_edge_count", "same_deg_seq", "deg_seq_diff"]
        for k in keys:
            vals = [r[k] for r in records]
            if isinstance(vals[0], bool):
                print(f"  {k:20s}  True: {sum(vals)}/{len(vals)} ({100*sum(vals)/len(vals):.0f}%)")
            else:
                arr = np.array(vals, dtype=float)
                print(f"  {k:20s}  mean={arr.mean():.2f}  std={arr.std():.2f}  "
                      f"min={arr.min():.0f}  max={arr.max():.0f}")

    for key in ["correct_iso", "wrong_iso", "correct_noniso", "wrong_noniso"]:
        summarise(groups[key], key)

    # spotlight: wrong non-iso pairs — what makes them hard?
    wrong_noniso = groups["wrong_noniso"]
    if wrong_noniso:
        print(f"\n{'─'*60}")
        print("Wrong non-iso pairs — degree sequence diff distribution:")
        diffs = [r["deg_seq_diff"] for r in wrong_noniso]
        for d in sorted(set(diffs)):
            count = diffs.count(d)
            print(f"  deg_seq_diff={d:3d}  count={count}")
        same = sum(r["same_deg_seq"] for r in wrong_noniso)
        print(f"\n  same degree sequence: {same}/{len(wrong_noniso)}")
        same_e = sum(r["same_edge_count"] for r in wrong_noniso)
        print(f"  same edge count:      {same_e}/{len(wrong_noniso)}")

    wrong_iso = groups["wrong_iso"]
    if wrong_iso:
        print(f"\n{'─'*60}")
        print("Wrong iso pairs — mean-degree gap (G1 vs G2 after attention):")
        gaps = [abs(r["mean_deg_g1"] - r["mean_deg_g2"]) for r in wrong_iso]
        print(f"  mean gap={np.mean(gaps):.3f}  max={np.max(gaps):.3f}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    config = GNNConfig.from_yaml("configs/iso_adj.yaml")
    data_list = load_or_create("isomorphism",
                               node_features=config.node_features,
                               lpe_dim=config.lpe_dim)

    print(f"\nTraining for {config.epochs} epochs...\n")
    test_ds, preds, labels = train_and_collect(config, data_list, device, epochs=config.epochs)

    total = len(labels)
    correct = sum(p == l for p, l in zip(preds, labels))
    print(f"\nFinal test accuracy: {correct}/{total} = {correct/total:.4f}")

    analyse(test_ds, preds, labels)
