"""Run YEHUDAI's exact model + training loop on OUR connectedness_hard data.

The symmetric counterpart to running our model on their data. We already showed our
adjacency-rows GNN reaches 1.00 on their connectivity set; this checks whether their
TransformerModel (plain nn.TransformerEncoder + mean-pool, AdamW, BCE) can learn ours.
If it also fails, the dataset — not the implementation — is conclusively the hard part.

Reuses their TransformerModel / get_graph_tokens / train_model / evaluate_model
unchanged; only swaps in our data. Run from inside yehudai/.

Usage:
    python run_on_our_data.py --rep_type adj_rows  --data ../data/connectedness_hard_lpe_dim0.pt
    python run_on_our_data.py --rep_type edge_list --data ../data/connectedness_hard_lpe_dim0.pt
    python run_on_our_data.py --rep_type lap_full  --data ../data/connectedness_hard_fixed_lpe_dim0.pt
"""
import os
import sys
import random
from unittest.mock import MagicMock

# stub unused-for-connectivity deps so importing their module succeeds
for _n in ("wandb", "ogb", "ogb.graphproppred", "matplotlib", "matplotlib.pyplot"):
    sys.modules[_n] = MagicMock()
sys.modules["matplotlib"].pyplot = sys.modules["matplotlib.pyplot"]

import torch
from torch_geometric.loader import DataLoader
import connectivity_adj_mat as conn


class _Args:
    pass


def main(a):
    torch.manual_seed(a.seed)
    random.seed(a.seed)

    data = torch.load(a.data, weights_only=False)
    random.shuffle(data)
    n = len(data)
    train, val, test = data[:int(0.8 * n)], data[int(0.8 * n):int(0.9 * n)], data[int(0.9 * n):]

    max_nodes = max(int(g.num_nodes) for g in data)
    node_feat_dim = data[0].num_features
    input_dim = (max_nodes + node_feat_dim) if a.rep_type in ("adj_rows", "lap_full") \
        else 2 * (max_nodes + node_feat_dim)

    print(f"OUR data: {a.data}")
    print(f"  graphs {n} | train {len(train)} val {len(val)} test {len(test)} | "
          f"max_nodes {max_nodes} | feat_dim {node_feat_dim} | input_dim {input_dim}")

    train_loader = DataLoader(train, batch_size=a.batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=a.batch_size)
    test_loader = DataLoader(test, batch_size=a.batch_size)

    args = _Args()
    args.d_model, args.nhead, args.num_encoder_layers = a.d_model, a.nhead, a.num_encoder_layers
    model = conn.TransformerModel(input_dim=input_dim, args=args)
    print(f"  THEIR TransformerModel: {sum(p.numel() for p in model.parameters()):,} params, "
          f"{a.num_encoder_layers} layer(s), rep={a.rep_type}\n")

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.1)

    best = 0.0
    for ep in range(1, a.num_epochs + 1):
        loss = conn.train_model(model, criterion, optimizer, train_loader, max_nodes, device, a.rep_type)
        if ep == 1 or ep % 10 == 0:
            va = conn.evaluate_model(model, val_loader, max_nodes, device, a.rep_type)
            te = conn.evaluate_model(model, test_loader, max_nodes, device, a.rep_type)
            best = max(best, te)
            print(f"Epoch {ep:03d}  loss={loss:.4f}  val={va:.4f}  test={te:.4f}")
    print(f"\nBest test acc: {best:.4f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--rep_type", required=True, choices=["adj_rows", "edge_list", "lap_full"])
    p.add_argument("--data", required=True, help="path to our PyG .pt dataset")
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--nhead", type=int, default=1)
    p.add_argument("--num_encoder_layers", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=1)
    main(p.parse_args())
