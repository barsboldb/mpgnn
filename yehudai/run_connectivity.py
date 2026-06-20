"""Runner for Yehudai et al. 2025 connectivity experiment, without touching their source.

Their `connectivity_adj_mat.py` has three friction points for running locally:
  1. top-level imports of `ogb` / `wandb` (unused for connectivity unless --use_wandb),
     and `matplotlib` inside the data generator (only draws a degree histogram);
  2. a name bug: line 207 calls `create_data.create_connectivity_dataset` but the
     module is imported as `create_connectivity_data`;
  3. it writes into `connectivity_dataset/` without creating the directory.

This wrapper stubs the unused modules, patches the alias, makes the dirs, then calls
their `main()` unchanged. Run from inside the yehudai/ directory.

Usage:
    python run_connectivity.py --rep_type edge_list --n_nodes 50 --n_train 100 --num_epochs 100
    python run_connectivity.py --rep_type adj_rows  --n_nodes 50 --n_train 100 --num_epochs 100
    python run_connectivity.py --rep_type lap_full  ...
"""
import os
import sys
from unittest.mock import MagicMock

# 1. stub modules that connectivity doesn't actually need at runtime
for name in ("wandb", "ogb", "ogb.graphproppred", "matplotlib", "matplotlib.pyplot"):
    sys.modules[name] = MagicMock()
# pyplot is accessed as matplotlib.pyplot; make the attribute resolve to the stub
sys.modules["matplotlib"].pyplot = sys.modules["matplotlib.pyplot"]

# 2. make the output directories their code assumes exist
os.makedirs("connectivity_dataset", exist_ok=True)
os.makedirs("graph_dataset", exist_ok=True)

# 3. import their modules and patch two bugs without editing their source
import torch
import create_connectivity_data
import connectivity_adj_mat as conn


class _CorrectedData:
    """Shim passed in place of the missing `create_data` reference.

    Fixes both:
      - the NameError (line 207 calls `create_data.create_connectivity_dataset`);
      - the cache-reload bug, where val/test mistakenly load the *train* file.
        Their generator saves correct _val_/_test_ .pt files, so on a cache hit we
        simply load each split from its own file instead.
    """
    @staticmethod
    def create_connectivity_dataset(n_nodes):
        d = "connectivity_dataset"
        train_p = f"{d}/{n_nodes}connectivity_train_data.pt"
        val_p   = f"{d}/{n_nodes}connectivity_val_data.pt"
        test_p  = f"{d}/{n_nodes}connectivity_test_data.pt"
        if all(os.path.exists(p) for p in (train_p, val_p, test_p)):
            print(f"Loading cached splits for n_nodes={n_nodes} (train/val/test from own files)")
            # weights_only=False: these are pickled PyG Data objects, rejected by
            # torch>=2.6's safe-loading default.
            load = lambda p: torch.load(p, weights_only=False)
            return (load(train_p), load(val_p), load(test_p))
        # not cached -> their generator builds and returns correct fresh splits
        return create_connectivity_data.create_connectivity_dataset(n_nodes)


conn.create_data = _CorrectedData

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Run Yehudai connectivity (patched)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--nhead", type=int, default=1)
    p.add_argument("--num_encoder_layers", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--rep_type", type=str, required=True,
                   choices=["adj_rows", "edge_list", "lap_full"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n_nodes", type=int, default=50)
    p.add_argument("--n_train", type=int, default=100)
    args = p.parse_args()

    conn.main(args)
