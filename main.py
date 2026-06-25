"""
mpGNN experiment runner.

Usage:
    python main.py --config config.yaml --dataset connectedness
    python main.py --layer gat --task graph --dataset mutag
    python main.py --results          # print comparison table of all saved runs
"""
import argparse
import os
from dataclasses import asdict
import torch
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.loader import DataLoader

from src.config import GNNConfig
from src.model import build_model
from src.train import run_node_experiment, run_graph_experiment, run_connectivity_experiment
from src.dataset import GENERATORS, load_or_create
from src.logger import RunLogger, print_results_table


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")


def make_default_config(layer: str, task: str) -> GNNConfig:
    if task == "node":
        return GNNConfig(
            in_channels=1433,
            out_channels=7,
            hidden_channels=64,
            task="node",
            dropout=0.5,
            layers=[{"type": layer}, {"type": layer}],
        )
    else:
        return GNNConfig(
            in_channels=7,
            out_channels=2,
            hidden_channels=64,
            task="graph",
            pooling="mean",
            dropout=0.5,
            norm_type="batch",
            layers=[{"type": layer}, {"type": layer}, {"type": layer}],
        )


def node_experiment(config: GNNConfig, dataset_name: str):
    dataset = Planetoid(root="/tmp/Cora", name="Cora")
    data = dataset[0]
    print(f"\nDataset: Cora  |  Nodes: {data.num_nodes}  Edges: {data.num_edges}  Features: {data.num_node_features}")
    model = build_model(config).to(DEVICE)
    print(f"Model params: {model.num_parameters():,}\n")
    print(config.describe())
    print()

    logger = RunLogger(dataset_name, config)
    run_node_experiment(model, data, DEVICE,
                        epochs=config.epochs, lr=config.lr, weight_decay=config.weight_decay,
                        logger=logger)
    logger.save()


def graph_experiment(config: GNNConfig, dataset_name: str, overfit: int = 0, limit: int = 0):
    if dataset_name in GENERATORS:
        data_list = load_or_create(dataset_name,
                                   node_features=config.node_features,
                                   lpe_dim=config.lpe_dim,
                                   in_channels=config.in_channels)
        if limit > 0 and limit < len(data_list):
            # quick run on a subset; the on-disk cache is untouched. The generator
            # alternates labels (i % 2), so a head slice stays class-balanced.
            data_list = data_list[:limit]
            print(f"[--limit {limit}] using {len(data_list)} of the cached graphs")
        if overfit > 0:
            # balanced: take equal numbers from each class
            class0 = [d for d in data_list if d.y.item() == 0][:overfit // 2]
            class1 = [d for d in data_list if d.y.item() == 1][:overfit // 2]
            data_list = class0 + class1
            train_ds = test_ds = data_list
            print(f"\n[OVERFIT MODE] Dataset: {dataset_name}  |  Graphs: {len(data_list)}  (all used for train+test)")
        else:
            n = len(data_list)
            train_ds = data_list[:int(0.8 * n)]
            test_ds  = data_list[int(0.8 * n):]
            print(f"\nDataset: {dataset_name}  |  Graphs: {n}  Train: {len(train_ds)}  Test: {len(test_ds)}")
    else:
        dataset = TUDataset(root=f"/tmp/{dataset_name.upper()}", name=dataset_name.upper())
        dataset = dataset.shuffle()
        n = len(dataset)
        train_ds = dataset[:int(0.8 * n)]
        test_ds  = dataset[int(0.8 * n):]
        print(f"\nDataset: {dataset_name.upper()}  |  Graphs: {n}  Classes: {dataset.num_classes}")

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=config.batch_size)

    model = build_model(config).to(DEVICE)
    print(f"Model params: {model.num_parameters():,}\n")
    print(config.describe())
    print()

    tag = f"overfit{overfit}_" if overfit > 0 else ""
    logger = RunLogger(dataset_name, config, tag=tag)
    run_graph_experiment(model, train_loader, test_loader, DEVICE,
                         epochs=config.epochs, lr=config.lr, weight_decay=config.weight_decay,
                         logger=logger)
    logger.save()


def _attach_components(data_list):
    """Attach per-node connected-component labels (data.comp [n]) — the target for
    the connectivity task: R_ij = 1 iff comp[i] == comp[j]. As a node attribute it
    batches cleanly (unlike a ragged n x n matrix)."""
    for g in data_list:
        n = int(g.num_nodes)
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        ei = g.edge_index
        for a, b in zip(ei[0].tolist(), ei[1].tolist()):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        remap, comp = {}, []
        for i in range(n):
            r = find(i)
            if r not in remap:
                remap[r] = len(remap)
            comp.append(remap[r])
        g.comp = torch.tensor(comp, dtype=torch.long)


def save_checkpoint(model, config: GNNConfig, run_id: str, train_dataset: str) -> str:
    """Persist weights + config so the model can be reloaded and evaluated on
    other (e.g. uncontrolled / OOD) datasets later."""
    os.makedirs("checkpoints", exist_ok=True)
    path = os.path.join("checkpoints", f"{run_id}.pt")
    torch.save({"config": asdict(config), "state_dict": model.state_dict(),
                "train_dataset": train_dataset}, path)
    print(f"Saved model -> {path}")
    return path


def evaluate_checkpoint(ckpt_path: str, dataset_name: str, limit: int = 0):
    """Load a trained model and evaluate it on `dataset_name` (no training)."""
    ckpt = torch.load(ckpt_path, weights_only=False)
    config = GNNConfig.from_dict(ckpt["config"])
    model = build_model(config).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    data_list = load_or_create(dataset_name,
                               node_features=config.node_features,
                               lpe_dim=config.lpe_dim,
                               in_channels=config.in_channels)
    if limit > 0 and limit < len(data_list):
        data_list = data_list[:limit]
    loader = DataLoader(data_list, batch_size=config.batch_size)

    trained_on = ckpt.get("train_dataset", "?")
    print(f"\nLoaded {ckpt_path}  (task={config.task}, trained on {trained_on})")
    print(f"Evaluating on {dataset_name}  |  {len(data_list)} graphs")
    if config.task == "connectivity":
        from src.train import _connectivity_eval
        _attach_components(data_list)
        em = _connectivity_eval(model, loader, DEVICE)
        print(f"  connectivity exact-match = {em:.4f}")
    else:
        from src.train import eval_graph
        acc = eval_graph(model, loader, DEVICE)
        print(f"  accuracy = {acc:.4f}")


def _print_matrix(name, M):
    print(f"  {name}:")
    for row in M.cpu().int().tolist():
        print("    " + "".join("1" if v else "·" for v in row))


def inspect_checkpoint(ckpt_path: str, dataset_name: str, limit: int = 0, show: int = 2):
    """Print failing DISCONNECTED examples: true R vs predicted R, and the mean
    predicted P(reachable) within vs across components. Cross-component ~1.0 means
    the model collapsed to all-ones (heuristic); low means it really separated the
    components but missed some entries (a near-miss, not all-ones)."""
    ckpt = torch.load(ckpt_path, weights_only=False)
    config = GNNConfig.from_dict(ckpt["config"])
    model = build_model(config).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    data_list = load_or_create(dataset_name, node_features=config.node_features,
                               lpe_dim=config.lpe_dim, in_channels=config.in_channels)
    if limit > 0 and limit < len(data_list):
        data_list = data_list[:limit]
    _attach_components(data_list)
    loader = DataLoader(data_list, batch_size=1)

    print(f"\nInspecting {ckpt_path}\n  on {dataset_name} — first {show} failing disconnected graph(s):")
    shown = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(DEVICE)
            logits = model(data)[0]                       # [n, n]
            comp = data.comp
            ncomp = int(comp.max().item()) + 1
            R = (comp[:, None] == comp[None, :]).float()
            pred = (logits > 0).float()
            if ncomp > 1 and not bool((pred == R).all().item()):
                probs = torch.sigmoid(logits)
                within = probs[R == 1].mean().item()
                cross = probs[R == 0].mean().item()
                n = R.size(0)
                print(f"\n  --- n={n}, components={ncomp}, "
                      f"mismatches={int((pred != R).sum().item())}/{n*n} ---")
                print(f"  pred density={pred.mean().item():.3f}  (true={R.mean().item():.3f})")
                print(f"  mean P(reachable):  within-comp={within:.3f}   cross-comp={cross:.3f}")
                print(f"  (cross~1.0 => ALL-ONES heuristic;  cross<0.5 => components separated)")
                _print_matrix("true R", R)
                _print_matrix("pred R", pred)
                shown += 1
                if shown >= show:
                    break
    if shown == 0:
        print("  (no failing disconnected example found)")


def connectivity_experiment(config: GNNConfig, dataset_name: str, limit: int = 0):
    data_list = load_or_create(dataset_name,
                               node_features=config.node_features,
                               lpe_dim=config.lpe_dim,
                               in_channels=config.in_channels)
    if limit > 0 and limit < len(data_list):
        data_list = data_list[:limit]
    _attach_components(data_list)
    n = len(data_list)
    train_ds, test_ds = data_list[:int(0.8 * n)], data_list[int(0.8 * n):]
    print(f"\nDataset: {dataset_name}  |  Graphs: {n}  Train: {len(train_ds)}  Test: {len(test_ds)}  (connectivity-matrix task)")

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=config.batch_size)

    model = build_model(config).to(DEVICE)
    print(f"Model params: {model.num_parameters():,}\n")
    print(config.describe())
    print()

    logger = RunLogger(dataset_name, config, tag="conn_")
    run_connectivity_experiment(model, train_loader, test_loader, DEVICE,
                                epochs=config.epochs, lr=config.lr, weight_decay=config.weight_decay,
                                logger=logger)
    logger.save()
    save_checkpoint(model, config, logger.run_id, dataset_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  type=str, default=None)
    parser.add_argument("--layer",   type=str, default="gcn",
                        choices=["gcn", "sage", "gat", "gin", "global_attn"])
    parser.add_argument("--task",    type=str, default="node", choices=["node", "graph"])
    parser.add_argument("--dataset", type=str, default="cora",
                        help="cora | mutag | connectedness | isomorphism")
    parser.add_argument("--results", action="store_true",
                        help="Print comparison table of all saved runs and exit")
    parser.add_argument("--overfit", type=int, default=0,
                        help="Use only N graphs (balanced), same set for train+test, to check memorisation capacity")
    parser.add_argument("--limit", type=int, default=0,
                        help="Cap the dataset to the first N cached graphs (balanced) for a quick run; cache is untouched")
    parser.add_argument("--eval", type=str, default=None,
                        help="Path to a saved checkpoint (.pt); evaluate it on --dataset and exit (no training)")
    parser.add_argument("--inspect", type=str, default=None,
                        help="Path to a checkpoint (.pt); print failing disconnected examples on --dataset and exit")
    args = parser.parse_args()

    if args.results:
        print_results_table()
        return

    if args.eval:
        evaluate_checkpoint(args.eval, args.dataset, limit=args.limit)
        return

    if args.inspect:
        inspect_checkpoint(args.inspect, args.dataset, limit=args.limit)
        return

    if args.config:
        config = GNNConfig.from_yaml(args.config)
    else:
        config = make_default_config(args.layer, args.task)

    if config.task == "node":
        node_experiment(config, args.dataset)
    elif config.task == "connectivity":
        connectivity_experiment(config, args.dataset, limit=args.limit)
    else:
        graph_experiment(config, args.dataset, overfit=args.overfit, limit=args.limit)


if __name__ == "__main__":
    main()
