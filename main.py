"""
mpGNN experiment runner.

Usage:
    python main.py --config config.yaml --dataset connectedness
    python main.py --layer gat --task graph --dataset mutag
    python main.py --results          # print comparison table of all saved runs
"""
import argparse
import torch
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.loader import DataLoader

from config import GNNConfig
from model import GNN, build_model
from train import run_node_experiment, run_graph_experiment
from dataset import GENERATORS, load_or_create
from logger import RunLogger, print_results_table


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
    model = GNN(config).to(DEVICE)
    print(f"Model params: {model.num_parameters():,}\n")
    print(config.describe())
    print()

    logger = RunLogger(dataset_name, config)
    run_node_experiment(model, data, DEVICE,
                        epochs=config.epochs, lr=config.lr, weight_decay=config.weight_decay,
                        logger=logger)
    logger.save()


def graph_experiment(config: GNNConfig, dataset_name: str, overfit: int = 0):
    if dataset_name in GENERATORS:
        data_list = load_or_create(dataset_name,
                                   node_features=config.node_features,
                                   lpe_dim=config.lpe_dim)
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
    args = parser.parse_args()

    if args.results:
        print_results_table()
        return

    if args.config:
        config = GNNConfig.from_yaml(args.config)
    else:
        config = make_default_config(args.layer, args.task)

    if config.task == "node":
        node_experiment(config, args.dataset)
    else:
        graph_experiment(config, args.dataset, overfit=args.overfit)


if __name__ == "__main__":
    main()
