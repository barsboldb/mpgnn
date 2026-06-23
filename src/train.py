from __future__ import annotations
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .logger import RunLogger


# ── Node classification ────────────────────────────────────────────────────────

def train_node(model, data, optimizer, device):
    data = data.to(device)
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def eval_node(model, data, device):
    data = data.to(device)
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    results = {}
    for split in ("train_mask", "val_mask", "test_mask"):
        mask = getattr(data, split)
        acc = (pred[mask] == data.y[mask]).float().mean().item()
        results[split.replace("_mask", "")] = acc
    return results


def run_node_experiment(model, data, device, epochs=200, lr=0.01, weight_decay=5e-4,
                        logger: RunLogger | None = None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        loss = train_node(model, data, optimizer, device)
        metrics = eval_node(model, data, device)
        if logger is not None:
            logger.log(epoch, loss=round(loss, 4), **{k: round(v, 4) for k, v in metrics.items()})
        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}  loss={loss:.4f}  "
                f"train={metrics['train']:.4f}  val={metrics['val']:.4f}  test={metrics['test']:.4f}"
            )

    summary = logger._summary() if logger else {}
    print(f"\nBest val: {summary.get('val', '-')}  |  Test at best val: {summary.get('test', '-')}")


# ── Graph classification ───────────────────────────────────────────────────────

def train_graph(model, loader: DataLoader, optimizer, device):
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_graph(model, loader: DataLoader, device):
    model.eval()
    correct = total = 0
    for data in loader:
        data = data.to(device)
        pred = model(data).argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.num_graphs
    return correct / total


def run_graph_experiment(model, train_loader, test_loader, device,
                         epochs=200, lr=0.01, weight_decay=5e-4,
                         logger: RunLogger | None = None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        loss = train_graph(model, train_loader, optimizer, device)
        test_acc = eval_graph(model, test_loader, device)
        if logger is not None:
            logger.log(epoch, loss=round(loss, 4), test=round(test_acc, 4))
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}  loss={loss:.4f}  test_acc={test_acc:.4f}")

    summary = logger._summary() if logger else {}
    print(f"\nBest test acc: {summary.get('test', '-')}  at epoch {summary.get('best_epoch', '-')}")
