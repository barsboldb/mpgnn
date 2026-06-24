from __future__ import annotations
import time
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .logger import RunLogger


def _sync(device):
    """Block until queued device work finishes, so timers measure real compute.

    CUDA/MPS kernels are dispatched asynchronously; without a sync the host
    clock stops before the GPU does and every duration is meaningless.
    """
    t = device.type if hasattr(device, "type") else str(device)
    if t == "cuda":
        torch.cuda.synchronize()
    elif t == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


@torch.no_grad()
def benchmark_inference(model, sample, device, warmup: int = 5, iters: int = 30):
    """Average latency of one forward pass on a fixed input (eval mode).

    `sample` is a graph batch (graph task) or the single graph (node task).
    Returns per-call latency plus a per-unit normalisation (per graph / per node).
    """
    model.eval()
    sample = sample.to(device)
    n_graphs = int(getattr(sample, "num_graphs", 1))
    n_nodes = int(sample.num_nodes)

    for _ in range(warmup):
        model(sample)
    _sync(device)

    t0 = time.perf_counter()
    for _ in range(iters):
        model(sample)
    _sync(device)
    per_call = (time.perf_counter() - t0) / iters

    return {
        "per_call_ms": round(per_call * 1e3, 5),
        "per_graph_ms": round(per_call / n_graphs * 1e3, 5),
        "per_node_ms": round(per_call / max(n_nodes, 1) * 1e3, 5),
        "graphs_per_call": n_graphs,
        "nodes_per_call": n_nodes,
        "warmup": warmup,
        "iters": iters,
    }


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
        _sync(device); t0 = time.perf_counter()
        loss = train_node(model, data, optimizer, device)
        _sync(device); t1 = time.perf_counter()
        metrics = eval_node(model, data, device)
        _sync(device); t2 = time.perf_counter()
        if logger is not None:
            logger.log(epoch, loss=round(loss, 4),
                       **{k: round(v, 4) for k, v in metrics.items()},
                       train_time_s=round(t1 - t0, 5), eval_time_s=round(t2 - t1, 5))
        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}  loss={loss:.4f}  "
                f"train={metrics['train']:.4f}  val={metrics['val']:.4f}  test={metrics['test']:.4f}"
            )

    if logger is not None:
        logger.set_timing(device, benchmark_inference(model, data, device))

    summary = logger._summary() if logger else {}
    print(f"\nBest val: {summary.get('val', '-')}  |  Test at best val: {summary.get('test', '-')}")


# ── Graph classification ───────────────────────────────────────────────────────

def train_graph(model, loader: DataLoader, optimizer, device):
    """One training epoch. Returns (mean loss, train accuracy).

    Train accuracy is accumulated from the same forward passes used for the loss
    (free), so it is measured in train mode (dropout on) — a slight underestimate,
    but enough to separate memorisation (train→1, test flat) from underfitting.
    """
    model.train()
    total_loss = 0.0
    correct = total = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        correct += (out.argmax(dim=1) == data.y).sum().item()
        total += data.num_graphs
    return total_loss / total, correct / total


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
        _sync(device); t0 = time.perf_counter()
        loss, train_acc = train_graph(model, train_loader, optimizer, device)
        _sync(device); t1 = time.perf_counter()
        test_acc = eval_graph(model, test_loader, device)
        _sync(device); t2 = time.perf_counter()
        if logger is not None:
            logger.log(epoch, loss=round(loss, 4),
                       train=round(train_acc, 4), test=round(test_acc, 4),
                       train_time_s=round(t1 - t0, 5), eval_time_s=round(t2 - t1, 5))
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}  loss={loss:.4f}  train_acc={train_acc:.4f}  "
                  f"test_acc={test_acc:.4f}  gap={train_acc - test_acc:+.3f}")

    if logger is not None:
        sample = next(iter(test_loader))
        logger.set_timing(device, benchmark_inference(model, sample, device))

    summary = logger._summary() if logger else {}
    print(f"\nBest test acc: {summary.get('test', '-')}  at epoch {summary.get('best_epoch', '-')}")
