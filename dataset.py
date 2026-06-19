import os
import torch
import numpy as np
from torch_geometric.data import Data


def _is_connected(num_nodes: int, edge_index: torch.Tensor) -> bool:
    if num_nodes <= 1:
        return True
    if edge_index.size(1) == 0:
        return False

    adj: list[list[int]] = [[] for _ in range(num_nodes)]
    for i in range(edge_index.size(1)):
        adj[edge_index[0, i].item()].append(edge_index[1, i].item())  # type: ignore[arg-type]

    visited = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for neighbor in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)

    return len(visited) == num_nodes


def make_connectedness_dataset(
    num_graphs: int = 1000,
    min_nodes: int = 5,
    max_nodes: int = 20,
    seed: int = 42,
) -> list[Data]:
    """
    Generates random Erdos-Renyi graphs labeled by connectedness.

    Edge probability is sampled around the connectivity threshold log(n)/n,
    which gives a natural mix of connected and disconnected graphs.

    Node features: normalised degree of each node [degree / (n-1)].
    Pure constant features (all ones) cause global attention to produce
    identical outputs for every node regardless of graph structure, making
    the task unsolvable. Degree gives each node a structural identity.
    """
    rng = np.random.default_rng(seed)
    data_list = []
    counts = [0, 0]

    for _ in range(num_graphs):
        n = int(rng.integers(min_nodes, max_nodes + 1))
        threshold = np.log(n) / n
        p = rng.uniform(0.3 * threshold, 3.0 * threshold)

        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    edges.append([i, j])
                    edges.append([j, i])

        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        # degree of each node, normalised by max possible degree (n-1)
        deg = torch.zeros(n, dtype=torch.float)
        if edge_index.size(1) > 0:
            deg.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1)))
        x = (deg / (n - 1)).unsqueeze(1)  # [n, 1]

        label = int(_is_connected(n, edge_index))
        counts[label] += 1

        data_list.append(Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor([label], dtype=torch.long),
        ))

    print(f"Generated {num_graphs} graphs  |  connected: {counts[1]}  disconnected: {counts[0]}")
    return data_list


def _random_undirected_edges(n: int, p: float, rng: np.random.Generator) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p]


def _degree_sequence(n: int, edges: list[tuple[int, int]]) -> tuple[int, ...]:
    deg = [0] * n
    for i, j in edges:
        deg[i] += 1
        deg[j] += 1
    return tuple(sorted(deg))


def make_isomorphism_dataset(
    num_graphs: int = 1000,
    min_nodes: int = 6,
    max_nodes: int = 15,
    seed: int = 42,
) -> list[Data]:
    """
    Each sample is a pair of graphs (G1, G2) encoded as one disconnected graph.
    Label: 1 if G1 ≅ G2 (isomorphic), 0 otherwise.

    Isomorphic pairs are created by randomly permuting node labels of G1.
    Non-isomorphic pairs are generated independently and verified to have
    different degree sequences (which is a necessary condition for isomorphism).

    Node features: [1, 0] for G1 nodes, [0, 1] for G2 nodes — the only way
    the GNN can tell the two components apart. Structure must do the rest.

    This directly probes WL-test expressiveness: GIN (sum aggregation) is
    as powerful as 1-WL, but 1-WL still fails on some non-isomorphic pairs
    that share the same degree sequence at every iteration.
    """
    rng = np.random.default_rng(seed)
    data_list = []
    counts = [0, 0]

    for i in range(num_graphs):
        label = i % 2  # strictly alternating for balanced classes
        n = int(rng.integers(min_nodes, max_nodes + 1))

        edges1 = _random_undirected_edges(n, rng.uniform(0.2, 0.5), rng)

        if label == 1:
            # isomorphic: permute node labels — structure is identical
            perm = rng.permutation(n)
            edges2 = [(int(perm[a]), int(perm[b])) for a, b in edges1]
        else:
            # non-isomorphic: generate independently, ensure degree sequences differ
            edges2 = _random_undirected_edges(n, rng.uniform(0.2, 0.5), rng)
            for _ in range(200):
                if _degree_sequence(n, edges1) != _degree_sequence(n, edges2):
                    break
                edges2 = _random_undirected_edges(n, rng.uniform(0.2, 0.5), rng)

        # combine into one disconnected graph: G2 node indices are offset by n
        all_edges: list[list[int]] = []
        for a, b in edges1:
            all_edges += [[a, b], [b, a]]
        for a, b in edges2:
            all_edges += [[a + n, b + n], [b + n, a + n]]

        edge_index = (
            torch.tensor(all_edges, dtype=torch.long).t().contiguous()
            if all_edges else torch.zeros((2, 0), dtype=torch.long)
        )

        # one-hot graph membership as node features
        x = torch.cat([
            torch.tensor([[1., 0.]] * n),   # G1 nodes
            torch.tensor([[0., 1.]] * n),   # G2 nodes
        ])

        counts[label] += 1
        data_list.append(Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor([label], dtype=torch.long),
        ))

    print(f"Generated {num_graphs} graph pairs  |  isomorphic: {counts[1]}  non-isomorphic: {counts[0]}")
    return data_list


def load_or_create(
    name: str,
    cache_dir: str = "data",
    **kwargs,
) -> list[Data]:
    """
    Load dataset from disk if it exists, otherwise generate and save it.
    This ensures all experiments run on the exact same graphs.
    """
    os.makedirs(cache_dir, exist_ok=True)
    # encode key params into filename so different configs don't collide
    suffix = "_".join(f"{k}{v}" for k, v in sorted(kwargs.items()))
    path = os.path.join(cache_dir, f"{name}_{suffix}.pt" if suffix else f"{name}.pt")

    if os.path.exists(path):
        data_list = torch.load(path, weights_only=False)
        labels = [d.y.item() for d in data_list]
        n_classes = len(set(labels))
        dist = "  ".join(f"class {c}: {labels.count(c)}" for c in range(n_classes))
        print(f"Loaded {len(data_list)} samples from {path}  |  {dist}")
        return data_list  # type: ignore[return-value]

    data_list = GENERATORS[name](**kwargs)
    torch.save(data_list, path)
    print(f"Saved to {path}")
    return data_list


GENERATORS = {
    "connectedness": make_connectedness_dataset,
    "isomorphism":   make_isomorphism_dataset,
}
