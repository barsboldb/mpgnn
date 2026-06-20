import os
import torch
import numpy as np
from torch_geometric.data import Data
from features import laplacian_positional_encoding


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
    lpe_dim: int = 0,
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

        pe = laplacian_positional_encoding(edge_index, n, lpe_dim) if lpe_dim > 0 else None
        data_list.append(Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor([label], dtype=torch.long),
            pe=pe,
        ))

    print(f"Generated {num_graphs} graphs  |  connected: {counts[1]}  disconnected: {counts[0]}")
    return data_list


def _connected_component_edges(
    nodes: list[int], rng: np.random.Generator, extra_p: float = 0.3
) -> list[tuple[int, int]]:
    """A connected graph over `nodes` with minimum degree >= 2.

    Built as a random Hamiltonian cycle (guarantees connectivity and min degree 2)
    plus random chords. This removes any degree-0 / low-degree signal, so the
    presence of a disconnected component can NOT be detected from local degree.
    """
    perm = list(rng.permutation(nodes))
    edges: set[tuple[int, int]] = set()
    for i in range(len(perm)):
        u, v = perm[i], perm[(i + 1) % len(perm)]
        edges.add((min(u, v), max(u, v)))
    for a_idx in range(len(perm)):
        for b_idx in range(a_idx + 1, len(perm)):
            if rng.random() < extra_p:
                u, v = perm[a_idx], perm[b_idx]
                edges.add((min(u, v), max(u, v)))
    return list(edges)


def make_connectedness_hard_dataset(
    num_graphs: int = 1000,
    min_nodes: int = 12,
    max_nodes: int = 24,
    seed: int = 42,
    lpe_dim: int = 0,
) -> list[Data]:
    """
    Connectedness without the local degree shortcut.

    Every graph is two dense, internally-connected blobs (each min degree >= 2).
    - label 1 (connected):    one bridge edge joins the two blobs.
    - label 0 (disconnected): no bridge; instead one extra intra-blob edge so the
                              total edge count (and degree sequence distribution)
                              matches the connected class.

    There is NO isolated node in either class and the degree statistics are
    matched, so "min degree == 0" and "mean degree" are both useless. The only way
    to tell the classes apart is to trace reachability across the whole graph —
    genuine global reasoning. A lossy bottleneck (e.g. hidden=2) should fail here.

    Node features: normalised degree, same as `make_connectedness`.
    """
    rng = np.random.default_rng(seed)
    data_list = []
    counts = [0, 0]

    for i in range(num_graphs):
        label = i % 2  # balanced
        n = int(rng.integers(min_nodes, max_nodes + 1))
        na = int(rng.integers(3, n - 2))  # each blob has >= 3 nodes
        a_nodes = list(range(na))
        b_nodes = list(range(na, n))

        edges: set[tuple[int, int]] = set(_connected_component_edges(a_nodes, rng))
        edges |= set(_connected_component_edges(b_nodes, rng))

        if label == 1:
            # bridge the two blobs -> connected
            u, v = int(rng.choice(a_nodes)), int(rng.choice(b_nodes))
            edges.add((min(u, v), max(u, v)))
        else:
            # keep disconnected, but add one intra-blob edge so edge counts match
            for _ in range(100):
                blob = a_nodes if rng.random() < 0.5 else b_nodes
                u, v = (int(x) for x in rng.choice(blob, size=2, replace=False))
                e = (min(u, v), max(u, v))
                if e not in edges:
                    edges.add(e)
                    break

        all_edges: list[list[int]] = []
        for a, b in edges:
            all_edges += [[a, b], [b, a]]
        edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()

        deg = torch.zeros(n, dtype=torch.float)
        deg.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1)))
        x = (deg / (n - 1)).unsqueeze(1)

        counts[label] += 1
        pe = laplacian_positional_encoding(edge_index, n, lpe_dim) if lpe_dim > 0 else None
        data_list.append(Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor([label], dtype=torch.long),
            pe=pe,
        ))

    print(f"Generated {num_graphs} graphs  |  connected: {counts[1]}  disconnected: {counts[0]}")
    return data_list


def make_connectedness_hard_adj_dataset(
    num_graphs: int = 1000,
    min_nodes: int = 12,
    max_nodes: int = 24,
    seed: int = 42,
    lpe_dim: int = 0,
) -> list[Data]:
    """
    Same graphs as make_connectedness_hard_dataset but with adjacency rows as
    node features instead of normalised degree.

    Node features: x[i] = i-th row of the adjacency matrix, zero-padded to
    max_nodes. This gives each node a structural identity derived purely from
    its neighbourhood — no learned position embeddings needed. Two nodes in
    the same blob share common neighbours; the bridge endpoints are the only
    nodes with a non-zero entry pointing across the gap.
    """
    rng = np.random.default_rng(seed)
    data_list = []
    counts = [0, 0]

    for i in range(num_graphs):
        label = i % 2
        n = int(rng.integers(min_nodes, max_nodes + 1))
        na = int(rng.integers(3, n - 2))
        a_nodes = list(range(na))
        b_nodes = list(range(na, n))

        edges: set[tuple[int, int]] = set(_connected_component_edges(a_nodes, rng))
        edges |= set(_connected_component_edges(b_nodes, rng))

        if label == 1:
            u, v = int(rng.choice(a_nodes)), int(rng.choice(b_nodes))
            edges.add((min(u, v), max(u, v)))
        else:
            for _ in range(100):
                blob = a_nodes if rng.random() < 0.5 else b_nodes
                u, v = (int(x) for x in rng.choice(blob, size=2, replace=False))
                e = (min(u, v), max(u, v))
                if e not in edges:
                    edges.add(e)
                    break

        all_edges: list[list[int]] = []
        for a, b in edges:
            all_edges += [[a, b], [b, a]]
        edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()

        # adjacency row for each node, padded to max_nodes
        x = torch.zeros(n, max_nodes)
        for a, b in edges:
            x[a, b] = 1.0
            x[b, a] = 1.0

        counts[label] += 1
        pe = laplacian_positional_encoding(edge_index, n, lpe_dim) if lpe_dim > 0 else None
        data_list.append(Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor([label], dtype=torch.long),
            pe=pe,
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
    lpe_dim: int = 0,
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
        # LPE for combined graph (both components together)
        pe = laplacian_positional_encoding(edge_index, 2 * n, lpe_dim) if lpe_dim > 0 else None
        data_list.append(Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor([label], dtype=torch.long),
            pe=pe,
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
    "connectedness":          make_connectedness_dataset,
    "connectedness_hard":     make_connectedness_hard_dataset,
    "connectedness_hard_adj": make_connectedness_hard_adj_dataset,
    "isomorphism":            make_isomorphism_dataset,
}
