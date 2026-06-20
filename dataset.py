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


# ── Raw graph generators ──────────────────────────────────────────────────────
# Generators produce only structural data: edge_index, y, and any metadata
# (e.g. n1 for isomorphism pairs).  Node features (x) and positional encodings
# (pe) are applied separately by tokenize_dataset() after loading, so the same
# cached graphs can be re-tokenized without regenerating.

def make_connectedness_dataset(
    num_graphs: int = 1000,
    min_nodes: int = 5,
    max_nodes: int = 20,
    seed: int = 42,
) -> list[Data]:
    """Random Erdős–Rényi graphs labeled by connectedness.

    Edge probability sampled around the connectivity threshold log(n)/n.
    Stores only edge_index and y — node features applied by tokenize_dataset.
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

        edge_index = (
            torch.tensor(edges, dtype=torch.long).t().contiguous()
            if edges else torch.zeros((2, 0), dtype=torch.long)
        )
        label = int(_is_connected(n, edge_index))
        counts[label] += 1
        data_list.append(Data(edge_index=edge_index, y=torch.tensor([label], dtype=torch.long)))

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
) -> list[Data]:
    """Connectedness without the local degree shortcut.

    Every graph is two dense, internally-connected blobs (each min degree >= 2).
    - label 1 (connected):    one bridge edge joins the two blobs.
    - label 0 (disconnected): no bridge; instead one extra intra-blob edge so the
                              total edge count (and degree sequence distribution)
                              matches the connected class.

    There is NO isolated node in either class and the degree statistics are
    matched, so "min degree == 0" and "mean degree" are both useless. The only way
    to tell the classes apart is to trace reachability across the whole graph —
    genuine global reasoning.
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

        counts[label] += 1
        data_list.append(Data(edge_index=edge_index, y=torch.tensor([label], dtype=torch.long)))

    print(f"Generated {num_graphs} graphs  |  connected: {counts[1]}  disconnected: {counts[0]}")
    return data_list


def make_connectedness_hard_fixed_dataset(
    num_graphs: int = 1000,
    seed: int = 42,
) -> list[Data]:
    """connectedness_hard with a FIXED graph size (n=20 for every graph)."""
    return make_connectedness_hard_dataset(
        num_graphs=num_graphs, min_nodes=20, max_nodes=20, seed=seed,
    )


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
    """Graph pairs labeled by isomorphism.

    Each sample encodes (G1, G2) as one disconnected graph: G1 at nodes 0..n-1,
    G2 at nodes n..2n-1. Stores n1=n so tokenize_dataset can split the pair.
    - label 1: G2 is a random permutation of G1 (isomorphic)
    - label 0: G2 generated independently with a different degree sequence
    """
    rng = np.random.default_rng(seed)
    data_list = []
    counts = [0, 0]

    for i in range(num_graphs):
        label = i % 2
        n = int(rng.integers(min_nodes, max_nodes + 1))

        edges1 = _random_undirected_edges(n, rng.uniform(0.2, 0.5), rng)

        if label == 1:
            perm = rng.permutation(n)
            edges2 = [(int(perm[a]), int(perm[b])) for a, b in edges1]
        else:
            edges2 = _random_undirected_edges(n, rng.uniform(0.2, 0.5), rng)
            for _ in range(200):
                if _degree_sequence(n, edges1) != _degree_sequence(n, edges2):
                    break
                edges2 = _random_undirected_edges(n, rng.uniform(0.2, 0.5), rng)

        all_edges: list[list[int]] = []
        for a, b in edges1:
            all_edges += [[a, b], [b, a]]
        for a, b in edges2:
            all_edges += [[a + n, b + n], [b + n, a + n]]

        edge_index = (
            torch.tensor(all_edges, dtype=torch.long).t().contiguous()
            if all_edges else torch.zeros((2, 0), dtype=torch.long)
        )

        counts[label] += 1
        data_list.append(Data(
            edge_index=edge_index,
            y=torch.tensor([label], dtype=torch.long),
            n1=torch.tensor(n, dtype=torch.long),  # G1 = nodes 0..n-1, G2 = nodes n..2n-1
        ))

    print(f"Generated {num_graphs} graph pairs  |  isomorphic: {counts[1]}  non-isomorphic: {counts[0]}")
    return data_list


# ── Yehudai et al. 2025 connectivity data ────────────────────────────────────

_YEHUDAI_DIR = "yehudai/connectivity_dataset"
_YEHUDAI_N = 50


def _load_yehudai_pool(num_graphs: int) -> list[Data]:
    paths = [f"{_YEHUDAI_DIR}/{_YEHUDAI_N}connectivity_{s}_data.pt" for s in ("train", "test")]
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found — generate it first:\n"
                f"  cd yehudai && python run_connectivity.py --rep_type adj_rows "
                f"--n_nodes 50 --num_epochs 1")
    pool = []
    for p in paths:
        pool += list(torch.load(p, weights_only=False))
    conn = [g for g in pool if int(g.y.item()) == 1][:num_graphs // 2]
    disc = [g for g in pool if int(g.y.item()) == 0][:num_graphs // 2]
    return conn + disc


def make_yehudai_connectivity_dataset(
    num_graphs: int = 1000, seed: int = 42,
) -> list[Data]:
    """Yehudai's connectivity graphs — raw edge_index + y only."""
    graphs = _load_yehudai_pool(num_graphs)
    out = [Data(edge_index=g.edge_index, y=g.y.view(1)) for g in graphs]
    print(f"Loaded {len(out)} Yehudai connectivity graphs (raw, n={_YEHUDAI_N})")
    return out


# ── Tokenization ──────────────────────────────────────────────────────────────

def tokenize_dataset(
    data_list: list[Data],
    node_features: str = "degree",
    lpe_dim: int = 0,
) -> list[Data]:
    """Apply node features and LPE to a list of raw Data objects in memory.

    node_features options:
      "degree"     — normalised degree [n, 1].  Safe default; breaks symmetry
                     without encoding global structure.
      "constant"   — ones [n, 1].  No structural signal; useful as an ablation
                     or when LPE/edge-tokens carry all the structure.
      "adj_rows"   — each node's adjacency row, zero-padded to the largest
                     num_nodes across the dataset [n, max_nodes].
      "membership" — one-hot component flag [n, 2]: [1,0] for G1 nodes,
                     [0,1] for G2 nodes.  Requires data.n1 (isomorphism dataset).

    lpe_dim > 0 appends Laplacian eigenvectors as data.pe.
    """
    max_n = max(g.num_nodes for g in data_list)

    for g in data_list:
        n = g.num_nodes

        if node_features == "degree":
            deg = torch.zeros(n, dtype=torch.float)
            if g.edge_index.size(1) > 0:
                deg.scatter_add_(0, g.edge_index[0], torch.ones(g.edge_index.size(1)))
            g.x = (deg / max(n - 1, 1)).unsqueeze(1)

        elif node_features == "constant":
            g.x = torch.ones(n, 1)

        elif node_features == "adj_rows":
            x = torch.zeros(n, max_n)
            if g.edge_index.size(1) > 0:
                row, col = g.edge_index
                # clamp col to max_n in case a graph's node indices exceed the padded width
                valid = col < max_n
                x[row[valid], col[valid]] = 1.0
            g.x = x

        elif node_features == "membership":
            n1 = int(g.n1.item())
            g.x = torch.cat([
                torch.tensor([[1., 0.]] * n1),
                torch.tensor([[0., 1.]] * (n - n1)),
            ])

        else:
            raise ValueError(f"Unknown node_features: '{node_features}'. "
                             f"Choose from: degree, constant, adj_rows, membership")

        g.pe = laplacian_positional_encoding(g.edge_index, n, lpe_dim) if lpe_dim > 0 else None

    return data_list


# ── Dataset registry and caching ─────────────────────────────────────────────

GENERATORS: dict[str, object] = {
    "connectedness":           make_connectedness_dataset,
    "connectedness_hard":      make_connectedness_hard_dataset,
    "connectedness_hard_fixed": make_connectedness_hard_fixed_dataset,
    "isomorphism":             make_isomorphism_dataset,
    "yehudai_connectivity":    make_yehudai_connectivity_dataset,
}


def load_or_create(
    name: str,
    node_features: str = "degree",
    lpe_dim: int = 0,
    cache_dir: str = "data",
    **structural_kwargs,
) -> list[Data]:
    """Load (or generate) a dataset, then tokenize in memory.

    The on-disk cache stores only raw graph structure — edge_index, y, and
    any metadata (e.g. n1 for isomorphism).  Node features and LPE are applied
    after loading, so changing node_features or lpe_dim never requires a re-run
    of the (potentially slow) graph generation step.

    Cache key = name + structural_kwargs only.  node_features and lpe_dim are
    intentionally excluded from the filename.
    """
    os.makedirs(cache_dir, exist_ok=True)
    suffix = "_".join(f"{k}{v}" for k, v in sorted(structural_kwargs.items()))
    path = os.path.join(cache_dir, f"{name}_{suffix}.pt" if suffix else f"{name}.pt")

    if os.path.exists(path):
        data_list = torch.load(path, weights_only=False)
        print(f"Loaded {len(data_list)} raw graphs from {path}")
    else:
        generator = GENERATORS[name]
        data_list = generator(**structural_kwargs)  # type: ignore[operator]
        torch.save(data_list, path)
        print(f"Saved raw graphs to {path}")

    data_list = tokenize_dataset(data_list, node_features=node_features, lpe_dim=lpe_dim)

    labels = [d.y.item() for d in data_list]
    n_classes = len(set(labels))
    dist = "  ".join(f"class {c}: {labels.count(c)}" for c in range(n_classes))
    print(f"{len(data_list)} samples  |  {dist}")
    return data_list
