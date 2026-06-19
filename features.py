"""Structural features computed once per graph (no learnable params here).

These are the graph-specific inputs that a vanilla transformer lacks:
  - Laplacian positional encoding: where a node sits in the graph's structure.
  - Shortest-path distances: how far apart two nodes are, used later as an
    attention bias.
"""

import collections

import torch


def laplacian_positional_encoding(edge_index, num_nodes, k):
    """k smallest non-trivial eigenvectors of the symmetric normalized Laplacian.

    L = I - D^{-1/2} A D^{-1/2}. The first eigenvector (eigenvalue ~0) is constant
    and carries no positional information, so it is skipped. The result is padded
    with zeros if the graph has fewer than k+1 nodes.
    """
    A = torch.zeros(num_nodes, num_nodes)
    A[edge_index[0], edge_index[1]] = 1.0
    A = ((A + A.t()) > 0).float()  # symmetrize, drop multiplicity

    deg = A.sum(dim=1)
    d_inv_sqrt = deg.pow(-0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    L = torch.eye(num_nodes) - d_inv_sqrt.unsqueeze(1) * A * d_inv_sqrt.unsqueeze(0)

    # eigh because L is real symmetric; eigenvalues come out ascending.
    _, eigvecs = torch.linalg.eigh(L)
    pe = eigvecs[:, 1:k + 1]
    if pe.shape[1] < k:  # tiny graph: pad
        pe = torch.cat([pe, torch.zeros(num_nodes, k - pe.shape[1])], dim=1)
    return pe  # (num_nodes, k)


def shortest_path_distances(edge_index, num_nodes, max_dist):
    """All-pairs shortest-path lengths via BFS, clamped to `max_dist`.

    Unreachable pairs (and anything farther than max_dist) collapse to max_dist,
    which becomes its own learnable bias bucket.
    """
    adj = [[] for _ in range(num_nodes)]
    for s, t in edge_index.t().tolist():
        adj[s].append(t)
        adj[t].append(s)

    dist = torch.full((num_nodes, num_nodes), max_dist, dtype=torch.long)
    for src in range(num_nodes):
        dist[src, src] = 0
        seen = {src}
        q = collections.deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    dist[src, v] = min(int(dist[src, u]) + 1, max_dist)
                    q.append(v)
    return dist  # (num_nodes, num_nodes) long


def spd_batch(edge_index: torch.Tensor, batch: torch.Tensor | None,
              num_nodes: int, max_dist: int) -> torch.Tensor:
    """Build a block-diagonal SPD matrix for a batch of graphs.

    Each graph's all-pairs distances sit on the diagonal block;
    cross-graph entries are set to max_dist (treated as unreachable).
    Returns shape [N_total, N_total].
    """
    device = edge_index.device
    out = torch.full((num_nodes, num_nodes), max_dist, dtype=torch.long, device=device)

    if batch is None:
        spd = shortest_path_distances(edge_index.cpu(), num_nodes, max_dist)
        return spd.to(device)

    for g in range(int(batch.max().item()) + 1):
        node_mask = batch == g
        node_idx = node_mask.nonzero(as_tuple=True)[0]  # global node indices
        n = node_idx.size(0)
        offset = int(node_idx[0].item())

        # extract edges within this graph and remap to local 0..n-1 indices
        edge_mask = node_mask[edge_index[0]]
        local_ei = edge_index[:, edge_mask] - offset

        spd = shortest_path_distances(local_ei.cpu(), n, max_dist).to(device)
        out[offset:offset + n, offset:offset + n] = spd

    return out
