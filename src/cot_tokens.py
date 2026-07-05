"""Discrete token sequences for the autoregressive chain-of-thought experiments.

A graph becomes a prompt the decoder-only transformer reads, followed by a
supervised BFS trace and the answer it must emit (teacher-forced at training,
generated greedily at eval):

    N v_0 .. v_{n-1}  E u_1 w_1 .. u_m w_m  TRACE  l0 SEP l1 SEP .. lk  ANS YES|NO EOS

The trace is the canonical BFS from the lowest node id, frontier by frontier,
nodes within a level sorted ascending — deterministic, so teacher forcing has a
unique target. The number of SEPs equals the BFS depth from the start node,
which is the sequential-steps quantity the depth-vs-diameter argument needs.

Node ids are randomly permuted per graph (permute=True) because the generators
place components on contiguous id ranges (connectedness_hard blobs, the
diameter_controlled backbone 0..d) — literal ids would leak the answer to a
model that never traces reachability.

These sequences do NOT go through the PyG DataLoader (it concatenates node
tensors); use a plain torch DataLoader with `collate_cot`.
"""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data


class CoTVocab:
    """Fixed token layout: node ids 0..max_nodes-1, then the special tokens."""

    SPECIALS = ("PAD", "N", "E", "TRACE", "SEP", "ANS", "YES", "NO", "EOS", "EXP")

    def __init__(self, max_nodes: int):
        self.max_nodes = max_nodes
        self.PAD = max_nodes
        self.N = max_nodes + 1
        self.E = max_nodes + 2
        self.TRACE = max_nodes + 3
        self.SEP = max_nodes + 4
        self.ANS = max_nodes + 5
        self.YES = max_nodes + 6
        self.NO = max_nodes + 7
        self.EOS = max_nodes + 8
        self.EXP = max_nodes + 9   # parent marker for trace_format: bfs_expand
        self.size = max_nodes + len(self.SPECIALS)

    def answer_token(self, label: int) -> int:
        return self.YES if label == 1 else self.NO

    def decode(self, ids) -> str:
        names = []
        for t in (ids.tolist() if torch.is_tensor(ids) else ids):
            names.append(str(t) if t < self.max_nodes else self.SPECIALS[t - self.max_nodes])
        return " ".join(names)


def bfs_levels(n: int, edges: list[tuple[int, int]], start: int) -> tuple[list[list[int]], int]:
    """Canonical BFS: levels (each sorted ascending) from `start`, and the
    connectedness answer (1 iff every node was reached)."""
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    seen = {start}
    levels, frontier = [[start]], [start]
    while frontier:
        nxt = sorted({v for u in frontier for v in adj[u] if v not in seen})
        seen.update(nxt)
        if nxt:
            levels.append(nxt)
        frontier = nxt
    return levels, int(len(seen) == n)


def bfs_expand_trace(n: int, edges: list[tuple[int, int]], start: int,
                     vocab: CoTVocab) -> tuple[list[int], int]:
    """Verbose BFS trace: each level is `EXP parent [new children sorted] ...`
    for EVERY previous-level node (empty expansions included), levels separated
    by SEP. Same information as bfs_levels but every next token is LOCALLY
    computable: parents are a copy of the previous level's children in order
    (induction head), children are a lookup keyed by the adjacent parent token —
    no global sorted-set-minimum per token. This is the Bachmann & Nagarajan
    (2024) lesson: teacher forcing only learns computations the trace makes
    explicit; compressed targets leave the hard step ungraded."""
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    seen = {start}
    tokens = [start]
    frontier = [start]
    while frontier:
        nxt: list[int] = []
        tokens.append(vocab.SEP)
        for u in frontier:
            tokens.append(vocab.EXP)
            tokens.append(u)
            for v in sorted(set(adj[u]) - seen):
                seen.add(v)
                nxt.append(v)
                tokens.append(v)
        frontier = nxt
    # the last round is all-empty expansions — kept on purpose: it is the local,
    # explicit "frontier exhausted" proof right before ANS
    return tokens, int(len(seen) == n)


def _undirected_edges(edge_index: torch.Tensor) -> list[tuple[int, int]]:
    und = edge_index[0] < edge_index[1]
    return list(zip(edge_index[0][und].tolist(), edge_index[1][und].tolist()))


def build_cot_sequences(
    data_list: list[Data],
    vocab: CoTVocab,
    permute: bool = True,
    trace: bool = True,
    seed: int = 0,
    max_seq_len: int = 0,
    trace_format: str = "bfs_levels",
    roster: bool = True,
    drop_overlong: bool = False,
) -> list[dict]:
    """Token sequences for every graph: {tokens, prompt_len, y, diam}.

    trace=False emits only `ANS YES|NO EOS` after the prompt — the answer-only
    ablation with the identical architecture and prompt distribution.
    trace_format: 'bfs_levels' (compact sorted frontiers) or 'bfs_expand'
    (verbose `EXP parent children` rounds; ~2x longer, every token local).
    max_seq_len > 0 asserts every sequence fits (catches config/dataset drift
    before the position table does, with a message that names the culprit).
    drop_overlong=True skips oversized graphs with a report instead — for
    evaluating a trained checkpoint (fixed position table) on denser datasets,
    where one big graph must not kill the whole probe. Never use in training.
    """
    rng = np.random.default_rng(seed)
    out = []
    dropped = 0
    for g in data_list:
        n = int(g.num_nodes)
        assert n <= vocab.max_nodes, \
            f"graph has {n} nodes but the vocab covers max_nodes={vocab.max_nodes}"
        edges = _undirected_edges(g.edge_index)

        if permute:
            perm = rng.permutation(n)
            edges = [(int(perm[u]), int(perm[v])) for u, v in edges]
        edges = [(min(u, v), max(u, v)) for u, v in edges]
        rng.shuffle(edges)

        # roster=False drops `N v_0..v_{n-1}`: at fixed n it is informationless
        # distractor mass (a guaranteed non-edge occurrence of every node id)
        prompt = [vocab.N, *range(n), vocab.E] if roster else [vocab.E]
        for u, v in edges:
            prompt += [u, v]
        prompt.append(vocab.TRACE)

        if trace and trace_format == "bfs_expand":
            completion, answer = bfs_expand_trace(n, edges, start=0, vocab=vocab)
        elif trace and trace_format == "bfs_l1":
            # diagnostic probe: emit ONLY the sorted neighbours of node 0 — the
            # atomic lookup circuit, isolated from all BFS composition. trace_em
            # then measures exactly "can the model do one content lookup".
            levels, answer = bfs_levels(n, edges, start=0)
            completion = list(levels[1]) if len(levels) > 1 else []
        else:
            levels, answer = bfs_levels(n, edges, start=0)
            completion = []
            if trace:
                for i, level in enumerate(levels):
                    if i > 0:
                        completion.append(vocab.SEP)
                    completion += level
        assert answer == int(g.y.item()), \
            "BFS-from-0 answer disagrees with the dataset label — is y not connectedness?"
        completion += [vocab.ANS, vocab.answer_token(answer), vocab.EOS]

        tokens = torch.tensor(prompt + completion, dtype=torch.long)
        if max_seq_len > 0 and tokens.numel() > max_seq_len:
            if drop_overlong:
                dropped += 1
                continue
            raise AssertionError(
                f"sequence of length {tokens.numel()} (n={n}, m={len(edges)}) exceeds "
                f"max_seq_len={max_seq_len}; raise max_seq_len in the config")
        diam = int(g.diam.item()) if hasattr(g, "diam") and g.diam is not None else -1
        out.append({"tokens": tokens, "prompt_len": len(prompt),
                    "y": int(g.y.item()), "diam": diam})
    if dropped:
        print(f"[cot] dropped {dropped}/{len(data_list)} graphs whose sequences exceed "
              f"max_seq_len={max_seq_len} (eval on the {len(out)} that fit)")
    return out


def collate_cot(batch: list[dict], pad_id: int):
    """Right-pad a list of sequence dicts into dense tensors."""
    L = max(item["tokens"].numel() for item in batch)
    tokens = torch.full((len(batch), L), pad_id, dtype=torch.long)
    for i, item in enumerate(batch):
        tokens[i, : item["tokens"].numel()] = item["tokens"]
    return {
        "tokens": tokens,
        "prompt_len": torch.tensor([b["prompt_len"] for b in batch], dtype=torch.long),
        "y": torch.tensor([b["y"] for b in batch], dtype=torch.long),
        "diam": torch.tensor([b["diam"] for b in batch], dtype=torch.long),
    }


def make_cot_loader(sequences: list[dict], vocab: CoTVocab,
                    batch_size: int, shuffle: bool) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        sequences, batch_size=batch_size, shuffle=shuffle,
        collate_fn=lambda b: collate_cot(b, vocab.PAD))
