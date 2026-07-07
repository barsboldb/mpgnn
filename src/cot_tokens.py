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


def bfs_check_trace(n: int, edges: list[tuple[int, int]], start: int,
                    vocab: CoTVocab) -> tuple[list[int], int]:
    """bfs_expand with the visited-set subtraction made explicit — the next
    rung of the trace-locality ladder (CHANGELOG 2026-07-07).

    On dense graphs bfs_expand's per-parent children are `sorted(adj[u] - seen)`:
    every REJECTED neighbor is a membership test against the whole trace so far
    that emits no token, so the hardest op never gets its own gradient (levels
    2-3 collapse to ~0.25 tf acc at mean degree ~9). Here each parent is an
    explicit scan — every neighbor in ascending order, each followed by a
    supervised verdict:

        SEP  [per parent u: EXP u  [v YES | v NO  for each v in sorted(adj[u])]]

    YES = v is new (it joins the next frontier), NO = already visited. The v
    tokens are a local scan of the parent's prompt edges (the circuit that
    formed at 0.998); each verdict is a 1-bit membership lookup graded with
    partial credit. YES/NO are reused from the answer slot — no vocab change,
    no parsing ambiguity (the answer is always the token after ANS). ~2x the
    bfs_expand length (2 tokens per edge-visit instead of ~1 per accepted
    child); the final all-NO round is the explicit frontier-exhausted proof."""
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
            tokens += [vocab.EXP, u]
            for v in sorted(adj[u]):
                tokens.append(v)
                if v in seen:
                    tokens.append(vocab.NO)
                else:
                    tokens.append(vocab.YES)
                    seen.add(v)
                    nxt.append(v)
        frontier = nxt
    return tokens, int(len(seen) == n)


def wl_expand_trace(n: int, n1: int, edges: list[tuple[int, int]], rounds: int,
                    vocab: CoTVocab) -> tuple[list[int], int]:
    """Verbose 1-WL colour-refinement trace on the disjoint union of a graph
    pair (G1 = nodes 0..n1-1, G2 = nodes n1..n-1) — the isomorphism analog of
    bfs_expand. Colours reuse the node-id token range (canonical ints, first
    appearance in node-id order mints the next colour, scanning the union so
    colour names are shared across the pair). Each round:

        SEP  [per node u: EXP u c_old(u) [sorted neighbour colours] c_new(u)]

    Every token is locally computable: c_old is a copy of the previous round's
    c_new at EXP u (induction head), neighbour colours are lookups keyed by the
    prompt's edge tokens, and c_new is an associative match against this
    round's earlier identical signature (or the next fresh colour) — the same
    retrieval-circuit family the BFS recipe trains. Exactly `rounds` rounds are
    emitted for EVERY pair regardless of when refinement stabilises or
    diverges, so trace length depends only on (n, m) — never on the label.

    The final section makes the multiset comparison explicit instead of leaving
    it as one unsupervised global step over the last round (the Bachmann &
    Nagarajan pitfall the compact bfs_levels target died of):

        SEP SEP [sorted final colours of G1] SEP [sorted final colours of G2]

    Answer: YES iff the two sorted colour lists match (a copy-compare over the
    just-emitted section). Known residual globality: the first token of each
    sorted list is a set-minimum. If trace_em stalls exactly there, switch the
    section to per-colour counts (fixed 0..C-1 enumeration) — noted in the
    config."""
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    colors = [0] * n
    tokens: list[int] = []
    for _ in range(rounds):
        sigs = [(colors[u], tuple(sorted(colors[v] for v in adj[u]))) for u in range(n)]
        palette: dict = {}
        for s in sigs:
            palette.setdefault(s, len(palette))
        new = [palette[s] for s in sigs]
        tokens.append(vocab.SEP)
        for u in range(n):
            tokens += [vocab.EXP, u, colors[u],
                       *sorted(colors[v] for v in adj[u]), new[u]]
        colors = new
    h1, h2 = sorted(colors[:n1]), sorted(colors[n1:])
    tokens += [vocab.SEP, vocab.SEP, *h1, vocab.SEP, *h2]
    return tokens, int(h1 == h2)


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

        # pair graphs (isomorphism): G1 lives at ids 0..n1-1, G2 at n1..n-1.
        # Permute WITHIN each half, independently — membership must stay
        # readable from the id range, but any literal id correspondence between
        # the halves (negatives are rewires of G1) must not survive.
        n1 = int(g.n1.item()) if hasattr(g, "n1") and g.n1 is not None else 0

        if permute:
            perm = (np.concatenate([rng.permutation(n1), n1 + rng.permutation(n - n1)])
                    if n1 else rng.permutation(n))
            edges = [(int(perm[u]), int(perm[v])) for u, v in edges]
        edges = [(min(u, v), max(u, v)) for u, v in edges]
        rng.shuffle(edges)

        # roster=False drops `N v_0..v_{n-1}`: at fixed n it is informationless
        # distractor mass (a guaranteed non-edge occurrence of every node id)
        prompt = [vocab.N, *range(n), vocab.E] if roster else [vocab.E]
        for u, v in edges:
            prompt += [u, v]
        prompt.append(vocab.TRACE)

        if trace_format == "wl_expand":
            assert n1 > 0, \
                "trace_format 'wl_expand' needs pair graphs with n1 (dataset: iso_wl)"
            rounds = int(g.wl_R.item())
            completion, answer = wl_expand_trace(n, n1, edges, rounds, vocab)
            if not trace:            # answer-only ablation: same prompt, no trace
                completion = []
            assert answer == int(g.y.item()), \
                "WL histogram answer disagrees with the dataset label — was the pair " \
                "not screened for WL-divergence within wl_R rounds?"
        elif trace and trace_format == "bfs_expand":
            completion, answer = bfs_expand_trace(n, edges, start=0, vocab=vocab)
        elif trace and trace_format == "bfs_check":
            completion, answer = bfs_check_trace(n, edges, start=0, vocab=vocab)
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
        if trace_format != "wl_expand":
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
        # difficulty knob rides the `diam` field so by-diameter bucketing works
        # unchanged: for WL pairs it is the divergence round (negatives) /
        # stabilisation round (positives), not a graph diameter.
        if hasattr(g, "wl_round") and g.wl_round is not None:
            diam = int(g.wl_round.item())
        else:
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
