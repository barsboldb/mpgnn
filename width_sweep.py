"""Width sweeps for the thesis questions (docs/WIDTH-QUESTIONS.md).

Reuses the connectivity-matrix task from src/connectivity.py (A+I row tokens ->
n x n reachability, exact-match metric) but trains with a train/val/test split so
hyperparameters are picked on validation, never on test.

Every run appends one JSON line to results/width/<sweep>.jsonl; reruns skip runs
already on disk, so a sweep can be killed and resumed.

Usage:
    python width_sweep.py q1                 # run (or resume) the Q1 sweep
    python width_sweep.py q1 --smoke         # 2 epochs, 1 seed, 2 widths, 2 LRs
    python width_sweep.py q1 --analyze       # tables from the saved runs
    python width_sweep.py q1b --shard 0/4    # worker 0 of 4 (run 4 in parallel)

Sweeps may add axes: `ns` (graph sizes), `depths` (ints or "log" = ceil(log2 n)),
`train_sizes`; analysis reports one table per (n, depth, train size) cell.
"""
import argparse
import glob
import itertools
import math
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse.csgraph import shortest_path

from src.connectivity import DEVICE, ConnectivityTransformer, make_set, reachability
from src.dataset import (_cycle_blob_edges, make_connectedness_hard_dataset,
                         make_connectedness_hard_diam_dataset)

OUT_DIR = os.path.join("results", "width")

SWEEPS = {
    # Q1: is a fixed learning rate confounding width? Fixed LR = the 1e-3 column,
    # tuned = best-validation LR per width. Head dim fixed so only m moves.
    "q1": dict(dist="hard", n=24, depth=2, head_dim=8,
               widths=[8, 16, 32, 64, 128, 256],
               lrs=[3e-4, 1e-3, 3e-3, 1e-2],
               seeds=[0, 1, 2],
               train=2000, val=400, test=400, epochs=300, batch=128,
               fixed_lr=1e-3),
    # Q1b: Q1 plus linear warm-up and cosine decay (per step), to test whether the
    # wide-model drop in Q1 is optimisation instability rather than width. Widths 2
    # and 4 added to find where narrow models fail; 3e-2 added since warm-up lets
    # higher rates survive.
    "q1b": dict(dist="hard", n=24, depth=2, head_dim=8,
                widths=[2, 4, 8, 16, 32, 64, 128, 256],
                lrs=[3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
                seeds=[0, 1, 2],
                train=2000, val=400, test=400, epochs=300, batch=128,
                fixed_lr=1e-3, warmup_frac=0.05, schedule="cosine"),
    # Q1c: width x training-set size. If the Q1b wide-model drop is memorization, it
    # should move to larger m as data grows. Optimizer steps are fixed (= Q1b's 4800)
    # so more data never means more training.
    "q1c": dict(dist="hard", n=24, depth=2, head_dim=8,
                widths=[8, 16, 32, 64, 128, 256],
                lrs=[1e-3, 3e-3, 1e-2, 3e-2],
                seeds=[0, 1, 2], train_sizes=[500, 2000, 8000], steps=4800,
                train=2000, val=400, test=400, epochs=300, batch=128,
                fixed_lr=1e-3, warmup_frac=0.05, schedule="cosine"),
    # Q3: critical width vs graph size n. Graphs are relabelled `swap` blobs:
    # locally indistinguishable classes (degree and short-cycle statistics at
    # chance, k-hop heuristics fail up to k=8; audit_width_data.py) with diameter
    # growing ~log n. 8000 train graphs keep the Q1c data ceiling out of the way.
    # depth 2 (fixed) vs ceil(log2 n): can width stand in for missing depth?
    "q3": dict(gen="swap", chord_frac=0.25, dmin=6, ns=[32, 64, 128], depths=[2, "log"],
               head_dim=8, widths=[4, 8, 16, 32, 64, 128],
               lrs=[1e-3, 3e-3, 1e-2],
               seeds=[0, 1, 2], train_sizes=[8000], steps=4800,
               train=8000, val=400, test=400, epochs=77, batch=128,
               fixed_lr=1e-3, warmup_frac=0.05, schedule="cosine"),
}
# Pilot 1 (2026-10-02): relabelled hard_diam at n = 16/32/64 — diameter grew ~n/2 and
# nothing generalized at n >= 32. Kept for the record.
SWEEPS["q3pilot"] = dict(SWEEPS["q3"], gen="hard_diam", ns=[16, 32, 64], widths=[8, 32, 128],
                         lrs=[3e-3], seeds=[0], fixed_lr=3e-3)
# Pilot 2: the q3 data at one seed, one LR, three widths — is it learnable at all?
SWEEPS["q3pilot2"] = dict(SWEEPS["q3"], widths=[8, 32, 128], lrs=[3e-3], seeds=[0],
                          fixed_lr=3e-3)
# Learnability probe: pilot 2 learned only at n=32, depth 5, m=128 (pair 0.87). Is it a
# budget problem? 5x the steps, 4x the data, wider models; read the learning curves
# (--curves) for whether and when exact-match leaves 0.5.
SWEEPS["q3probe"] = dict(SWEEPS["q3"], ns=[32], depths=["log"], widths=[64, 128, 256],
                         lrs=[1e-3, 3e-3], seeds=[0], train_sizes=[32000], train=32000,
                         steps=24000, fixed_lr=3e-3)
# Trimmed Q3 (the probe set the budget): 32 000 graphs, 12 000 steps (the probe was at
# 0.95-0.98 halfway), depth ceil(log2 n) only, the narrow end of the width range where
# m* lives, 2 LRs x 2 seeds. Depth 2 (can width replace depth?) moves to Q7.
SWEEPS["q3trim"] = dict(SWEEPS["q3"], depths=["log"], widths=[8, 16, 32, 64, 128],
                        lrs=[1e-3, 3e-3], seeds=[0, 1], train_sizes=[32000], train=32000,
                        steps=12000, fixed_lr=1e-3)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_size(r):
    return r.get("train_size", r["config"]["train"])


def run_key(r):
    return (r["width"], r["lr"], r["seed"], train_size(r), r["config"]["n"], r["config"]["depth"])


def cell(r):
    """The (n, depth, train size) group a run belongs to."""
    return r["config"]["n"], r["config"]["depth"], train_size(r)


def resolve_depth(depth, n):
    return math.ceil(math.log2(n)) if depth == "log" else depth


def point(cfg, n, depth, size):
    """cfg for one (n, depth, train size); with `steps`, epochs keep the step count fixed."""
    c = dict(cfg, n=n, depth=resolve_depth(depth, n), train=size)
    if "steps" in cfg:
        c["epochs"] = math.ceil(cfg["steps"] / math.ceil(size / cfg["batch"]))
    return c


def load_runs(sweep):
    """All runs of a sweep: <sweep>.jsonl plus any per-shard <sweep>.shard*.jsonl."""
    runs = []
    for path in sorted(glob.glob(os.path.join(OUT_DIR, f"{sweep}.jsonl")) +
                       glob.glob(os.path.join(OUT_DIR, f"{sweep}.shard*.jsonl"))):
        with open(path) as f:
            runs += [json.loads(line) for line in f if line.strip()]
    return runs


def sparse_blob_graphs(num, n, seed, chord_frac):
    """Two sparse blobs +/- one bridge, like hard_diam, but each blob gets
    round(chord_frac * size) chords so its diameter grows ~log(size) rather than
    ~size/2 — n can grow without the paths outrunning every depth. Label 1: one
    bridge; label 0: one extra intra-blob chord instead (edge counts and degree
    sequences matched). Blob split na ~ U[n/4, 3n/4]. Yields (adjacency, label)."""
    rng = np.random.default_rng(seed)
    for i in range(num):
        label = i % 2
        na = int(rng.integers(n // 4, n - n // 4 + 1))
        blobs = (list(range(na)), list(range(na, n)))
        edges = set()
        for b in blobs:
            edges |= _cycle_blob_edges(b, round(chord_frac * len(b)), rng)
        if label == 1:
            u, v = int(rng.choice(blobs[0])), int(rng.choice(blobs[1]))
            edges.add((min(u, v), max(u, v)))
        else:
            while True:
                b = blobs[int(rng.integers(2))]
                u, v = (int(x) for x in rng.choice(b, size=2, replace=False))
                e = (min(u, v), max(u, v))
                if e not in edges:
                    edges.add(e)
                    break
        A = np.zeros((n, n), dtype=np.float32)
        for u, v in edges:
            A[u, v] = A[v, u] = 1.0
        yield A, label


def _far_pair(A, nodes, rng):
    """A random node of `nodes`, a node at maximum distance from it within the blob,
    and that distance."""
    sub = A[np.ix_(nodes, nodes)]
    i = int(rng.integers(len(nodes)))
    d = shortest_path(sub, unweighted=True, directed=False, indices=i)
    dmax = d[np.isfinite(d)].max()
    far = np.flatnonzero(d == dmax)
    return nodes[i], nodes[int(rng.choice(far))], int(dmax)


def swap_blob_graphs(num, n, seed, chord_frac, dmin=6):
    """Locally indistinguishable classes (the 1-cycle vs 2-cycle idea). Two sparse
    blobs (cycle + round(chord_frac * size) chords); pick a far-apart pair a1, a2 in
    blob A and b1, b2 in blob B. Label 0 adds a1-a2 and b1-b2 (two components);
    label 1 adds a1-b1 and a2-b2 (connected). Same four endpoints gain one degree in
    both classes, and every cycle the new edges close is long, so degree and
    short-cycle statistics carry no label signal. Each pair must be >= dmin hops
    apart, so every cycle the new edges close has >= dmin + 1 edges (invisible to
    closed-walk counts up to length dmin); graphs that miss it are redrawn before
    the label is applied, so the filter is label-independent. Yields (adjacency, label)."""
    rng = np.random.default_rng(seed)
    for i in range(num):
        label = i % 2
        for _ in range(1000):
            na = int(rng.integers(n // 4, n - n // 4 + 1))
            blobs = (list(range(na)), list(range(na, n)))
            A = np.zeros((n, n), dtype=np.float32)
            for b in blobs:
                for u, v in _cycle_blob_edges(b, round(chord_frac * len(b)), rng):
                    A[u, v] = A[v, u] = 1.0
            a1, a2, da = _far_pair(A, blobs[0], rng)
            b1, b2, db = _far_pair(A, blobs[1], rng)
            if min(da, db) >= dmin:
                break
        else:
            raise ValueError(f"swap: no blob pair >= {dmin} hops apart at n={n}, "
                             f"chord_frac={chord_frac}; lower dmin or chord_frac")
        new = [(a1, a2), (b1, b2)] if label == 0 else [(a1, b1), (a2, b2)]
        for u, v in new:
            A[u, v] = A[v, u] = 1.0
        yield A, label


def raw_graphs(gen, num, n, seed, chord_frac=0.5, dmin=6):
    """(adjacency, label) pairs from a named generator, before relabelling."""
    if gen == "sparse":
        yield from sparse_blob_graphs(num, n, seed, chord_frac)
        return
    if gen == "swap":
        yield from swap_blob_graphs(num, n, seed, chord_frac, dmin)
        return
    make = {"hard": make_connectedness_hard_dataset,
            "hard_diam": make_connectedness_hard_diam_dataset}[gen]
    for g in make(num_graphs=num, min_nodes=n, max_nodes=n, seed=seed):
        A = np.zeros((n, n), dtype=np.float32)
        ei = g.edge_index.numpy()
        A[ei[0], ei[1]] = 1.0
        yield A, int(g.y)


def relabelled(gen, num, n, seed, chord_frac=0.5, dmin=6):
    """Graphs with node labels shuffled per graph, so component membership can't be
    read off node indices. Returns (A + I, R) tensors."""
    rng = np.random.default_rng(seed + 1)
    As, Rs = [], []
    for A, _ in raw_graphs(gen, num, n, seed, chord_frac, dmin):
        perm = rng.permutation(n)
        A = A[np.ix_(perm, perm)]
        As.append(A + np.eye(n, dtype=np.float32))
        Rs.append(reachability(A))
    return torch.tensor(np.array(As)), torch.tensor(np.array(Rs))


@torch.no_grad()
def metrics(model, A, R, bs=128):
    """Exact-match (whole matrix right) and pair accuracy (fraction of off-diagonal
    entries right) in one pass. A, R may be float or uint8, on any device."""
    model.eval()
    n = A.size(1)
    off = ~torch.eye(n, dtype=torch.bool, device=DEVICE)
    exact = pair = 0.0
    for i in range(0, A.size(0), bs):
        lo = model(A[i:i+bs].to(DEVICE).float())
        ok = (lo > 0) == (R[i:i+bs].to(DEVICE) > 0.5)
        exact += ok.all(dim=(1, 2)).float().sum().item()
        pair += ok[:, off].float().mean(dim=1).sum().item()
    return exact / A.size(0), pair / A.size(0)


CACHE_DIR = os.environ.get("WIDTH_CACHE", os.path.join("data", "width_cache"))


def cached(key, build, wait_s=7200):
    """Build-once tensor cache shared by parallel shards: the first shard to take the
    lock builds and saves (atomically); the others wait for the file and load it."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{key}.pt")
    lock = path + ".lock"
    t0 = time.time()
    while not os.path.exists(path):
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            if time.time() - t0 > wait_s:
                raise TimeoutError(f"waited {wait_s}s for {path}; stale {lock}?")
            time.sleep(5)
            continue
        try:
            data = build()
            torch.save(data, path + ".tmp")
            os.replace(path + ".tmp", path)
            return data
        finally:
            os.remove(lock)
    return torch.load(path)


def make_splits(cfg, seed):
    """Same graphs for every width/LR at a given seed; val and test use their own seeds."""
    if "gen" in cfg:
        g, n = cfg["gen"], cfg["n"]
        kw = dict(chord_frac=cfg.get("chord_frac", 0.5), dmin=cfg.get("dmin", 6))
        tag = f"{g}_n{n}_cf{kw['chord_frac']}_d{kw['dmin']}"
        # 0/1 matrices stored as uint8 (4x smaller); batches are cast to float on use.
        return tuple(cached(f"{tag}_N{num}_s{sd}_u8", lambda num=num, sd=sd:
                            tuple(t.to(torch.uint8) for t in relabelled(g, num, n, sd, **kw)))
                     for num, sd in ((cfg["train"], seed), (cfg["val"], seed + 5555),
                                     (cfg["test"], seed + 9999)))
    rng = np.random.default_rng(seed)
    kw = dict(n=cfg["n"], p=0.12, cap=10**6, dist=cfg["dist"])
    tr = make_set(cfg["train"], rng=rng, seed=seed, **kw)
    va = make_set(cfg["val"], rng=rng, seed=seed + 5555, **kw)
    te = make_set(cfg["test"], rng=rng, seed=seed + 9999, **kw)
    return tr, va, te


def train_one(cfg, width, lr, seed, splits):
    (Atr, Rtr), (Ava, Rva), (Ate, Rte) = splits
    set_seed(seed)
    heads = max(1, width // cfg["head_dim"])
    model = ConnectivityTransformer(cfg["n"], width, cfg["depth"], heads).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    Atr_d, Rtr_d = Atr.to(DEVICE), Rtr.to(DEVICE)
    N, bs, epochs = Atr.size(0), cfg["batch"], cfg["epochs"]
    sched = None
    if cfg.get("schedule") == "cosine":
        total = epochs * math.ceil(N / bs)
        warm = max(1, int(cfg.get("warmup_frac", 0.0) * total))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: (t + 1) / warm if t < warm
            else 0.5 * (1 + math.cos(math.pi * (t - warm) / max(1, total - warm))))
    history, t0 = [], time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(N, device=DEVICE)
        tot = 0.0
        for i in range(0, N, bs):
            idx = perm[i:i+bs]
            loss = F.binary_cross_entropy_with_logits(model(Atr_d[idx].float()),
                                                      Rtr_d[idx].float())
            opt.zero_grad(); loss.backward(); opt.step()
            if sched is not None:
                sched.step()
            tot += loss.item() * idx.size(0)
        if epoch % max(1, epochs // 15) == 0 or epoch in (1, epochs):
            test, test_pair = metrics(model, Ate, Rte, bs)
            history.append({"epoch": epoch, "loss": tot / N,
                            "train": metrics(model, Atr_d, Rtr_d, bs)[0],
                            "val": metrics(model, Ava, Rva, bs)[0],
                            "test": test, "test_pair": test_pair})

    best = max(history, key=lambda h: h["val"])
    return {"width": width, "lr": lr, "seed": seed, "heads": heads, "train_size": N,
            "params": model.num_parameters(), "seconds": round(time.time() - t0, 1),
            "final": history[-1], "best_val": best, "history": history}


def run_sweep(name, smoke=False, shard=(0, 1)):
    cfg = dict(SWEEPS[name])
    if smoke:
        cfg.update(widths=cfg["widths"][:2], lrs=cfg["lrs"][:2], seeds=cfg["seeds"][:1],
                   epochs=2, train=256, val=64, test=64)
        cfg.pop("steps", None)
        if "train_sizes" in cfg:
            cfg["train_sizes"] = [128, 256]
        name = f"{name}_smoke"
    os.makedirs(OUT_DIR, exist_ok=True)
    # Parallel shards each append to their own file so lines never interleave.
    suffix = f".shard{shard[0]}of{shard[1]}" if shard[1] > 1 else ""
    path = os.path.join(OUT_DIR, f"{name}{suffix}.jsonl")
    done = {run_key(r) for r in load_runs(name)}
    sizes = cfg.get("train_sizes", [cfg["train"]])
    ns, depths = cfg.get("ns", [cfg.get("n")]), cfg.get("depths", [cfg.get("depth")])
    todo = [(w, lr, s, size, n, d) for s, n, d, size, w, lr in
            itertools.product(cfg["seeds"], ns, depths, sizes, cfg["widths"], cfg["lrs"])
            if (w, lr, s, size, n, resolve_depth(d, n)) not in done]
    todo = todo[shard[0]::shard[1]]
    print(f"Device {DEVICE} | sweep {name} shard {shard[0]}/{shard[1]}: "
          f"{len(todo)} runs to go ({len(done)} on disk)")

    splits = {}
    for k, (w, lr, s, size, n, d) in enumerate(todo, 1):
        c = point(cfg, n, d, size)
        if (s, n, size) not in splits:
            splits.clear()             # one dataset in memory at a time (n=128 is large)
            print(f"  loading data for seed {s}, n={n}, {size} train graphs...")
            splits[(s, n, size)] = make_splits(c, s)
        r = train_one(c, w, lr, s, splits[(s, n, size)])
        r["sweep"], r["config"] = name, {k2: v for k2, v in c.items() if k2 not in
                                         ("widths", "lrs", "seeds", "train_sizes", "ns", "depths")}
        with open(path, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(f"  [{k}/{len(todo)}] n={n:<3d} L={c['depth']:<2d} N={size:<5d} m={w:<4d} H={r['heads']:<3d} lr={lr:<7g} seed={s}  "
              f"train={r['final']['train']:.3f} val={r['best_val']['val']:.3f} "
              f"test@bestval={r['best_val']['test']:.3f} test@final={r['final']['test']:.3f}  "
              f"({r['seconds']}s)")


def _prepare_one(args):
    c, s = args
    make_splits(c, s)
    return c["n"], c["train"], s


def prepare(name, workers=None):
    """Build every cached dataset a sweep needs, in parallel, before training starts."""
    from multiprocessing import Pool
    cfg = SWEEPS[name]
    if "gen" not in cfg:
        print(f"{name}: no cached data to prepare")
        return
    jobs = {(n, size, s): (point(cfg, n, cfg.get("depths", [2])[0], size), s)
            for s in cfg["seeds"] for n in cfg.get("ns", [cfg.get("n")])
            for size in cfg.get("train_sizes", [cfg["train"]])}
    workers = workers or min(len(jobs), os.cpu_count() or 1)
    print(f"{name}: preparing {len(jobs)} datasets with {workers} processes -> {CACHE_DIR}")
    t0 = time.time()
    with Pool(workers) as pool:
        for n, size, s in pool.imap_unordered(_prepare_one, list(jobs.values())):
            print(f"  ready: n={n} train={size} seed={s}  ({time.time() - t0:.0f}s)", flush=True)


def analyze(name, thresholds=(0.90, 0.95, 0.99)):
    runs = load_runs(name)
    if not runs:
        print(f"no runs for {name}")
        return
    cells = sorted({cell(r) for r in runs})
    tuned = {}
    for c in cells:
        label = f"{name} (n={c[0]}, depth={c[1]}, train={c[2]})" if len(cells) > 1 else name
        tuned[c] = report([r for r in runs if cell(r) == c], label, thresholds)
    if len(cells) > 1:
        widths = sorted({r["width"] for r in runs})
        heads = [f"n{c[0]} L{c[1]} N{c[2]}" for c in cells]
        print("\nTuned test exact-match by width x (n, depth, train size); train acc in brackets:")
        print("  m     " + "".join(f"{h:>17}" for h in heads))
        for w in widths:
            row = []
            for c in cells:
                t = tuned[c].get(w)
                row.append(f"{np.mean(t[0]):10.3f} ({np.mean(t[1]):.2f})" if t else f"{'-':>17}")
            print(f"  {w:<6d}" + "".join(row))


def curves(name):
    """Test exact-match / pair accuracy at every evaluation, one line per run."""
    for r in sorted(load_runs(name), key=lambda r: (*cell(r), r["width"], r["lr"], r["seed"])):
        c = r["config"]
        print(f"\nn={c['n']} L={c['depth']} N={train_size(r)} m={r['width']} lr={r['lr']:g} "
              f"seed={r['seed']}  ({r['seconds']}s)")
        print("  epoch  " + " ".join(f"{h['epoch']:>6d}" for h in r["history"]))
        for key in ("train", "test", "test_pair"):
            print(f"  {key:<7}" + " ".join(f"{h.get(key, float('nan')):6.3f}" for h in r["history"]))


def report(runs, name, thresholds):
    """Tables for one training-set size; returns {width: (tuned test, tuned train)}."""
    cfg = runs[0]["config"]
    widths = sorted({r["width"] for r in runs})
    lrs = sorted({r["lr"] for r in runs})
    seeds = sorted({r["seed"] for r in runs})
    by = {run_key(r)[:3]: r for r in runs}   # one training-set size here

    def test(r):
        return r["best_val"]["test"]

    print(f"\n{name}: n={cfg['n']} depth={cfg['depth']} head_dim={cfg['head_dim']} "
          f"epochs={cfg['epochs']} | {len(runs)} runs")
    print("\nMean test exact-match (best-val epoch) over seeds, by width x LR:")
    print("  m     " + "".join(f"{lr:>9g}" for lr in lrs))
    for w in widths:
        row = []
        for lr in lrs:
            v = [test(by[(w, lr, s)]) for s in seeds if (w, lr, s) in by]
            row.append(f"{np.mean(v):9.3f}" if v else f"{'-':>9}")
        print(f"  {w:<6d}" + "".join(row))

    # Fixed LR vs per-seed tuned LR (picked on validation, reported on test).
    fixed = cfg["fixed_lr"]
    per_width, tuned = {}, {}
    print(f"\nFixed LR ({fixed:g}) vs tuned LR (best val per width and seed):")
    print("  m      fixed_test   tuned_test   tuned_pair   tuned_lrs")
    for w in widths:
        f_acc, t_acc, t_lr, t_pair = [], [], [], []
        for s in seeds:
            cands = [by[(w, lr, s)] for lr in lrs if (w, lr, s) in by]
            if not cands:
                continue
            if (w, fixed, s) in by:
                f_acc.append(test(by[(w, fixed, s)]))
            b = max(cands, key=lambda r: r["best_val"]["val"])
            t_acc.append(test(b)); t_lr.append(b["lr"])
            t_pair.append(b["best_val"].get("test_pair", float("nan")))
            tuned.setdefault(w, ([], []))[0].append(test(b))
            tuned[w][1].append(b["final"]["train"])
        per_width[w] = (f_acc, t_acc)
        print(f"  {w:<6d} {np.mean(f_acc):10.3f}   {np.mean(t_acc):10.3f}   "
              f"{np.mean(t_pair):10.3f}   {', '.join(f'{x:g}' for x in t_lr)}")

    print("\nCritical width m* (smallest m where >= 2/3 of seeds reach the threshold):")
    for th in thresholds:
        out = []
        for label, i in (("fixed", 0), ("tuned", 1)):
            ms = [w for w in widths
                  if per_width[w][i] and np.mean(np.array(per_width[w][i]) >= th) >= 2 / 3]
            out.append(f"{label}={ms[0] if ms else 'none'}")
        print(f"  threshold {th:.2f}: " + "  ".join(out))
    return tuned


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep", choices=sorted(SWEEPS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--shard", default="0/1", help="i/k: run every k-th pending run from i")
    ap.add_argument("--curves", action="store_true", help="per-run learning curves")
    ap.add_argument("--prepare", action="store_true", help="build the sweep's datasets first")
    a = ap.parse_args()
    if a.prepare:
        prepare(a.sweep)
    elif a.curves:
        curves(f"{a.sweep}_smoke" if a.smoke else a.sweep)
    elif a.analyze:
        analyze(f"{a.sweep}_smoke" if a.smoke else a.sweep)
    else:
        i, k = map(int, a.shard.split("/"))
        run_sweep(a.sweep, smoke=a.smoke, shard=(i, k))
