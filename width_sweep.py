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

from src.connectivity import DEVICE, ConnectivityTransformer, evaluate, make_set, reachability
from src.dataset import make_connectedness_hard_diam_dataset

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
    # Q3: critical width vs graph size n. Graphs are relabelled hard_diam (sparse
    # blobs, diameter ~n/2, no index leak); 8000 train graphs so the Q1c data
    # ceiling stays out of the way. depth 2 (fixed) vs ceil(log2 n) (theory's
    # depth for connectivity): can width stand in for missing depth?
    "q3": dict(gen="hard_diam", ns=[16, 32, 64], depths=[2, "log"], head_dim=8,
               widths=[4, 8, 16, 32, 64, 128],
               lrs=[1e-3, 3e-3, 1e-2],
               seeds=[0, 1, 2], train_sizes=[8000], steps=4800,
               train=8000, val=400, test=400, epochs=77, batch=128,
               fixed_lr=1e-3, warmup_frac=0.05, schedule="cosine"),
}
# Q3 pilot: one seed, one LR, three widths — is relabelled hard_diam learnable at all?
SWEEPS["q3pilot"] = dict(SWEEPS["q3"], widths=[8, 32, 128], lrs=[3e-3], seeds=[0],
                         fixed_lr=3e-3)


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


def relabelled_hard_diam(num, n, seed):
    """hard_diam graphs at fixed n with node labels shuffled per graph, so component
    membership can't be read off node indices. Returns (A + I, R) tensors."""
    rng = np.random.default_rng(seed + 1)
    As, Rs = [], []
    for g in make_connectedness_hard_diam_dataset(num_graphs=num, min_nodes=n, max_nodes=n,
                                                  seed=seed):
        A = np.zeros((n, n), dtype=np.float32)
        ei = g.edge_index.numpy()
        A[ei[0], ei[1]] = 1.0
        perm = rng.permutation(n)
        A = A[np.ix_(perm, perm)]
        As.append(A + np.eye(n, dtype=np.float32))
        Rs.append(reachability(A))
    return torch.tensor(np.array(As)), torch.tensor(np.array(Rs))


def pair_accuracy(model, A, R, bs=128):
    """Fraction of off-diagonal entries of R predicted correctly (exact-match's soft twin)."""
    model.eval()
    n = A.size(1)
    off = ~torch.eye(n, dtype=torch.bool, device=DEVICE)
    right = 0.0
    with torch.no_grad():
        for i in range(0, A.size(0), bs):
            lo = model(A[i:i+bs].to(DEVICE))
            ok = (lo > 0).float() == R[i:i+bs].to(DEVICE)
            right += ok[:, off].float().mean(dim=1).sum().item()
    return right / A.size(0)


def make_splits(cfg, seed):
    """Same graphs for every width/LR at a given seed; val and test use their own seeds."""
    if cfg.get("gen") == "hard_diam":
        n = cfg["n"]
        return (relabelled_hard_diam(cfg["train"], n, seed),
                relabelled_hard_diam(cfg["val"], n, seed + 5555),
                relabelled_hard_diam(cfg["test"], n, seed + 9999))
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
            loss = F.binary_cross_entropy_with_logits(model(Atr_d[idx]), Rtr_d[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            if sched is not None:
                sched.step()
            tot += loss.item() * idx.size(0)
        if epoch % max(1, epochs // 15) == 0 or epoch in (1, epochs):
            history.append({"epoch": epoch, "loss": tot / N,
                            "train": evaluate(model, Atr, Rtr, bs),
                            "val": evaluate(model, Ava, Rva, bs),
                            "test": evaluate(model, Ate, Rte, bs),
                            "test_pair": pair_accuracy(model, Ate, Rte, bs)})

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
            print(f"  generating data for seed {s}, n={n}, {size} train graphs...")
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
    a = ap.parse_args()
    if a.analyze:
        analyze(f"{a.sweep}_smoke" if a.smoke else a.sweep)
    else:
        i, k = map(int, a.shard.split("/"))
        run_sweep(a.sweep, smoke=a.smoke, shard=(i, k))
