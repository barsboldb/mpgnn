"""Reproduction of Sanford et al. 2024a's connectivity experiment (NeurIPS 2024,
"Understanding Transformer Reasoning Capabilities via Graph Algorithms", §4 and App. E).

Their setting, as far as the paper states it:
  data   GraphQA "connectivity" = google-research/graphqa `Reachability`: Erdos-Renyi
         graphs, n ~ U{5..19} (size bucket, then node count), p ~ U(0, 1), one random
         node pair, "is there a path from s to t?". 1 000 train / 500 dev / 500 test,
         and a 100 000-graph train set from the same generator.
  tokens vertex tokens + edge tokens + a task token (their Fig. 1); answer read at
         the task token.
  model  decoder-only transformer, L = 12, m = 768, H = 12, GLU, dropout 0.1, AdamW,
         LR 5e-4, 1 000 000 steps (~60M parameters). Reported test accuracy: 92.9 (1K),
         98.0 (100K).
Not stated, so chosen here: batch 64, weight decay 0.01, 1 000 warm-up steps then a
constant rate, gradient clipping at 1, learned positions, a 2-way head on the task
token (equivalent to predicting the Yes/No answer token), and a shorter step budget.

The generator is re-implemented (same distribution, not the same graphs: their code
pulls in TensorFlow for I/O). `--audit` scores no-learning rules on the test set.

Every run appends one JSON line to results/sanford/<sweep>.jsonl; reruns skip runs
already on disk.

Usage:
    python sanford_repro.py a1k                  # part A, 1 000 training graphs
    python sanford_repro.py a100k --shard 0/2    # worker 0 of 2
    python sanford_repro.py bwidth --smoke       # tiny run to check the pipeline
    python sanford_repro.py a1k --audit          # shortcut rules on the test set
    python sanford_repro.py bwidth --analyze
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
OUT_DIR = os.path.join("results", "sanford")
CACHE_DIR = os.environ.get("WIDTH_CACHE", os.path.join("data", "width_cache"))
VERTEX, EDGE, QUERY = 0, 1, 2

BASE = dict(data="graphqa", depth=12, width=768, head_dim=64, dropout=0.1, lr=5e-4,
            weight_decay=0.01, batch=64, warmup=1000, val=500, test=2000, seeds=[0])
SWEEPS = {
    # Part A: their model and data at both training-set sizes. They fit the training
    # set long before 1M steps (App. E.4.1), so the budget is cut to what a T4 affords.
    # max_minutes stops a run early (evaluated and saved) inside Kaggle's 12 h session.
    "a1k": dict(BASE, train=1000, steps=20000, eval_every=1000, max_minutes=300),
    "a100k": dict(BASE, train=100000, steps=60000, eval_every=3000, max_minutes=600),
    # a100k (constant 5e-4 after warm-up) never fitted its training set (train ~0.92,
    # loss flat at ~0.19 from step 3k) and went NaN at ~31k under fp16. Rerun with the
    # rate decayed (cosine to 10 %), at their peak rate and at 1e-4.
    "a100k_cos": dict(BASE, train=100000, steps=60000, eval_every=3000, max_minutes=600,
                      schedule="cosine"),
    "a100k_lr1e4": dict(BASE, train=100000, steps=60000, eval_every=3000, max_minutes=600,
                        schedule="cosine", lr=1e-4),
    # Result: 1e-4 reproduces their 98.0 (0.982); 5e-4 + cosine still stuck at 0.93.
    # Part B1: their claim is about width, their experiment fixes m = 768. Same model
    # and 100K data at the rate that worked, widths 8 ... 768 (heads = m / 64, at
    # least 1). Narrow models may want a higher rate (Q1), so m <= 128 also get 1e-3.
    "bwidth": dict(BASE, train=100000, steps=30000, eval_every=3000, max_minutes=240,
                   schedule="cosine", lr=1e-4,
                   lrs={1e-4: [8, 16, 32, 64, 128, 256, 768], 1e-3: [8, 16, 32, 64, 128]}),
}


# ── data ──────────────────────────────────────────────────────────────────────

def graphqa_reachability(num, seed):
    """GraphQA `Reachability` examples: (n, edges [E, 2], s, t, label)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(num):
        lo = 5 + 5 * int(rng.integers(3))            # size bucket: 5-9, 10-14, 15-19
        n = int(rng.integers(lo, lo + 5))
        p = rng.uniform(0, 1)
        iu = np.triu_indices(n, 1)
        keep = rng.random(len(iu[0])) < p
        edges = np.stack([iu[0][keep], iu[1][keep]], 1)
        s, t = (int(x) for x in rng.choice(n, size=2, replace=False))
        A = csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
        _, comp = connected_components(A, directed=False)
        out.append((n, edges, s, t, int(comp[s] == comp[t])))
    return out


def tokenize(examples, max_len):
    """Right-padded token tensors: type, endpoint a, endpoint b [num, max_len]; the
    task token sits at index length - 1. Vertex v -> (v, v); edge uv -> (u, v)."""
    num = len(examples)
    typ = torch.zeros(num, max_len, dtype=torch.uint8)
    ab = torch.zeros(num, max_len, 2, dtype=torch.uint8)
    length = torch.zeros(num, dtype=torch.int16)
    y = torch.zeros(num, dtype=torch.uint8)
    for k, (n, edges, s, t, label) in enumerate(examples):
        L = n + len(edges) + 1
        assert L <= max_len, (L, max_len)
        ab[k, :n, 0] = ab[k, :n, 1] = torch.arange(n)
        typ[k, n:n + len(edges)] = EDGE
        ab[k, n:n + len(edges)] = torch.as_tensor(edges, dtype=torch.uint8).reshape(-1, 2)
        typ[k, L - 1] = QUERY
        ab[k, L - 1] = torch.tensor([s, t])
        length[k], y[k] = L, label
    return typ, ab, length, y


MAX_LEN = {"graphqa": 192}      # 19 vertices + C(19, 2) = 171 edges + task token
MAX_NODES = {"graphqa": 20}


def make_split(cfg, num, seed):
    key = f"sanford_{cfg['data']}_N{num}_s{seed}"
    path = os.path.join(CACHE_DIR, key + ".pt")
    if os.path.exists(path):
        return torch.load(path)
    data = tokenize(graphqa_reachability(num, seed), MAX_LEN[cfg["data"]])
    os.makedirs(CACHE_DIR, exist_ok=True)
    torch.save(data, path + ".tmp")
    os.replace(path + ".tmp", path)
    return data


def make_splits(cfg, seed):
    """Train depends on the run seed; dev and test are fixed across runs."""
    return (make_split(cfg, cfg["train"], seed), make_split(cfg, cfg["val"], 5555),
            make_split(cfg, cfg["test"], 9999))


# ── model ─────────────────────────────────────────────────────────────────────

class Block(nn.Module):
    """Pre-norm decoder block: causal self-attention + GLU feed-forward."""

    def __init__(self, d, heads, dropout):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.up, self.down = nn.Linear(d, 4 * d), nn.Linear(2 * d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, L, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(B, L, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        x = x + self.drop(self.out(a.transpose(1, 2).reshape(B, L, d)))
        return x + self.drop(self.down(F.glu(self.up(self.norm2(x)), dim=-1)))


class GraphDecoder(nn.Module):
    """Sanford et al.'s encoding: token = type + id(a) + id(b) + position."""

    def __init__(self, max_nodes, max_len, width, depth, heads, dropout):
        super().__init__()
        self.typ = nn.Embedding(3, width)
        self.node = nn.Embedding(max_nodes, width)
        self.pos = nn.Embedding(max_len, width)
        self.blocks = nn.ModuleList([Block(width, heads, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 2)

    def forward(self, typ, ab, length):
        L = typ.size(1)
        x = (self.typ(typ) + self.node(ab[..., 0]) + self.node(ab[..., 1])
             + self.pos(torch.arange(L, device=typ.device)))
        for blk in self.blocks:
            x = blk(x)
        last = x[torch.arange(x.size(0), device=x.device), length - 1]
        return self.head(self.norm(last))

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ── training ──────────────────────────────────────────────────────────────────

def epoch_batches(length, bs, pool=50):
    """Shuffled batches of similar length: sort each random pool of bs * pool examples
    by length, cut it into batches, shuffle the batch order. Padding drops ~3x on
    GraphQA (lengths 6-191, mean ~50)."""
    perm = torch.randperm(length.size(0))
    batches = []
    for i in range(0, len(perm), bs * pool):
        chunk = perm[i:i + bs * pool]
        chunk = chunk[torch.argsort(length[chunk])]
        batches += [chunk[j:j + bs] for j in range(0, len(chunk), bs) if len(chunk[j:j + bs]) == bs]
    return [batches[k] for k in torch.randperm(len(batches))]


def batch_to(data, idx):
    """Gather a batch and crop the padding to its longest example."""
    typ, ab, length, y = (t[idx] for t in data)
    L = int(length.max())
    return (typ[:, :L].to(DEVICE).long(), ab[:, :L].to(DEVICE).long(),
            length.to(DEVICE).long(), y.to(DEVICE).long())


@torch.no_grad()
def correct(model, data, bs=256):
    """Per-example right/wrong, as a bool tensor."""
    model.eval()
    out = []
    for i in range(0, data[0].size(0), bs):
        typ, ab, length, y = batch_to(data, torch.arange(i, min(i + bs, data[0].size(0))))
        with autocast():
            out.append((model(typ, ab, length).argmax(-1) == y).cpu())
    return torch.cat(out)


def accuracy(model, data):
    return correct(model, data).float().mean().item()


def distance_buckets(data):
    """s-t distance of each example, bucketed: "1", "2", "3", "4+", "no path, isolated"
    (s or t has no edge) and "no path, both have edges". The 3-hop rule is wrong on
    exactly "4+", the degree rule on exactly "no path, both have edges"; a model right
    on both does more than either shortcut."""
    typ, ab, length, _ = data
    out = []
    for k in range(typ.size(0)):
        L = int(length[k])
        n = int((typ[k, :L] == VERTEX).sum())
        e = ab[k, :L][typ[k, :L] == EDGE].long().numpy()
        s, t = (int(x) for x in ab[k, L - 1])
        A = csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
        d = shortest_path(A, directed=False, unweighted=True, indices=s)[t]
        if np.isfinite(d):
            out.append(str(int(d)) if d <= 3 else "4+")
        else:
            deg = np.bincount(e.ravel(), minlength=n)
            out.append("no path, both have edges" if deg[s] and deg[t] else "no path, isolated")
    return np.array(out)


def by_bucket(ok, buckets):
    return {b: {"acc": round(float(ok[buckets == b].float().mean()), 4),
                "count": int((buckets == b).sum())}
            for b in BUCKETS if (buckets == b).any()}


BUCKETS = ("1", "2", "3", "4+", "no path, isolated", "no path, both have edges")
HARD = ("4+", "no path, both have edges")


def make_hard_split(cfg, per_bucket=1000, seed=7777):
    """Only the examples both shortcuts can't settle: per_bucket of each HARD bucket,
    drawn from the same generator (they are ~2-3 % of it)."""
    key = f"sanford_{cfg['data']}_hard{per_bucket}_s{seed}"
    path = os.path.join(CACHE_DIR, key + ".pt")
    if os.path.exists(path):
        return torch.load(path)
    keep, have, sd = [], {b: 0 for b in HARD}, seed
    while min(have.values()) < per_bucket:
        ex = graphqa_reachability(20000, sd)
        sd += 1
        for e, b in zip(ex, distance_buckets(tokenize(ex, MAX_LEN[cfg["data"]]))):
            if b in have and have[b] < per_bucket:
                keep.append(e); have[b] += 1
    data = tokenize(keep, MAX_LEN[cfg["data"]])
    os.makedirs(CACHE_DIR, exist_ok=True)
    torch.save(data, path + ".tmp")
    os.replace(path + ".tmp", path)
    return data


def autocast():
    return torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda")


def train_one(cfg, width, lr, seed, splits):
    tr, va, te = splits
    torch.manual_seed(seed)
    heads = max(1, width // cfg["head_dim"])
    model = GraphDecoder(MAX_NODES[cfg["data"]], MAX_LEN[cfg["data"]], width, cfg["depth"],
                         heads, cfg["dropout"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["weight_decay"])
    warm, total = cfg["warmup"], cfg["steps"]

    def rate(t):    # linear warm-up, then constant or cosine down to 10 %
        if t < warm:
            return (t + 1) / warm
        if cfg.get("schedule") == "cosine":
            return 0.1 + 0.45 * (1 + math.cos(math.pi * (t - warm) / max(1, total - warm)))
        return 1.0
    sched = torch.optim.lr_scheduler.LambdaLR(opt, rate)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    N, bs = tr[0].size(0), cfg["batch"]
    sub = tuple(t[:2000] for t in tr)
    history, t0, batches, tot, since, bad = [], time.time(), [], 0.0, 0, 0
    for step in range(1, cfg["steps"] + 1):
        model.train()
        if not batches:
            batches = epoch_batches(tr[2], min(bs, N))
        typ, ab, length, y = batch_to(tr, batches.pop())
        with autocast():
            loss = F.cross_entropy(model(typ, ab, length), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        if math.isfinite(loss.item()):
            tot += loss.item(); since += 1
        else:
            bad += 1          # fp16 overflow: GradScaler skips the step
        out_of_time = time.time() - t0 > 60 * cfg.get("max_minutes", float("inf"))
        if step % cfg["eval_every"] == 0 or step == cfg["steps"] or out_of_time:
            h = {"step": step, "loss": tot / max(since, 1), "nonfinite": bad,
                 "train": accuracy(model, sub),
                 "val": accuracy(model, va), "test": accuracy(model, te)}
            history.append(h)
            diverged = bad > since         # mostly NaN/inf since the last evaluation
            tot, since, bad = 0.0, 0, 0
            print(f"    step {step:>7d}  loss {h['loss']:.4f}  train {h['train']:.3f}  "
                  f"val {h['val']:.3f}  test {h['test']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
            if out_of_time or diverged:
                print(f"    {'out of time' if out_of_time else 'diverged'} at step {step}", flush=True)
                break
    best = max(history, key=lambda h: h["val"])
    test_by_dist = by_bucket(correct(model, te), distance_buckets(te))
    hard = make_hard_split(cfg)
    hard_by_dist = by_bucket(correct(model, hard), distance_buckets(hard))
    for label, d in (("test", test_by_dist), ("hard set", hard_by_dist)):
        print(f"    final {label} by s-t distance: " + "  ".join(
            f"{b} {v['acc']:.3f} (n={v['count']})" for b, v in d.items()), flush=True)
    return {"width": width, "lr": lr, "depth": cfg["depth"], "seed": seed, "heads": heads,
            "train_size": N, "params": model.num_parameters(), "steps": history[-1]["step"],
            "seconds": round(time.time() - t0, 1), "diverged": diverged,
            "final": history[-1], "best_val": best, "final_test_by_dist": test_by_dist,
            "final_hard_by_dist": hard_by_dist,
            "history": history}


def load_runs(sweep):
    runs = []
    for path in sorted(glob.glob(os.path.join(OUT_DIR, f"{sweep}.jsonl")) +
                       glob.glob(os.path.join(OUT_DIR, f"{sweep}.shard*.jsonl"))):
        with open(path) as f:
            runs += [json.loads(line) for line in f if line.strip()]
    return runs


def run_sweep(name, smoke=False, shard=(0, 1)):
    cfg = dict(SWEEPS[name])
    if smoke:
        cfg.update(train=512, val=128, test=128, steps=60, eval_every=30, warmup=10,
                   widths=[16, 64], depth=2)
        name = f"{name}_smoke"
    os.makedirs(OUT_DIR, exist_ok=True)
    suffix = f".shard{shard[0]}of{shard[1]}" if shard[1] > 1 else ""
    path = os.path.join(OUT_DIR, f"{name}{suffix}.jsonl")
    done = {(r["width"], r.get("lr", r["config"]["lr"]), r["seed"]) for r in load_runs(name)}
    lrs = cfg.get("lrs", {cfg["lr"]: cfg.get("widths", [cfg["width"]])})
    if smoke:
        lrs = {lr: cfg["widths"] for lr in list(lrs)[:1]}
    todo = [(w, lr, s) for s in cfg["seeds"] for lr, ws in lrs.items() for w in ws
            if (w, lr, s) not in done]
    todo = todo[shard[0]::shard[1]]
    print(f"Device {DEVICE} | sweep {name} shard {shard[0]}/{shard[1]}: "
          f"{len(todo)} runs to go ({len(done)} on disk)", flush=True)
    for k, (w, lr, s) in enumerate(todo, 1):
        r = train_one(cfg, w, lr, s, make_splits(cfg, s))
        r["sweep"], r["config"] = name, {k2: v for k2, v in cfg.items()
                                         if k2 not in ("widths", "seeds", "lrs")}
        with open(path, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(f"  [{k}/{len(todo)}] m={w:<4d} lr={lr:g} L={cfg['depth']} N={cfg['train']} seed={s}  "
              f"params={r['params']/1e6:.1f}M  train={r['final']['train']:.3f}  "
              f"test@bestval={r['best_val']['test']:.3f}  test@final={r['final']['test']:.3f}  "
              f"({r['seconds']}s)", flush=True)


def analyze(name):
    runs = sorted(load_runs(name), key=lambda r: (r["width"], r.get("lr", r["config"]["lr"]), r["seed"]))
    print(f"\n{name}: test accuracy (dev-selected step / final step; hard set by bucket)")
    for r in runs:
        dist = "  ".join(f"{b}:{v['acc']:.3f}" for b, v in r.get("final_hard_by_dist", {}).items())
        print(f"  m={r['width']:<4d} lr={r.get('lr', r['config']['lr']):<6g} L={r['depth']:<2d} "
              f"N={r['train_size']:<6d} seed={r['seed']}  params {r['params']/1e6:6.2f}M  "
              f"train {r['final']['train']:.3f}  test {r['best_val']['test']:.3f} / "
              f"{r['final']['test']:.3f}  {dist}")


def _features(examples):
    """Per-example features of growing reach, the s-t distance and the label."""
    rows, d, y = [], [], []
    for n, edges, s, t, label in examples:
        A = csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
        A = ((A + A.T) > 0).astype(np.int32)
        deg = np.asarray(A.sum(1)).ravel()
        n1 = [set(A[v].indices) for v in (s, t)]
        n2 = [set().union(*(A[u].indices for u in nb)) | nb if nb else set() for nb in n1]
        m = len(edges)
        rows.append([n, m, m / (n * (n - 1) / 2),                       # graph size
                     deg[s], deg[t], min(deg[s], deg[t]),               # 1 hop
                     int(t in n1[0]), len(n1[0] & n1[1]),               # 2 hops
                     len(n2[0]), len(n2[1]), int(t in n2[0])])
        d.append(shortest_path(A, directed=False, unweighted=True, indices=s)[t])
        y.append(label)
    return np.array(rows, dtype=float), np.array(d), np.array(y)


FEATURE_SETS = {"graph size (n, edges, density)": [0, 1, 2],
                "+ endpoint degrees": [0, 1, 2, 3, 4, 5],
                "+ 2-hop neighbourhoods": list(range(11))}


def audit(name, train=50000):
    """Shortcut audit of the GraphQA reachability data: fixed rules, then gradient-
    boosted classifiers on features of growing reach (fit on a train-distribution
    sample, scored on the test distribution and on the hard set)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    cfg = SWEEPS[name]
    Xtr, _, ytr = _features(graphqa_reachability(train, 0))
    test = graphqa_reachability(20000, 9999)
    Xte, dte, yte = _features(test)
    hard = make_hard_split(cfg)
    hard_ex = []
    for k in range(hard[0].size(0)):          # decode the tokenized hard set
        typ, ab, L = hard[0][k], hard[1][k], int(hard[2][k])
        n = int((typ[:L] == VERTEX).sum())
        e = ab[:L][typ[:L] == EDGE].long().numpy().reshape(-1, 2)
        s, t = (int(x) for x in ab[L - 1])
        hard_ex.append((n, e, s, t, int(hard[3][k])))
    Xh, dh, yh = _features(hard_ex)
    far, cut = np.isfinite(dh), ~np.isfinite(dh)   # 4+ hops / no path, both have edges
    print(f"test distribution: {len(test)} examples, mean nodes {Xte[:, 0].mean():.2f}, "
          f"mean edges {Xte[:, 1].mean():.1f}, yes {yte.mean():.3f}")
    print(f"hard set: {far.sum()} connected pairs >= 4 hops, {cut.sum()} no-path pairs "
          f"with both endpoints' degree > 0\n")
    print(f"  {'':46s} {'test':>6s} {'hard: 4+':>9s} {'hard: no path':>14s}")

    def row(label, pt, ph):
        print(f"  {label:46s} {(pt == yte).mean():6.3f} {(ph[far] == yh[far]).mean():9.3f} "
              f"{(ph[cut] == yh[cut]).mean():14.3f}")
    row("always yes", np.ones_like(yte), np.ones_like(yh))
    row("yes iff both degrees > 0", Xte[:, 5] > 0, Xh[:, 5] > 0)
    for k in (1, 2, 3):
        row(f"yes iff within {k} hops", dte <= k, dh <= k)
    for label, cols in FEATURE_SETS.items():
        clf = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(Xtr[:, cols], ytr)
        row(f"boosted trees: {label}", clf.predict(Xte[:, cols]), clf.predict(Xh[:, cols]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep", choices=sorted(SWEEPS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--prepare", action="store_true", help="build the sweep's datasets first")
    ap.add_argument("--shard", default="0/1", help="i/k: run every k-th pending run from i")
    a = ap.parse_args()
    if a.analyze:
        analyze(f"{a.sweep}_smoke" if a.smoke else a.sweep)
    elif a.audit:
        audit(a.sweep)
    elif a.prepare:
        for sd in SWEEPS[a.sweep]["seeds"]:
            make_splits(SWEEPS[a.sweep], sd)
        make_hard_split(SWEEPS[a.sweep])
    else:
        i, k = map(int, a.shard.split("/"))
        run_sweep(a.sweep, smoke=a.smoke, shard=(i, k))
