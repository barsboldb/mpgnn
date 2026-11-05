"""Checkpoint surgery on stalled edge-bias models (2026-11-05).

eb_adj_big runs at n >= 96 either leave a ~0.9 pair-accuracy plateau or stay on it. Resume a
stalled checkpoint with one change each and see which one unsticks it (research report D8):

  ctrl   resume as is, peak LR 1e-4                     (does more training alone help?)
  bias   b += 1.5 in every layer and head, peak 1e-4    (H1: diluted edge bias)
  lr     resume as is, peak LR 1e-3                     (H3: cosine schedule ran out)
  deep   + 6 identity-initialised blocks (depth 12), peak 1e-4   (H5: too few hops)
  loop   run the 6 trained blocks twice (weights tied), peak 1e-4 (H5 without parameters)

Each: fresh Adam, `steps` steps (default 6000, a quarter of the original budget), 5% warmup then
cosine to 0, on the checkpoint's own training graphs. Results append to
results/width/eb_surgery[.shard<i>of<k>].jsonl (history every 500 steps, test pair at the end).

Usage:
    python eb_surgery.py --models n128_L6_m32_lr0.001_s1 n96_L6_m16_lr0.001_s0 --treat ctrl bias lr deep loop
    python eb_surgery.py ... --shard 0/2          # every 2nd (model, treatment) pair from 0
"""
import argparse
import glob
import json
import math
import os
import re
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import width_sweep as ws
from src.transformer import _EncoderBlock

PEAK = {"ctrl": 1e-4, "bias": 1e-4, "lr": 1e-3, "deep": 1e-4, "loop": 1e-4}


def identity_block(width, heads):
    blk = _EncoderBlock(width, heads, 0.0)
    for p in (blk.attn.out_proj.weight, blk.attn.out_proj.bias, blk.ffn[2].weight, blk.ffn[2].bias):
        nn.init.zeros_(p)   # both residual branches write 0: the block starts as the identity
    return blk


def apply(model, treat, width, heads):
    if treat == "bias":
        model.edge_bias.data += 1.5
    elif treat == "deep":
        for _ in range(6):
            model.inner.blocks.append(identity_block(width, heads))
        # new layers start where trained models put their early-layer b (~3.5, eb_diagnose.py)
        model.edge_bias = nn.Parameter(torch.cat([model.edge_bias.data, torch.full((6, heads), 3.5)]))
    elif treat == "loop":
        model.loops = 2
    return model


def run(name, treat, steps, out, train=128000, sweep="eb_adj_big"):
    n, L, mw, s = map(int, re.match(r"n(\d+)_L(\d+)_m(\d+)_lr[\d.e-]+_s(\d+)", name).groups())
    cfg = dict(ws.SWEEPS[sweep], n=n, depth=L, train=train)
    (Atr, Rtr), _, (Ate, Rte) = ws.make_splits(cfg, s)
    heads = max(1, mw // cfg["head_dim"])
    ws.set_seed(1000 + s)
    model = ws.build_model(cfg, mw, heads, s)
    model.load_state_dict(torch.load(os.path.join(ws.OUT_DIR, "models", sweep, name + ".pt"), map_location="cpu"))
    model = apply(model, treat, mw, heads).to(ws.DEVICE)
    start = ws.metrics(model, Ate, Rte)[1]
    opt = torch.optim.Adam(model.parameters(), lr=PEAK[treat])
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: (t + 1) / warm if t < warm
        else 0.5 * (1 + math.cos(math.pi * (t - warm) / max(1, steps - warm))))
    Atr_d, Rtr_d = Atr.to(ws.DEVICE), Rtr.to(ws.DEVICE)
    hist, tot, t0 = [], 0.0, time.time()
    for t in range(1, steps + 1):
        idx = torch.randint(0, Atr_d.size(0), (cfg["batch"],), device=ws.DEVICE)
        model.train()
        loss = F.binary_cross_entropy_with_logits(model(Atr_d[idx].float()), Rtr_d[idx].float())
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        tot += loss.item()
        if t % 500 == 0:
            hist.append({"step": t, "loss": tot / 500, "test_pair": ws.metrics(model, Ate, Rte)[1]})
            tot = 0.0
    final = ws.metrics(model, Ate, Rte)
    r = {"model": name, "sweep": sweep, "n": n, "width": mw, "seed": s, "treat": treat, "steps": steps,
         "peak_lr": PEAK[treat], "start_pair": start, "test": final[0], "test_pair": final[1],
         "b_mean": model.edge_bias.detach().mean(1).tolist(), "history": hist,
         "seconds": round(time.time() - t0, 1)}
    with open(out, "a") as f:
        f.write(json.dumps(r) + "\n")
    print(f"{name} {treat:5s} start={start:.3f} -> test_pair={final[1]:.3f} exact={final[0]:.3f} "
          f"({r['seconds']}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--treat", nargs="+", default=list(PEAK))
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--train", type=int, default=128000, help="smoke tests: a few hundred")
    a = ap.parse_args()
    i, k = map(int, a.shard.split("/"))
    # one file per shard so parallel workers never interleave lines
    out = os.path.join(ws.OUT_DIR, f"eb_surgery.shard{i}of{k}.jsonl" if k > 1 else "eb_surgery.jsonl")
    done = {(r["model"], r["treat"]) for p in glob.glob(os.path.join(ws.OUT_DIR, "eb_surgery*.jsonl"))
            for r in map(json.loads, open(p))}
    todo = [(mn, t) for mn in a.models for t in a.treat if (mn, t) not in done][i::k]
    for mn, t in todo:
        run(mn, t, a.steps, out, a.train)


if __name__ == "__main__":
    main()
