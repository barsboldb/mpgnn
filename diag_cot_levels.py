"""Per-position teacher-forced accuracy for an AR-CoT checkpoint.

The trace-level microscope: buckets every supervised completion position by what
it is — node tokens by BFS level, bfs_expand parents (pure copy of the previous
level = induction) split from children (edge lookup) — and reports teacher-forced
accuracy per bucket. This is what separates "the format is learned" (SEP/ANS/EOS
near 1.0) from "the computation is learned" (level/child buckets), and localizes
*which* circuit is missing: parents low = induction copying absent; parents high
with children low = content lookup is the wall.

Usage:
    python diag_cot_levels.py checkpoints/<run>.pt [--dataset NAME] [--limit N]

Defaults to the checkpoint's own training dataset and dataset_kwargs.
"""
import argparse
from collections import defaultdict

import torch

from main import load_checkpoint, _prepare_cot_sequences, split_head_tail, DEVICE
from src.cot_tokens import make_cot_loader


def level_table(model, loader, vocab) -> dict[str, tuple[int, int]]:
    """{bucket: (hits, total)} of teacher-forced next-token accuracy."""
    hit: dict[str, int] = defaultdict(int)
    tot: dict[str, int] = defaultdict(int)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            tokens = batch["tokens"].to(DEVICE)
            pl = batch["prompt_len"]
            pred = model(tokens[:, :-1]).argmax(dim=-1).cpu()
            for i in range(tokens.size(0)):
                row = tokens[i].cpu().tolist()
                level = 0
                for t in range(int(pl[i]), len(row)):
                    tok = row[t]
                    if tok == vocab.PAD:
                        break
                    if tok == vocab.SEP:
                        key = "SEP"; level += 1
                    elif tok == vocab.EXP:
                        key = "EXP"
                    elif tok == vocab.ANS:
                        key = "ANS"
                    elif tok in (vocab.YES, vocab.NO):
                        key = "YES/NO"
                    elif tok == vocab.EOS:
                        key = "EOS"
                    elif row[t - 1] == vocab.EXP:
                        key = "parent(copy)"
                    else:
                        key = f"level {level}" if level <= 6 else "level 7+"
                    hit[key] += int(pred[i, t - 1] == tok)
                    tot[key] += 1
    return {k: (hit[k], tot[k]) for k in tot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--dataset", default=None,
                    help="defaults to the checkpoint's training dataset")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    model, config, ckpt = load_checkpoint(args.checkpoint)
    assert config.cot_mode == "autoregressive", "diag_cot_levels reads AR-CoT checkpoints"
    dataset = args.dataset or ckpt.get("train_dataset") or config.dataset
    seqs = _prepare_cot_sequences(config, dataset, model.vocab, limit=args.limit,
                                  gen_kwargs=None if dataset == config.dataset else {})
    _, test_seq = split_head_tail(seqs, config.train_frac)
    loader = make_cot_loader(test_seq, model.vocab, config.batch_size, shuffle=False)

    table = level_table(model, loader, model.vocab)
    order = [f"level {k}" for k in range(7)] + ["level 7+", "parent(copy)", "EXP",
                                                "SEP", "ANS", "YES/NO", "EOS"]
    print(f"\n{dataset} test split ({len(test_seq)} seqs), teacher-forced accuracy by position class:")
    print(f"{'position class':>14}  {'tf acc':>7}  {'n':>6}")
    for k in order:
        if k in table:
            h, n = table[k]
            print(f"{k:>14}  {h / n:7.3f}  {n:6d}")


if __name__ == "__main__":
    main()
