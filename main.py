"""
mpGNN experiment runner.

Usage:
    python main.py --config configs/connectivity_gat.yaml            # dataset from YAML
    python main.py --config configs/mpgnn.yaml --dataset isomorphism  # CLI dataset override
    python main.py --config c.yaml -o epochs=50 -o lr=0.003          # quick config tweaks
    python main.py -o dataset_kwargs.num_graphs=500 --config c.yaml  # generator params
    python main.py --layer gat --task graph --dataset mutag
    python main.py --results          # print comparison table of all saved runs
"""
import argparse
import os
import random
import signal
from dataclasses import asdict

import numpy as np
import torch
import yaml
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.loader import DataLoader

from src.config import GNNConfig
from src.model import build_model
from src.train import (run_node_experiment, run_graph_experiment,
                       run_connectivity_experiment, run_cot_experiment, eval_cot)
from src.dataset import GENERATORS, load_or_create, attach_components
from src.cot_tokens import CoTVocab, build_cot_sequences, make_cot_loader
from src.logger import RunLogger, print_results_table


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")


def set_seed(seed: int):
    """Seed python/numpy/torch so weight init and the per-load 'random' node
    features are reproducible. Generators seed their own rng from dataset_kwargs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_default_config(layer: str, task: str) -> GNNConfig:
    if task == "node":
        return GNNConfig(
            in_channels=1433,
            out_channels=7,
            hidden_channels=64,
            task="node",
            dropout=0.5,
            layers=[{"type": layer}, {"type": layer}],
        )
    else:
        return GNNConfig(
            in_channels=7,
            out_channels=2,
            hidden_channels=64,
            task="graph",
            pooling="mean",
            dropout=0.5,
            norm_type="batch",
            layers=[{"type": layer}, {"type": layer}, {"type": layer}],
        )


def apply_overrides(config_dict: dict, overrides: list[str]) -> dict:
    """Apply `-o key=value` pairs onto the raw config dict before validation.

    Values are parsed as YAML (so `lr=0.003`, `residual=true`, `layers=[{type: gat}]`
    all coerce correctly). One level of dotting reaches into dict fields:
    `dataset_kwargs.num_graphs=500`.
    """
    for item in overrides:
        key, _, raw = item.partition("=")
        if not _:
            raise SystemExit(f"--override needs key=value, got '{item}'")
        value = yaml.safe_load(raw)
        if "." in key:
            outer, inner = key.split(".", 1)
            config_dict.setdefault(outer, {})[inner] = value
        else:
            config_dict[key] = value
        print(f"[override] {key} = {value}")
    return config_dict


def load_config(args) -> GNNConfig:
    if args.config:
        d = GNNConfig.yaml_dict(args.config)  # resolves `extends:` chains
    else:
        d = asdict(make_default_config(args.layer, args.task))
    d = apply_overrides(d, args.override)
    return GNNConfig(**d)


# ── Data preparation (shared by graph / connectivity / eval / inspect) ────────

def prepare_generated(config: GNNConfig, dataset_name: str, limit: int = 0):
    """Load-or-generate a synthetic dataset per the config and cap it to `limit`.
    The generator alternates labels (i % 2), so a head slice stays class-balanced
    and the on-disk cache is untouched."""
    data_list = load_or_create(dataset_name,
                               node_features=config.node_features,
                               lpe_dim=config.lpe_dim,
                               in_channels=config.in_channels,
                               **config.dataset_kwargs)
    if 0 < limit < len(data_list):
        data_list = data_list[:limit]
        print(f"[--limit {limit}] using {len(data_list)} of the cached graphs")
    return data_list


def split_head_tail(data_list, train_frac: float):
    n = int(train_frac * len(data_list))
    return data_list[:n], data_list[n:]


def make_loaders(train_ds, test_ds, batch_size: int):
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(test_ds, batch_size=batch_size))


def build_and_describe(config: GNNConfig) -> torch.nn.Module:
    model = build_model(config).to(DEVICE)
    print(f"Model params: {model.num_parameters():,}\n")
    print(config.describe())
    print()
    return model


def load_checkpoint(ckpt_path: str):
    """Rebuild (model, config, ckpt) from a saved checkpoint. `state_dict` holds
    the BEST-epoch weights (new checkpoints; old ones stored final weights)."""
    ckpt = torch.load(ckpt_path, weights_only=False)
    config = GNNConfig.from_dict(ckpt["config"])
    model = build_model(config).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    if "best_epoch" in ckpt:
        print(f"(checkpoint weights = best epoch {ckpt['best_epoch']})")
    return model, config, ckpt


def save_checkpoint(model, config: GNNConfig, run_id: str, train_dataset: str,
                    final_state: dict | None = None, best_epoch: int | None = None) -> str:
    """Persist weights + config so the model can be reloaded and evaluated on
    other (e.g. uncontrolled / OOD) datasets later. The model is expected to
    hold the best-epoch weights (the primary `state_dict`); the last-epoch
    weights ride along as `final_state_dict`."""
    os.makedirs("checkpoints", exist_ok=True)
    path = os.path.join("checkpoints", f"{run_id}.pt")
    payload = {"config": asdict(config), "state_dict": model.state_dict(),
               "train_dataset": train_dataset}
    if final_state is not None:
        payload["final_state_dict"] = final_state
    if best_epoch is not None:
        payload["best_epoch"] = best_epoch
    torch.save(payload, path)
    print(f"Saved model -> {path}")
    return path


# ── Experiments ────────────────────────────────────────────────────────────────

def node_experiment(config: GNNConfig, dataset_name: str):
    dataset = Planetoid(root="/tmp/Cora", name="Cora")
    data = dataset[0]
    print(f"\nDataset: Cora  |  Nodes: {data.num_nodes}  Edges: {data.num_edges}  Features: {data.num_node_features}")
    model = build_and_describe(config)

    logger = RunLogger(dataset_name, config, params=model.num_parameters())
    run_node_experiment(model, data, DEVICE,
                        epochs=config.epochs, lr=config.lr, weight_decay=config.weight_decay,
                        logger=logger)
    logger.save()


def graph_experiment(config: GNNConfig, dataset_name: str, overfit: int = 0, limit: int = 0):
    if dataset_name in GENERATORS:
        data_list = prepare_generated(config, dataset_name, limit=limit)
        if overfit > 0:
            # balanced: take equal numbers from each class
            class0 = [d for d in data_list if d.y.item() == 0][:overfit // 2]
            class1 = [d for d in data_list if d.y.item() == 1][:overfit // 2]
            data_list = class0 + class1
            train_ds = test_ds = data_list
            print(f"\n[OVERFIT MODE] Dataset: {dataset_name}  |  Graphs: {len(data_list)}  (all used for train+test)")
        else:
            train_ds, test_ds = split_head_tail(data_list, config.train_frac)
            print(f"\nDataset: {dataset_name}  |  Graphs: {len(data_list)}  Train: {len(train_ds)}  Test: {len(test_ds)}")
    else:
        dataset = TUDataset(root=f"/tmp/{dataset_name.upper()}", name=dataset_name.upper())
        dataset = dataset.shuffle()
        train_ds, test_ds = split_head_tail(dataset, config.train_frac)
        print(f"\nDataset: {dataset_name.upper()}  |  Graphs: {len(dataset)}  Classes: {dataset.num_classes}")

    train_loader, test_loader = make_loaders(train_ds, test_ds, config.batch_size)
    model = build_and_describe(config)

    tag = f"overfit{overfit}_" if overfit > 0 else ""
    logger = RunLogger(dataset_name, config, tag=tag, params=model.num_parameters())
    info = run_graph_experiment(model, train_loader, test_loader, DEVICE,
                                epochs=config.epochs, lr=config.lr, weight_decay=config.weight_decay,
                                logger=logger)
    logger.save()
    save_checkpoint(model, config, logger.run_id, dataset_name,
                    final_state=info["final_state"], best_epoch=info["best_epoch"])


def connectivity_experiment(config: GNNConfig, dataset_name: str, limit: int = 0):
    data_list = prepare_generated(config, dataset_name, limit=limit)
    attach_components(data_list)
    train_ds, test_ds = split_head_tail(data_list, config.train_frac)
    print(f"\nDataset: {dataset_name}  |  Graphs: {len(data_list)}  Train: {len(train_ds)}  "
          f"Test: {len(test_ds)}  (connectivity-matrix task)")

    train_loader, test_loader = make_loaders(train_ds, test_ds, config.batch_size)
    model = build_and_describe(config)

    logger = RunLogger(dataset_name, config, tag="conn_", params=model.num_parameters())
    info = run_connectivity_experiment(model, train_loader, test_loader, DEVICE,
                                       epochs=config.epochs, lr=config.lr, weight_decay=config.weight_decay,
                                       logger=logger)
    ood_probe(model, config, logger, trained_on=dataset_name)
    logger.save()
    save_checkpoint(model, config, logger.run_id, dataset_name,
                    final_state=info["final_state"], best_epoch=info["best_epoch"])


def _prepare_cot_sequences(config: GNNConfig, dataset_name: str, vocab: CoTVocab,
                           limit: int = 0, seed: int | None = None,
                           gen_kwargs: dict | None = None,
                           drop_overlong: bool = False):
    """Raw graphs -> attach_components (diam) -> token sequences.
    gen_kwargs=None uses config.dataset_kwargs; pass {} for generator defaults
    (the OOD probe — its generator doesn't take the training set's kwargs).
    drop_overlong=True for eval on datasets denser than the training one
    (a trained position table can't grow; skip what doesn't fit, loudly)."""
    data_list = load_or_create(dataset_name, node_features="constant",
                               **(config.dataset_kwargs if gen_kwargs is None else gen_kwargs))
    if 0 < limit < len(data_list):
        data_list = data_list[:limit]
        print(f"[--limit {limit}] using {len(data_list)} of the cached graphs")
    attach_components(data_list)
    return build_cot_sequences(
        data_list, vocab,
        permute=config.permute_node_ids,
        trace=config.max_trace_len > 0,
        seed=config.seed if seed is None else seed,
        max_seq_len=config.max_seq_len,
        trace_format=config.trace_format,
        roster=config.prompt_roster,
        drop_overlong=drop_overlong)


def cot_experiment(config: GNNConfig, dataset_name: str, overfit: int = 0, limit: int = 0):
    """Autoregressive-CoT run: graphs are serialized to token sequences (with the
    supervised BFS trace unless max_trace_len == 0) and a decoder-only LM is
    trained with teacher forcing; eval decodes greedily."""
    vocab = CoTVocab(config.max_nodes)
    sequences = _prepare_cot_sequences(config, dataset_name, vocab, limit=limit)

    if overfit > 0:
        class0 = [s for s in sequences if s["y"] == 0][:overfit // 2]
        class1 = [s for s in sequences if s["y"] == 1][:overfit // 2]
        train_seq = test_seq = class0 + class1
        print(f"\n[OVERFIT MODE] Dataset: {dataset_name}  |  Sequences: {len(train_seq)}  "
              f"(all used for train+test, AR-CoT task)")
    else:
        train_seq, test_seq = split_head_tail(sequences, config.train_frac)
        print(f"\nDataset: {dataset_name}  |  Sequences: {len(sequences)}  "
              f"Train: {len(train_seq)}  Test: {len(test_seq)}  (AR-CoT task)")

    train_loader = make_cot_loader(train_seq, vocab, config.batch_size, shuffle=True)
    test_loader = make_cot_loader(test_seq, vocab, config.batch_size, shuffle=False)
    model = build_and_describe(config)

    tag = ("cot_ar_" if config.max_trace_len > 0 else "cot_ar_ansonly_") \
        + (f"overfit{overfit}_" if overfit > 0 else "")
    logger = RunLogger(dataset_name, config, tag=tag, params=model.num_parameters())
    max_new = config.max_trace_len + 8  # trace + `SEP.. ANS YES|NO EOS` slack
    info = run_cot_experiment(model, train_loader, test_loader, DEVICE,
                              epochs=config.epochs, lr=config.lr,
                              weight_decay=config.weight_decay,
                              answer_loss_weight=config.answer_loss_weight,
                              max_new=max_new, eval_every=config.cot_eval_every,
                              logger=logger)
    if not info.get("interrupted"):
        cot_ood_probe(model, config, logger, vocab, trained_on=dataset_name, max_new=max_new)
    logger.save()
    save_checkpoint(model, config, logger.run_id, dataset_name,
                    final_state=info["final_state"], best_epoch=info["best_epoch"])


def cot_ood_probe(model, config: GNNConfig, logger: RunLogger, vocab: CoTVocab,
                  trained_on: str, max_new: int, ood_dataset: str = "er",
                  n_graphs: int = 400):
    """Decode the (best-epoch) CoT model on the ER OOD set and attach the result
    to the run JSON. A different permutation seed so OOD sequences never reuse
    the training relabelings."""
    if trained_on == ood_dataset:
        return
    try:
        sequences = _prepare_cot_sequences(config, ood_dataset, vocab,
                                           limit=n_graphs, seed=config.seed + 1,
                                           gen_kwargs={}, drop_overlong=True)
        loader = make_cot_loader(sequences, vocab, config.batch_size, shuffle=False)
        stats = eval_cot(model, loader, DEVICE, max_new=max_new, by_diameter=True)
        by_diam = stats.pop("by_diameter", None)
        result = {"dataset": ood_dataset, "n": len(sequences),
                  **{k: round(v, 4) for k, v in stats.items()},
                  "by_diameter": by_diam}
        logger.set_extra(ood=result)
        print(f"OOD probe on {ood_dataset} (n={len(sequences)}): "
              f"answer_acc={result['answer_acc']}  trace_em={result['trace_em']}")
    except Exception as e:
        print(f"(OOD probe on {ood_dataset} skipped: {e})")


def inspect_cot_checkpoint(ckpt_path: str, dataset_name: str, limit: int = 0, show: int = 4):
    """Print prompt / gold trace / decoded trace side by side for a few test
    graphs — the direct view of whether the model performs BFS or guesses."""
    model, config, _ = load_checkpoint(ckpt_path)
    vocab = model.vocab
    sequences = _prepare_cot_sequences(config, dataset_name, vocab, limit=limit or 512)
    _, test_seq = split_head_tail(sequences, config.train_frac)
    loader = make_cot_loader(test_seq[:show], vocab, batch_size=show, shuffle=False)

    print(f"\nInspecting {ckpt_path} on {dataset_name} — {show} test sequence(s):")
    batch = next(iter(loader))
    tokens = batch["tokens"].to(DEVICE)
    prompt_len = batch["prompt_len"].to(DEVICE)
    out = model.generate(tokens, prompt_len, max_new=config.max_trace_len + 8)
    for i in range(tokens.size(0)):
        pl = int(prompt_len[i].item())
        gold = [t for t in tokens[i, pl:].tolist() if t != vocab.PAD]
        gen = out[i, pl:].tolist()
        if vocab.EOS in gen:
            gen = gen[:gen.index(vocab.EOS) + 1]
        print(f"\n  --- graph {i}  (y={int(batch['y'][i])}, diam={int(batch['diam'][i])}) ---")
        print(f"  prompt:  {vocab.decode(tokens[i, :pl])}")
        print(f"  gold:    {vocab.decode(gold)}")
        print(f"  decoded: {vocab.decode(gen)}"
              + ("   [MATCH]" if gen == gold else "   [DIFF]"))


def ood_probe(model, config: GNNConfig, logger: RunLogger, trained_on: str,
              ood_dataset: str = "er", n_graphs: int = 400):
    """Evaluate the (best-epoch) model on the ER OOD set and attach the result
    to the run JSON, so distribution shift is measured on every run instead of
    living in terminal scrollback. Skips quietly when the OOD set is
    incompatible with the model's input width."""
    if trained_on == ood_dataset:
        return
    from src.train import connectivity_stats
    try:
        data_list = load_or_create(ood_dataset,
                                   node_features=config.node_features,
                                   lpe_dim=config.lpe_dim,
                                   in_channels=config.in_channels)[:n_graphs]
        attach_components(data_list)
        loader = DataLoader(data_list, batch_size=config.batch_size)
        stats = connectivity_stats(model, loader, DEVICE)
        result = {"dataset": ood_dataset, "n": len(data_list),
                  **{k: round(v, 4) for k, v in stats.items() if v is not None}}
        logger.set_extra(ood=result)
        print(f"OOD probe on {ood_dataset} (n={len(data_list)}): "
              f"em={result.get('em')}  verdict={result.get('verdict')}")
    except Exception as e:
        print(f"(OOD probe on {ood_dataset} skipped: {e})")


# ── Checkpoint evaluation / inspection ─────────────────────────────────────────

def evaluate_checkpoint(ckpt_path: str, dataset_name: str, limit: int = 0):
    """Load a trained model and evaluate it on `dataset_name` (no training)."""
    model, config, ckpt = load_checkpoint(ckpt_path)
    if config.cot_mode == "autoregressive":
        vocab = model.vocab
        sequences = _prepare_cot_sequences(config, dataset_name, vocab, limit=limit,
                                           gen_kwargs=None if dataset_name == config.dataset else {},
                                           drop_overlong=True)
        loader = make_cot_loader(sequences, vocab, config.batch_size, shuffle=False)
        print(f"\nLoaded {ckpt_path}  (AR-CoT, trained on {ckpt.get('train_dataset', '?')})")
        print(f"Evaluating on {dataset_name}  |  {len(sequences)} sequences")
        from src.train import eval_cot as _eval_cot
        stats = _eval_cot(model, loader, DEVICE, max_new=config.max_trace_len + 8,
                          by_diameter=True)
        by_diam = stats.pop("by_diameter", None)
        for k, v in stats.items():
            print(f"  {k:12s} = {v:.4f}")
        if by_diam:
            print("  by diameter:")
            for d, b in by_diam.items():
                print(f"    diam {d:>2}:  acc={b['acc']:.3f}  (n={b['n']})")
        return
    data_list = prepare_generated(config, dataset_name, limit=limit)
    loader = DataLoader(data_list, batch_size=config.batch_size)

    trained_on = ckpt.get("train_dataset", "?")
    print(f"\nLoaded {ckpt_path}  (task={config.task}, trained on {trained_on})")
    print(f"Evaluating on {dataset_name}  |  {len(data_list)} graphs")
    if config.task == "connectivity":
        from src.train import connectivity_stats
        attach_components(data_list)
        stats = connectivity_stats(model, loader, DEVICE, by_diameter=True)
        by_diam = stats.pop("by_diameter", None)
        for k, v in stats.items():
            if v is not None:
                print(f"  {k:12s} = {v:.4f}")
        if by_diam:
            print("  by diameter:")
            for d, b in by_diam.items():
                print(f"    diam {d:>2}:  em={b['em']:.3f}  verdict={b['verdict']:.3f}  (n={b['n']})")
    else:
        from src.train import eval_graph
        acc = eval_graph(model, loader, DEVICE)
        print(f"  accuracy = {acc:.4f}")


def _print_matrix(name, M):
    print(f"  {name}:")
    for row in M.cpu().int().tolist():
        print("    " + "".join("1" if v else "·" for v in row))


def inspect_checkpoint(ckpt_path: str, dataset_name: str, limit: int = 0, show: int = 2):
    """Print failing DISCONNECTED examples: true R vs predicted R, and the mean
    predicted P(reachable) within vs across components. Cross-component ~1.0 means
    the model collapsed to all-ones (heuristic); low means it really separated the
    components but missed some entries (a near-miss, not all-ones)."""
    model, config, _ = load_checkpoint(ckpt_path)
    data_list = prepare_generated(config, dataset_name, limit=limit)
    attach_components(data_list)
    loader = DataLoader(data_list, batch_size=1)

    print(f"\nInspecting {ckpt_path}\n  on {dataset_name} — first {show} failing disconnected graph(s):")
    shown = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(DEVICE)
            logits = model(data)[0]                       # [n, n]
            comp = data.comp
            ncomp = int(comp.max().item()) + 1
            R = (comp[:, None] == comp[None, :]).float()
            pred = (logits > 0).float()
            if ncomp > 1 and not bool((pred == R).all().item()):
                probs = torch.sigmoid(logits)
                within = probs[R == 1].mean().item()
                cross = probs[R == 0].mean().item()
                n = R.size(0)
                print(f"\n  --- n={n}, components={ncomp}, "
                      f"mismatches={int((pred != R).sum().item())}/{n*n} ---")
                print(f"  pred density={pred.mean().item():.3f}  (true={R.mean().item():.3f})")
                print(f"  mean P(reachable):  within-comp={within:.3f}   cross-comp={cross:.3f}")
                print(f"  (cross~1.0 => ALL-ONES heuristic;  cross<0.5 => components separated)")
                _print_matrix("true R", R)
                _print_matrix("pred R", pred)
                shown += 1
                if shown >= show:
                    break
    if shown == 0:
        print("  (no failing disconnected example found)")


def main():
    # `kill <pid>` behaves like Ctrl-C: the CoT training loop catches
    # KeyboardInterrupt, restores best-so-far weights, and the run still saves
    # its checkpoint + results JSON instead of dying with nothing.
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt))

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  type=str, default=None)
    parser.add_argument("--layer",   type=str, default="gcn",
                        choices=["gcn", "sage", "gat", "gin", "global_attn"])
    parser.add_argument("--task",    type=str, default="node", choices=["node", "graph"])
    parser.add_argument("--dataset", type=str, default=None,
                        help="cora | mutag | connectedness | isomorphism | ... "
                             "(overrides the config's `dataset` field)")
    parser.add_argument("-o", "--override", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="Override a config field (YAML-parsed value); repeatable. "
                             "One dot reaches into dicts: dataset_kwargs.num_graphs=500")
    parser.add_argument("--results", action="store_true",
                        help="Print comparison table of all saved runs and exit")
    parser.add_argument("--overfit", type=int, default=0,
                        help="Use only N graphs (balanced), same set for train+test, to check memorisation capacity")
    parser.add_argument("--limit", type=int, default=0,
                        help="Cap the dataset to the first N cached graphs (balanced) for a quick run; cache is untouched")
    parser.add_argument("--eval", type=str, default=None,
                        help="Path to a saved checkpoint (.pt); evaluate it on --dataset and exit (no training)")
    parser.add_argument("--inspect", type=str, default=None,
                        help="Path to a checkpoint (.pt); print failing disconnected examples on --dataset and exit")
    args = parser.parse_args()

    if args.results:
        print_results_table()
        return

    if args.eval or args.inspect:
        if not args.dataset:
            raise SystemExit("--eval/--inspect need --dataset to pick the evaluation set")
        if args.eval:
            evaluate_checkpoint(args.eval, args.dataset, limit=args.limit)
        else:
            ckpt = torch.load(args.inspect, weights_only=False)
            if ckpt["config"].get("cot_mode") == "autoregressive":
                inspect_cot_checkpoint(args.inspect, args.dataset, limit=args.limit)
            else:
                inspect_checkpoint(args.inspect, args.dataset, limit=args.limit)
        return

    config = load_config(args)
    set_seed(config.seed)

    dataset_name = args.dataset or config.dataset \
        or ("cora" if config.task == "node" else None)
    if dataset_name is None:
        raise SystemExit("No dataset: pass --dataset or set `dataset:` in the config YAML")

    if config.task == "node":
        node_experiment(config, dataset_name)
    elif config.task == "connectivity":
        connectivity_experiment(config, dataset_name, limit=args.limit)
    elif config.cot_mode == "autoregressive":
        cot_experiment(config, dataset_name, overfit=args.overfit, limit=args.limit)
    else:
        graph_experiment(config, dataset_name, overfit=args.overfit, limit=args.limit)


if __name__ == "__main__":
    main()
