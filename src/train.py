from __future__ import annotations
import time
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .logger import RunLogger


def _sync(device):
    """Block until queued device work finishes, so timers measure real compute.

    CUDA/MPS kernels are dispatched asynchronously; without a sync the host
    clock stops before the GPU does and every duration is meaningless.
    """
    t = device.type if hasattr(device, "type") else str(device)
    if t == "cuda":
        torch.cuda.synchronize()
    elif t == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


@torch.no_grad()
def benchmark_inference(model, sample, device, warmup: int = 5, iters: int = 30):
    """Average latency of one forward pass on a fixed input (eval mode).

    `sample` is a graph batch (graph task) or the single graph (node task).
    Returns per-call latency plus a per-unit normalisation (per graph / per node).
    """
    model.eval()
    sample = sample.to(device)
    n_graphs = int(getattr(sample, "num_graphs", 1))
    n_nodes = int(sample.num_nodes)

    for _ in range(warmup):
        model(sample)
    _sync(device)

    t0 = time.perf_counter()
    for _ in range(iters):
        model(sample)
    _sync(device)
    per_call = (time.perf_counter() - t0) / iters

    return {
        "per_call_ms": round(per_call * 1e3, 5),
        "per_graph_ms": round(per_call / n_graphs * 1e3, 5),
        "per_node_ms": round(per_call / max(n_nodes, 1) * 1e3, 5),
        "graphs_per_call": n_graphs,
        "nodes_per_call": n_nodes,
        "warmup": warmup,
        "iters": iters,
    }


def _grad_norm(model) -> float:
    """Global L2 norm of all parameter gradients (call between backward and step).
    The direct measurement for 'is the useful signal (e.g. the bridge-edge
    gradient) vanishing relative to the rest'."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float((p.grad.detach() ** 2).sum())
    return total ** 0.5


def _cpu_state(model) -> dict:
    """Detached CPU copy of the model weights (for best-epoch snapshots)."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# ── Node classification ────────────────────────────────────────────────────────

def train_node(model, data, optimizer, device):
    data = data.to(device)
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def eval_node(model, data, device):
    data = data.to(device)
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    results = {}
    for split in ("train_mask", "val_mask", "test_mask"):
        mask = getattr(data, split)
        acc = (pred[mask] == data.y[mask]).float().mean().item()
        results[split.replace("_mask", "")] = acc
    return results


def run_node_experiment(model, data, device, epochs=200, lr=0.01, weight_decay=5e-4,
                        logger: RunLogger | None = None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        _sync(device); t0 = time.perf_counter()
        loss = train_node(model, data, optimizer, device)
        _sync(device); t1 = time.perf_counter()
        metrics = eval_node(model, data, device)
        _sync(device); t2 = time.perf_counter()
        if logger is not None:
            logger.log(epoch, loss=round(loss, 4),
                       **{k: round(v, 4) for k, v in metrics.items()},
                       train_time_s=round(t1 - t0, 5), eval_time_s=round(t2 - t1, 5))
        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}  loss={loss:.4f}  "
                f"train={metrics['train']:.4f}  val={metrics['val']:.4f}  test={metrics['test']:.4f}"
            )

    if logger is not None:
        logger.set_timing(device, benchmark_inference(model, data, device))

    summary = logger._summary() if logger else {}
    print(f"\nBest val: {summary.get('val', '-')}  |  Test at best val: {summary.get('test', '-')}")


# ── Graph classification ───────────────────────────────────────────────────────

def train_graph(model, loader: DataLoader, optimizer, device):
    """One training epoch. Returns (mean loss, train accuracy, mean grad norm).

    Train accuracy is accumulated from the same forward passes used for the loss
    (free), so it is measured in train mode (dropout on) — a slight underestimate,
    but enough to separate memorisation (train→1, test flat) from underfitting.
    """
    model.train()
    total_loss = gn_sum = 0.0
    correct = total = batches = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        gn_sum += _grad_norm(model)
        batches += 1
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        correct += (out.argmax(dim=1) == data.y).sum().item()
        total += data.num_graphs
    return total_loss / total, correct / total, gn_sum / max(batches, 1)


@torch.no_grad()
def eval_graph(model, loader: DataLoader, device):
    model.eval()
    correct = total = 0
    for data in loader:
        data = data.to(device)
        pred = model(data).argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.num_graphs
    return correct / total


def run_graph_experiment(model, train_loader, test_loader, device,
                         epochs=200, lr=0.01, weight_decay=5e-4,
                         logger: RunLogger | None = None) -> dict:
    """Returns {"best_epoch", "final_state"}; the model is left holding the
    BEST-epoch weights (final weights are in the returned dict)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_acc, best_epoch, best_state = -1.0, 0, None

    for epoch in range(1, epochs + 1):
        _sync(device); t0 = time.perf_counter()
        loss, train_acc, gnorm = train_graph(model, train_loader, optimizer, device)
        _sync(device); t1 = time.perf_counter()
        test_acc = eval_graph(model, test_loader, device)
        _sync(device); t2 = time.perf_counter()
        if test_acc > best_acc:
            best_acc, best_epoch, best_state = test_acc, epoch, _cpu_state(model)
        if logger is not None:
            logger.log(epoch, loss=round(loss, 4),
                       train=round(train_acc, 4), test=round(test_acc, 4),
                       grad_norm=round(gnorm, 4),
                       train_time_s=round(t1 - t0, 5), eval_time_s=round(t2 - t1, 5))
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}  loss={loss:.4f}  train_acc={train_acc:.4f}  "
                  f"test_acc={test_acc:.4f}  gap={train_acc - test_acc:+.3f}")

    if logger is not None:
        sample = next(iter(test_loader))
        logger.set_timing(device, benchmark_inference(model, sample, device))

    final_state = _cpu_state(model)
    if best_state is not None:
        model.load_state_dict(best_state)

    summary = logger._summary() if logger else {}
    print(f"\nBest test acc: {summary.get('test', '-')}  at epoch {summary.get('best_epoch', '-')}"
          f"  (model restored to best epoch for downstream evals)")
    return {"best_epoch": best_epoch, "final_state": final_state}


# ── Autoregressive chain-of-thought (cot_mode: autoregressive) ────────────────

def _cot_targets(tokens, prompt_len, pad_id):
    """Next-token targets with prompt and padding positions masked to -100.

    logits[i, t] predicts tokens[i, t+1], so the target row is tokens[:, 1:] and
    position t is supervised iff t+1 >= prompt_len[i] (the completion starts at
    the first token AFTER the prompt; the prompt itself is given, not predicted).
    """
    targets = tokens[:, 1:].clone()
    B, Lm1 = targets.shape
    pos = torch.arange(1, Lm1 + 1, device=tokens.device).expand(B, Lm1)
    targets[(pos < prompt_len[:, None]) | (targets == pad_id)] = -100
    return targets


def _tf_answer_acc(logits, tokens, y, vocab) -> tuple[int, int]:
    """Teacher-forced answer accuracy: at the position holding ANS, does the
    model predict the correct YES/NO next token? Cheap per-epoch proxy for the
    decoded accuracy (it conditions on the GOLD trace, so it is an upper bound)."""
    ans_pos = (tokens == vocab.ANS).float().argmax(dim=1)      # [B], one ANS per row
    b = torch.arange(tokens.size(0), device=tokens.device)
    pred = logits[b, ans_pos].argmax(dim=-1)                   # logits index t predicts t+1
    gold = torch.where(y.bool(), torch.tensor(vocab.YES, device=tokens.device),
                       torch.tensor(vocab.NO, device=tokens.device))
    return int((pred == gold).sum().item()), tokens.size(0)


def train_cot(model, loader, optimizer, device, answer_loss_weight: float = 1.0):
    """One teacher-forced epoch. Returns (mean loss, tf answer acc, mean grad norm)."""
    vocab = model.vocab
    model.train()
    total_loss = gn_sum = 0.0
    correct = total = batches = 0
    for batch in loader:
        tokens = batch["tokens"].to(device)
        prompt_len = batch["prompt_len"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad()
        logits = model(tokens[:, :-1])
        targets = _cot_targets(tokens, prompt_len, vocab.PAD)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                               ignore_index=-100)
        if answer_loss_weight != 1.0:
            # extra CE on just the YES/NO position, scaled by (weight - 1) so
            # weight=1 is a no-op and the trace loss is otherwise untouched
            ans_pos = (tokens == vocab.ANS).float().argmax(dim=1)
            b = torch.arange(tokens.size(0), device=device)
            loss = loss + (answer_loss_weight - 1.0) * F.cross_entropy(
                logits[b, ans_pos], targets[b, ans_pos])
        loss.backward()
        gn_sum += _grad_norm(model)
        batches += 1
        optimizer.step()

        n = tokens.size(0)
        total_loss += loss.item() * n
        c, t = _tf_answer_acc(logits, tokens, y, vocab)
        correct += c
        total += t
    return total_loss / total, correct / total, gn_sum / max(batches, 1)


@torch.no_grad()
def eval_cot_teacher_forced(model, loader, device) -> float:
    """Answer accuracy with the gold trace in context (one forward per batch)."""
    model.eval()
    correct = total = 0
    for batch in loader:
        tokens = batch["tokens"].to(device)
        logits = model(tokens[:, :-1])
        c, t = _tf_answer_acc(logits, tokens, batch["y"].to(device), model.vocab)
        correct += c
        total += t
    return correct / total


def _parse_generated(row: list[int], prompt_len: int, vocab):
    """Split a generated row into (trace tokens, predicted label or None).
    The prediction is the token right after ANS; None if ANS never appeared or
    is followed by garbage — parse failures count as wrong."""
    gen = row[prompt_len:]
    if vocab.EOS in gen:
        gen = gen[:gen.index(vocab.EOS) + 1]
    if vocab.ANS not in gen:
        return gen, None
    k = gen.index(vocab.ANS)
    trace = gen[:k]
    if k + 1 >= len(gen) or gen[k + 1] not in (vocab.YES, vocab.NO):
        return trace, None
    return trace, int(gen[k + 1] == vocab.YES)


@torch.no_grad()
def eval_cot(model, loader, device, max_new: int, by_diameter: bool = False) -> dict:
    """Greedy-decode eval:

      answer_acc  — fraction of graphs whose decoded YES/NO is correct (primary;
                    parse failures count as wrong).
      trace_em    — decoded trace exactly matches the canonical BFS trace
                    (only meaningful when training with traces).
      mean_levels — mean number of BFS levels the model emitted (#SEP + 1);
                    should track the true eccentricity if the trace is real.
      parse_fail  — fraction with no well-formed `ANS YES|NO`.
      by_diameter — optional {diam: {n, acc}} buckets.
    """
    vocab = model.vocab
    model.eval()
    correct = em = fails = total = 0
    level_sum = 0
    buckets: dict[int, dict] = {}
    for batch in loader:
        tokens = batch["tokens"].to(device)
        prompt_len = batch["prompt_len"].to(device)
        out = model.generate(tokens, prompt_len, max_new=max_new)
        for i in range(tokens.size(0)):
            pl = int(prompt_len[i].item())
            trace, pred = _parse_generated(out[i].tolist(), pl, vocab)
            gold_full = tokens[i, pl:].tolist()
            if vocab.PAD in gold_full:
                gold_full = gold_full[:gold_full.index(vocab.PAD)]
            gold_trace = gold_full[:gold_full.index(vocab.ANS)] if vocab.ANS in gold_full else []
            y = int(batch["y"][i].item())

            ok = pred is not None and pred == y
            correct += ok
            em += int(trace == gold_trace)
            fails += int(pred is None)
            level_sum += trace.count(vocab.SEP) + 1 if trace else 0
            total += 1
            if by_diameter:
                d = int(batch["diam"][i].item())
                b = buckets.setdefault(d, {"n": 0, "acc": 0})
                b["n"] += 1
                b["acc"] += ok
    out = {
        "answer_acc": correct / total,
        "trace_em": em / total,
        "mean_levels": level_sum / total,
        "parse_fail": fails / total,
    }
    if by_diameter:
        out["by_diameter"] = {d: {"n": b["n"], "acc": round(b["acc"] / b["n"], 4)}
                              for d, b in sorted(buckets.items())}
    return out


def run_cot_experiment(model, train_loader, test_loader, device,
                       epochs=200, lr=1e-3, weight_decay=0.0,
                       answer_loss_weight=1.0, max_new=64, eval_every=1,
                       logger: RunLogger | None = None) -> dict:
    """Autoregressive-CoT training loop. Greedy-decode eval every `eval_every`
    epochs (the `test` metric — decoded answer accuracy — exists only on those
    epochs, so best-epoch selection sees decoded numbers only); teacher-forced
    test accuracy rides along every epoch as `test_tf`. Returns
    {"best_epoch", "final_state", "interrupted"} and leaves the model on the
    best weights.

    Ctrl-C / SIGTERM (main.py maps it to KeyboardInterrupt) stops the epoch
    loop but NOT the run: best-so-far weights are restored and returned so the
    caller still saves the checkpoint and results JSON. The slow best-weights
    decode breakdown is skipped on interrupt — a second Ctrl-C during the save
    tail kills for real."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_acc, best_epoch, best_state = -1.0, 0, None
    interrupted, epoch = False, 0

    try:
        for epoch in range(1, epochs + 1):
            _sync(device); t0 = time.perf_counter()
            loss, tf_train, gnorm = train_cot(model, train_loader, optimizer, device,
                                              answer_loss_weight=answer_loss_weight)
            _sync(device); t1 = time.perf_counter()
            decode = epoch % eval_every == 0 or epoch == epochs
            tf_test = eval_cot_teacher_forced(model, test_loader, device)
            stats = eval_cot(model, test_loader, device, max_new=max_new) if decode else None
            _sync(device); t2 = time.perf_counter()

            if stats is not None and stats["answer_acc"] > best_acc:
                best_acc, best_epoch, best_state = stats["answer_acc"], epoch, _cpu_state(model)
            if logger is not None:
                row = dict(loss=round(loss, 4), train=round(tf_train, 4),
                           test_tf=round(tf_test, 4), grad_norm=round(gnorm, 4),
                           train_time_s=round(t1 - t0, 5), eval_time_s=round(t2 - t1, 5))
                if stats is not None:
                    row.update(test=round(stats["answer_acc"], 4),
                               trace_em=round(stats["trace_em"], 4),
                               mean_levels=round(stats["mean_levels"], 2),
                               parse_fail=round(stats["parse_fail"], 4))
                logger.log(epoch, **row)
            if epoch % 10 == 0:
                dec = f"decoded={stats['answer_acc']:.4f}  trace_em={stats['trace_em']:.4f}" \
                    if stats is not None else "decoded=—"
                print(f"Epoch {epoch:03d}  loss={loss:.4f}  tf_train={tf_train:.4f}  "
                      f"tf_test={tf_test:.4f}  {dec}")
    except KeyboardInterrupt:
        interrupted = True
        print(f"\n[interrupted] stopping at epoch {epoch}/{epochs} — saving best-so-far "
              f"(decoded {best_acc:.4f} @ epoch {best_epoch}); Ctrl-C again to abandon")

    final_state = _cpu_state(model)
    if best_state is not None:
        model.load_state_dict(best_state)
    if not interrupted:
        breakdown = eval_cot(model, test_loader, device, max_new=max_new, by_diameter=True)
        by_diam = breakdown.pop("by_diameter", None)
        if logger is not None:
            logger.set_extra(
                best_eval={k: round(v, 4) for k, v in breakdown.items()},
                diameter_breakdown=by_diam,
            )
        print(f"\nBest decoded answer acc: {best_acc:.4f}  at epoch {best_epoch}"
              f"  (model restored to best epoch for downstream evals)")
        if by_diam:
            print("Decoded accuracy by graph diameter (best weights):")
            for d, b in by_diam.items():
                print(f"  diam {d:>2}:  acc={b['acc']:.3f}  (n={b['n']})")
    if logger is not None:
        logger.set_extra(interrupted_at_epoch=epoch if interrupted else None)
    return {"best_epoch": best_epoch, "final_state": final_state,
            "interrupted": interrupted}


# ── Connectivity matrix (Ye et al. 2026) ───────────────────────────────────────

def _connectivity_loss_acc(model, loader, device, train=False, optimizer=None):
    """Per-graph BCE over the n_g x n_g reachability target (from node component
    labels data.comp) + exact-match accuracy + mean grad norm (train only).
    data.comp must be attached upstream."""
    total_loss = exact = total = gn_sum = 0.0
    batches = 0
    for data in loader:
        data = data.to(device)
        logits = model(data)                       # list of [n_g, n_g]
        batch = data.batch
        comp = data.comp
        loss = 0.0
        for g, lg in enumerate(logits):
            cg = comp[batch == g]
            Rg = (cg[:, None] == cg[None, :]).float()
            loss = loss + F.binary_cross_entropy_with_logits(lg, Rg)
            exact += float(((lg > 0).float() == Rg).all().item())
            total += 1
        loss = loss / len(logits)
        if train:
            optimizer.zero_grad()
            loss.backward()
            gn_sum += _grad_norm(model)
            batches += 1
            optimizer.step()
        total_loss += loss.item() * len(logits)
    return total_loss / total, exact / total, gn_sum / max(batches, 1)


@torch.no_grad()
def connectivity_stats(model, loader, device, by_diameter=False) -> dict:
    """One eval pass over the connectivity readout, computing everything:

      em         — fraction of graphs whose entire n x n matrix is correct (strict).
      verdict    — fraction whose connected/disconnected verdict is correct
                   (predicted-connected iff every entry > 0; strictly easier than em).
      p_within   — mean predicted P(reachable) over true same-component pairs.
      p_cross    — mean predicted P(reachable) over true cross-component pairs.
                   p_cross ~ 1 with high p_within = the all-ones collapse;
                   p_cross < 0.5 = the components genuinely separated.
      fp_cross   — fraction of true-unreachable pairs predicted reachable (the
                   all-ones heuristic as an error rate; the 'hard part' of the task).
      bdh_sparsity — occupancy of the positive-sparse neuron space (BDH models
                   only; fraction of nonzero activations on the last eval batch).
      by_diameter — optional {diam: {n, em, verdict}} buckets (needs data.diam):
                   the depth-vs-reachability curve the capacity argument rests on.

    p_cross / fp_cross are None when the split contains no disconnected graph."""
    model.eval()
    em = verdict = total = 0.0
    pw_sum = pc_sum = fp = 0.0
    pw_cnt = pc_cnt = 0
    buckets: dict[int, dict] = {}
    for data in loader:
        data = data.to(device)
        logits = model(data)                       # list of [n_g, n_g]
        batch = data.batch
        comp = data.comp
        diams = data.diam.tolist() if by_diameter and hasattr(data, "diam") else None
        for g, lg in enumerate(logits):
            cg = comp[batch == g]
            Rg = (cg[:, None] == cg[None, :]).float()
            pred = (lg > 0).float()
            g_em = bool((pred == Rg).all().item())
            g_verdict = pred.bool().all().item() == Rg.bool().all().item()
            probs = torch.sigmoid(lg)
            within, cross = Rg == 1, Rg == 0
            pw_sum += probs[within].sum().item(); pw_cnt += int(within.sum())
            if bool(cross.any()):
                pc_sum += probs[cross].sum().item(); pc_cnt += int(cross.sum())
                fp += pred[cross].sum().item()
            em += g_em; verdict += g_verdict; total += 1
            if diams is not None:
                b = buckets.setdefault(int(diams[g]), {"n": 0, "em": 0, "verdict": 0})
                b["n"] += 1; b["em"] += g_em; b["verdict"] += g_verdict
    out = {
        "em": em / total,
        "verdict": verdict / total,
        "p_within": pw_sum / max(pw_cnt, 1),
        "p_cross": pc_sum / pc_cnt if pc_cnt else None,
        "fp_cross": fp / pc_cnt if pc_cnt else None,
    }
    sparsity = getattr(model, "last_sparsity", None)
    if sparsity is not None:
        out["bdh_sparsity"] = sparsity
    if by_diameter:
        out["by_diameter"] = {d: {"n": b["n"], "em": round(b["em"] / b["n"], 4),
                                  "verdict": round(b["verdict"] / b["n"], 4)}
                              for d, b in sorted(buckets.items())}
    return out


@torch.no_grad()
def connectivity_metrics(model, loader, device):
    """Back-compat wrapper: (exact_match, conn_verdict_acc)."""
    s = connectivity_stats(model, loader, device)
    return s["em"], s["verdict"]


def run_connectivity_experiment(model, train_loader, test_loader, device,
                                epochs=200, lr=1e-3, weight_decay=0.0,
                                logger: RunLogger | None = None) -> dict:
    """Train the pairwise reachability readout (R_ij = same component) with the
    dense matrix target — the supervision that makes connectivity learnable.

    Per epoch, logs the full connectivity_stats readout (verdict accuracy,
    within/cross probabilities, cross-pair FP rate, grad norm) alongside EM.
    Afterwards restores the BEST-epoch weights, runs the diameter-stratified
    breakdown on them, and returns {"best_epoch", "final_state"}."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_em, best_epoch, best_state = -1.0, 0, None

    for epoch in range(1, epochs + 1):
        model.train()
        _sync(device); t0 = time.perf_counter()
        loss, train_acc, gnorm = _connectivity_loss_acc(model, train_loader, device,
                                                        train=True, optimizer=optimizer)
        _sync(device); t1 = time.perf_counter()
        stats = connectivity_stats(model, test_loader, device)
        _sync(device); t2 = time.perf_counter()
        if stats["em"] > best_em:
            best_em, best_epoch, best_state = stats["em"], epoch, _cpu_state(model)
        if logger is not None:
            extra = {k: round(v, 4) for k, v in stats.items() if v is not None and k != "em"}
            logger.log(epoch, loss=round(loss, 4),
                       train=round(train_acc, 4), test=round(stats["em"], 4),
                       grad_norm=round(gnorm, 4), **extra,
                       train_time_s=round(t1 - t0, 5), eval_time_s=round(t2 - t1, 5))
        if epoch % 10 == 0:
            pc = stats["p_cross"]
            print(f"Epoch {epoch:03d}  loss={loss:.4f}  train_EM={train_acc:.4f}  "
                  f"test_EM={stats['em']:.4f}  verdict={stats['verdict']:.4f}  "
                  f"p_within={stats['p_within']:.3f}  p_cross={pc if pc is None else round(pc, 3)}")

    if logger is not None:
        sample = next(iter(test_loader))
        logger.set_timing(device, benchmark_inference(model, sample, device))

    # final evals run on the best weights, not wherever training happened to end
    final_state = _cpu_state(model)
    if best_state is not None:
        model.load_state_dict(best_state)
    breakdown = connectivity_stats(model, test_loader, device, by_diameter=True)
    by_diam = breakdown.pop("by_diameter", None)
    if logger is not None:
        logger.set_extra(
            best_eval={k: round(v, 4) for k, v in breakdown.items() if v is not None},
            diameter_breakdown=by_diam,
        )

    print(f"\nBest test exact-match: {best_em:.4f}  at epoch {best_epoch}"
          f"  (model restored to best epoch for downstream evals)")
    if by_diam:
        print("EM by graph diameter (best weights):")
        for d, b in by_diam.items():
            print(f"  diam {d:>2}:  em={b['em']:.3f}  verdict={b['verdict']:.3f}  (n={b['n']})")
    return {"best_epoch": best_epoch, "final_state": final_state}
