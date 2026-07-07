import json
import os
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime


def _git(*args) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _provenance() -> dict:
    """What produced this run: code version, exact invocation, library versions.
    A results JSON must be traceable to the code that made it — configs alone
    don't identify uncommitted model changes."""
    import torch
    return {
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "argv": " ".join(sys.argv),
        "torch": torch.__version__,
        "python": sys.version.split()[0],
    }


class RunLogger:
    """
    Records one experiment run: config, per-epoch metrics, provenance, and a
    final summary. Saves to results/<timestamp>_<dataset>_<layers>.json on .save().

    `config` may be a GNNConfig dataclass or a plain dict (e.g. a standalone
    experiment's argparse settings). `params` is the model's trainable-parameter
    count. Arbitrary end-of-run artifacts (OOD evals, diameter breakdowns, ...)
    are attached with set_extra() and land as top-level payload keys.
    """
    def __init__(self, dataset: str, config, tag: str = "", params: int | None = None):
        self.dataset = dataset
        self.config = asdict(config) if is_dataclass(config) else dict(config)
        self.params = params
        self.history: list[dict] = []
        self.timing: dict = {}
        self.extra: dict = {}
        self.provenance = _provenance()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        layer_types = "-".join(l["type"] for l in self.config.get("layers", []))
        prefix = f"{tag}" if tag else ""
        self.run_id = f"{self.timestamp}_{prefix}{dataset}_{layer_types}"

    def log(self, epoch: int, **metrics):
        self.history.append({"epoch": epoch, **metrics})
        # flush on every epoch so a killed run loses nothing: the JSON on disk
        # is always the full history so far (atomic replace, no torn files)
        self.save(quiet=True)

    def set_extra(self, **kwargs):
        """Attach end-of-run artifacts (each becomes a top-level key in the JSON)."""
        self.extra.update({k: v for k, v in kwargs.items() if v is not None})

    def set_timing(self, device, inference: dict):
        """Record the device and the single-inference benchmark. Per-epoch
        aggregates are derived from history at save time."""
        self.timing = {"device": str(device), "inference": inference}

    def _timing(self) -> dict:
        out = dict(self.timing)
        tt = [h["train_time_s"] for h in self.history if "train_time_s" in h]
        et = [h["eval_time_s"] for h in self.history if "eval_time_s" in h]
        if tt:
            out["epochs_timed"] = len(tt)
            out["total_train_s"] = round(sum(tt), 4)
            out["mean_epoch_train_s"] = round(sum(tt) / len(tt), 5)
        if et:
            out["mean_epoch_eval_s"] = round(sum(et) / len(et), 5)
        return out

    def save(self, results_dir: str = "results", quiet: bool = False) -> str:
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, f"{self.run_id}.json")
        payload = {
            "run_id": self.run_id,
            "dataset": self.dataset,
            "params": self.params,
            "provenance": self.provenance,
            "config": self.config,
            "summary": self._summary(),
            **self.extra,
            "timing": self._timing(),
            "history": self.history,
        }
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)   # atomic: a kill mid-write never corrupts the JSON
        if not quiet:
            print(f"Saved to {path}")
        return path

    def _summary(self) -> dict:
        if not self.history:
            return {}
        # pick epoch with best val acc (node tasks) or best test acc (graph tasks)
        key = "val" if "val" in self.history[0] else "test"
        best = max(self.history, key=lambda r: r.get(key, 0))
        return {
            "best_epoch": best["epoch"],
            **{k: round(v, 4) for k, v in best.items() if k != "epoch"},
        }


# ── Comparison view ────────────────────────────────────────────────────────────

def print_results_table(results_dir: str = "results"):
    files = sorted(f for f in os.listdir(results_dir) if f.endswith(".json"))
    if not files:
        print("No results yet.")
        return

    rows = []
    skipped = []
    for fname in files:
        with open(os.path.join(results_dir, fname)) as f:
            run = json.load(f)
        if "run_id" not in run:  # standalone scripts (e.g. ye_connectivity) use another schema
            skipped.append(fname)
            continue
        layers = " → ".join(l["type"] for l in run["config"].get("layers", []))
        s = run.get("summary", {})
        t = run.get("timing", {})
        inf = t.get("inference", {})
        rows.append({
            "run_id":   run["run_id"],
            "dataset":  run["dataset"],
            "layers":   layers,
            "hidden":   run["config"].get("hidden_channels", "-"),
            "epochs":   run["config"].get("epochs", "-"),
            **{k: v for k, v in s.items() if k != "best_epoch"},
            "best_epoch": s.get("best_epoch", "-"),
            "ms/epoch": round(t["mean_epoch_train_s"] * 1e3, 1) if "mean_epoch_train_s" in t else "-",
            "infer_ms": inf.get("per_graph_ms", inf.get("per_call_ms", "-")),
        })

    if skipped:
        print(f"(skipped {len(skipped)} non-RunLogger result files: {', '.join(skipped)})")
    if not rows:
        return

    # dynamic column widths over the union of keys (tasks report different metrics)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    widths = {c: max(len(c), max(len(str(r.get(c, "-"))) for r in rows)) for c in cols}
    sep = "  ".join("-" * widths[c] for c in cols)
    header = "  ".join(c.ljust(widths[c]) for c in cols)

    print(f"\n{'='*len(sep)}")
    print(f"Experiment results ({len(rows)} runs)")
    print(f"{'='*len(sep)}")
    print(header)
    print(sep)
    for r in rows:
        print("  ".join(str(r.get(c, "-")).ljust(widths[c]) for c in cols))
    print()
