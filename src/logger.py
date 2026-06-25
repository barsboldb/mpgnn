import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime


class RunLogger:
    """
    Records one experiment run: config, per-epoch metrics, and final summary.
    Saves to results/<timestamp>_<dataset>_<layers>.json on .save().

    `config` may be a GNNConfig dataclass or a plain dict (e.g. a standalone
    experiment's argparse settings).
    """
    def __init__(self, dataset: str, config, tag: str = ""):
        self.dataset = dataset
        self.config = asdict(config) if is_dataclass(config) else dict(config)
        self.history: list[dict] = []
        self.timing: dict = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        layer_types = "-".join(l["type"] for l in self.config.get("layers", []))
        prefix = f"{tag}" if tag else ""
        self.run_id = f"{self.timestamp}_{prefix}{dataset}_{layer_types}"

    def log(self, epoch: int, **metrics):
        self.history.append({"epoch": epoch, **metrics})

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

    def save(self, results_dir: str = "results") -> str:
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, f"{self.run_id}.json")
        payload = {
            "run_id": self.run_id,
            "dataset": self.dataset,
            "config": self.config,
            "summary": self._summary(),
            "timing": self._timing(),
            "history": self.history,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
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
    for fname in files:
        with open(os.path.join(results_dir, fname)) as f:
            run = json.load(f)
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

    if not rows:
        return

    # dynamic column widths
    cols = list(rows[0].keys())
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
    sep = "  ".join("-" * widths[c] for c in cols)
    header = "  ".join(c.ljust(widths[c]) for c in cols)

    print(f"\n{'='*len(sep)}")
    print(f"Experiment results ({len(rows)} runs)")
    print(f"{'='*len(sep)}")
    print(header)
    print(sep)
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))
    print()
