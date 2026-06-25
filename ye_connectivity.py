"""CLI for the connectivity-matrix task (Ye et al. 2026).

The implementation lives in src/connectivity.py (also usable via
`main.py --config <connectivity.yaml>`). This is just an argparse front-end.

Usage:
    python ye_connectivity.py --dist hard --regime within --n 24 --depth 2 --epochs 300
    python ye_connectivity.py --dist diam --regime within --local --depth 6
"""
import argparse

from src.connectivity import run_connectivity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["within", "raw"], default="within")
    ap.add_argument("--dist", choices=["er", "diam", "hard"], default="diam")
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--capacity", type=int, default=0)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--p", type=float, default=0.12)
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--test", type=int, default=400)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    run_connectivity(vars(ap.parse_args()))


if __name__ == "__main__":
    main()
