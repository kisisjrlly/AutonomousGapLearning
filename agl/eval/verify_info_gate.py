"""Backward-compatible alias for the no-wind regression check."""
import argparse
import copy
import json

import torch

from ..config import load_config
from .verify_no_wind import run_check


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", type=int, default=8)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    out = run_check(args.pairs, args.device, args.steps)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
