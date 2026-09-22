"""Run a small deterministic checkpoint monitor episode and open it in Rerun.

This is intentionally separate from the high-throughput training hot path. It is
safe to run manually against ckpt_latest.pt while training is paused or when the
extra device load is acceptable.
"""
import argparse
import copy
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from ..config import load_config
from ..models.policy import Policy
from ..viz.episode import load_eval_episode
from ..viz.rerun_episode import log_episode
from .evaluate import run_eval


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--split", choices=("id", "ood_geom", "ood_dyn"), default="id")
    p.add_argument("--task", type=int, default=0,
                   help="fixed task-bank index; small indices are recommended")
    p.add_argument("--device", default="cuda")
    p.add_argument("--noise-seed", type=int, default=1234)
    p.add_argument("--out", type=Path, help="write .rrd instead of spawning Viewer")
    args = p.parse_args()
    if args.task < 0:
        raise ValueError("task must be non-negative")

    ck = torch.load(args.ckpt, map_location=args.device)
    cfg = load_config(overrides=ck["cfg"])
    model = Policy(cfg).to(args.device)
    model.load_state_dict(ck["model"])
    model.eval()

    n = args.task + 1
    ecfg = copy.deepcopy(cfg)
    rec, task, steps = run_eval(
        model, ecfg, args.split, n, args.device,
        wipe_context=False,
        reset_between_attempts=cfg.model.reset_between_attempts,
        noise_seed=args.noise_seed,
        save_frames=n,
    )
    meta = {
        "ckpt": str(args.ckpt), "split": args.split, "n": n,
        "wipe": False, "train_steps": ck.get("steps", -1),
        "noise_seed": args.noise_seed, "save_frames": n,
        "dt_ctrl": cfg.sim.dt_ctrl, "retry_x": cfg.sim.retry_x,
        "succ_margin": cfg.sim.succ_margin, "body_r": cfg.sim.body_r,
        "body_hh": cfg.sim.body_hh,
        "info_gate_enabled": cfg.task.info_gate_enabled,
        "info_probe_distance": cfg.task.info_probe_distance,
        "info_probe_ramp": cfg.task.info_probe_ramp,
    }

    with tempfile.TemporaryDirectory(prefix="agl-viz-") as tmp:
        path = Path(tmp) / "monitor.npz"
        np.savez_compressed(
            path, steps=steps,
            **{f"rec_{k}": v for k, v in rec.items()},
            **{f"task_{k}": v for k, v in task.items()},
            meta=json.dumps(meta),
        )
        ep = load_eval_episode(path, args.task)
        log_episode(ep, out=args.out, spawn=args.out is None)
        if args.out:
            print(f"saved {args.out}")
        else:
            print(
                f"opened checkpoint monitor task {args.task}: "
                f"{ep.steps} valid steps, phase={ep.phase(ep.steps-1)}"
            )


if __name__ == "__main__":
    main()
