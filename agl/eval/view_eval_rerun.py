"""Open a real GapEnv evaluation episode in Rerun or export it to .rrd.

Examples:
  python -m agl.eval.view_eval_rerun --input results/run/eval_id.npz --task 0
  python -m agl.eval.view_eval_rerun --input results/run/eval_id.npz --task 0 --out /tmp/task0.rrd
"""
import argparse
from pathlib import Path

from ..viz.episode import load_eval_episode
from ..viz.rerun_episode import log_episode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--out", type=Path, help="write .rrd instead of spawning a viewer")
    args = p.parse_args()
    ep = load_eval_episode(args.input, args.task)
    if args.out:
        log_episode(ep, out=args.out, spawn=False)
        print(f"saved {args.out}")
    else:
        log_episode(ep, spawn=True)
        print(
            f"opened task {args.task}: {ep.steps} valid steps "
            f"(viewer data is clipped at the episode boundary)"
        )


if __name__ == "__main__":
    main()
