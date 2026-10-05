"""Run the auditable no-wind posture-controller diagnostic.

This is a low-level controller check, not a learned-policy evaluation. It
records the actual GapEnv trajectory, reports whether the body ever crossed
the success plane without contact, and optionally leaves the NPZ ready for
``animate_eval_3d`` or the Rerun viewer.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..attempt.executor import ControllerConfig, execute_attempt_batch
from ..attempt.spec import AttemptSpec
from ..config import load_config
from ..data.generate_attempt_dataset import _set_v0_fixed_geometry, save_viewer_trace
from ..sim.env import GapEnv
from .verify_narrow_gap import scan


def _fixture(device: str):
    cfg = load_config()
    cfg.sim.n_envs = 1
    cfg.sim.device = device
    cfg.sim.ep_len = 1400
    cfg.task.delay_max_steps = 0
    env = GapEnv(cfg, device, difficulty=1.0)
    _set_v0_fixed_geometry(env.task, cfg)
    dyn = env.task["dyn"]
    dyn["mass"].fill_(0.775)
    dyn["tmax"].fill_(0.775 * 9.81 * 2.8)
    dyn["wind_steady"].zero_()
    dyn["probe_wind"].zero_()
    dyn["gust_sigma"].zero_()
    dyn["delay"].zero_()
    dyn["tau_rate"].fill_(0.03)
    dyn["tau_thrust"].fill_(0.04)
    dyn["alpha_max"].fill_(55.0)
    dyn["kd_lin"].fill_(0.05)
    dyn["kd_quad"].fill_(0.005)
    return cfg, env


@torch.no_grad()
def run(*, out: str | Path | None = None, start_y: float = 0.4,
        entry_speed: float = 2.0, roll_target_deg: float = 65.0,
        device: str = "cpu"):
    cfg, env = _fixture(device)
    controller = ControllerConfig(
        accel_limit=3.0,
        roll_target_rad=float(np.deg2rad(roll_target_deg)),
        roll_start_distance=.9,
        roll_hold_distance=.4,
        roll_prebias_rad=0.0,
        roll_prebias_distance=.1,
        roll_release_distance=.4,
        roll_kp=10.0,
        roll_kd=.2,
    )
    spec = AttemptSpec(float(start_y), 0.0, 0.0, float(entry_speed), 1.2, 3.0)
    batch = execute_attempt_batch(
        env, [spec], max_approach_steps=500, max_retreat_steps=200,
        return_batch=True, record=out is not None, controller=controller,
    )
    outcome = batch.outcomes[0].to_dict()
    report, *_ = scan(cfg)
    result = {
        "scope": "low_level_posture_controller_diagnostic",
        "wind_model": "disabled",
        "static_geometry": report,
        "spec": spec.to_dict(),
        "controller": asdict(controller),
        "outcome": outcome,
        "success_plane_reached": bool(outcome["success"]),
        "collision_free_success": bool(outcome["success"] and not outcome["contact"]),
        "interpretation": (
            "A success row would establish only this fixed controller and fixture "
            "rollout; it would not establish learning or real-flight safety."
        ),
    }
    if out is not None:
        root = Path(out)
        root.mkdir(parents=True, exist_ok=False)
        save_viewer_trace(root / "trajectory.npz", batch.trace)
        result["trajectory"] = str(root / "trajectory.npz")
        result["summary"] = str(root / "summary.json")
        (root / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path)
    p.add_argument("--start-y", type=float, default=.4)
    p.add_argument("--entry-speed", type=float, default=2.0)
    p.add_argument("--roll-target-deg", type=float, default=65.0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    print(json.dumps(run(out=args.out, start_y=args.start_y,
                         entry_speed=args.entry_speed,
                         roll_target_deg=args.roll_target_deg,
                         device=args.device), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
