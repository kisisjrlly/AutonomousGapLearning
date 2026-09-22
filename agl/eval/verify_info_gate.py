"""Smoke-check the information gate without training a policy.

Creates matched task pairs that differ only in latent probe-wind sign, verifies
that the hidden factor has no effect outside the probe zone, then applies the
same hover command from a matched near-gap state and reports the induced lateral
velocity divergence. This is environment validation, not adaptation evidence.
"""
import argparse
import copy
import json

import torch

from ..config import load_config
from ..sim import dynamics, scene
from ..sim.env import GapEnv


@torch.no_grad()
def run_check(n_pairs=8, device="cpu", steps=20, seed=20260922):
    if n_pairs <= 0 or steps <= 0:
        raise ValueError("n_pairs and steps must be positive")
    torch.manual_seed(seed)
    cfg = load_config()
    cfg = copy.deepcopy(cfg)
    cfg.sim.n_envs = 2 * n_pairs
    cfg.sim.device = device
    cfg.curriculum.enabled = False
    cfg.task.info_gate_enabled = True

    env = GapEnv(cfg, device, difficulty=1.0)
    bank = scene.paired_information_tasks(n_pairs, cfg, 1.0, device)
    ids = torch.arange(2 * n_pairs, device=device)
    env._reset_envs(ids, tasks=bank)

    # Remove unrelated stochastic disturbances for this structural check.
    env.task["dyn"]["wind_steady"].zero_()
    env.task["dyn"]["gust_sigma"].zero_()
    env.state["wind"].zero_()

    # Pairwise-identical physical state, outside the information gate.
    x_far = env.task["wall_x"] - cfg.task.info_probe_distance - 0.2
    env.state["p"][:, 0] = x_far
    env.state["p"][:, 1] = 0.0
    env.state["p"][:, 2] = 1.5
    env.state["v"].zero_()
    far = env._effective_wind_steady().clone()
    far_pair_delta = (far[0::2] - far[1::2]).abs().max().item()

    # Move both members to the same near-gap state where the latent factor is on.
    x_near = (env.task["wall_x"] - cfg.task.info_probe_distance
              + cfg.task.info_probe_ramp + 0.05)
    env.state["p"][:, 0] = x_near
    env.state["p"][:, 1] = 0.0
    env.state["p"][:, 2] = 1.5
    env.state["v"].zero_()
    env.state["w"].zero_()
    env.state["wind"].zero_()
    env.state["q"].zero_()
    env.state["q"][:, 0] = 1.0
    env.state["thrust"] = env.task["dyn"]["mass"] * 9.81

    near = env._effective_wind_steady().clone()
    near_pair_delta = (near[0::2, 1] - near[1::2, 1]).abs()
    sign_product = near[0::2, 1] * near[1::2, 1]

    hover = dynamics.hover_thrust_action(env.task["dyn"])
    action = torch.zeros(2 * n_pairs, 4, device=device)
    action[:, 0] = hover
    env.prev_action = action.clone()
    env.delay_buf[:] = action.unsqueeze(1)

    for _ in range(steps):
        _, _, done, info = env.step(action)
        if done.any() or info['collision'].any() or not torch.isfinite(info['clearance']).all() or (info['clearance'] <= 0).any():
            raise RuntimeError('information-gate check invalid: contact, terminal/reset or invalid clearance')

    vy = env.state["v"][:, 1].clone()
    pair_vy_separation = (vy[0::2] - vy[1::2]).abs()
    opposite_motion = vy[0::2] * vy[1::2] < 0

    return {
        "scope": "environment_smoke_NOT_adaptation_evidence",
        "n_pairs": n_pairs,
        "steps": steps,
        "far_pair_max_wind_delta": far_pair_delta,
        "near_pair_min_wind_delta": float(near_pair_delta.min().item()),
        "near_pair_opposite_wind_fraction": float((sign_product < 0).float().mean().item()),
        "pair_min_lateral_velocity_separation": float(pair_vy_separation.min().item()),
        "pair_opposite_lateral_motion_fraction": float(opposite_motion.float().mean().item()),
    }


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
