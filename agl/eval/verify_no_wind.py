"""Regression check for the disabled wind model and legacy compatibility slots."""
import argparse
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
    cfg.sim.n_envs = 2 * n_pairs
    cfg.sim.device = device
    cfg.curriculum.enabled = False
    cfg.task.wind_max = 10.0
    cfg.task.wind_gust_sigma = 5.0
    cfg.task.info_probe_wind = 3.0
    env = GapEnv(cfg, device, difficulty=1.0)
    bank = scene.paired_information_tasks(n_pairs, cfg, 1.0, device)
    env._reset_envs(torch.arange(env.n, device=device), tasks=bank)
    sampled = max(float(bank["dyn"][k].abs().max())
                  for k in ("wind_steady", "gust_sigma", "probe_wind"))
    if float(bank["dyn"]["wind_tau"]) != 0.0:
        raise RuntimeError("wind_tau compatibility slot must be zero")
    env.state["p"][:, 0] = env.task["wall_x"] - .65
    env.state["p"][:, 1] = env.task["gap_cy"]
    env.state["p"][:, 2] = env.task["gap_cz"]
    for k in ("v", "w", "spec_force"):
        env.state[k].zero_()
    env.state["q"].zero_(); env.state["q"][:, 0] = 1.0
    env.state["thrust"] = env.task["dyn"]["mass"] * dynamics.G
    env.state["wind"][1::2] = torch.tensor([4., -3., 2.], device=device)
    env.task["dyn"]["wind_steady"][1::2] = 8.0
    env.task["dyn"]["probe_wind"][1::2] = -6.0
    env.task["dyn"]["gust_sigma"][1::2] = 5.0
    action = torch.zeros(env.n, 4, device=device)
    action[:, 0] = dynamics.hover_thrust_action(env.task["dyn"])
    env.prev_action.copy_(action); env.delay_buf.copy_(action[:, None].expand_as(env.delay_buf))
    max_delta = 0.0
    for _ in range(steps):
        _, _, done, info = env.step(action)
        if done.any() or info["collision"].any():
            raise RuntimeError("no-wind check contacted or terminated")
        max_delta = max(max_delta, float((env.state["p"][0::2] - env.state["p"][1::2]).abs().max()))
    out = {
        "scope": "no_wind_regression_NOT_traversal_evidence",
        "wind_model": "disabled", "sampled_wind_max": sampled,
        "effective_wind_max": float(env._effective_wind_steady().abs().max()),
        "paired_motion_max_abs_delta": max_delta,
        "final_state_wind_max": float(env.state["wind"].abs().max()),
    }
    out["passed"] = all(out[k] == 0.0 for k in
                         ("sampled_wind_max", "effective_wind_max",
                          "paired_motion_max_abs_delta", "final_state_wind_max"))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", type=int, default=8); p.add_argument("--steps", type=int, default=20)
    p.add_argument("--device", default="cpu")
    a = p.parse_args(); out = run_check(a.pairs, a.device, a.steps)
    print(json.dumps(out, indent=2))
    if not out["passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
