"""Verify that hidden task evidence reaches actor-visible sensors.

This is a structural observability check, not a learned classifier and not a
noise-robust deployment claim. Observation noise/pixel noise are disabled so
matched latent task pairs must be exactly identical outside the information
zone. Inside the zone, the only task difference (probe-wind sign) acts through
the 6-DoF dynamics and must create a measurable difference in the actor's
18-D observation vector.
"""
import argparse
import json

import torch

from ..config import load_config
from ..sim import dynamics, render, scene
from ..sim.env import GapEnv


def _pair_max_delta(x):
    return float((x[0::2] - x[1::2]).abs().max().item())


def _sync_frame_queue(env):
    fresh = render.render(
        env.state["p"], env.state["q"], env.task, env.rays, env.cfg.sensor
    )
    env.frame.copy_(fresh)
    env.frame_delay_buf.copy_(fresh.unsqueeze(1).expand_as(env.frame_delay_buf))


@torch.no_grad()
def run(n_pairs=8, steps=20, seed=20260922):
    if n_pairs <= 0 or steps <= 0:
        raise ValueError("n_pairs and steps must be positive")
    torch.manual_seed(seed)
    cfg = load_config()
    cfg.sim.n_envs = 2 * n_pairs
    cfg.sim.device = "cpu"
    cfg.curriculum.enabled = False
    cfg.task.info_gate_enabled = True
    # Structural check first: remove observation/render noise, keep paired
    # task sensor biases and the real dynamics path.
    cfg.sensor.gyro_noise = 0.0
    cfg.sensor.acc_noise = 0.0
    cfg.sensor.gdir_noise = 0.0
    cfg.sensor.vel_noise = 0.0
    cfg.sensor.z_noise = 0.0
    cfg.sensor.vel_bias_sigma = 0.0
    cfg.sensor.z_bias_sigma = 0.0

    env = GapEnv(cfg, "cpu", difficulty=1.0)
    bank = scene.paired_information_tasks(n_pairs, cfg, 1.0, "cpu")
    env._reset_envs(torch.arange(env.n), tasks=bank)
    env.task["dyn"]["wind_steady"].zero_()
    env.task["dyn"]["gust_sigma"].zero_()
    env.task["vis"]["px_noise"].zero_()
    env.state["wind"].zero_()
    env.v_bias[1::2].copy_(env.v_bias[0::2])
    env.z_bias[1::2].copy_(env.z_bias[0::2])

    st = env.state
    st["p"][:, 0] = env.task["wall_x"] - cfg.task.info_probe_distance - .20
    st["p"][:, 1] = env.task["gap_cy"]
    st["p"][:, 2] = env.task["gap_cz"]
    for k in ("v", "w", "wind", "spec_force"):
        st[k].zero_()
    st["q"].zero_()
    st["q"][:, 0] = 1.0
    st["thrust"] = env.task["dyn"]["mass"] * dynamics.G
    _sync_frame_queue(env)
    far = env.observe()
    far_vec_delta = _pair_max_delta(far["vec"])
    far_img_delta = _pair_max_delta(far["img"])

    # Move both pair members to the same fully active information-zone state.
    st["p"][:, 0] = (
        env.task["wall_x"] - cfg.task.info_probe_distance
        + cfg.task.info_probe_ramp + .05
    )
    st["v"].zero_()
    st["w"].zero_()
    st["spec_force"].zero_()
    _sync_frame_queue(env)

    action = torch.zeros(env.n, 4)
    action[:, 0] = dynamics.hover_thrust_action(env.task["dyn"])
    env.prev_action.copy_(action)
    env.delay_buf.copy_(action.unsqueeze(1).expand_as(env.delay_buf))

    obs = env.observe()
    min_clear = float("inf")
    for _ in range(steps):
        obs, _, done, info = env.step(action)
        min_clear = min(min_clear, float(info["clearance"].min()))
        if done.any() or info["collision"].any():
            raise RuntimeError("sensor-information probe terminated/contacted")
    near_vec_delta = _pair_max_delta(obs["vec"])
    # v_body_y normalized channel: gyro 0:3, acc 3:6, gravity 6:9,
    # body velocity 9:12 -> y is index 10.
    near_vbody_y_pair_delta = float(
        (obs["vec"][0::2, 10] - obs["vec"][1::2, 10]).abs().min().item()
    )

    return {
        "scope": "actor_sensor_structural_observability_NOT_noise_robust_learning_evidence",
        "n_pairs": n_pairs,
        "steps": steps,
        "far_obs_vec_pair_max_abs_delta": far_vec_delta,
        "far_ego_rgb_pair_max_abs_delta": far_img_delta,
        "near_obs_vec_pair_max_abs_delta": near_vec_delta,
        "near_vbody_y_pair_min_abs_delta": near_vbody_y_pair_delta,
        "min_clearance_m": min_clear,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", type=int, default=8)
    p.add_argument("--steps", type=int, default=20)
    a = p.parse_args()
    print(json.dumps(run(a.pairs, a.steps), indent=2))


if __name__ == "__main__":
    main()
