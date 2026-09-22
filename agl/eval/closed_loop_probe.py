"""Privileged dynamic-braking recovery baseline for GapEnv.

This is an offline simulation control baseline, NOT learned adaptation and NOT a
deployable safety shield. It deliberately uses ground-truth p/v/q/w, true mass
and thrust calibration, and known gap/wall position in order to answer a narrow
engineering question first:

Can the 6-DoF simulator execute a non-zero-speed near-gap approach, brake before
contact, retreat, settle, and save a deterministic recovery snapshot?

Only accepted rollouts produce recovery.pt. The policy-facing RGB/observation
vector are recorded alongside the privileged state for audit in the Viewer.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..config import load_config
from ..sim import collision, dynamics, render, scene
from ..sim.env import GapEnv
from ..sim.maths import body_z_world, quat_rotate_inv


PHASE = {0: "approach_probe", 1: "brake", 2: "retreat", 3: "settled"}
OBS_VEC_LABELS = (
    "gyro_x_norm", "gyro_y_norm", "gyro_z_norm",
    "acc_x_norm", "acc_y_norm", "acc_z_norm",
    "gravity_x", "gravity_y", "gravity_z",
    "vbody_x_norm", "vbody_y_norm", "vbody_z_norm",
    "z_norm",
    "prev_thrust_action", "prev_roll_rate_action",
    "prev_pitch_rate_action", "prev_yaw_rate_action",
    "thrust_fraction",
)


def _action_from_accel(env, accel):
    """Convert desired world acceleration to the existing CTBR action interface."""
    st = env.state
    accel = accel.clamp(-1.8, 1.8)
    force = accel.clone()
    force[:, 2] += dynamics.G
    desired_z = force / force.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    axis_error = torch.cross(body_z_world(st["q"]), desired_z, dim=-1)
    rates = 5.0 * quat_rotate_inv(st["q"], axis_error) - 0.30 * st["w"]
    action = torch.zeros(env.n, 4, device=env.dev)
    action[:, 0] = (
        2.0 * force.norm(dim=-1) * env.task["dyn"]["mass"]
        / env.task["dyn"]["tmax"] - 1.0
    )
    action[:, 1:] = rates / dynamics.OMEGA_MAX.to(env.dev)
    return action.clamp(-1.0, 1.0)


def position_feedback(env, target):
    """Ground-truth position/velocity feedback used for brake/retreat settling."""
    st = env.state
    accel = 2.2 * (target - st["p"]) - 2.8 * st["v"]
    return _action_from_accel(env, accel)


def velocity_approach(env, speed, yz_target):
    """Track positive x velocity while stabilizing lateral/vertical position."""
    st = env.state
    accel = torch.zeros_like(st["p"])
    accel[:, 0] = 2.6 * (float(speed) - st["v"][:, 0])
    accel[:, 1:] = 2.0 * (yz_target - st["p"][:, 1:]) - 2.5 * st["v"][:, 1:]
    return _action_from_accel(env, accel)


def ready(state, target, pos_tol=.08, speed_tol=.08, rate_tol=.12):
    return (
        (state["p"] - target).norm(dim=-1) < pos_tol
    ) & (state["v"].norm(dim=-1) < speed_tol) & (
        state["w"].norm(dim=-1) < rate_tol
    )


def _sync_controlled_initial_state(env, home):
    """Synchronize caches, camera latency buffers and runtime paired biases."""
    st, cfg = env.state, env.cfg
    env.prev_x.copy_(st["p"][:, 0])
    env.prev_dist.copy_((st["p"] - env._target()).norm(dim=-1))
    # Runtime OU biases are environment state, not sampled task fields. Match
    # them within each latent pair so the pair differs only by hidden wind.
    env.v_bias[1::2].copy_(env.v_bias[0::2])
    env.z_bias[1::2].copy_(env.z_bias[0::2])
    fresh = render.render(st["p"], st["q"], env.task, env.rays, cfg.sensor)
    env.frame.copy_(fresh)
    env.frame_delay_buf.copy_(fresh.unsqueeze(1).expand_as(env.frame_delay_buf))
    hover = position_feedback(env, home)
    env.prev_action.copy_(hover)
    env.delay_buf.copy_(hover.unsqueeze(1).expand_as(env.delay_buf))
    return env.observe()


def _phase_names(phase):
    return np.asarray([PHASE[int(x)] for x in phase.cpu().tolist()])


@torch.no_grad()
def run(
    n_pairs=2,
    seed=0,
    phase_limit=360,
    dwell_steps=12,
    approach_speed=.45,
    probe_wall_distance=.60,
    clearance_margin=.18,
    min_brake_entry_speed=.25,
    out_dir=None,
):
    """Run the privileged dynamic-braking protocol.

    The brake trigger is a fixed center-to-wall distance, not a safety proof.
    A rollout is accepted only if every environment:
      1) reaches the trigger with non-zero forward speed,
      2) brakes and settles without crossing the clearance margin,
      3) retreats to home and settles,
      4) never terminates, contacts, or becomes non-finite.
    """
    if min(n_pairs, phase_limit, dwell_steps) <= 0:
        raise ValueError("counts must be positive")
    if approach_speed <= 0 or probe_wall_distance <= 0 or clearance_margin <= 0:
        raise ValueError("approach/safety parameters must be positive")
    dest = Path(out_dir) if out_dir else None
    if dest:
        dest.mkdir(parents=True, exist_ok=False)

    torch.manual_seed(seed)
    cfg = load_config()
    cfg.sim.device = "cpu"
    cfg.sim.n_envs = 2 * n_pairs
    cfg.sim.ep_len = 3 * phase_limit + 64
    cfg.task.info_gate_enabled = True
    cfg.curriculum.enabled = False

    env = GapEnv(cfg, "cpu", difficulty=1.0)
    env._reset_envs(
        torch.arange(env.n),
        tasks=scene.paired_information_tasks(n_pairs, cfg, 1.0, "cpu"),
    )
    # First validate the controlled recovery mechanism without unrelated gusts.
    env.task["dyn"]["wind_steady"].zero_()
    env.task["dyn"]["gust_sigma"].zero_()
    env.state["wind"].zero_()

    st = env.state
    home = torch.stack(
        [
            env.task["wall_x"] - cfg.task.info_probe_distance - .30,
            env.task["gap_cy"],
            env.task["gap_cz"],
        ],
        dim=-1,
    )
    st["p"].copy_(home)
    for k in ("v", "w", "wind", "spec_force"):
        st[k].zero_()
    st["q"].zero_()
    st["q"][:, 0] = 1.0
    st["thrust"] = env.task["dyn"]["mass"] * dynamics.G
    obs = _sync_controlled_initial_state(env, home)

    trigger_x = env.task["wall_x"] - float(probe_wall_distance)
    brake_target = torch.stack(
        [trigger_x, env.task["gap_cy"], env.task["gap_cz"]], dim=-1
    )
    yz_target = brake_target[:, 1:].clone()

    phase = torch.zeros(env.n, dtype=torch.long)
    dwell = torch.zeros(env.n, dtype=torch.long)
    phase_steps = torch.zeros(env.n, dtype=torch.long)
    brake_start_x = torch.full((env.n,), float("nan"))
    brake_entry_speed = torch.full((env.n,), float("nan"))
    brake_peak_x = torch.full((env.n,), -float("inf"))
    phase_peak_speed = torch.zeros(env.n, 4)
    phase_peak_rate = torch.zeros(env.n, 4)

    trace = {
        k: [] for k in (
            "p", "q", "v", "act", "clear_pre", "clear", "collision", "done",
            "attempt_id", "phase", "obs_vec", "frames",
        )
    }
    task_data = {
        f"task_{k}": v.numpy().copy()
        for k, v in env.task.items() if torch.is_tensor(v)
    }
    task_data["task_probe_wind"] = env.task["dyn"]["probe_wind"].numpy().copy()

    reason = None
    minimum = float("inf")
    closest_wall_distance = float("inf")
    max_steps = 3 * phase_limit + 64

    for _ in range(max_steps):
        # Promote approach -> brake before computing the next command.
        entering = (phase == 0) & (st["p"][:, 0] >= trigger_x)
        if entering.any():
            brake_start_x[entering] = st["p"][entering, 0]
            brake_entry_speed[entering] = st["v"][entering, 0]
            if (brake_entry_speed[entering] < min_brake_entry_speed).any():
                reason = "insufficient_brake_entry_speed"
                break
            phase[entering] = 1
            dwell[entering] = 0
            phase_steps[entering] = 0

        active = phase < 3
        phase_steps[active] += 1
        if (phase_steps[active] > phase_limit).any():
            reason = "phase_timeout"
            break

        clear_pre = collision.clearance(
            st["p"], st["q"], env.task, env.bpts,
            cfg.sim.arena_y, cfg.sim.arena_z,
        )
        if not all(torch.isfinite(v).all() for v in st.values()) or not torch.isfinite(clear_pre).all():
            reason = "nonfinite_state"
            break
        if (clear_pre <= clearance_margin).any():
            reason = "clearance_margin"
            break

        action_approach = velocity_approach(env, approach_speed, yz_target)
        action_brake = position_feedback(env, brake_target)
        action_retreat = position_feedback(env, home)
        action = action_retreat.clone()
        action[phase == 0] = action_approach[phase == 0]
        action[phase == 1] = action_brake[phase == 1]
        action[phase == 2] = action_retreat[phase == 2]
        action[phase == 3] = action_retreat[phase == 3]

        for key in ("p", "q", "v"):
            trace[key].append(st[key].numpy().copy())
        trace["act"].append(action.numpy().copy())
        trace["clear_pre"].append(clear_pre.numpy().copy())
        trace["phase"].append(_phase_names(phase))
        trace["obs_vec"].append(obs["vec"].cpu().numpy().copy())
        trace["frames"].append(
            (obs["img"].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy().copy()
        )

        obs, _, done, info = env.step(action)
        for key, value in (
            ("clear", info["clearance"]),
            ("collision", info["collision"]),
            ("done", done),
            ("attempt_id", info["attempt_id"]),
        ):
            trace[key].append(value.cpu().numpy().copy())

        minimum = min(minimum, float(info["clearance"].min()))
        wall_distance = env.task["wall_x"] - st["p"][:, 0]
        closest_wall_distance = min(
            closest_wall_distance, float(wall_distance.min().item())
        )

        for code in range(4):
            mask = phase == code
            if mask.any():
                phase_peak_speed[mask, code] = torch.maximum(
                    phase_peak_speed[mask, code], st["v"][mask].norm(dim=-1)
                )
                phase_peak_rate[mask, code] = torch.maximum(
                    phase_peak_rate[mask, code], st["w"][mask].norm(dim=-1)
                )
        braking = phase == 1
        if braking.any():
            brake_peak_x[braking] = torch.maximum(
                brake_peak_x[braking], st["p"][braking, 0]
            )

        if done.any() or info["collision"].any():
            reason = "terminal_or_contact"
            break
        if not all(torch.isfinite(v).all() for v in st.values()) or not torch.isfinite(info["clearance"]).all():
            reason = "nonfinite_state"
            break
        if (info["clearance"] <= clearance_margin).any():
            reason = "clearance_margin"
            break

        # Brake must genuinely settle from a non-zero entry speed.
        braking = phase == 1
        brake_ready = braking & ready(st, brake_target)
        dwell[braking] = torch.where(
            brake_ready[braking], dwell[braking] + 1, torch.zeros_like(dwell[braking])
        )
        brake_done = braking & (dwell >= dwell_steps)
        if brake_done.any():
            phase[brake_done] = 2
            dwell[brake_done] = 0
            phase_steps[brake_done] = 0

        retreating = phase == 2
        retreat_ready = retreating & ready(st, home)
        dwell[retreating] = torch.where(
            retreat_ready[retreating], dwell[retreating] + 1,
            torch.zeros_like(dwell[retreating]),
        )
        recovered = retreating & (dwell >= dwell_steps)
        if recovered.any():
            phase[recovered] = 3
            phase_steps[recovered] = 0

        if (phase == 3).all():
            break
    else:
        reason = "protocol_timeout"

    if reason is None and not (phase == 3).all():
        reason = "protocol_timeout"

    valid_brake = torch.isfinite(brake_start_x) & torch.isfinite(brake_peak_x)
    stopping_distance = torch.full((env.n,), float("nan"))
    stopping_distance[valid_brake] = (
        brake_peak_x[valid_brake] - brake_start_x[valid_brake]
    ).clamp_min(0.0)

    accepted = reason is None and bool((phase == 3).all())
    summary = {
        "scope": "privileged_dynamic_braking_baseline_NOT_learned_NOT_safety_guarantee",
        "seed": seed,
        "n_envs": env.n,
        "protocol_completed": accepted,
        "failure_reason": reason,
        "acceptance": {
            "clearance_margin_m": clearance_margin,
            "position_error_m": .08,
            "terminal_speed_mps": .08,
            "terminal_body_rate_radps": .12,
            "dwell_steps": dwell_steps,
            "min_brake_entry_speed_mps": min_brake_entry_speed,
        },
        "approach_speed_command_mps": approach_speed,
        "probe_wall_distance_m": probe_wall_distance,
        "brake_entry_speed_min_mps": (
            float(torch.nan_to_num(brake_entry_speed, nan=float("inf")).min())
            if torch.isfinite(brake_entry_speed).any() else None
        ),
        "brake_entry_speed_max_mps": (
            float(torch.nan_to_num(brake_entry_speed, nan=-float("inf")).max())
            if torch.isfinite(brake_entry_speed).any() else None
        ),
        "stopping_distance_max_m": (
            float(torch.nan_to_num(stopping_distance, nan=-float("inf")).max())
            if torch.isfinite(stopping_distance).any() else None
        ),
        "closest_center_to_wall_m": (
            closest_wall_distance if np.isfinite(closest_wall_distance) else None
        ),
        "min_clearance_m": minimum if np.isfinite(minimum) else None,
        "terminal_speed_max_mps": float(st["v"].norm(dim=-1).max()),
        "terminal_body_rate_max_radps": float(st["w"].norm(dim=-1).max()),
        "phase_peak_speed_mps": {
            PHASE[i]: float(phase_peak_speed[:, i].max()) for i in range(4)
        },
        "phase_peak_body_rate_radps": {
            PHASE[i]: float(phase_peak_rate[:, i].max()) for i in range(4)
        },
        "state_feedback": "ground truth p/v/q/w; true gap pose, mass and thrust calibration",
        "safety": "pre-step clearance rejection plus post-step checks; NOT a predictive recovery shield",
    }

    if dest:
        meta = {
            "scope": summary["scope"],
            "dt_ctrl": cfg.sim.dt_ctrl,
            "seed": seed,
            "body_r": cfg.sim.body_r,
            "retry_x": cfg.sim.retry_x,
            "succ_margin": cfg.sim.succ_margin,
            "info_gate_enabled": True,
            "info_probe_distance": cfg.task.info_probe_distance,
            "info_probe_ramp": cfg.task.info_probe_ramp,
            "probe_wall_distance": probe_wall_distance,
            "approach_speed_command": approach_speed,
            "obs_vec_labels": list(OBS_VEC_LABELS),
            "state_timing": "pre_action state/obs; clear/collision/done describe following control interval",
        }
        if trace["p"]:
            np.savez_compressed(
                dest / "trajectory.npz",
                **{f"rec_{k}": np.stack(v) for k, v in trace.items()},
                **task_data,
                steps=np.full(env.n, len(trace["p"])),
                meta=json.dumps(meta),
            )
        if accepted:
            torch.save(
                {"env": env.snapshot(), "config": cfg, "acceptance": summary},
                dest / "recovery.pt",
            )
        (dest / "summary.json").write_text(json.dumps(summary, indent=2))

    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pairs", type=int, default=2)
    p.add_argument("--approach-speed", type=float, default=.45)
    p.add_argument("--probe-wall-distance", type=float, default=.60)
    a = p.parse_args()
    print(json.dumps(run(
        n_pairs=a.pairs,
        seed=a.seed,
        approach_speed=a.approach_speed,
        probe_wall_distance=a.probe_wall_distance,
        out_dir=a.out,
    ), indent=2))


if __name__ == "__main__":
    main()
