"""Privileged attempt executor used for simulation data generation.

This is NOT the learned planner and NOT a deployable safety shield. It is a
controlled low-level executor that maps an AttemptSpec to a real GapEnv rollout
so the contextual world model can be trained on honest action/outcome pairs.

Each environment row receives one AttemptSpec. Rows may share the same task
when generating several candidate attempts for one hidden physical condition.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import torch

from .outcome import AttemptOutcome
from .spec import AttemptSpec
from ..sim import collision, dynamics, render
from ..sim.maths import body_z_world, quat_from_yaw, quat_rotate_inv


def _action_from_accel(env, accel):
    st = env.state
    accel = accel.clamp(-2.0, 2.0)
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


def _position_feedback(env, target):
    st = env.state
    accel = 2.2 * (target - st["p"]) - 2.8 * st["v"]
    return _action_from_accel(env, accel)


def _ready(state, target, pos_tol=.08, speed_tol=.08, rate_tol=.12):
    return (
        (state["p"] - target).norm(dim=-1) < pos_tol
    ) & (state["v"].norm(dim=-1) < speed_tol) & (
        state["w"].norm(dim=-1) < rate_tol
    )


def _spec_tensor(specs: Sequence[AttemptSpec], name: str, device):
    return torch.tensor(
        [float(getattr(s, name)) for s in specs],
        dtype=torch.float32,
        device=device,
    )


def _sync_initial_state(env, specs: Sequence[AttemptSpec], start_x):
    """Place each row on its candidate retry-side start pose and sync caches."""
    st, cfg = env.state, env.cfg
    sy = _spec_tensor(specs, "start_y_offset", env.dev)
    sz = _spec_tensor(specs, "start_z_offset", env.dev)
    yaw = _spec_tensor(specs, "entry_yaw", env.dev)

    home = torch.stack(
        [
            start_x,
            env.task["gap_cy"] + sy,
            env.task["gap_cz"] + sz,
        ],
        dim=-1,
    )
    st["p"].copy_(home)
    for key in ("v", "w", "wind", "spec_force"):
        st[key].zero_()
    st["q"].copy_(quat_from_yaw(yaw))
    st["thrust"] = env.task["dyn"]["mass"] * dynamics.G

    env.t_step.zero_()
    env.attempts.zero_()
    env.in_attempt.zero_()
    env.attempt_depth.zero_()
    env.retry_dwell.zero_()
    env.ep_min_clear.fill_(10.0)
    env.prev_x.copy_(st["p"][:, 0])
    env.prev_dist.copy_((st["p"] - env._target()).norm(dim=-1))

    fresh = render.render(st["p"], st["q"], env.task, env.rays, cfg.sensor)
    env.frame.copy_(fresh)
    env.frame_delay_buf.copy_(fresh.unsqueeze(1).expand_as(env.frame_delay_buf))
    env.frame_buf_ptr = 0

    hover = _position_feedback(env, home)
    env.prev_action.copy_(hover)
    env.delay_buf.copy_(hover.unsqueeze(1).expand_as(env.delay_buf))
    env.buf_ptr = 0
    return home


@torch.no_grad()
def execute_attempt_batch(
    env,
    specs: Sequence[AttemptSpec],
    *,
    abort_clearance: float = 0.04,
    settle_steps: int = 12,
    max_approach_steps: int = 360,
    max_retreat_steps: int = 320,
) -> list[AttemptOutcome]:
    """Execute one candidate attempt per environment row.

    The executor follows a straight, progressively centered approach. Forward
    speed is shaped by accel_early before the information-zone split and
    accel_late afterwards, capped by entry_speed. If measured simulator
    clearance reaches abort_clearance before success/contact, the row switches
    to a feedback retreat and is labelled recovered only after it settles behind
    the real retry plane.

    Collisions are allowed as simulation labels; they are not intended as a
    real-flight data-collection mechanism.
    """
    if len(specs) != env.n:
        raise ValueError(f"expected {env.n} specs, got {len(specs)}")
    if abort_clearance <= 0 or settle_steps <= 0:
        raise ValueError("abort_clearance and settle_steps must be positive")

    cfg, st, dev = env.cfg, env.state, env.dev
    n = env.n
    start_x = torch.full(
        (n,), float(cfg.sim.retry_x - 0.25), dtype=torch.float32, device=dev
    )
    home = _sync_initial_state(env, specs, start_x)

    entry_speed = _spec_tensor(specs, "entry_speed", dev)
    accel_early = _spec_tensor(specs, "accel_early", dev)
    accel_late = _spec_tensor(specs, "accel_late", dev)
    v_cmd = torch.minimum(entry_speed, torch.full_like(entry_speed, 0.08))

    if getattr(cfg.task, "info_gate_enabled", False):
        split_x = (
            env.task["wall_x"] - float(cfg.task.info_probe_distance)
            + float(cfg.task.info_probe_ramp)
        )
    else:
        split_x = env.task["wall_x"] - 0.50
    center_x = env.task["wall_x"] - 0.12

    phase = torch.zeros(n, dtype=torch.long, device=dev)  # 0 approach, 1 retreat, 2 done
    aborted = torch.zeros(n, dtype=torch.bool, device=dev)
    success = torch.zeros(n, dtype=torch.bool, device=dev)
    contact = torch.zeros(n, dtype=torch.bool, device=dev)
    recovered = torch.zeros(n, dtype=torch.bool, device=dev)
    dwell = torch.zeros(n, dtype=torch.long, device=dev)
    steps = torch.zeros(n, dtype=torch.long, device=dev)

    min_clear = torch.full((n,), float("inf"), device=dev)
    max_tilt = torch.zeros(n, device=dev)
    peak_x = st["p"][:, 0].clone()
    y_at_peak = st["p"][:, 1].clone()
    z_at_peak = st["p"][:, 2].clone()
    abort_x = torch.full((n,), float("nan"), device=dev)
    max_x_after_abort = torch.full((n,), -float("inf"), device=dev)
    terminal_speed = torch.zeros(n, device=dev)

    max_total = max_approach_steps + max_retreat_steps
    for _ in range(max_total):
        active = phase < 2
        if not active.any():
            break
        steps[active] += 1

        clear_pre = collision.clearance(
            st["p"], st["q"], env.task, env.bpts,
            cfg.sim.arena_y, cfg.sim.arena_z,
        )
        min_clear[active] = torch.minimum(min_clear[active], clear_pre[active])

        tilt = torch.acos(body_z_world(st["q"])[:, 2].clamp(-1.0, 1.0))
        max_tilt[active] = torch.maximum(max_tilt[active], tilt[active])

        approaching = phase == 0
        new_peak = approaching & (st["p"][:, 0] > peak_x)
        peak_x[new_peak] = st["p"][new_peak, 0]
        y_at_peak[new_peak] = st["p"][new_peak, 1]
        z_at_peak[new_peak] = st["p"][new_peak, 2]

        should_abort = approaching & (clear_pre <= abort_clearance)
        timeout_abort = approaching & (steps >= max_approach_steps)
        switch = should_abort | timeout_abort
        if switch.any():
            aborted[switch] = True
            abort_x[switch] = st["p"][switch, 0]
            max_x_after_abort[switch] = st["p"][switch, 0]
            phase[switch] = 1
            dwell[switch] = 0

        approaching = phase == 0
        retreating = phase == 1

        # Candidate forward-speed profile.
        accel = torch.where(st["p"][:, 0] < split_x, accel_early, accel_late)
        v_cmd = (v_cmd + accel * float(cfg.sim.dt_ctrl)).clamp_min(0.05)
        v_cmd = torch.minimum(v_cmd, entry_speed)

        denom = (center_x - start_x).clamp_min(0.1)
        alpha = ((st["p"][:, 0] - start_x) / denom).clamp(0.0, 1.0)
        target_y = (1.0 - alpha) * home[:, 1] + alpha * env.task["gap_cy"]
        target_z = (1.0 - alpha) * home[:, 2] + alpha * env.task["gap_cz"]

        accel_world = torch.zeros(n, 3, device=dev)
        accel_world[:, 0] = 2.6 * (v_cmd - st["v"][:, 0])
        accel_world[:, 1] = (
            2.2 * (target_y - st["p"][:, 1]) - 2.6 * st["v"][:, 1]
        )
        accel_world[:, 2] = (
            2.2 * (target_z - st["p"][:, 2]) - 2.6 * st["v"][:, 2]
        )
        action_approach = _action_from_accel(env, accel_world)
        action_retreat = _position_feedback(env, home)

        action = action_retreat.clone()
        action[approaching] = action_approach[approaching]

        speed_pre = st["v"].norm(dim=-1).clone()
        _, _, done, info = env.step(action)
        min_clear[active] = torch.minimum(min_clear[active], info["clearance"][active])

        succ_now = active & info["success"]
        contact_now = active & info["collision"]
        other_done = active & done & ~succ_now & ~contact_now

        success[succ_now] = True
        contact[contact_now] = True
        terminal_speed[succ_now | contact_now | other_done] = speed_pre[
            succ_now | contact_now | other_done
        ]
        phase[succ_now | contact_now | other_done] = 2

        # Only live rows have meaningful post-step state; done rows were auto-reset.
        live_retreat = (phase == 1) & ~done
        if live_retreat.any():
            max_x_after_abort[live_retreat] = torch.maximum(
                max_x_after_abort[live_retreat], st["p"][live_retreat, 0]
            )
            ready = live_retreat & _ready(st, home)
            retry_state = (
                ready
                & (st["p"][:, 0] < cfg.sim.retry_x)
                & (~env.in_attempt)
                & (env.attempts >= 1)
            )
            dwell[live_retreat] = torch.where(
                retry_state[live_retreat],
                dwell[live_retreat] + 1,
                torch.zeros_like(dwell[live_retreat]),
            )
            rec_now = live_retreat & (dwell >= settle_steps)
            if rec_now.any():
                recovered[rec_now] = True
                terminal_speed[rec_now] = st["v"][rec_now].norm(dim=-1)
                phase[rec_now] = 2

        # Retreat rows that fail to settle within their budget are terminal failures.
        retreat_timeout = (phase == 1) & (steps >= max_total)
        if retreat_timeout.any():
            terminal_speed[retreat_timeout] = st["v"][retreat_timeout].norm(dim=-1)
            phase[retreat_timeout] = 2

    # Rows that somehow survive the global budget are marked terminal failures.
    unfinished = phase < 2
    if unfinished.any():
        terminal_speed[unfinished] = st["v"][unfinished].norm(dim=-1)
        phase[unfinished] = 2

    lateral = y_at_peak - env.task["gap_cy"]
    vertical = z_at_peak - env.task["gap_cz"]
    stop_dist = torch.zeros(n, device=dev)
    valid_abort = aborted & torch.isfinite(max_x_after_abort)
    stop_dist[valid_abort] = (
        max_x_after_abort[valid_abort] - abort_x[valid_abort]
    ).clamp_min(0.0)

    outcomes = []
    for i in range(n):
        mc = float(min_clear[i].item())
        if not math.isfinite(mc):
            mc = 10.0
        outcomes.append(
            AttemptOutcome(
                success=bool(success[i].item()),
                recovered=bool(recovered[i].item()),
                contact=bool(contact[i].item()),
                aborted=bool(aborted[i].item()),
                lateral_drift=float(lateral[i].item()),
                vertical_drift=float(vertical[i].item()),
                min_clearance=mc,
                stopping_distance=float(stop_dist[i].item()),
                max_tilt=float(max_tilt[i].item()),
                terminal_speed=float(terminal_speed[i].item()),
                abort_x=float(abort_x[i].item()),
                steps=max(1, int(steps[i].item())),
            )
        )
    return outcomes
