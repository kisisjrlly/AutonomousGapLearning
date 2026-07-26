import math

import pytest
import torch

from agl.config import load_config
from agl.sim import collision, dynamics, render, scene
from agl.sim.maths import quat_from_yaw, quat_integrate, quat_mul, quat_rotate, quat_rotate_inv

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def make_cfg(n=8):
    cfg = load_config()
    cfg.sim.n_envs = n
    cfg.sim.device = DEV
    return cfg


# ---------------------------------------------------------------- quaternions
def test_quat_rotate_roundtrip():
    torch.manual_seed(0)
    q = torch.randn(64, 4)
    q = q / q.norm(dim=-1, keepdim=True)
    v = torch.randn(64, 3)
    assert torch.allclose(quat_rotate_inv(q, quat_rotate(q, v)), v, atol=1e-5)
    assert torch.allclose(quat_rotate(q, v).norm(dim=-1), v.norm(dim=-1), atol=1e-5)


def test_quat_yaw_and_integration():
    yaw = torch.tensor([math.pi / 2])
    q = quat_from_yaw(yaw)
    v = torch.tensor([[1.0, 0.0, 0.0]])
    out = quat_rotate(q, v)
    assert torch.allclose(out, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6)
    # integrate yaw rate 2 rad/s for pi/4 s -> quarter turn
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    w = torch.tensor([[0.0, 0.0, 2.0]])
    for _ in range(100):
        q = quat_integrate(q, w, math.pi / 4 / 100)
    out = quat_rotate(q, v)
    assert torch.allclose(out, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-3)


# ------------------------------------------------------------------ dynamics
def _dyn(n, dev):
    one = torch.ones(n, device=dev)
    return {
        "mass": 0.75 * one, "tmax": 0.75 * 9.81 * 3.0 * one,
        "tau_thrust": 0.04 * one, "tau_rate": 0.05 * one,
        "alpha_max": 40.0 * one, "kd_lin": 0.0 * one, "kd_quad": 0.0 * one,
        "wind_tau": 2.0, "gust_sigma": 0.0 * one,
        "wind_steady": torch.zeros(n, 3, device=dev),
        "delay": torch.zeros(n, dtype=torch.long, device=dev),
    }


def test_hover_equilibrium():
    n = 4
    dyn = _dyn(n, DEV)
    st = dynamics.make_state(n, DEV)
    st["p"][:, 2] = 1.5
    st["thrust"] = dyn["mass"] * 9.81
    t_cmd = dyn["mass"] * 9.81
    w_cmd = torch.zeros(n, 3, device=DEV)
    for _ in range(80):
        dynamics.step(st, t_cmd, w_cmd, dyn, 0.025, 5)
    assert st["v"].norm(dim=-1).max().item() < 1e-3
    assert (st["p"][:, 2] - 1.5).abs().max().item() < 1e-3


def test_free_fall_and_rate_response():
    n = 4
    dyn = _dyn(n, DEV)
    st = dynamics.make_state(n, DEV)
    t_cmd = torch.zeros(n, device=DEV)
    w_cmd = torch.tensor([[3.0, 0.0, 0.0]], device=DEV).expand(n, 3).contiguous()
    for _ in range(40):  # 1 s
        dynamics.step(st, t_cmd, w_cmd, dyn, 0.025, 5)
    assert abs(st["v"][:, 2].mean().item() + 9.81) < 0.2      # ~ -g t
    assert (st["w"][:, 0] - 3.0).abs().max().item() < 0.05    # rate converged
    # accelerometer in free fall reads ~0 specific force
    assert st["spec_force"].norm(dim=-1).max().item() < 0.1


# ----------------------------------------------------------------- collision
def _manual_task(n, dev, wall_x=3.0, thick=0.1, w=0.5, h=0.5, cy=0.0, cz=1.5, roll=0.0):
    t = lambda v: torch.full((n,), float(v), device=dev)
    return {"wall_x": t(wall_x), "thick": t(thick), "gap_w": t(w), "gap_h": t(h),
            "gap_cy": t(cy), "gap_cz": t(cz), "gap_roll": t(roll)}


def test_clearance_geometry():
    cfg = make_cfg()
    bpts = collision.body_points(cfg.sim.body_r, cfg.sim.body_hh, DEV)
    task = _manual_task(4, DEV)
    p = torch.tensor([[1.0, 0.0, 1.5],    # far from wall: ground dominates
                      [2.5, 2.0, 1.5],    # 0.5 m before wall face, off-gap
                      [3.05, 0.0, 1.5],   # centered inside gap
                      [3.02, 2.0, 1.5]],  # inside wall solid
                     device=DEV)
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=DEV).expand(4, 4).contiguous()
    c = collision.clearance(p, q, task, bpts, cfg.sim.arena_y, cfg.sim.arena_z)
    assert abs(c[0].item() - (1.5 - cfg.sim.body_hh)) < 0.02   # ground clearance
    assert abs(c[1].item() - (0.5 - cfg.sim.body_r)) < 0.02    # wall face clearance
    # centered in 0.5 gap: lateral slack = 0.25 - 0.16 = 0.09 (vertical 0.25-0.05=0.2)
    assert abs(c[2].item() - 0.09) < 0.02
    assert c[3].item() < 0.0


def test_ground_clearance():
    cfg = make_cfg()
    bpts = collision.body_points(cfg.sim.body_r, cfg.sim.body_hh, DEV)
    task = _manual_task(1, DEV)
    p = torch.tensor([[0.0, 0.0, 0.3]], device=DEV)
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=DEV)
    c = collision.clearance(p, q, task, bpts, cfg.sim.arena_y, cfg.sim.arena_z)
    assert abs(c[0].item() - (0.3 - cfg.sim.body_hh)) < 1e-3


def test_feasibility_label():
    feas, margin = scene.feasibility(
        torch.tensor([0.50, 0.20, 0.36, 0.34]), torch.tensor([0.50, 0.50, 0.36, 0.34]),
        0.16, 0.05, 0.03, 65.0, "cpu")
    assert feas.tolist() == [True, False, True, False]
    assert margin[0] > margin[2] > 0


# -------------------------------------------------------------------- render
def test_render_gap_visible():
    cfg = make_cfg(1)
    rays = render.camera_rays(cfg.sensor, DEV)
    task = _manual_task(1, DEV, wall_x=3.0, w=0.6, h=0.6)
    vis_n = 1
    task["vis"] = {
        "wall_alb": torch.full((vis_n, 3), 0.8, device=DEV),
        "inner_scale": torch.full((vis_n,), 0.6, device=DEV),
        "ground_a": torch.full((vis_n, 3), 0.2, device=DEV),
        "ground_b": torch.full((vis_n, 3), 0.4, device=DEV),
        "sky_top": torch.full((vis_n, 3), 0.1, device=DEV),
        "sky_hor": torch.full((vis_n, 3), 0.1, device=DEV),
        "stripe_amp": torch.zeros(vis_n, device=DEV),
        "stripe_freq": torch.ones(vis_n, device=DEV),
        "stripe_phase": torch.zeros(vis_n, device=DEV),
        "light": torch.ones(vis_n, device=DEV),
        "checker": torch.ones(vis_n, device=DEV),
        "px_noise": torch.zeros(vis_n, device=DEV),
    }
    p = torch.tensor([[0.0, 0.0, 1.5]], device=DEV)
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=DEV)
    img = render.render(p, q, task, rays, cfg.sensor)
    assert img.shape == (1, 3, cfg.sensor.img_h, cfg.sensor.img_w)
    H, W = cfg.sensor.img_h, cfg.sensor.img_w
    center = img[0, :, H // 2, W // 2]          # through gap -> sky/ground behind
    off = img[0, :, H // 2, W // 2 + 8]         # wall face
    assert (center - off).abs().max().item() > 0.1
    # wall face brightness ~ albedo * fade * light
    assert off.mean().item() > 0.3


def test_render_behind_wall_looks_back():
    cfg = make_cfg(1)
    rays = render.camera_rays(cfg.sensor, DEV)
    task = _manual_task(1, DEV)
    task["vis"] = {k: v for k, v in _default_vis(1).items()}
    p = torch.tensor([[5.0, 0.0, 1.5]], device=DEV)
    yaw = torch.tensor([math.pi], device=DEV)
    q = quat_from_yaw(yaw)
    img = render.render(p, q, task, rays, cfg.sensor)
    assert torch.isfinite(img).all()


def _default_vis(n):
    return {
        "wall_alb": torch.full((n, 3), 0.8, device=DEV),
        "inner_scale": torch.full((n,), 0.6, device=DEV),
        "ground_a": torch.full((n, 3), 0.2, device=DEV),
        "ground_b": torch.full((n, 3), 0.4, device=DEV),
        "sky_top": torch.full((n, 3), 0.5, device=DEV),
        "sky_hor": torch.full((n, 3), 0.7, device=DEV),
        "stripe_amp": torch.zeros(n, device=DEV),
        "stripe_freq": torch.ones(n, device=DEV),
        "stripe_phase": torch.zeros(n, device=DEV),
        "light": torch.ones(n, device=DEV),
        "checker": torch.ones(n, device=DEV),
        "px_noise": torch.zeros(n, device=DEV),
    }
