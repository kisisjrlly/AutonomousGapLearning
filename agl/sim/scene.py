"""Gap-task distribution: geometry, dynamics, visual randomization, feasibility label.

A task instance = one wall with a rotated rectangular gap + hidden dynamics params
+ visual appearance. The feasibility label (metadata only, never observed) uses a
quasi-static fit test: does the drone cross-section rectangle, rolled by any
beta <= feas_roll_max, fit in the gap with margin?
"""
import math
import torch


def _u(n, lo, hi, device, gen):
    return lo + (hi - lo) * torch.rand(n, device=device, generator=gen)


def lerp(a, b, t):
    return a + (b - a) * t


def feasibility(width, height, body_r, body_hh, margin, roll_max_deg, device):
    """Vectorized rotated-rectangle-in-rectangle fit. Returns (feasible, geo_margin)."""
    wd, hd = 2.0 * body_r, 2.0 * body_hh
    betas = torch.linspace(0.0, math.radians(roll_max_deg), 66, device=device)
    c, s = betas.cos(), betas.sin()
    w_req = wd * c + hd * s          # (B,)
    h_req = wd * s + hd * c
    slack_w = (width.unsqueeze(-1) - w_req) * 0.5
    slack_h = (height.unsqueeze(-1) - h_req) * 0.5
    geo_margin = torch.minimum(slack_w, slack_h).max(dim=-1).values
    return geo_margin >= margin, geo_margin


def sample_tasks(n: int, cfg, difficulty: float, device, gen=None) -> dict:
    t, lam = cfg.task, difficulty
    d = device
    width = _u(n, lerp(t.width_lo_easy, t.width_lo, lam), t.width_hi, d, gen)
    inf_mask = torch.rand(n, device=d, generator=gen) < (t.infeasible_frac * lam)
    width = torch.where(inf_mask, _u(n, t.infeasible_w_lo, t.infeasible_w_hi, d, gen), width)
    height = _u(n, t.height_lo, t.height_hi, d, gen)
    roll_max = math.radians(lerp(t.roll_max_deg_easy, t.roll_max_deg, lam))
    task = {
        "gap_w": width,
        "gap_h": height,
        "gap_roll": _u(n, -roll_max, roll_max, d, gen),
        "wall_x": _u(n, t.wall_x_lo, t.wall_x_hi, d, gen),
        "thick": _u(n, t.thick_lo, t.thick_hi, d, gen),
        "gap_cy": _u(n, -t.gap_cy, t.gap_cy, d, gen),
        "gap_cz": _u(n, t.gap_cz_lo, t.gap_cz_hi, d, gen),
    }
    feas, gm = feasibility(width, height, cfg.sim.body_r, cfg.sim.body_hh,
                           t.feas_margin, t.feas_roll_max_deg, d)
    task["feasible"] = feas
    task["geo_margin"] = gm
    # hidden dynamics
    mass = _u(n, t.mass_lo, t.mass_hi, d, gen)
    task["dyn"] = {
        "mass": mass,
        "tmax": mass * 9.81 * _u(n, t.twr_lo, t.twr_hi, d, gen),
        "tau_thrust": _u(n, t.tau_thrust_lo, t.tau_thrust_hi, d, gen),
        "tau_rate": _u(n, t.tau_rate_lo, t.tau_rate_hi, d, gen),
        "alpha_max": _u(n, t.alpha_max_lo, t.alpha_max_hi, d, gen),
        "kd_lin": _u(n, t.kd_lin_lo, t.kd_lin_hi, d, gen),
        "kd_quad": _u(n, t.kd_quad_lo, t.kd_quad_hi, d, gen),
        "wind_tau": t.wind_tau,
        "gust_sigma": _u(n, 0.0, t.wind_gust_sigma * lam, d, gen),
        "delay": torch.randint(0, t.delay_max_steps + 1, (n,), device=d, generator=gen),
    }
    wind_mag = _u(n, 0.0, t.wind_max * lam, d, gen)
    wind_dir = _u(n, 0.0, 2 * math.pi, d, gen)
    task["dyn"]["wind_steady"] = torch.stack(
        [wind_mag * wind_dir.cos(), wind_mag * wind_dir.sin(), torch.zeros(n, device=d)], dim=-1)
    # visual randomization
    task["vis"] = {
        "wall_alb": _u(n, 0.15, 0.85, d, gen).unsqueeze(-1) * torch.ones(1, 3, device=d)
        + 0.10 * (torch.rand(n, 3, device=d, generator=gen) - 0.5),
        "inner_scale": _u(n, 0.5, 0.9, d, gen),
        "ground_a": 0.1 + 0.5 * torch.rand(n, 3, device=d, generator=gen),
        "ground_b": 0.1 + 0.5 * torch.rand(n, 3, device=d, generator=gen),
        "sky_top": 0.3 + 0.6 * torch.rand(n, 3, device=d, generator=gen),
        "sky_hor": 0.3 + 0.6 * torch.rand(n, 3, device=d, generator=gen),
        "stripe_amp": _u(n, 0.0, 0.35, d, gen),
        "stripe_freq": _u(n, 2.0, 12.0, d, gen),
        "stripe_phase": _u(n, 0.0, 6.283, d, gen),
        "light": _u(n, 0.6, 1.3, d, gen),
        "checker": _u(n, 0.5, 1.5, d, gen),
        "px_noise": _u(n, cfg.sensor.px_noise_lo, cfg.sensor.px_noise_hi, d, gen),
    }
    # sensor biases (per-episode constants; OU biases live in env)
    task["sens"] = {
        "gyro_bias": cfg.sensor.gyro_bias * torch.randn(n, 3, device=d, generator=gen),
        "acc_bias": cfg.sensor.acc_bias * torch.randn(n, 3, device=d, generator=gen),
    }
    return task


def scatter_tasks(dst: dict, src: dict, idx: torch.Tensor):
    """Write src task fields (sampled for idx.numel() envs) into dst at idx."""
    for k, v in src.items():
        if isinstance(v, dict):
            scatter_tasks(dst[k], v, idx)
        elif torch.is_tensor(v):
            dst[k][idx] = v
