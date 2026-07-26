"""Batched quadrotor rigid-body dynamics with actuator lag and simplified rate loop.

Action interface (CTBR): normalized [-1,1]^4 -> (thrust fraction u in [0,1],
body-rate commands in rad/s). The inner rate loop is modeled as a first-order
response with per-env time constant and angular-acceleration limit; thrust has
first-order motor lag. Mass/inertia/latency effects are hidden from the policy.
"""
import torch

from .maths import quat_rotate, quat_rotate_inv, quat_integrate, body_z_world

G = 9.81
OMEGA_MAX = torch.tensor([6.0, 6.0, 3.0])  # rad/s command range (roll, pitch, yaw)


def make_state(n: int, device) -> dict:
    return {
        "p": torch.zeros(n, 3, device=device),
        "v": torch.zeros(n, 3, device=device),
        "q": torch.cat([torch.ones(n, 1, device=device),
                        torch.zeros(n, 3, device=device)], dim=-1),
        "w": torch.zeros(n, 3, device=device),          # body rates
        "thrust": torch.zeros(n, device=device),        # actual thrust (N)
        "wind": torch.zeros(n, 3, device=device),       # OU gust component
        "spec_force": torch.zeros(n, 3, device=device), # body-frame accelerometer
    }


def action_to_cmd(a: torch.Tensor, dyn: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalized action -> (thrust command N, rate command rad/s)."""
    a = a.clamp(-1.0, 1.0)
    u = 0.5 * (a[:, 0] + 1.0)
    t_cmd = u * dyn["tmax"]
    w_cmd = a[:, 1:4] * OMEGA_MAX.to(a.device)
    return t_cmd, w_cmd


def step(state: dict, t_cmd: torch.Tensor, w_cmd: torch.Tensor, dyn: dict,
         dt: float, substeps: int, gen: torch.Generator | None = None):
    """Advance physics by one control period (substeps * dt_phys)."""
    dtp = dt / substeps
    g_vec = torch.tensor([0.0, 0.0, -G], device=t_cmd.device)
    m = dyn["mass"]
    for _ in range(substeps):
        # actuators
        state["thrust"] += dtp * (t_cmd - state["thrust"]) / dyn["tau_thrust"]
        state["thrust"] = state["thrust"].clamp(torch.zeros_like(dyn["tmax"]), dyn["tmax"])
        alpha = (w_cmd - state["w"]) / dyn["tau_rate"].unsqueeze(-1)
        amax = dyn["alpha_max"].unsqueeze(-1)
        state["w"] += dtp * alpha.clamp(-amax, amax)
        # attitude
        state["q"] = quat_integrate(state["q"], state["w"], dtp)
        # translational
        f_world = body_z_world(state["q"]) * state["thrust"].unsqueeze(-1)
        v_air = state["v"] - (dyn["wind_steady"] + state["wind"])
        drag = dyn["kd_lin"].unsqueeze(-1) * v_air \
            + dyn["kd_quad"].unsqueeze(-1) * v_air.norm(dim=-1, keepdim=True) * v_air
        acc = g_vec + f_world / m.unsqueeze(-1) - drag
        state["v"] += dtp * acc
        state["p"] += dtp * state["v"]
    # accelerometer specific force (world acc minus gravity, in body frame)
    state["spec_force"] = quat_rotate_inv(state["q"], acc - g_vec)
    # OU gust process at control rate
    n = t_cmd.shape[0]
    noise = torch.randn(n, 3, device=t_cmd.device, generator=gen)
    tau_w = dyn["wind_tau"]
    sig = dyn["gust_sigma"].unsqueeze(-1) * torch.tensor([1.0, 1.0, 0.3], device=t_cmd.device)
    state["wind"] += dt * (-state["wind"] / tau_w) + (dt ** 0.5) * sig * noise
    return state


def hover_thrust_action(dyn: dict) -> torch.Tensor:
    """Normalized thrust action that hovers (for state init only)."""
    u = dyn["mass"] * G / dyn["tmax"]
    return (2.0 * u - 1.0).clamp(-1.0, 1.0)
