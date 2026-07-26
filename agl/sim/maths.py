"""Batched quaternion / rotation utilities. Quaternions are (w,x,y,z), body->world."""
import torch


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate body-frame v into world frame. q: (...,4), v: (...,3)."""
    w, u = q[..., :1], q[..., 1:]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def quat_rotate_inv(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate world-frame v into body frame."""
    w, u = q[..., :1], -q[..., 1:]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def quat_integrate(q: torch.Tensor, omega_body: torch.Tensor, dt: float) -> torch.Tensor:
    """Exact exponential-map integration of body rates."""
    theta = omega_body.norm(dim=-1, keepdim=True) * dt
    half = 0.5 * theta
    # small-angle-safe axis
    axis = omega_body / omega_body.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    dq = torch.cat([torch.cos(half), axis * torch.sin(half)], dim=-1)
    out = quat_mul(q, dq)
    return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    half = 0.5 * yaw
    z = torch.zeros_like(yaw)
    return torch.stack([torch.cos(half), z, z, torch.sin(half)], dim=-1)


def body_z_world(q: torch.Tensor) -> torch.Tensor:
    """Thrust axis in world frame (third column of R)."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([2 * (x * z + w * y), 2 * (y * z - w * x),
                        1 - 2 * (x * x + y * y)], dim=-1)
