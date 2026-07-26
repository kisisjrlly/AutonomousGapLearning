"""Clearance = min signed distance from drone body surface points to solid world.

World solid = wall slab (spanning the full arena cross-section) minus the gap
prism, plus the ground plane. SDF subtraction max(d_box, -d_prism) is exact in
the hole interior and conservative (never overestimates clearance, never
reports false contact) elsewhere.
"""
import math
import torch

from .maths import quat_rotate


def body_points(body_r: float, body_hh: float, device) -> torch.Tensor:
    """Sample points on the prop-guard cylinder surface. (P,3).

    Density requirement: adjacent-point spacing must stay below the minimum
    wall thickness (5 cm) so a thin slab cannot pass undetected between
    points at any attitude. 24-pt rims give 4.2 cm chords; cap mid-rings
    close the radial gap on the end faces for rolled traversal.
    """
    def ring(nang, r, z):
        ang = torch.arange(nang, device=device) * (2 * math.pi / nang)
        return torch.stack([r * ang.cos(), r * ang.sin(),
                            torch.full_like(ang, z)], -1)

    pts = [ring(24, body_r, z) for z in (-body_hh, 0.0, body_hh)]
    pts += [ring(12, 0.5 * body_r, z) for z in (-body_hh, body_hh)]
    pts.append(torch.tensor([[0.0, 0.0, -body_hh], [0.0, 0.0, body_hh]], device=device))
    return torch.cat(pts, dim=0)  # (98,3)


def wall_sdf(pts: torch.Tensor, task: dict, arena_y: float, arena_z: float) -> torch.Tensor:
    """SDF of wall-with-hole at world points. pts: (N,P,3) paired with task dims (N,...)."""
    wx = task["wall_x"].view(-1, 1)
    th = task["thick"].view(-1, 1)
    x, y, z = pts.unbind(-1)
    # slab box centered at (wx + th/2, 0, arena_z/2)
    qx = (x - wx - 0.5 * th).abs() - 0.5 * th
    qy = y.abs() - arena_y
    qz = (z - 0.5 * arena_z).abs() - 0.5 * arena_z
    qmax = torch.stack([qx, qy, qz], -1).clamp_min(0.0)
    d_box = qmax.norm(dim=-1) + torch.minimum(
        torch.maximum(qx, torch.maximum(qy, qz)), torch.zeros_like(qx))
    # gap prism (2D SDF in wall plane, rotated frame)
    cy = task["gap_cy"].view(-1, 1)
    cz = task["gap_cz"].view(-1, 1)
    cr = task["gap_roll"].cos().view(-1, 1)
    sr = task["gap_roll"].sin().view(-1, 1)
    dy, dz = y - cy, z - cz
    u = cr * dy + sr * dz
    v = -sr * dy + cr * dz
    hu = 0.5 * task["gap_w"].view(-1, 1)
    hv = 0.5 * task["gap_h"].view(-1, 1)
    q2u = u.abs() - hu
    q2v = v.abs() - hv
    q2max = torch.stack([q2u, q2v], -1).clamp_min(0.0)
    d_prism = q2max.norm(dim=-1) + torch.minimum(
        torch.maximum(q2u, q2v), torch.zeros_like(q2u))
    return torch.maximum(d_box, -d_prism)


def clearance(p: torch.Tensor, q: torch.Tensor, task: dict, bpts: torch.Tensor,
              arena_y: float, arena_z: float) -> torch.Tensor:
    """Min distance from body surface to solid (wall + ground). Negative = contact."""
    world = p.unsqueeze(1) + quat_rotate(q.unsqueeze(1), bpts.unsqueeze(0))
    d_wall = wall_sdf(world, task, arena_y, arena_z)
    d_ground = world[..., 2]
    return torch.minimum(d_wall, d_ground).min(dim=1).values
