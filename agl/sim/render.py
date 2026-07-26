"""Analytic raycast renderer: low-res flat-shaded RGB with domain randomization.

Scene = wall slab with rotated rectangular hole + checkered ground + sky.
The hole cross-section is convex and constant along x, so a ray whose slab
entry and exit points are both inside the hole passes through; otherwise the
2D exit of the (u,v) segment from the rectangle gives the inner-wall hit.
"""
import math
import torch

from .maths import quat_rotate

INF = 1e9


def camera_rays(cfg_sensor, device) -> torch.Tensor:
    """Unit ray directions in body frame (x fwd, y left, z up). (H*W, 3)."""
    W, H = cfg_sensor.img_w, cfg_sensor.img_h
    tan_h = math.tan(math.radians(cfg_sensor.fov_deg) / 2)
    tan_v = tan_h * H / W
    ii = (torch.arange(W, device=device) + 0.5) / W    # 0..1 left->right
    jj = (torch.arange(H, device=device) + 0.5) / H    # 0..1 top->bottom
    y = tan_h * (1.0 - 2.0 * ii)                       # +y = left
    z = tan_v * (1.0 - 2.0 * jj)                       # +z = up
    zz, yy = torch.meshgrid(z, y, indexing="ij")       # (H,W)
    pitch = math.radians(cfg_sensor.cam_pitch_deg)
    dirs = torch.stack([torch.ones_like(yy), yy, zz], dim=-1).reshape(-1, 3)
    if abs(pitch) > 1e-9:
        c, s = math.cos(pitch), math.sin(pitch)
        rot = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], device=device)
        dirs = dirs @ rot.T
    return dirs / dirs.norm(dim=-1, keepdim=True)


@torch.no_grad()
def render(p, q, task, rays_body, cfg_sensor, gen=None) -> torch.Tensor:
    """p:(N,3) q:(N,4) -> RGB (N,3,H,W) in [0,1]."""
    N = p.shape[0]
    H, W = cfg_sensor.img_h, cfg_sensor.img_w
    vis = task["vis"]
    d = quat_rotate(q.unsqueeze(1), rays_body.unsqueeze(0))       # (N,P,3)
    px, py, pz = p[:, 0:1], p[:, 1:2], p[:, 2:3]
    dx, dy, dz = d.unbind(-1)

    # ---- wall slab ----
    wx = task["wall_x"].unsqueeze(-1)
    th = task["thick"].unsqueeze(-1)
    safe_dx = torch.where(dx.abs() < 1e-6, torch.full_like(dx, 1e-6), dx)
    ta = (wx - px) / safe_dx
    tb = (wx + th - px) / safe_dx
    t_lo, t_hi = torch.minimum(ta, tb), torch.maximum(ta, tb)
    hit_slab = (t_hi > 1e-4) & (dx.abs() >= 1e-6)
    t_en = t_lo.clamp_min(1e-4)

    def hole_uv(t):
        y = py + t * dy
        z = pz + t * dz
        dyc = y - task["gap_cy"].unsqueeze(-1)
        dzc = z - task["gap_cz"].unsqueeze(-1)
        cr = task["gap_roll"].cos().unsqueeze(-1)
        sr = task["gap_roll"].sin().unsqueeze(-1)
        return cr * dyc + sr * dzc, -sr * dyc + cr * dzc, y, z

    hu = 0.5 * task["gap_w"].unsqueeze(-1)
    hv = 0.5 * task["gap_h"].unsqueeze(-1)
    u1, v1, y1, z1 = hole_uv(t_en)
    u2, v2, _, _ = hole_uv(t_hi)
    in1 = (u1.abs() < hu) & (v1.abs() < hv)
    in2 = (u2.abs() < hu) & (v2.abs() < hv)
    face_hit = hit_slab & ~in1
    thru = hit_slab & in1 & in2
    inner_hit = hit_slab & in1 & ~in2
    # 2D exit of segment (u1,v1)->(u2,v2) from rectangle (u1,v1 is interior)
    du, dv = u2 - u1, v2 - v1
    inf_t = torch.full_like(du, INF)
    su = torch.where(du > 1e-9, (hu - u1) / du,
                     torch.where(du < -1e-9, (-hu - u1) / du, inf_t))
    sv = torch.where(dv > 1e-9, (hv - v1) / dv,
                     torch.where(dv < -1e-9, (-hv - v1) / dv, inf_t))
    s = torch.minimum(su, sv).clamp(0.0, 1.0)
    t_inner = t_en + s * (t_hi - t_en)
    t_wall = torch.where(face_hit, t_en, torch.where(inner_hit, t_inner, torch.full_like(t_en, INF)))
    t_wall = torch.where(hit_slab & ~thru, t_wall, torch.full_like(t_wall, INF))

    # ---- ground ----
    t_g = torch.where(dz < -1e-6, -pz / dz, torch.full_like(dz, INF))
    t_g = torch.where(t_g > 1e-4, t_g, torch.full_like(t_g, INF))

    # ---- shading ----
    light = vis["light"].view(N, 1, 1)
    wall_alb = vis["wall_alb"].view(N, 1, 3)
    stripe = 1.0 + vis["stripe_amp"].unsqueeze(-1) * torch.sin(
        vis["stripe_freq"].unsqueeze(-1) * y1 + vis["stripe_phase"].unsqueeze(-1))
    col_face = wall_alb * stripe.unsqueeze(-1)
    col_inner = wall_alb * vis["inner_scale"].view(N, 1, 1)
    col_wall = torch.where(face_hit.unsqueeze(-1), col_face, col_inner)

    gx = px + t_g * dx
    gy = py + t_g * dy
    cell = vis["checker"].unsqueeze(-1)
    check = ((torch.floor(gx / cell) + torch.floor(gy / cell)) % 2).unsqueeze(-1)
    col_ground = vis["ground_a"].view(N, 1, 3) * (1 - check) + vis["ground_b"].view(N, 1, 3) * check

    sky_f = dz.clamp(0.0, 1.0).pow(0.6).unsqueeze(-1)
    col_sky = vis["sky_hor"].view(N, 1, 3) * (1 - sky_f) + vis["sky_top"].view(N, 1, 3) * sky_f

    t_best = torch.minimum(t_wall, t_g)
    col = torch.where(
        (t_wall <= t_g).unsqueeze(-1) & (t_wall < INF).unsqueeze(-1), col_wall,
        torch.where((t_g < INF).unsqueeze(-1), col_ground, col_sky))
    fade = (0.55 + 0.45 / (1.0 + 0.12 * t_best.clamp_max(60.0))).unsqueeze(-1)
    col = col * fade * light.view(N, 1, 1)
    noise = vis["px_noise"].view(N, 1, 1) * torch.randn(N, H * W, 3, device=p.device, generator=gen)
    img = (col + noise).clamp(0.0, 1.0)
    return img.view(N, H, W, 3).permute(0, 3, 1, 2).contiguous()
