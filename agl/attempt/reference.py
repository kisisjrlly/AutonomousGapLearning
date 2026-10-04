"""Smooth path and speed reference; no hidden task dynamics are read here."""
import torch


def path_reference(x, start, wall_x, gap_y, gap_z, entry_yaw):
    """Cubic Hermite path. dy/dx at the wall equals tan(entry_yaw).

    Start tangent is zero, end point is the aperture center. Past the front
    plane extend the tangent rather than snapping every candidate to one path.
    """
    length = (wall_x - start[:, 0]).clamp_min(.1)
    u = ((x - start[:, 0]) / length).clamp(0, 1)
    h00, h01, h11 = 2*u**3-3*u**2+1, -2*u**3+3*u**2, u**3-u**2
    slope = torch.tan(entry_yaw)
    y = h00*start[:, 1] + h01*gap_y + h11*length*slope
    z = h00*start[:, 2] + h01*gap_z
    dy = ((6*u**2-6*u)*start[:, 1] + (-6*u**2+6*u)*gap_y)/length + (3*u**2-2*u)*slope
    dz = ((6*u**2-6*u)*start[:, 2] + (-6*u**2+6*u)*gap_z)/length
    past = x > wall_x
    y = torch.where(past, gap_y+(x-wall_x)*slope, y)
    z = torch.where(past, gap_z, z)
    dy = torch.where(past, slope, dy)
    dz = torch.where(past, torch.zeros_like(dz), dz)
    return torch.stack([y, z], -1), torch.stack([dy, dz], -1)


def speed_reference(v_cmd, x, wall_x, cap, early, late, dt, switch_distance=.8):
    """Reserve late-stage headroom so positive accel_late is not silently clipped.

    Early command cap is 0.65*entry_speed; late cap is entry_speed. Negative
    late acceleration decelerates to 0.05m/s. These are reference commands.
    """
    is_early = x < wall_x - switch_distance
    a = torch.where(is_early, early, late)
    local_cap = torch.where(is_early, .65*cap, cap)
    return torch.minimum((v_cmd + a*dt).clamp_min(.05), local_cap)
