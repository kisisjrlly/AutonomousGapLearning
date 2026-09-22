"""Conservative, independent safety-envelope calculations.

These functions are diagnostics/guards, not a proof of flight safety. They use
measured clearance and speed plus bounded reaction and braking assumptions.
"""
import torch


def braking_accel(task_dyn: dict) -> torch.Tensor:
    """Unvalidated thrust-margin proxy, NOT a braking lower bound."""
    return (task_dyn["tmax"] / task_dyn["mass"] - 9.81).clamp_min(0.)


def stopping_distance(speed: torch.Tensor, accel: torch.Tensor,
                      reaction_time: float, uncertainty: torch.Tensor | None = None):
    """Reaction plus constant-deceleration distance, conservatively bounded."""
    if reaction_time < 0:
        raise ValueError("reaction_time must be non-negative")
    d = speed.abs() * reaction_time + speed.square() / (2.0 * accel.clamp_min(1e-6))
    return d if uncertainty is None else d + uncertainty.clamp_min(0.0)


def safe_to_continue(clearance: torch.Tensor, forward_speed: torch.Tensor,
                     accel: torch.Tensor, reaction_time: float,
                     uncertainty: torch.Tensor | None = None,
                     margin: float = 0.0) -> torch.Tensor:
    """Whether braking before the obstacle remains possible."""
    if margin < 0:
        raise ValueError("margin must be non-negative")
    stop = stopping_distance(forward_speed.clamp_min(0.0), accel,
                             reaction_time, uncertainty)
    return clearance >= stop + margin


def retreat_gate(clearance: torch.Tensor, forward_speed: torch.Tensor,
                 accel: torch.Tensor, reaction_time: float,
                 uncertainty: torch.Tensor | None = None,
                 margin: float = 0.0) -> torch.Tensor:
    """Gate a candidate action from the current, pre-action state."""
    return (clearance > 0) & safe_to_continue(
        clearance, forward_speed, accel, reaction_time, uncertainty, margin)


def replay_trace(clearance_pre: torch.Tensor, forward_speed_pre: torch.Tensor,
                 accel: torch.Tensor, reaction_time: float,
                 contact_after: torch.Tensor,
                 uncertainty: torch.Tensor | None = None,
                 margin: float = 0.0) -> dict:
    """Replay the gate with transition-aligned data.

    clearance_pre and forward_speed_pre are measured immediately before action
    t. contact_after[t] reports whether action t produced contact during its
    following control interval (including physics substeps).
    """
    if clearance_pre.shape != forward_speed_pre.shape:
        raise ValueError("clearance_pre and forward_speed_pre must have equal shape")
    if contact_after.shape != clearance_pre.shape:
        raise ValueError("contact_after must match trace shape")
    if accel.ndim == 0:
        accel = accel.expand_as(clearance_pre)
    else:
        try:
            accel = torch.broadcast_to(accel, clearance_pre.shape)
        except RuntimeError as exc:
            raise ValueError("accel must broadcast to trace shape") from exc
    if uncertainty is not None and uncertainty.ndim == 0:
        uncertainty = uncertainty.expand_as(clearance_pre)
    gate = retreat_gate(clearance_pre, forward_speed_pre, accel, reaction_time,
                        uncertainty, margin)
    contact = contact_after.bool()
    false_negative = contact & gate
    false_positive = (~contact) & (~gate)
    return {
        "gate": gate,
        "contact": contact,
        "contact_allowed": false_negative,
        "noncontact_rejected": false_positive,
        "n": int(clearance_pre.numel()),
        "contacts": int(contact.sum().item()),
        "allowed": int(gate.sum().item()),
        "contact_allowed_count": int(false_negative.sum().item()),
        "noncontact_rejected_count": int(false_positive.sum().item()),
    }
