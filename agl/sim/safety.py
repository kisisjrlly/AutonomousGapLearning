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
    """Whether braking before the obstacle remains possible.

    A negative forward speed is retreating and is therefore safe under this
    one-dimensional gate. The caller must separately verify lateral/attitude
    and a reachable retreat trajectory.
    """
    if margin < 0:
        raise ValueError("margin must be non-negative")
    stop = stopping_distance(forward_speed.clamp_min(0.0), accel,
                             reaction_time, uncertainty)
    return clearance >= stop + margin


def retreat_gate(clearance: torch.Tensor, forward_speed: torch.Tensor,
                 accel: torch.Tensor, reaction_time: float,
                 uncertainty: torch.Tensor | None = None,
                 margin: float = 0.0) -> torch.Tensor:
    """Gate a forward action while allowing already-retreating motion."""
    return (clearance > 0) & safe_to_continue(
        clearance, forward_speed, accel, reaction_time, uncertainty, margin)


def replay_trace(clearance: torch.Tensor, forward_speed: torch.Tensor,
                 accel: torch.Tensor, reaction_time: float,
                 uncertainty: torch.Tensor | None = None,
                 margin: float = 0.0) -> dict:
    """Replay a recorded trace against the conservative forward gate.

    Inputs are ``(T,N)`` or ``(T,)`` and represent the state *before* each
    action. ``clearance`` is the measured body clearance after collision
    checking; it is never used to train the policy. A contact is a negative
    clearance. This function deliberately reports both gate decisions and
    observed contacts: passing the gate is not treated as proof of safety.
    """
    if clearance.shape != forward_speed.shape:
        raise ValueError("clearance and forward_speed must have equal shape")
    if accel.ndim == 0:
        accel = accel.expand_as(clearance)
    else:
        try:
            accel = torch.broadcast_to(accel, clearance.shape)
        except RuntimeError as exc:
            raise ValueError("accel must broadcast to trace shape") from exc
    if uncertainty is not None and uncertainty.ndim == 0:
        uncertainty = uncertainty.expand_as(clearance)
    gate = retreat_gate(clearance, forward_speed, accel, reaction_time,
                        uncertainty, margin)
    contact = clearance < 0.0
    # A false-negative is a contact on a state the gate would have allowed.
    false_negative = contact & gate
    false_positive = (~contact) & (~gate)
    return {
        "gate": gate,
        "contact": contact,
        "contact_allowed": false_negative,
        "noncontact_rejected": false_positive,
        "n": int(clearance.numel()),
        "contacts": int(contact.sum().item()),
        "allowed": int(gate.sum().item()),
        "contact_allowed_count": int(false_negative.sum().item()),
        "noncontact_rejected_count": int(false_positive.sum().item()),
    }
