"""Observed trial labels, NOT estimates of intrinsic feasibility or safety.

None means unobserved/not applicable. In arrays it is encoded as 0 PLUS a false
mask. Labels must not be used as onboard context. 'success=False' says only
that THIS controller/abort rule did not finish within THIS trial budget. It
never labels the unexecuted continuation of an abort as certain failure.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import numpy as np

TARGET_FIELDS = ("lateral_drift", "vertical_drift", "min_clearance", "stopping_distance",
                 "max_tilt", "terminal_speed", "success", "recovered", "contact", "aborted")
STATUSES = ("success", "recovered", "contact", "environment_terminal", "recovery_timeout",
            "invalid_state", "invalid_start")


@dataclass(frozen=True)
class AttemptOutcome:
    status: str
    success: bool
    recovered: bool
    contact: bool
    aborted: bool
    lateral_drift: float | None
    vertical_drift: float | None
    min_clearance: float | None
    stopping_distance: float | None
    max_tilt: float | None
    terminal_speed: float | None
    abort_x: float | None
    steps: int
    abort_reason: str | None = None

    def __post_init__(self):
        if self.status not in STATUSES or self.steps < 0:
            raise ValueError("invalid status/step count")
        for name in TARGET_FIELDS[:6] + ("abort_x",):
            v = getattr(self, name)
            if v is not None and not math.isfinite(float(v)):
                raise ValueError(f"{name}: use None + mask, not NaN/Inf")
        for name in TARGET_FIELDS[6:]:
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be observed bool, not a probability")
        if self.success != (self.status == "success") or self.recovered != (self.status == "recovered"):
            raise ValueError("status conflicts with success/recovered")
        if self.contact != (self.status == "contact"):
            raise ValueError("status conflicts with contact")
        if self.recovered and not self.aborted:
            raise ValueError("recovered requires an actual recovery request")
        if self.stopping_distance is not None and (not self.aborted or self.stopping_distance < 0):
            raise ValueError("stopping distance requires a measured braking event")

    @property
    def safe_terminal(self):
        """Observed contact-free success/recovery, NOT a prospective guarantee."""
        return (self.success or self.recovered) and not self.contact

    @property
    def context_eligible(self):
        # A passed task ends the online mission; use only recovered trials for retry contexts.
        return self.recovered and not self.contact and self.steps > 0

    def as_target_mask(self):
        m = np.array([getattr(self, k) is not None for k in TARGET_FIELDS], dtype=bool)
        # Recovery was not tested in trials that never requested it.
        m[TARGET_FIELDS.index("recovered")] = self.aborted
        if self.status in ("invalid_state", "invalid_start"):
            m[:] = False
        return m

    def as_target_vector(self, dtype=np.float32):
        m = self.as_target_mask()
        return np.array([float(getattr(self, k)) if m[i] else 0.0
                         for i, k in enumerate(TARGET_FIELDS)], dtype=dtype)

    def to_dict(self):
        return asdict(self)  # standard JSON: null for unavailable data; no nonstandard NaN
