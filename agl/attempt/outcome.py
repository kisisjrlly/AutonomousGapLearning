"""Observed/labelled result of one traversal attempt."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

import numpy as np


TARGET_FIELDS = (
    "lateral_drift",
    "vertical_drift",
    "min_clearance",
    "stopping_distance",
    "max_tilt",
    "terminal_speed",
    "success",
    "recovered",
    "contact",
    "aborted",
)


@dataclass(frozen=True)
class AttemptOutcome:
    success: bool
    recovered: bool
    contact: bool
    aborted: bool
    lateral_drift: float
    vertical_drift: float
    min_clearance: float
    stopping_distance: float
    max_tilt: float
    terminal_speed: float
    abort_x: float
    steps: int

    def __post_init__(self):
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        for name in (
            "lateral_drift", "vertical_drift", "min_clearance",
            "stopping_distance", "max_tilt", "terminal_speed",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if not (math.isfinite(float(self.abort_x)) or math.isnan(float(self.abort_x))):
            raise ValueError("abort_x must be finite or NaN")

    @property
    def safe_terminal(self) -> bool:
        return bool(self.success or self.recovered) and not self.contact

    def to_dict(self) -> dict:
        return asdict(self)

    def as_target_vector(self, dtype=np.float32) -> np.ndarray:
        values = []
        for name in TARGET_FIELDS:
            v = getattr(self, name)
            values.append(float(v))
        return np.asarray(values, dtype=dtype)

    @classmethod
    def from_target_vector(cls, values: Iterable[float], *,
                           abort_x: float = float("nan"),
                           steps: int = 1) -> "AttemptOutcome":
        values = list(values)
        if len(values) != len(TARGET_FIELDS):
            raise ValueError(f"expected {len(TARGET_FIELDS)} targets, got {len(values)}")
        d = dict(zip(TARGET_FIELDS, values))
        return cls(
            success=bool(round(float(d["success"]))),
            recovered=bool(round(float(d["recovered"]))),
            contact=bool(round(float(d["contact"]))),
            aborted=bool(round(float(d["aborted"]))),
            lateral_drift=float(d["lateral_drift"]),
            vertical_drift=float(d["vertical_drift"]),
            min_clearance=float(d["min_clearance"]),
            stopping_distance=float(d["stopping_distance"]),
            max_tilt=float(d["max_tilt"]),
            terminal_speed=float(d["terminal_speed"]),
            abort_x=float(abort_x),
            steps=int(steps),
        )
