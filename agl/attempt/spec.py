"""Versioned six-dimensional attempt space, in SI units.

This is a SIMULATION trial specification, not a command to teleport a real UAV.
entry_yaw now specifies the path tangent angle at the front wall (radians),
not just the initial camera yaw. entry_speed is a speed COMMAND CAP, not a
promise about the measured crossing speed. Controllers are versioned separately.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
from numbers import Real
import numpy as np

SPEC_VERSION = "attempt_spec_v1_path_tangent"
SPEC_FIELDS = ("start_y_offset", "start_z_offset", "entry_yaw", "entry_speed",
               "accel_early", "accel_late")


def _finite_real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite real number")


@dataclass(frozen=True)
class AttemptSpec:
    start_y_offset: float
    start_z_offset: float
    entry_yaw: float
    entry_speed: float
    accel_early: float
    accel_late: float

    def __post_init__(self):
        for k in SPEC_FIELDS:
            _finite_real(getattr(self, k), k)
        if not 0 < self.entry_speed <= 2.0 or not 0 < self.accel_early <= 3.0:
            raise ValueError("speed/early acceleration outside V1 simulation bounds")
        if abs(self.entry_yaw) > .5 or abs(self.accel_late) > 3.0:
            raise ValueError("entry tangent/late acceleration outside V1 bounds")
        if abs(self.start_y_offset) > 1.0 or abs(self.start_z_offset) > .5:
            raise ValueError("start offsets outside V1 simulation bounds")

    def as_vector(self, dtype=np.float32):
        return np.array([getattr(self, k) for k in SPEC_FIELDS], dtype=dtype)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_vector(cls, values):
        values = list(values)
        if len(values) != len(SPEC_FIELDS):
            raise ValueError("expected six attempt parameters")
        return cls(**dict(zip(SPEC_FIELDS, values)))


@dataclass(frozen=True)
class AttemptRanges:
    start_y_offset: tuple[float, float] = (-.30, .30)
    start_z_offset: tuple[float, float] = (-.18, .18)
    entry_yaw: tuple[float, float] = (-.22, .22)
    entry_speed: tuple[float, float] = (.25, .75)
    accel_early: tuple[float, float] = (.25, 1.20)
    accel_late: tuple[float, float] = (-.45, .55)

    def __post_init__(self):
        for k in SPEC_FIELDS:
            lo, hi = getattr(self, k)
            _finite_real(lo, k); _finite_real(hi, k)
            if lo >= hi:
                raise ValueError(f"invalid range: {k}")
        # Validate endpoints, not just the sampled interior.
        for side in (0, 1):
            AttemptSpec(**{k: getattr(self, k)[side] for k in SPEC_FIELDS})


def sample_specs(n, rng, ranges=None):
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    ranges = ranges or AttemptRanges()
    # Sample by record so regrouping records does not permute the random columns.
    return [AttemptSpec(**{k: float(rng.uniform(*getattr(ranges, k))) for k in SPEC_FIELDS})
            for _ in range(n)]
