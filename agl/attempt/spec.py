"""Low-dimensional traversal-attempt parameterization.

The first contextual-world-model milestone deliberately keeps the search space
small. start_y/z are offsets from the perceived gap center at the retry-side
start plane. entry_yaw is relative to world +x. The two acceleration parameters
shape the commanded forward-speed profile before/after the information zone.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

import numpy as np


SPEC_FIELDS = (
    "start_y_offset",
    "start_z_offset",
    "entry_yaw",
    "entry_speed",
    "accel_early",
    "accel_late",
)


@dataclass(frozen=True)
class AttemptSpec:
    start_y_offset: float
    start_z_offset: float
    entry_yaw: float
    entry_speed: float
    accel_early: float
    accel_late: float

    def __post_init__(self):
        values = [getattr(self, k) for k in SPEC_FIELDS]
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError("AttemptSpec values must be finite")
        if self.entry_speed <= 0:
            raise ValueError("entry_speed must be positive")
        if self.accel_early <= 0:
            raise ValueError("accel_early must be positive")

    def as_vector(self, dtype=np.float32) -> np.ndarray:
        return np.asarray([getattr(self, k) for k in SPEC_FIELDS], dtype=dtype)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_vector(cls, values: Iterable[float]) -> "AttemptSpec":
        values = list(values)
        if len(values) != len(SPEC_FIELDS):
            raise ValueError(f"expected {len(SPEC_FIELDS)} values, got {len(values)}")
        return cls(**dict(zip(SPEC_FIELDS, map(float, values))))


@dataclass(frozen=True)
class AttemptRanges:
    """Sampling ranges for the first six-dimensional attempt space."""

    start_y_offset: tuple[float, float] = (-0.30, 0.30)
    start_z_offset: tuple[float, float] = (-0.18, 0.18)
    entry_yaw: tuple[float, float] = (-0.22, 0.22)
    entry_speed: tuple[float, float] = (0.25, 0.75)
    accel_early: tuple[float, float] = (0.25, 1.20)
    accel_late: tuple[float, float] = (-0.45, 0.55)

    def __post_init__(self):
        for name in SPEC_FIELDS:
            lo, hi = getattr(self, name)
            if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
                raise ValueError(f"invalid range for {name}: {(lo, hi)}")
        if self.entry_speed[0] <= 0 or self.accel_early[0] <= 0:
            raise ValueError("entry_speed and accel_early ranges must stay positive")


def sample_specs(n: int, rng: np.random.Generator,
                 ranges: AttemptRanges | None = None) -> list[AttemptSpec]:
    if n <= 0:
        raise ValueError("n must be positive")
    ranges = ranges or AttemptRanges()
    cols = {}
    for name in SPEC_FIELDS:
        lo, hi = getattr(ranges, name)
        cols[name] = rng.uniform(lo, hi, size=n)
    return [
        AttemptSpec(**{name: float(cols[name][i]) for name in SPEC_FIELDS})
        for i in range(n)
    ]
