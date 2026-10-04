"""Attempt-level abstraction for contextual world-model adaptation.

An AttemptSpec describes *how* to try a traversal. AttemptOutcome records what
actually happened. These objects are intentionally independent of PPO so they
can be shared by simulation data generation, planners and real-flight logging.
"""

from .spec import AttemptRanges, AttemptSpec, SPEC_FIELDS, sample_specs
from .outcome import AttemptOutcome, TARGET_FIELDS
from .executor import execute_attempt_batch

__all__ = [
    "AttemptRanges", "AttemptSpec", "AttemptOutcome",
    "SPEC_FIELDS", "TARGET_FIELDS", "sample_specs", "execute_attempt_batch",
]
