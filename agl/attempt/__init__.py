"""Attempt schema. Importing it does not load the simulator or PyTorch."""
from .spec import AttemptSpec, AttemptRanges, SPEC_FIELDS, SPEC_VERSION, sample_specs
from .outcome import AttemptOutcome, TARGET_FIELDS

__all__ = ["AttemptSpec", "AttemptRanges", "SPEC_FIELDS", "SPEC_VERSION", "sample_specs",
           "AttemptOutcome", "TARGET_FIELDS", "execute_attempt_batch"]


def execute_attempt_batch(*args, **kwargs):
    from .executor import execute_attempt_batch as execute
    return execute(*args, **kwargs)
