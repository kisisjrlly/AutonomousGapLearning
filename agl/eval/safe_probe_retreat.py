"""Deprecated compatibility entry point.

Use agl.eval.closed_loop_probe for the current dynamic-braking recovery
baseline. The old open-loop behavior lives in
agl.eval.legacy_open_loop_probe_retreat solely as a negative example.
"""
import warnings

from .legacy_open_loop_probe_retreat import run as _legacy_run


def run(*args, **kwargs):
    warnings.warn(
        "safe_probe_retreat is deprecated and is only an open-loop negative example; "
        "use agl.eval.closed_loop_probe",
        DeprecationWarning,
        stacklevel=2,
    )
    return _legacy_run(*args, **kwargs)


def main():
    from .legacy_open_loop_probe_retreat import main as legacy_main
    warnings.warn(
        "safe_probe_retreat is deprecated; use python -m agl.eval.closed_loop_probe",
        DeprecationWarning,
        stacklevel=2,
    )
    legacy_main()


if __name__ == "__main__":
    main()
