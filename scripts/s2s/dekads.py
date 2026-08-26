"""Dekad math — re-exported from :mod:`africas2s.time`.

This logic moved into the library, where the completion engine and any other
caller can reach it. The names stay importable from here so the S2S scripts and
their tests keep working; new code should import from ``africas2s.time``.
"""

from africas2s.time import (  # noqa: F401
    dekad_start_for as _dekad_start_for,
    dekad_window,
    dekads_for_issuance,
)

__all__ = ["dekad_window", "dekads_for_issuance"]
