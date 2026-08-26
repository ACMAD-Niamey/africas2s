"""Daily-rainfall aggregations: season timing and dry spells.

Access is module-qualified, following ``africas2s.time`` and
``africas2s.tercile``::

    from africas2s import aggregations as agg

    onset = agg.onset(daily, season="MAM")
"""
from .spells import DrySpellResult, dry_spell
from .timing import ONSET_DEFAULTS, TimingResult, cessation, onset, season_length

__all__ = [
    "onset",
    "cessation",
    "season_length",
    "dry_spell",
    "TimingResult",
    "DrySpellResult",
    "ONSET_DEFAULTS",
]
