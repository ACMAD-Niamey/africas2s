"""Reporting subpackage — PDF assembly for SkillReport / ComparisonReport.

Page-rendering primitives live in `_pages` (private). The SVSLRF
composition lives in `svslrf` and is the canonical WMO-style PDF.

All rendering code requires the [plotting] extra; functions gate at
entry via `africas2s._optional.require_optional`.
"""
