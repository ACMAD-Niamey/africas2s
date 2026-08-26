"""Reusable styling for tercile-forecast maps (palette + masks + clip + lakes).

Region-agnostic: callers supply the discrete palette, probability-bin edges, and
optional dry/country/lake styling. Nothing here encodes a specific region or
outlook convention.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TercileStyle:
    below_colors: list[str]
    normal_colors: list[str]
    above_colors: list[str]
    prob_bins: list[float]                 # percent edges; len == n_colors + 1
    dry_mask: Any = None                   # bool DataArray/ndarray; True -> dry_color.
                                            # A coordinate-bearing DataArray (lat/lon
                                            # coords) is aligned to the plotted field's
                                            # grid by nearest-neighbor interpolation on
                                            # coordinate value, so it need not share the
                                            # field's resolution, offset, or lat order.
                                            # A bare ndarray (no coords) must match the
                                            # field's shape exactly.
    dry_color: str = "#bebebe"
    clip_to: Any = None                    # list of country NAMEs, or a shapely geometry
    lakes: bool = False
    lake_color: str = "#78b8f8"
    nodata_color: str = "#ffffff"
    extent: Any = None                     # (lon_w, lon_e, lat_s, lat_n)

    def __post_init__(self):
        n = len(self.prob_bins) - 1
        for name in ("below_colors", "normal_colors", "above_colors"):
            if len(getattr(self, name)) != n:
                raise ValueError(
                    f"{name} must have {n} entries (len(prob_bins)-1), "
                    f"got {len(getattr(self, name))}."
                )
