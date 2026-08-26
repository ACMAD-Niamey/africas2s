"""Reusable styling for tercile-forecast maps (palette + masks + clip + lakes).

Region-agnostic: callers supply the discrete palette, probability-bin edges, and
optional dry/country/lake styling. Nothing here encodes a specific region or
outlook convention — institutional colour languages (GHACOF, ACMAD, ...) live
in JSON style files owned by the workflows that use them and load via
:meth:`TercileStyle.from_json`; `examples/styles/` in the repository carries
reference copies of the two in current operational use.
"""
from __future__ import annotations

import dataclasses
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .._optional import require_optional

_HINT = "pip install africas2s[plotting]"


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

    @classmethod
    def from_json(cls, path, **overrides):
        """Build a style from a JSON file of ``TercileStyle`` fields.

        The file is a single JSON object whose keys are field names
        (``below_colors``, ``prob_bins``, ``clip_to``, ``extent``, ...); keys
        beginning with ``_`` are ignored, so files can carry provenance notes.
        Keyword ``overrides`` win over the file — the place to attach the
        fields JSON cannot hold (a ``dry_mask`` array, a shapely ``clip_to``
        geometry) or to vary a shared file per figure::

            style = TercileStyle.from_json("styles/ghacof.json",
                                           dry_mask=too_dry,
                                           clip_to=ECCAS_COUNTRIES,
                                           extent=(5, 32, -12, 8))

        Unknown keys in the file raise ``ValueError`` (typo protection).
        """
        data = json.loads(Path(path).read_text())
        if not isinstance(data, dict):
            raise ValueError(
                f"{path}: a style file must be a JSON object of TercileStyle "
                f"fields, got {type(data).__name__}")
        data = {k: v for k, v in data.items() if not k.startswith("_")}
        valid = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - valid)
        if unknown:
            raise ValueError(
                f"{path}: unknown TercileStyle field(s) {unknown}; "
                f"valid fields are {sorted(valid)}")
        data.update(overrides)
        return cls(**data)


def tercile_diverging_cmap(style, *, name="tercile_diverging", n=256):
    """Diverging colormap from a style's strongest below → white → strongest above.

    The derived ramp the replication workflows use for anomaly, tercile-tilt
    difference, and skill-versus-neutral panels, so continuous maps stay in the
    same color language as the tercile maps beside them. Returns a Matplotlib
    ``LinearSegmentedColormap``.
    """
    require_optional("matplotlib", _HINT)
    mcolors = importlib.import_module("matplotlib.colors")
    return mcolors.LinearSegmentedColormap.from_list(
        name, [style.below_colors[-1], "#ffffff", style.above_colors[-1]], N=n)
