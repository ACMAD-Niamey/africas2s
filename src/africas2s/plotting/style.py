"""Reusable styling for tercile-forecast maps (palette + masks + clip + lakes).

Region-agnostic: callers supply the discrete palette, probability-bin edges, and
optional dry/country/lake styling. Nothing here encodes a specific region.
Institutional colour languages ship as JSON style files inside the package
(``plotting/styles/``: ``icpac`` — the default — ``icpac-temperature``,
``noaa-nmme``, ``ghacof``, ``acmad``) and load by name via
:meth:`TercileStyle.named`; a workflow that owns its own palette loads it from
a file of the same shape via :meth:`TercileStyle.from_json`.
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

# Packaged institutional colour languages, one JSON file each. ``icpac`` is the
# default of ``TercileStyle.named()``.
_STYLES_DIR = Path(__file__).resolve().parent / "styles"
_DEFAULT_NAMED = "icpac"


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
    secondary_max: Any = None              # percent. When set, a cell shows its leading
                                            # category only if every OTHER category is
                                            # under this value; contested cells render
                                            # as nodata ("no dominant category"). The
                                            # NMME rule is 38% leading (prob_bins[0])
                                            # with the rest under 33.

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

    @classmethod
    def named(cls, name=_DEFAULT_NAMED, **overrides):
        """Build one of the colour languages shipped with the package.

        ``name`` is a key of :meth:`list_named` — ``"icpac"`` (the default:
        ICPAC's rainfall palette, six 40–100 % bands), ``"icpac-temperature"``,
        ``"noaa-nmme"`` (the NOAA CPC NMME tercile-summary rules as used by
        CAPC-AC: 38 % leading / others under 33 %, blue wet, orange dry, green
        normal), ``"ghacof"`` (the GHACOF outlook graphics) or ``"acmad"``.
        Keyword ``overrides`` behave exactly as in :meth:`from_json`: they carry
        the fields JSON cannot hold (``dry_mask``, a geometry ``clip_to``) and
        win over the file::

            style = TercileStyle.named("noaa-nmme", dry_mask=too_dry,
                                       clip_to=ECCAS, extent=(6, 32, -18, 24))

        Unknown names raise ``ValueError`` listing what is available.
        """
        path = _STYLES_DIR / f"{name}.json"
        if not path.exists():
            raise ValueError(
                f"unknown style {name!r}; available: {sorted(cls.list_named())}")
        return cls.from_json(path, **overrides)

    @classmethod
    def list_named(cls):
        """``{name: provenance}`` for every packaged style."""
        out = {}
        for path in sorted(_STYLES_DIR.glob("*.json")):
            out[path.stem] = json.loads(path.read_text()).get("_provenance", "")
        return out


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
