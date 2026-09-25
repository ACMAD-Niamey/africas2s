"""Reusable styling for forecast maps: tercile styles and classified field scales.

Region-agnostic: callers supply the discrete palette, probability-bin edges, and
optional dry/country/lake styling. Nothing here encodes a specific region.

:class:`TercileStyle` is the palette + masks + clip + lakes of a tercile map.
:class:`FieldScale` is the colours + bin edges of a classified continuous map
(anomaly, total, onset date, spread), drawn on the same styled basemap.
Institutional colour languages ship as JSON files inside the package and load
by name — tercile styles from ``plotting/styles/`` via
:meth:`TercileStyle.named` (``icpac`` — the default — ``icpac-temperature``,
``icpac-onset``, ``noaa-cpc``, ``acmad``), field scales from
``plotting/scales/`` via :meth:`FieldScale.named` (``noaa-cpc-anomaly``,
``ucsb-chirps-anomaly``, ``ucsb-chirps-total``, ``icpac-onset-date``,
``icpac-onset-spread``). A workflow that owns its own palette loads it from a
file of the same shape via ``from_json``.
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
# default of ``TercileStyle.named()``; field scales have no default.
_STYLES_DIR = Path(__file__).resolve().parent / "styles"
_SCALES_DIR = Path(__file__).resolve().parent / "scales"
_DEFAULT_NAMED = "icpac"
# Former names, so a stale notebook gets told where the style went rather
# than "unknown style".
_RENAMED_STYLES = {"noaa-nmme": "noaa-cpc"}


def _from_json(cls, path, overrides):
    """Build ``cls`` from a JSON object of its fields (see ``from_json``)."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(
            f"{path}: a style file must be a JSON object of {cls.__name__} "
            f"fields, got {type(data).__name__}")
    data = {k: v for k, v in data.items() if not k.startswith("_")}
    valid = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - valid)
    if unknown:
        raise ValueError(
            f"{path}: unknown {cls.__name__} field(s) {unknown}; "
            f"valid fields are {sorted(valid)}")
    data.update(overrides)
    return cls(**data)


def _named(cls, directory, name, overrides, renamed=None):
    path = directory / f"{name}.json"
    if not path.exists():
        if renamed and name in renamed:
            raise ValueError(
                f"style {name!r} is now named {renamed[name]!r}: use "
                f"{cls.__name__}.named({renamed[name]!r}, ...)")
        raise ValueError(
            f"unknown {cls.__name__} {name!r}; available: "
            f"{sorted(_list_named(directory))}")
    return _from_json(cls, path, overrides)


def _list_named(directory):
    out = {}
    for path in sorted(directory.glob("*.json")):
        out[path.stem] = json.loads(path.read_text()).get("_provenance", "")
    return out


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

            style = TercileStyle.from_json("styles/my-org.json",
                                           dry_mask=too_dry,
                                           clip_to=ECCAS_COUNTRIES,
                                           extent=(5, 32, -12, 8))

        Unknown keys in the file raise ``ValueError`` (typo protection).
        """
        return _from_json(cls, path, overrides)

    @classmethod
    def named(cls, name=_DEFAULT_NAMED, **overrides):
        """Build one of the colour languages shipped with the package.

        ``name`` is a key of :meth:`list_named` — ``"icpac"`` (the default:
        ICPAC's rainfall palette, six 40–100 % bands), ``"icpac-temperature"``,
        ``"icpac-onset"`` (early / normal / late onset, seven bands from
        33.3 %; plot with ``variable_kind="onset"``), ``"noaa-cpc"`` (the NOAA
        CPC seasonal-outlook legend as used by CAPC-AC: green above, brown
        below, grey near-normal in seven bands from 33.3 %, and a contested
        cell — opposite outer tercile at 33 % or more — left white as "equal
        chances") or ``"acmad"`` (the ACMAD continental palette).
        Keyword ``overrides`` behave exactly as in :meth:`from_json`: they carry
        the fields JSON cannot hold (``dry_mask``, a geometry ``clip_to``) and
        win over the file::

            style = TercileStyle.named("noaa-cpc", dry_mask=too_dry,
                                       clip_to=ECCAS, extent=(6, 32, -18, 24))

        Unknown names raise ``ValueError`` listing what is available; a
        former name (``"noaa-nmme"``) says what it is called now.
        """
        return _named(cls, _STYLES_DIR, name, overrides, _RENAMED_STYLES)

    @classmethod
    def list_named(cls):
        """``{name: provenance}`` for every packaged style."""
        return _list_named(_STYLES_DIR)


_EXTEND_SLOTS = {"neither": 0, "min": 1, "max": 1, "both": 2}


@dataclass
class FieldScale:
    """A classified colour scale for a continuous field: colours + bin edges.

    The counterpart of :class:`TercileStyle` for anomaly, total, onset-date
    and spread maps. ``levels`` are the bin edges (increasing); ``colors`` is
    one colour per bin plus one per open end named by ``extend`` (``"min"``,
    ``"max"`` or ``"both"``: the colorbar's out-of-range arrows), in value
    order — so ``len(colors) == len(levels) - 1 + (number of open ends)``.
    ``label`` is the colorbar caption.

    Pass one as ``scale=`` to :func:`plot_field` / :func:`plot_matrix`, which
    then paint exactly these colours bin for bin in both the cell and the
    ``smooth`` contour renderer. :attr:`cmap` / :attr:`norm` are the matching
    Matplotlib objects for a hand-built figure (``pcolormesh(..., cmap=s.cmap,
    norm=s.norm)``, ``contourf(..., levels=s.levels, cmap=s.cmap, norm=s.norm,
    extend=s.extend)``).

    Institutional scales ship as JSON in ``plotting/scales/`` and load with
    :meth:`named` (:meth:`list_named` enumerates them); a workflow that owns
    one keeps a file of the same shape and loads it with :meth:`from_json`.
    """
    colors: list[str]
    levels: list[float]
    extend: str = "neither"
    label: Any = None

    def __post_init__(self):
        if self.extend not in _EXTEND_SLOTS:
            raise ValueError(
                f"extend must be one of {sorted(_EXTEND_SLOTS)}, got {self.extend!r}")
        self.levels = [float(v) for v in self.levels]
        if len(self.levels) < 2:
            raise ValueError("levels needs at least two bin edges")
        if any(b <= a for a, b in zip(self.levels[:-1], self.levels[1:])):
            raise ValueError(f"levels must be strictly increasing, got {self.levels}")
        self.colors = list(self.colors)
        want = len(self.levels) - 1 + _EXTEND_SLOTS[self.extend]
        if len(self.colors) != want:
            raise ValueError(
                f"colors must have {want} entries (len(levels)-1 bins plus "
                f"{_EXTEND_SLOTS[self.extend]} for extend={self.extend!r}), "
                f"got {len(self.colors)}.")

    @property
    def bin_colors(self):
        """The colours of the closed bins only, open ends dropped."""
        lo = 1 if self.extend in ("min", "both") else 0
        hi = len(self.colors) - (1 if self.extend in ("max", "both") else 0)
        return self.colors[lo:hi]

    @property
    def cmap(self):
        """``ListedColormap`` of every colour, open ends included, with the
        end colours also registered as ``under`` / ``over``."""
        require_optional("matplotlib", _HINT)
        mcolors = importlib.import_module("matplotlib.colors")
        cmap = mcolors.ListedColormap(self.colors, name=str(self.label or "field_scale"))
        if self.extend in ("min", "both"):
            cmap.set_under(self.colors[0])
        if self.extend in ("max", "both"):
            cmap.set_over(self.colors[-1])
        return cmap

    @property
    def norm(self):
        """``BoundaryNorm`` over ``levels`` sized to :attr:`cmap`, so bin *i*
        maps to colour *i* (offset by one when the lower end is open)."""
        require_optional("matplotlib", _HINT)
        mcolors = importlib.import_module("matplotlib.colors")
        return mcolors.BoundaryNorm(self.levels, len(self.colors), extend=self.extend)

    @classmethod
    def from_json(cls, path, **overrides):
        """Build a scale from a JSON file of ``FieldScale`` fields (``colors``,
        ``levels``, ``extend``, ``label``); keys beginning with ``_`` are
        provenance notes and ignored, unknown keys raise, keyword
        ``overrides`` win over the file."""
        return _from_json(cls, path, overrides)

    @classmethod
    def named(cls, name, **overrides):
        """Build one of the scales shipped with the package: a key of
        :meth:`list_named` — ``"noaa-cpc-anomaly"`` (CPC's precipitation
        anomaly bar, mm), ``"ucsb-chirps-anomaly"`` and ``"ucsb-chirps-total"``
        (the Climate Hazards Center CHIRPS anomaly and total bars, mm),
        ``"icpac-onset-date"`` (one colour per dekad since the search-window
        start) and ``"icpac-onset-spread"`` (onset standard deviation, days).
        Unknown names raise ``ValueError`` listing what is available."""
        return _named(cls, _SCALES_DIR, name, overrides)

    @classmethod
    def list_named(cls):
        """``{name: provenance}`` for every packaged scale."""
        return _list_named(_SCALES_DIR)


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
