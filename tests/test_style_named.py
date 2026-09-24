"""Packaged colour languages (TercileStyle.named) and the dominance rules.

ICPAC's 40 % first bin and the NMME "38 % leading, opposite outer tercile under
33 %" rule both leave some valid cells unfilled — "no dominant category" — which
the renderer must honour rather than clip up into the weakest band.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from africas2s.plotting.forecasts import _no_dominant_label, _tercile_codes
from africas2s.plotting.style import TercileStyle

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "styles"
PACKAGED = {"icpac", "icpac-temperature", "icpac-onset", "noaa-cpc", "ghacof", "acmad"}


def _probs(triples):
    """(tercile, lat=1, lon=k) from a list of (below, normal, above) triples."""
    return np.asarray(triples, dtype=float).T[:, None, :]


def test_named_default_is_icpac():
    style = TercileStyle.named()
    assert style.prob_bins[0] == 40 and len(style.below_colors) == 6
    assert style.lakes and style.lake_color.upper() == "#73B2FF"
    assert style.secondary_max is None


def test_every_packaged_style_is_listed_and_constructs():
    listed = TercileStyle.list_named()
    assert PACKAGED <= set(listed)
    for name in listed:
        style = TercileStyle.named(name)          # __post_init__ length checks
        assert len(style.prob_bins) == len(style.below_colors) + 1
        assert listed[name]                        # provenance text present


def test_noaa_cpc_carries_the_cpc_legend_and_the_contested_rule():
    style = TercileStyle.named("noaa-cpc")
    assert style.prob_bins[0] == pytest.approx(33.33) and style.secondary_max == 33
    assert len(style.prob_bins) == 8                   # seven bands, 33-40 ... 90-100
    assert style.above_colors[0].upper() == "#B3D9AB"  # CPC's 'leaning above' green
    assert style.below_colors[-1].upper() == "#4F2F2F" # CPC's darkest 'likely below'
    assert style.normal_colors[:2] == ["#CCCCCC", "#9C9C9C"]
    assert not style.lakes


def test_overrides_win_over_the_file():
    style = TercileStyle.named("noaa-cpc", extent=(6, 32, -18, 24), lakes=True)
    assert style.extent == (6, 32, -18, 24) and style.lakes


def test_unknown_name_lists_what_exists():
    with pytest.raises(ValueError, match="icpac"):
        TercileStyle.named("nope")


@pytest.mark.parametrize("name", ["ghacof", "acmad"])
def test_packaged_copies_match_the_example_files(name):
    packaged = TercileStyle.named(name)
    example = TercileStyle.from_json(EXAMPLES / f"{name}.json")
    assert packaged == example


def test_leading_category_under_the_first_bin_is_not_dominant():
    probs = _probs([(0.36, 0.34, 0.30), (0.45, 0.30, 0.25)])
    code, valid = _tercile_codes(probs, [40, 50, 60, 70, 80, 90, 100.01])
    assert list(valid[0]) == [False, True] and code[0, 0] == -1 and code[0, 1] >= 0
    # The GHACOF edge (33.3) never triggers: the largest of three is >= 1/3.
    code, valid = _tercile_codes(probs, [33.3, 40, 50, 60, 70, 100.01])
    assert valid.all() and (code >= 0).all()


def test_secondary_max_blanks_contested_cells():
    bins = [38, 50, 60, 70, 80, 90, 100.01]
    probs = _probs([(0.40, 0.21, 0.39),    # both outer terciles over 38: contested
                    (0.40, 0.30, 0.30),    # opposite outer under 33: dominant
                    (0.20, 0.20, 0.60)])   # clearly dominant above
    code, valid = _tercile_codes(probs, bins, secondary_max=33)
    assert list(valid[0]) == [False, True, True]
    assert code[0, 0] == -1
    code_no_rule, _ = _tercile_codes(probs, bins)
    assert code_no_rule[0, 0] >= 0               # the rule is opt-in


def test_secondary_max_constrains_only_the_opposite_outer_tercile():
    """NOAA CPC's rule names the opposite outer tercile, not "every other".

    "A and B contours show when one class has >38% of ensemble members, and the
    opposite class is below 33%. In the case that A is >38% and N is >33%, A
    will be shown." -- cpc.ncep.noaa.gov/products/NMME/NMME_PROB_descr.html.
    A near-normal runner-up therefore does not blank an outer tercile; only the
    other *outer* tercile can. Neutral still needs both outers under the bar.
    """
    bins = [38, 50, 60, 70, 80, 90, 100.01]
    probs = _probs([
        (0.25, 0.34, 0.41),   # above leads, below far under 33, normal over 33 -> shown
        (0.41, 0.34, 0.25),   # below leads, above under 33, normal over 33     -> shown
        (0.40, 0.21, 0.39),   # both outer terciles over 38                     -> white
        (0.30, 0.42, 0.28),   # normal leads, both outers under 33              -> shown
        (0.34, 0.40, 0.26),   # normal leads but below reaches 33               -> white
    ])
    code, valid = _tercile_codes(probs, bins, secondary_max=33)
    assert list(valid[0]) == [True, True, False, True, False]
    assert code[0, 0] >= 0 and code[0, 1] >= 0
    assert code[0, 2] == -1 and code[0, 4] == -1


def test_no_dominant_legend_label_only_when_a_rule_applies():
    assert _no_dominant_label(TercileStyle.named("ghacof")) is None
    assert "40" in _no_dominant_label(TercileStyle.named("icpac"))
    label = _no_dominant_label(TercileStyle.named("noaa-cpc"))
    assert "33" in label and "38" not in label        # only the contested rule remains


def test_smooth_render_honours_the_rules():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import xarray as xr
    from africas2s.plotting.forecasts import plot_tercile_forecast

    rng = np.random.default_rng(1)
    p = rng.dirichlet([1, 1, 1], size=(6, 7)).transpose(2, 0, 1)
    probs = xr.DataArray(p, dims=("tercile", "lat", "lon"),
                         coords={"tercile": [0, 1, 2], "lat": np.linspace(-4, 4, 6),
                                 "lon": np.linspace(30, 42, 7)})
    for name in ("noaa-cpc", "icpac"):
        fig = plot_tercile_forecast(probs, style=TercileStyle.named(name), smooth=2)
        assert fig is not None
        plt.close(fig)


def test_legend_handles_include_the_no_dominant_patch():
    pytest.importorskip("matplotlib")
    from africas2s.plotting.panels import tercile_legend_handles
    labels = [h.get_label() for h in tercile_legend_handles(TercileStyle.named("noaa-cpc"))]
    assert any(l.startswith("No dominant category") for l in labels)
    labels = [h.get_label() for h in tercile_legend_handles(TercileStyle.named("ghacof"))]
    assert not any(l.startswith("No dominant category") for l in labels)
