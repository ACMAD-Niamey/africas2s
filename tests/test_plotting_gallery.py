"""Tests for the colour-scheme reference sheet (plotting/gallery.py)."""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from africas2s.plotting import show_schemes
from africas2s.plotting.gallery import _collect, _edge_label, _org, _section, main
from africas2s.plotting.style import FieldScale, TercileStyle


def test_collect_covers_every_named_scheme():
    entries = _collect()
    names = [n for _, _, _, n, _ in entries]
    assert sorted(names) == sorted(list(TercileStyle.list_named()) +
                                   list(FieldScale.list_named()))
    # every entry carries a non-empty provenance note
    assert all(prov for _, _, _, _, prov in entries)


def test_sections_group_by_value_type():
    assert _section("icpac-onset", "style") == "Onset"
    assert _section("icpac-onset-date", "scale") == "Onset"
    assert _section("ghacof", "style") == "Tercile probabilities"
    assert _section("noaa-cpc-anomaly", "scale") == "Anomalies & totals"
    # display order: terciles, then continuous, then onset
    sections = [s for s, _, _, _, _ in _collect()]
    firsts = sorted(set(sections), key=sections.index)
    assert firsts == ["Tercile probabilities", "Anomalies & totals", "Onset"]


def test_org_labels():
    assert _org("ghacof") == "ICPAC"
    assert _org("icpac") == "RCC default"
    assert _org("icpac-temperature") == "RCC default"
    assert _org("icpac-onset-date") == "ICPAC"
    assert _org("noaa-cpc-anomaly") == "NOAA CPC"
    assert _org("ucsb-chirps-total") == "UCSB Climate Hazards Center"
    assert _org("acmad") == "ACMAD"


def test_tercile_subsections_split_precip_and_temperature():
    from africas2s.plotting.gallery import _subsection
    assert _subsection("icpac-temperature", "Tercile probabilities") == "Temperature"
    assert _subsection("ghacof", "Tercile probabilities") == "Precipitation"
    assert _subsection("icpac-onset-date", "Onset") is None
    subs = [sub for s, sub, _, _, _ in _collect() if s == "Tercile probabilities"]
    firsts = sorted(set(subs), key=subs.index)
    assert firsts == ["Precipitation", "Temperature"]


def test_edge_label_reads_probability_cap_as_100():
    assert _edge_label(100.01) == "100"
    assert _edge_label(33.33) == "33.3"
    assert _edge_label(40) == "40"


def test_show_schemes_draws_all_bars(tmp_path):
    fig = show_schemes(save=tmp_path / "schemes.png")
    try:
        n_styles = len(TercileStyle.list_named())
        n_scales = len(FieldScale.list_named())
        # three ramps per tercile style, one colorbar axes per field scale
        assert len(fig.axes) == 3 * n_styles + n_scales
        titles = " ".join(t.get_text() for t in fig.texts)
        for name in list(TercileStyle.list_named()) + list(FieldScale.list_named()):
            assert name in titles
        assert (tmp_path / "schemes.png").exists()
    finally:
        plt.close(fig)


def test_main_writes_file(tmp_path, capsys):
    out = tmp_path / "gallery.png"
    main([str(out)])
    assert out.exists()
    assert str(out) in capsys.readouterr().out
    plt.close("all")
