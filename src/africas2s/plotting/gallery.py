"""Colour-scheme reference sheet: every packaged convention on one page.

``show_schemes()`` renders the full catalogue of named colour conventions —
the :class:`~africas2s.plotting.style.TercileStyle` files in ``styles/`` and
the :class:`~africas2s.plotting.style.FieldScale` files in ``scales/`` —
grouped by the kind of value they paint (tercile probabilities; anomalies &
totals; onset), one entry per convention with the issuing organization, its
provenance note, and the colour bar(s) themselves. It is the visual
counterpart of ``TercileStyle.list_named()`` / ``FieldScale.list_named()``:
run it once to pick a colour language by eye instead of by name.

From a shell::

    python -m africas2s.plotting.gallery              # opens a window
    python -m africas2s.plotting.gallery schemes.png  # writes a file
"""
from __future__ import annotations

import textwrap

from .style import FieldScale, TercileStyle

__all__ = ["show_schemes"]

# Section headers, in display order, by the kind of value a scheme paints.
_SECTIONS = ("Tercile probabilities", "Anomalies & totals", "Onset")

# Issuing organization + reference label per scheme-name prefix (longest match
# wins). New packaged schemes that follow the naming convention need no edit.
_ORGS = (
    ("icpac-onset", "ICPAC / GHACOF"),
    ("icpac", "ICPAC / GHACOF"),
    ("ghacof", "ICPAC / GHACOF"),
    ("acmad", "ACMAD"),
    ("noaa-cpc", "NOAA CPC"),
    ("ucsb-chirps", "UCSB Climate Hazards Center"),
)

# Category captions for tercile ramps, by scheme flavour.
_TERCILE_LABELS = {
    "onset": ("Early", "Normal", "Late"),
    "temperature": ("Cooler", "Normal", "Warmer"),
    "precip": ("Below", "Normal", "Above"),
}


def _org(name):
    for prefix, org in _ORGS:
        if name.startswith(prefix):
            return org
    return name.split("-")[0].upper()


def _section(name, kind):
    if "onset" in name:
        return "Onset"
    return "Tercile probabilities" if kind == "style" else "Anomalies & totals"


def _labels(name):
    for key, labels in _TERCILE_LABELS.items():
        if key in name:
            return labels
    return _TERCILE_LABELS["precip"]


def _collect():
    """[(section, kind, name, provenance)] in display order.

    kind is ``"style"`` (TercileStyle) or ``"scale"`` (FieldScale); order is
    section order, styles before scales within a section, then by name.
    """
    entries = [("style", n, prov) for n, prov in TercileStyle.list_named().items()]
    entries += [("scale", n, prov) for n, prov in FieldScale.list_named().items()]
    out = [(_section(n, k), k, n, prov) for k, n, prov in entries]
    order = {s: i for i, s in enumerate(_SECTIONS)}
    out.sort(key=lambda e: (order.get(e[0], 99), e[1] != "style", e[2]))
    return out


def _edge_label(e):
    """Bin-edge tick text: probability caps like 100.01 read as 100."""
    if abs(e - 100.01) < 0.1:
        return "100"
    return f"{round(e, 1):g}"


def _draw_ramp(ax, colors, edges):
    """One horizontal segmented ramp with the bin edges as ticks."""
    import numpy as np
    for i, c in enumerate(colors):
        ax.axvspan(i, i + 1, color=c)
    ax.set_xlim(0, len(colors))
    ax.set_xticks(np.arange(len(edges)))
    ax.set_xticklabels([_edge_label(e) for e in edges], fontsize=6.5)
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("0.6")
        s.set_linewidth(0.6)
    ax.tick_params(length=2, pad=1.5)


def show_schemes(save=None):
    """Render the reference sheet of every packaged colour convention.

    Sections by value type (tercile probabilities; anomalies & totals; onset);
    per convention: the issuing organization and scheme name, the provenance
    note from its JSON file, and the colour bar(s) — three category ramps for
    a :class:`TercileStyle` (with its probability-bin edges), a single
    classified colorbar for a :class:`FieldScale` (with its levels, units and
    out-of-range arrows).

    ``save``: optional path; when given the figure is also written there.
    Returns the Matplotlib figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm

    entries = _collect()

    HEADER, ENTRY, PAD = 0.42, 1.18, 0.10          # row heights, inches
    width = 9.0
    heights, rows = [], []
    last_section = None
    for section, kind, name, prov in entries:
        if section != last_section:
            rows.append(("header", section))
            heights.append(HEADER)
            last_section = section
        rows.append(("entry", (kind, name, prov)))
        heights.append(ENTRY)
    total = sum(heights) + PAD * 2 + 0.5
    fig = plt.figure(figsize=(width, total))
    fig.suptitle("africas2s packaged colour schemes", fontsize=13,
                 fontweight="bold", y=1 - 0.12 / total)

    y = total - 0.55                                # top margin below suptitle
    for (rtype, payload), h in zip(rows, heights):
        y -= h
        if rtype == "header":
            fig.text(0.035, (y + 0.10) / total, payload, fontsize=11.5,
                     fontweight="bold")
            fig.lines.extend(plt.Line2D(
                [0.035, 0.965], [(y + 0.04) / total] * 2,
                transform=fig.transFigure, color="0.75", linewidth=0.8)
                for _ in (0,))
            continue
        kind, name, prov = payload
        title_y = (y + h - 0.24) / total
        fig.text(0.05, title_y, f"{_org(name)} — {name}", fontsize=9.5,
                 fontweight="bold")
        note = textwrap.shorten(str(prov), width=150, placeholder=" …")
        fig.text(0.05, title_y - 0.17 / total, note, fontsize=7,
                 color="0.35")
        bar_y, bar_h = (y + 0.16) / total, 0.30 / total
        if kind == "style":
            st = TercileStyle.named(name)
            labels = _labels(name)
            ramps = (st.below_colors, st.normal_colors, st.above_colors)
            gap, x0, span = 0.035, 0.05, 0.90
            w = (span - 2 * gap) / 3
            for i, (ramp, lab) in enumerate(zip(ramps, labels)):
                ax = fig.add_axes([x0 + i * (w + gap), bar_y, w, bar_h])
                _draw_ramp(ax, ramp, st.prob_bins)
                ax.set_title(lab, fontsize=7.5, pad=2)
        else:
            sc = FieldScale.named(name)
            ax = fig.add_axes([0.05, bar_y, 0.90, bar_h])
            cb = fig.colorbar(cm.ScalarMappable(norm=sc.norm, cmap=sc.cmap),
                              cax=ax, orientation="horizontal", extend=sc.extend)
            cb.set_ticks(sc.levels)
            cb.ax.tick_params(labelsize=6.5, length=2, pad=1.5)
            if sc.label:
                cb.ax.set_title(str(sc.label), fontsize=7.5, pad=2, loc="right")
            cb.outline.set_edgecolor("0.6")
            cb.outline.set_linewidth(0.6)
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight",
                    facecolor="white")
    return fig


def main(argv=None):
    import sys
    args = sys.argv[1:] if argv is None else argv
    if args:
        show_schemes(save=args[0])
        print(f"wrote {args[0]}")
    else:
        import matplotlib.pyplot as plt
        show_schemes()
        plt.show()


if __name__ == "__main__":
    main()
