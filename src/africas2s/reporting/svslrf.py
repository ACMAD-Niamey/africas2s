"""WMO-SVSLRF PDF composition.

Composes the SkillReport into a PDF following WMO-No. 1246 layout
conventions. All pages are US-letter portrait for consistent dimensions:

  1. Cover + mandatory triplet (one combined page).
  2. ROC curves + reliability diagram (one combined page, side-by-side).
  3. Spatial maps grid (one page, only if spatial maps are present).
  4. Member contributions (only if diagrams['member_contributions'] present).
  5. Secondary metrics table (only if secondary scores are present).
"""

from .._optional import require_optional


_MANDATORY = ("rpss", "roc_bn", "roc_nn", "roc_an", "reliability")


def render(report, path):
    """Render a SkillReport to `path` as a WMO-SVSLRF-formatted PDF."""
    require_optional("matplotlib", "pip install africas2s[plotting]")
    from matplotlib.backends.backend_pdf import PdfPages

    from . import _pages

    mandatory_scores = {k: v for k, v in report.scores.items() if k in _MANDATORY}
    secondary_scores = {k: v for k, v in report.scores.items() if k not in _MANDATORY}

    # Filter spatial to DataArrays only — roc/reliability store scalars via
    # scores.update(result) when spatial=True, which would confuse map_grid_page.
    import xarray as xr
    spatial_maps = {k: v for k, v in report.spatial.items()
                    if isinstance(v, xr.DataArray)}

    roc = report.diagrams.get("roc")
    rel = report.diagrams.get("reliability")

    with PdfPages(str(path)) as pdf:
        _pages.cover_and_triplet_page(
            pdf,
            title="Verification report",
            subtitle="WMO-SVSLRF format",
            metadata=report.metadata,
            mandatory_scores=mandatory_scores,
        )

        _pages.diagrams_page(pdf, roc, rel)

        if spatial_maps:
            _pages.map_grid_page(pdf, spatial_maps)

        member_contribs = report.diagrams.get("member_contributions")
        if member_contribs:
            _pages.member_contributions_page(pdf, member_contribs)

        if secondary_scores:
            _pages.scalar_table_page(pdf, secondary_scores, title="Secondary metrics")
