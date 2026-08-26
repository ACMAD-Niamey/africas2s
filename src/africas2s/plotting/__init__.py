"""Plot helpers for africas2s outputs.

This subpackage's functions require optional dependencies (matplotlib,
cartopy, rioxarray) that aren't installed by default. Install them with:

    pip install africas2s[plotting]

Importing this package itself does NOT load the optional deps; each
function calls `africas2s._optional.require_optional` lazily.
"""

from .domains import plot_domains
from .skill import plot_skill_maps
from .forecasts import (
    plot_tercile_forecast,
    plot_field,
    plot_tercile_comparison,
    render_styled_terciles,
    plot_deterministic_forecast,
    plot_exceedance_probability,
    plot_flex_pdf,
)
from .reliability import plot_reliability_diagram
from .scenarios import plot_accumulation_scenarios, plot_index_scatter
from .maps import plot_field_map, plot_choropleth
from .modes import plot_eof_modes, plot_cca_modes
from .panels import plot_matrix, plot_components_objective, tercile_legend_handles
from .forecasts import region_masks
from .style import tercile_diverging_cmap
from .style import TercileStyle

__all__ = [
    "plot_domains",
    "plot_skill_maps",
    "plot_tercile_forecast",
    "plot_field",
    "plot_tercile_comparison",
    "plot_deterministic_forecast",
    "plot_exceedance_probability",
    "plot_flex_pdf",
    "plot_reliability_diagram",
    "plot_accumulation_scenarios",
    "plot_index_scatter",
    "plot_field_map",
    "plot_choropleth",
    "plot_eof_modes",
    "plot_cca_modes",
    "plot_matrix",
    "plot_components_objective",
    "tercile_legend_handles",
    "region_masks",
    "tercile_diverging_cmap",
    "TercileStyle",
    "render_styled_terciles",
]
