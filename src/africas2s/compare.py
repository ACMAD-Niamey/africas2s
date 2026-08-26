"""Multi-method skill comparison.

`skill_compare()` runs `skill()` once per forecast and bundles the results
into a `ComparisonReport`. Forecasts must share the same lat/lon grid as
the observations — `skill_compare` does not regrid (an opinion-laden step
that belongs to the caller).
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .skill import skill, SkillReport


def _check_grids(forecasts, obs):
    """Raise ValueError if any forecast's lat/lon coords don't match obs.

    Uses np.allclose on the raw coord arrays for a tolerant comparison.
    """
    obs_lat = obs["lat"].values
    obs_lon = obs["lon"].values
    for name, fcst in forecasts.items():
        f_lat = fcst["lat"].values
        f_lon = fcst["lon"].values
        if f_lat.shape != obs_lat.shape or not np.allclose(f_lat, obs_lat):
            raise ValueError(
                f"forecast {name!r} grid does not match obs grid: "
                f"obs lat shape={obs_lat.shape}, forecast lat shape={f_lat.shape}. "
                f"Caller is responsible for regridding before skill_compare()."
            )
        if f_lon.shape != obs_lon.shape or not np.allclose(f_lon, obs_lon):
            raise ValueError(
                f"forecast {name!r} grid does not match obs grid: "
                f"obs lon shape={obs_lon.shape}, forecast lon shape={f_lon.shape}. "
                f"Caller is responsible for regridding before skill_compare()."
            )


@dataclass
class ComparisonReport:
    reports: dict          # method -> SkillReport
    methods: list          # insertion-ordered method names
    metrics: list          # union of metric names across reports

    def to_table(self):
        """methods (rows) x metrics (cols) scalar DataFrame."""
        rows = []
        for m in self.methods:
            row = {"method": m}
            row.update(self.reports[m].scores)
            rows.append(row)
        df = pd.DataFrame(rows).set_index("method")
        # Ensure all union metrics are columns even if missing for some methods
        for metric in self.metrics:
            if metric not in df.columns:
                df[metric] = float("nan")
        return df[self.metrics]

    def to_heatmap(self, path=None, *, metric=None):
        """Render a methods × metrics scalar heatmap.

        If `metric` is given, render only that single column.
        Returns the matplotlib Figure. If `path` is given, also save the
        figure via fig.savefig(path).
        """
        from ._optional import require_optional
        require_optional("matplotlib", "pip install africas2s[plotting]")
        import matplotlib.pyplot as plt

        df = self.to_table()
        if metric is not None:
            if metric not in df.columns:
                raise KeyError(
                    f"metric {metric!r} not in comparison; available: {list(df.columns)}"
                )
            df = df[[metric]]

        fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(df.columns) + 2),
                                        max(3, 0.5 * len(df.index) + 2)))
        data = df.values.astype(float)
        im = ax.imshow(data, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
        ax.set_xticks(np.arange(len(df.columns)))
        ax.set_xticklabels(df.columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(df.index)))
        ax.set_yticklabels(df.index)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                            color="white" if abs(v) > 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        fig.tight_layout()
        if path is not None:
            fig.savefig(str(path))
        return fig

    def to_pdf(self, path, *, spatial_maps=False):
        """Render the comparison to PDF.

        Default composition: heatmap page + scalar summary page.
        With spatial_maps=True: also one comparison-map-grid page per metric
        for which all methods have a spatial map.
        """
        from ._optional import require_optional
        require_optional("matplotlib", "pip install africas2s[plotting]")
        from matplotlib.backends.backend_pdf import PdfPages
        from .reporting import _pages

        df = self.to_table()
        with PdfPages(str(path)) as pdf:
            _pages.heatmap_page(pdf, df, title="Method comparison")
            # Per-method, flat summary
            summary = {}
            for method in self.methods:
                for k, v in self.reports[method].scores.items():
                    summary[f"{method} / {k}"] = v
            _pages.scalar_table_page(pdf, summary, title="Per-method scores")

            if spatial_maps:
                for metric in self.metrics:
                    maps = {}
                    for method in self.methods:
                        spatial = self.reports[method].spatial.get(metric)
                        if spatial is not None and isinstance(spatial, xr.DataArray):
                            maps[method] = spatial
                    if maps:
                        _pages.comparison_map_grid_page(pdf, metric=metric, maps=maps)


def skill_compare(forecasts, obs, metrics=None, spatial=False):
    """Compute skill for each forecast and bundle into a ComparisonReport.

    forecasts: dict mapping method name to forecast DataArray/Dataset.
               Must share the same lat/lon grid as `obs`.
    obs:       observations DataArray/Dataset.
    metrics:   forwarded to skill() (preset name, list of names, or None).
    spatial:   forwarded to skill() (whether to retain spatial maps).
    """
    if not forecasts:
        raise ValueError("skill_compare requires at least one forecast")

    _check_grids(forecasts, obs)

    reports = {}
    metric_order = []
    for name, fcst in forecasts.items():
        rep = skill(fcst, obs, metrics=metrics, spatial=spatial)
        reports[name] = rep
        for k in rep.scores.keys():
            if k not in metric_order:
                metric_order.append(k)

    return ComparisonReport(
        reports=reports,
        methods=list(forecasts.keys()),
        metrics=metric_order,
    )
