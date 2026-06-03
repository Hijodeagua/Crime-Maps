"""
Per-capita incident choropleth map (headline visualization).

Uses Folium's Choropleth layer with quantile binning over skewed rate distributions.
Tracts with rate_suppressed=True (below min_population_for_rate) are rendered with a
distinct style and labeled as "Suppressed (low pop.)" in the legend.
"""

import logging

import folium
import geopandas as gpd
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_TILES = "CartoDB positron"


def base_map(center: tuple, zoom: int) -> folium.Map:
    return folium.Map(location=center, zoom_start=zoom, tiles=_DEFAULT_TILES)


def choropleth_layer(
    fmap: folium.Map,
    tract_data: pd.DataFrame,
    boundaries: gpd.GeoDataFrame,
    rate_col: str = "rate_per_1k",
    count_col: str = "count",
    show_rate: bool = True,
) -> folium.Map:
    """
    Add a choropleth layer to fmap.

    tract_data: output of aggregate.tract_counts() — must have geography_id, rate_per_1k,
                count, rate_suppressed.
    boundaries: GeoDataFrame with GEOID and geometry.
    show_rate:  True → color by rate_per_1k; False → color by raw count.
    """
    if tract_data.empty or boundaries is None or boundaries.empty:
        logger.warning("choropleth_layer: no data to render")
        return fmap

    geoid_col = _detect_geoid_col(boundaries)
    value_col = rate_col if show_rate else count_col

    # Exclude suppressed tracts from choropleth coloring (they get a separate style)
    plottable = tract_data[~tract_data["rate_suppressed"]].copy()
    plottable = plottable.dropna(subset=[value_col])

    if plottable.empty:
        logger.warning("choropleth_layer: all tracts suppressed or missing rates")
        return fmap

    # Merge boundaries with data for tooltip
    merged_geo = boundaries.merge(
        tract_data[["geography_id", count_col, rate_col, "rate_suppressed", "population"]],
        left_on=geoid_col,
        right_on="geography_id",
        how="left",
    )

    # Suppress style for low-pop tracts
    suppressed_geo = merged_geo[merged_geo["rate_suppressed"] == True]
    if not suppressed_geo.empty:
        folium.GeoJson(
            suppressed_geo.__geo_interface__,
            style_function=lambda _: {
                "fillColor": "#cccccc",
                "color": "#999999",
                "weight": 0.5,
                "fillOpacity": 0.5,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=[geoid_col, "population"],
                aliases=["Tract", "Population"],
                localize=True,
            ),
            name="Suppressed tracts (low pop.)",
        ).add_to(fmap)

    # Build Choropleth for non-suppressed tracts
    geo_data = boundaries[[geoid_col, "geometry"]].copy()

    n_plottable = len(plottable)
    n_bins = min(6, max(3, n_plottable // 5))

    label = "Per 1,000 residents" if show_rate else "Incident count"

    choropleth = folium.Choropleth(
        geo_data=geo_data.__geo_interface__,
        data=plottable,
        columns=["geography_id", value_col],
        key_on=f"feature.properties.{geoid_col}",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.3,
        nan_fill_color="#f5f5f5",
        bins=n_bins,
        legend_name=label,
        name="Incident concentration",
        highlight=True,
    )
    choropleth.add_to(fmap)

    # Tooltip overlay
    tooltip_fields = [geoid_col, "population"]
    tooltip_aliases = ["Tract GEOID", "Population"]
    if count_col in merged_geo.columns:
        tooltip_fields.append(count_col)
        tooltip_aliases.append("Incidents")
    if rate_col in merged_geo.columns:
        tooltip_fields.append(rate_col)
        tooltip_aliases.append("Rate (per 1k)")

    folium.GeoJson(
        merged_geo.__geo_interface__,
        style_function=lambda _: {"fillOpacity": 0, "weight": 0},
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=tooltip_aliases,
            localize=True,
            sticky=False,
        ),
    ).add_to(fmap)

    folium.LayerControl().add_to(fmap)
    return fmap


def _detect_geoid_col(gdf: gpd.GeoDataFrame) -> str:
    for c in ("GEOID", "geoid", "GEO_ID"):
        if c in gdf.columns:
            return c
    raise ValueError(f"No GEOID column found in boundaries. Columns: {list(gdf.columns)}")
