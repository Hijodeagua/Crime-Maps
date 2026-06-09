"""
Street-level heat maps from raw incident points.

Unlike the tract choropleth (analysis-grade, per-capita) and the KDE projection
(smoothed, recency-weighted), this layer renders the raw point cloud directly so
hot blocks and corridors are visible at street zoom. Three renderers:

  - heatmap_layer:          static folium HeatMap over the selected period
  - animated_heatmap:       HeatMapWithTime — one frame per period (week/month),
                            a time slider for scrubbing through up to 3 years
  - marker_cluster_layer:   clustered individual incidents with popups

Caveats rendered alongside in the UI: raw points are NOT population-adjusted —
dense areas light up partly because more people are there. Many agencies geocode
to the block midpoint, so apparent address-level precision is approximate.
"""

import logging
from typing import List, Optional, Tuple

import folium
import pandas as pd
from folium.plugins import HeatMap, HeatMapWithTime, MarkerCluster

logger = logging.getLogger(__name__)

_MAX_HEAT_POINTS = 60_000     # downsample beyond this to keep the page responsive
_MAX_MARKERS = 3_000

_CATEGORY_COLORS = {
    "Violent": "red",
    "Property": "orange",
    "Other": "blue",
    "Unknown": "gray",
}


def valid_points(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with usable coordinates (assigned-or-not; street view shows all points)."""
    if df.empty:
        return df
    return df[
        df["lat"].notna() & df["lon"].notna() &
        (df["lat"] != 0.0) & (df["lon"] != 0.0)
    ]


def _downsample(df: pd.DataFrame, max_points: int, seed: int = 7) -> Tuple[pd.DataFrame, bool]:
    if len(df) <= max_points:
        return df, False
    return df.sample(n=max_points, random_state=seed), True


def heatmap_layer(
    df: pd.DataFrame,
    radius: int = 11,
    blur: int = 14,
    max_points: int = _MAX_HEAT_POINTS,
) -> Tuple[Optional[folium.FeatureGroup], Optional[str]]:
    """Static street-level heat map. Returns (layer, note); layer is None when
    there are no plottable points, note carries downsampling/empty messages."""
    pts = valid_points(df)
    if pts.empty:
        return None, "No incidents with usable coordinates in the selected range."

    pts, sampled = _downsample(pts, max_points)
    note = (
        f"Showing a random sample of {max_points:,} of {len(valid_points(df)):,} points "
        "to keep the map responsive."
    ) if sampled else None

    layer = folium.FeatureGroup(name="Street-level heat map", show=True)
    HeatMap(
        pts[["lat", "lon"]].values.tolist(),
        radius=radius,
        blur=blur,
        min_opacity=0.25,
        max_zoom=18,
    ).add_to(layer)
    return layer, note


def animation_frames(
    df: pd.DataFrame,
    freq: str = "MS",
    max_points_per_frame: int = 8_000,
) -> Tuple[List[List[List[float]]], List[str]]:
    """
    Bucket points into time frames for HeatMapWithTime.

    freq: pandas offset alias — "MS" (month) or "W" (week).
    Returns (frames, labels): frames[i] is a list of [lat, lon] pairs, labels[i]
    a human-readable period label. Periods with zero incidents yield empty frames
    so the slider timeline stays continuous.
    """
    pts = valid_points(df)
    if pts.empty:
        return [], []

    freq_alias = "M" if freq == "MS" else "W"
    dt = pts["datetime"].dt.tz_localize(None)
    period = dt.dt.to_period(freq_alias)
    full_range = pd.period_range(period.min(), period.max(), freq=freq_alias)

    grouped = {p: g for p, g in pts.assign(_p=period.values).groupby("_p")}
    frames: List[List[List[float]]] = []
    labels: List[str] = []
    for p in full_range:
        g = grouped.get(p)
        if g is not None and len(g) > max_points_per_frame:
            g = g.sample(n=max_points_per_frame, random_state=7)
        frames.append(g[["lat", "lon"]].values.tolist() if g is not None else [])
        labels.append(str(p))
    return frames, labels


def animated_heatmap(
    fmap: folium.Map,
    df: pd.DataFrame,
    freq: str = "MS",
    radius: int = 13,
) -> Tuple[folium.Map, Optional[str]]:
    """Attach a HeatMapWithTime (time-slider heat map) to fmap."""
    frames, labels = animation_frames(df, freq=freq)
    if not frames:
        return fmap, "No incidents with usable coordinates to animate."
    HeatMapWithTime(
        frames,
        index=labels,
        radius=radius,
        min_opacity=0.25,
        auto_play=False,
        max_speed=10,
        position="bottomleft",
    ).add_to(fmap)
    return fmap, None


def marker_cluster_layer(
    df: pd.DataFrame,
    max_points: int = _MAX_MARKERS,
) -> Tuple[Optional[folium.FeatureGroup], Optional[str]]:
    """Clustered individual incident markers with offense/date popups."""
    pts = valid_points(df)
    if pts.empty:
        return None, "No incidents with usable coordinates."

    total = len(pts)
    pts, sampled = _downsample(pts, max_points)
    note = (
        f"Markers show a random sample of {max_points:,} of {total:,} incidents."
        if sampled else None
    )

    layer = folium.FeatureGroup(name="Incidents", show=True)
    cluster = MarkerCluster().add_to(layer)
    for row in pts.itertuples(index=False):
        when = "" if pd.isna(row.datetime) else row.datetime.strftime("%Y-%m-%d %H:%M")
        nibrs = getattr(row, "nibrs_code", None) or ""
        popup = folium.Popup(
            f"<b>{row.category}</b> {('· ' + nibrs) if nibrs else ''}<br>{when}",
            max_width=260,
        )
        folium.CircleMarker(
            location=(row.lat, row.lon),
            radius=4,
            color=_CATEGORY_COLORS.get(str(row.category), "gray"),
            fill=True,
            fill_opacity=0.7,
            weight=1,
            popup=popup,
        ).add_to(cluster)
    return layer, note


def cfs_map_layer(cfs_df: pd.DataFrame) -> Tuple[Optional[folium.FeatureGroup], Optional[str]]:
    """Markers for recent calls-for-service (live-activity view)."""
    pts = cfs_df[
        cfs_df["lat"].notna() & cfs_df["lon"].notna() &
        (cfs_df["lat"] != 0.0) & (cfs_df["lon"] != 0.0)
    ] if not cfs_df.empty else cfs_df

    if pts.empty:
        return None, (
            "No mappable calls — this feed may publish block addresses without "
            "coordinates. See the table below."
        )

    pts, _ = _downsample(pts, _MAX_MARKERS)
    layer = folium.FeatureGroup(name="Calls for service", show=True)
    cluster = MarkerCluster().add_to(layer)
    for row in pts.itertuples(index=False):
        when = "" if pd.isna(row.datetime) else row.datetime.strftime("%Y-%m-%d %H:%M")
        addr = row.address or ""
        folium.CircleMarker(
            location=(row.lat, row.lon),
            radius=5,
            color="purple",
            fill=True,
            fill_opacity=0.75,
            weight=1,
            popup=folium.Popup(f"<b>{row.call_type}</b><br>{when}<br>{addr}", max_width=260),
        ).add_to(cluster)
    return layer, None
