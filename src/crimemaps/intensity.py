"""
Recent-intensity projection — recency-weighted KDE surface.

This is a SECONDARY, derived visualization layer. It projects recent inertia forward
using kernel density estimation, NOT a statistical forecast. It is labeled accordingly
everywhere in the UI.

Key design choices
------------------
- Coordinates are projected to the city's planar CRS (e.g. UTM 17N) before fitting KDE,
  so bandwidth is meaningful in meters rather than degrees.
- Bandwidth is explicit and tunable (not Scott/Silverman, which ignore recency weights).
- Recency weights: exponential decay by incident age (configurable half-life in days).
- Guard: fewer than 10 points after filtering → return None with a message rather than
  raising from scipy.

Output: a folium HeatMap layer (lat/lon back in WGS-84) and optionally a grid DataFrame.
"""

import logging
from typing import Optional, Tuple

import folium
import numpy as np
import pandas as pd
from folium.plugins import HeatMap
from scipy.stats import gaussian_kde

from crimemaps import schema
from crimemaps.config import CityConfig
from crimemaps.geography import to_planar, planar_to_wgs84

logger = logging.getLogger(__name__)

_MIN_POINTS = 10
_GRID_N = 60   # grid resolution per side


def recency_weights(
    datetimes: pd.Series,
    reference: pd.Timestamp,
    halflife_days: float,
) -> np.ndarray:
    """Exponential decay weights — older incidents count less."""
    ages_days = (reference - datetimes).dt.total_seconds() / 86400.0
    ages_days = np.clip(ages_days.values, 0, None)
    weights = np.exp(-np.log(2) * ages_days / halflife_days)
    total = weights.sum()
    return weights / total if total > 0 else weights


def intensity_layer(
    df: pd.DataFrame,
    city: CityConfig,
    bandwidth_m: float = 800.0,
    halflife_days: float = 90.0,
    reference: Optional[pd.Timestamp] = None,
) -> Tuple[Optional[folium.FeatureGroup], Optional[str]]:
    """
    Build a recency-weighted KDE HeatMap layer in planar coords projected back to WGS-84.

    Returns (folium_layer, error_message). error_message is None on success.
    """
    # Filter to assigned, valid-coord rows only
    valid = df[
        (df["geography_id"] != schema.UNASSIGNED) &
        df["lat"].notna() & df["lon"].notna() &
        (df["lat"] != 0.0) & (df["lon"] != 0.0)
    ].copy()

    if len(valid) < _MIN_POINTS:
        msg = (
            f"Too few points ({len(valid)}) for intensity projection "
            f"(minimum {_MIN_POINTS}). Try widening the date range or category filter."
        )
        logger.warning(msg)
        return None, msg

    if reference is None:
        reference = valid["datetime"].max()
        if pd.isna(reference):
            reference = pd.Timestamp.now(tz="America/New_York")

    weights = recency_weights(valid["datetime"], reference, halflife_days)

    # Project to planar CRS for geometrically correct KDE
    x, y = to_planar(valid["lat"].values, valid["lon"].values, city)

    try:
        kde = gaussian_kde(
            np.vstack([x, y]),
            bw_method=bandwidth_m / np.std(np.concatenate([x, y])),
            weights=weights,
        )
    except Exception as exc:
        msg = f"KDE fitting failed: {exc}"
        logger.error(msg)
        return None, msg

    # Evaluate on a planar grid covering the point cloud with 20% margin
    margin = bandwidth_m * 2
    xi = np.linspace(x.min() - margin, x.max() + margin, _GRID_N)
    yi = np.linspace(y.min() - margin, y.max() + margin, _GRID_N)
    xx, yy = np.meshgrid(xi, yi)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    # Normalize to [0, 1] for HeatMap
    zz_norm = (zz - zz.min()) / (zz.max() - zz.min() + 1e-12)

    # Project grid back to WGS-84
    grid_lats, grid_lons = planar_to_wgs84(xx.ravel(), yy.ravel(), city)
    grid_lats = grid_lats.reshape(xx.shape)
    grid_lons = grid_lons.reshape(xx.shape)

    # Build HeatMap data: list of [lat, lon, intensity]
    heat_data = [
        [grid_lats[i, j], grid_lons[i, j], float(zz_norm[i, j])]
        for i in range(_GRID_N)
        for j in range(_GRID_N)
        if zz_norm[i, j] > 0.05   # skip very-low-intensity grid cells
    ]

    layer = folium.FeatureGroup(name="Recent-Intensity Projection", show=True)
    HeatMap(
        heat_data,
        min_opacity=0.2,
        max_zoom=16,
        radius=18,
        blur=20,
    ).add_to(layer)

    return layer, None
