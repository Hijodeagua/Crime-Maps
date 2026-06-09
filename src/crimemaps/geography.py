"""
Census tract geography: boundary loading, spatial join, and CRS projection utilities.

Canonical geography unit: 11-digit census tract GEOID (state + county + tract).

Spatial-join miss handling
--------------------------
Points that fall outside all tract polygons (bad geocodes, null/zero coordinates,
or genuine edge cases) receive geography_id = schema.UNASSIGNED rather than being
dropped or causing an error. Callers track and surface the unassigned rate.

Boundary source hierarchy
--------------------------
1. Local cache (data/cache/<city>/boundaries/tracts.geojson)
2. TIGERweb REST API (requires network)
3. Bundled fixture (data/fixtures/tracts_sample.geojson) — dev/CI only
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from pyproj import Transformer
from shapely.geometry import Point

from crimemaps import schema
from crimemaps.config import CityConfig
from crimemaps.http import retrying_session

logger = logging.getLogger(__name__)

_BASE = Path(__file__).parents[2]
_FIXTURE_PATH = _BASE / "data" / "fixtures" / "tracts_sample.geojson"
_REQUEST_TIMEOUT = 45

TZ_LOCAL = "America/New_York"


def boundary_cache_path(city: CityConfig) -> Path:
    return _BASE / "data" / "cache" / city.slug / "boundaries" / "tracts.geojson"


def load_boundaries(city: CityConfig) -> gpd.GeoDataFrame:
    """
    Load census tract boundaries as a GeoDataFrame in WGS-84.
    Falls back through: cache → live API → fixture.
    """
    # 1. Local cache
    cached = boundary_cache_path(city)
    if cached.exists():
        logger.debug("Loading boundaries from cache: %s", cached)
        return gpd.read_file(cached)

    # 2. Live API
    if city.boundary_url:
        gdf = _fetch_boundaries(city)
        if gdf is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            gdf.to_file(cached, driver="GeoJSON")
            return gdf

    # 3. Fixture fallback
    if _FIXTURE_PATH.exists():
        logger.warning("Using bundled tract fixture — not real boundaries")
        return gpd.read_file(_FIXTURE_PATH)

    raise RuntimeError(
        "No tract boundaries available. "
        "Network is unreachable and no fixture found at %s" % _FIXTURE_PATH
    )


def _fetch_boundaries(city: CityConfig) -> Optional[gpd.GeoDataFrame]:
    """Fetch tract GeoJSON from TIGERweb or the configured boundary_url."""
    try:
        # The TIGERweb REST endpoint needs explicit state/county filter
        url = (
            "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
            "tigerWMS_Census2020/MapServer/8/query"
        )
        params = {
            "where": f"STATE='{city.census_state_fips}' AND COUNTY='{city.census_county_fips}'",
            "outFields": "GEOID,NAME",
            "f": "geojson",
            "returnGeometry": "true",
        }
        resp = retrying_session().get(url, params=params, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        gdf = gpd.read_file(resp.text, driver="GeoJSON")
        logger.info("Fetched %d tract boundaries from TIGERweb", len(gdf))
        return gdf
    # RuntimeError covers fiona/pyogrio GeoJSON parse failures
    except (requests.RequestException, ValueError, OSError, RuntimeError) as exc:
        logger.warning("Failed to fetch boundaries from TIGERweb: %s", exc)
        return None


def assign_geography(
    df: pd.DataFrame, city: CityConfig
) -> Tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Spatial join of incident lat/lon to census tract GEOID.

    Points with null/zero coordinates or that fall outside all tracts are assigned
    geography_id = schema.UNASSIGNED. Returns (enriched_df, boundaries_gdf).
    """
    boundaries = load_boundaries(city)
    if boundaries.crs is None:
        boundaries = boundaries.set_crs(city.wgs84_crs)
    elif boundaries.crs.to_epsg() != 4326:
        boundaries = boundaries.to_crs(city.wgs84_crs)

    # Ensure GEOID column is present
    geoid_col = _detect_geoid_column(boundaries)

    df = df.copy()
    df["geography_id"] = schema.UNASSIGNED

    # Identify joinable rows: non-null, non-zero coords
    mask = (
        df["lat"].notna() & df["lon"].notna() &
        (df["lat"] != 0.0) & (df["lon"] != 0.0) &
        df["lat"].between(-90, 90) & df["lon"].between(-180, 180)
    )
    joinable = df[mask].copy()

    if joinable.empty:
        return df, boundaries

    # Build GeoDataFrame from incident points
    geometry = gpd.points_from_xy(joinable["lon"], joinable["lat"])
    incidents_gdf = gpd.GeoDataFrame(joinable, geometry=geometry, crs=city.wgs84_crs)

    # Spatial join — left join keeps all incidents, unmatched get NaN GEOID
    joined = gpd.sjoin(
        incidents_gdf,
        boundaries[[geoid_col, "geometry"]],
        how="left",
        predicate="within",
    )

    # Map back GEOID; unmatched remain UNASSIGNED
    geoid_series = joined[geoid_col].fillna(schema.UNASSIGNED)
    df.loc[mask, "geography_id"] = geoid_series.values

    n_unassigned = (df["geography_id"] == schema.UNASSIGNED).sum()
    n_total = len(df)
    logger.info(
        "Geography join: %d/%d unassigned (%.1f%%)",
        n_unassigned, n_total, 100 * n_unassigned / n_total if n_total else 0,
    )

    return df, boundaries


def to_planar(
    lats: np.ndarray,
    lons: np.ndarray,
    city: CityConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project WGS-84 lat/lon to the city's planar CRS (e.g. UTM 17N). Returns (x, y)."""
    transformer = Transformer.from_crs(city.wgs84_crs, city.planar_crs, always_xy=True)
    x, y = transformer.transform(lons, lats)
    return x, y


def planar_to_wgs84(
    x: np.ndarray,
    y: np.ndarray,
    city: CityConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project planar coords back to WGS-84. Returns (lats, lons)."""
    transformer = Transformer.from_crs(city.planar_crs, city.wgs84_crs, always_xy=True)
    lons, lats = transformer.transform(x, y)
    return lats, lons


def _detect_geoid_column(gdf: gpd.GeoDataFrame) -> str:
    for candidate in ("GEOID", "geoid", "TRACTCE", "GEO_ID"):
        if candidate in gdf.columns:
            return candidate
    raise ValueError(
        "Cannot find GEOID column in boundaries GeoDataFrame. "
        f"Available columns: {list(gdf.columns)}"
    )
