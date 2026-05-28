"""
Tests for geography spatial join and CRS projection.

Uses the bundled fixture GeoJSON so no network access is needed.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from crimemaps import schema
from crimemaps.config import CHARLOTTE
from crimemaps.geography import assign_geography, to_planar, planar_to_wgs84

FIXTURE_TRACTS = Path(__file__).parents[1] / "data" / "fixtures" / "tracts_sample.geojson"


def _fixture_boundaries():
    return gpd.read_file(FIXTURE_TRACTS)


def _make_incident_df(lats, lons, n=None):
    if n is None:
        n = len(lats)
    return pd.DataFrame({
        "measure_source": "cmpd_incidents",
        "datetime": pd.Timestamp("2024-01-15", tz="America/New_York"),
        "geography_id": schema.UNASSIGNED,
        "lat": lats,
        "lon": lons,
        "category": "Property",
        "nibrs_code": "23A",
        "value": 1.0,
        "source_record_id": [str(i) for i in range(n)],
        "retrieved_at": pd.Timestamp.now(tz="America/New_York"),
    })


class TestSpatialJoin:
    def test_point_in_tract_gets_geoid(self, monkeypatch):
        # Monkeypatch load_boundaries to return fixture
        import crimemaps.geography as geo_module
        monkeypatch.setattr(geo_module, "load_boundaries", lambda city: _fixture_boundaries())

        df = _make_incident_df([35.230], [-80.840])   # inside Tract 1 (Uptown)
        result, _ = assign_geography(df, CHARLOTTE)
        assert result.iloc[0]["geography_id"] == "37119000100"

    def test_point_outside_all_tracts_gets_unassigned(self, monkeypatch):
        import crimemaps.geography as geo_module
        monkeypatch.setattr(geo_module, "load_boundaries", lambda city: _fixture_boundaries())

        # Far outside all fixture tracts
        df = _make_incident_df([36.0], [-79.0])
        result, _ = assign_geography(df, CHARLOTTE)
        assert result.iloc[0]["geography_id"] == schema.UNASSIGNED

    def test_null_coords_get_unassigned(self, monkeypatch):
        import crimemaps.geography as geo_module
        monkeypatch.setattr(geo_module, "load_boundaries", lambda city: _fixture_boundaries())

        df = _make_incident_df([float("nan")], [float("nan")])
        result, _ = assign_geography(df, CHARLOTTE)
        assert result.iloc[0]["geography_id"] == schema.UNASSIGNED

    def test_zero_coords_get_unassigned(self, monkeypatch):
        import crimemaps.geography as geo_module
        monkeypatch.setattr(geo_module, "load_boundaries", lambda city: _fixture_boundaries())

        df = _make_incident_df([0.0], [0.0])
        result, _ = assign_geography(df, CHARLOTTE)
        assert result.iloc[0]["geography_id"] == schema.UNASSIGNED

    def test_unassigned_rows_retained_in_output(self, monkeypatch):
        import crimemaps.geography as geo_module
        monkeypatch.setattr(geo_module, "load_boundaries", lambda city: _fixture_boundaries())

        # Mix of inside and outside
        df = _make_incident_df([35.230, 36.0], [-80.840, -79.0])
        result, _ = assign_geography(df, CHARLOTTE)
        assert len(result) == 2   # no rows dropped
        assert result.iloc[1]["geography_id"] == schema.UNASSIGNED


class TestProjection:
    def test_round_trip_wgs84_to_planar_and_back(self):
        lats = np.array([35.2271, 35.25, 35.30])
        lons = np.array([-80.8431, -80.80, -80.75])
        x, y = to_planar(lats, lons, CHARLOTTE)
        lats2, lons2 = planar_to_wgs84(x, y, CHARLOTTE)

        np.testing.assert_allclose(lats, lats2, atol=1e-6)
        np.testing.assert_allclose(lons, lons2, atol=1e-6)

    def test_planar_units_are_meters(self):
        # Two points ~1 km apart in Charlotte
        lat1, lon1 = 35.2271, -80.8431
        lat2, lon2 = 35.2271, -80.8341   # ~0.009 deg longitude ≈ 770 m

        x1, y1 = to_planar(np.array([lat1]), np.array([lon1]), CHARLOTTE)
        x2, y2 = to_planar(np.array([lat2]), np.array([lon2]), CHARLOTTE)

        dist_m = np.sqrt((x2[0] - x1[0])**2 + (y2[0] - y1[0])**2)
        assert 700 < dist_m < 900, f"Expected ~770 m, got {dist_m:.0f} m"
