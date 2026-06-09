"""Tests for street-level heat map data preparation."""

import folium
import numpy as np
import pandas as pd
import pytest

from crimemaps import streetmap


def _points_df(n=50, start="2024-01-01", days=90, seed=0):
    rng = np.random.default_rng(seed)
    dt = pd.to_datetime(start).tz_localize("America/New_York") + pd.to_timedelta(
        rng.integers(0, days * 24, n), unit="h"
    )
    return pd.DataFrame({
        "datetime": dt,
        "lat": 35.2 + rng.normal(0, 0.05, n),
        "lon": -80.8 + rng.normal(0, 0.05, n),
        "category": rng.choice(["Violent", "Property", "Other"], n),
        "nibrs_code": "23A",
        "geography_id": "37119000100",
        "value": 1.0,
    })


class TestValidPoints:
    def test_filters_null_and_zero_coords(self):
        df = _points_df(10)
        df.loc[0, "lat"] = np.nan
        df.loc[1, "lon"] = 0.0
        out = streetmap.valid_points(df)
        assert len(out) == 8

    def test_empty_in_empty_out(self):
        df = _points_df(0)
        assert streetmap.valid_points(df).empty


class TestHeatmapLayer:
    def test_returns_layer_for_valid_points(self):
        layer, note = streetmap.heatmap_layer(_points_df(100))
        assert isinstance(layer, folium.FeatureGroup)
        assert note is None

    def test_returns_none_when_no_points(self):
        layer, note = streetmap.heatmap_layer(_points_df(0))
        assert layer is None
        assert "No incidents" in note

    def test_downsamples_with_note(self):
        layer, note = streetmap.heatmap_layer(_points_df(200), max_points=50)
        assert layer is not None
        assert "sample" in note


class TestAnimationFrames:
    def test_monthly_frames_cover_full_range(self):
        df = _points_df(200, start="2024-01-01", days=120)
        frames, labels = streetmap.animation_frames(df, freq="MS")
        # 120 days from Jan 1 spans Jan..Apr (4-5 months depending on hours)
        assert len(frames) == len(labels)
        assert len(labels) >= 4
        assert labels == sorted(labels)
        assert sum(len(f) for f in frames) == 200

    def test_empty_period_yields_empty_frame(self):
        df = pd.concat([
            _points_df(10, start="2024-01-01", days=20),
            _points_df(10, start="2024-04-01", days=20),
        ])
        frames, labels = streetmap.animation_frames(df, freq="MS")
        # Feb/March gap must be present as empty frames, not skipped
        assert "2024-02" in labels
        assert frames[labels.index("2024-02")] == []

    def test_weekly_frames(self):
        df = _points_df(60, start="2024-01-01", days=28)
        frames, labels = streetmap.animation_frames(df, freq="W")
        assert 4 <= len(labels) <= 6

    def test_no_points_returns_empty(self):
        frames, labels = streetmap.animation_frames(_points_df(0))
        assert frames == [] and labels == []

    def test_frame_points_are_lat_lon_pairs(self):
        frames, _ = streetmap.animation_frames(_points_df(30))
        pt = next(p for f in frames for p in f)
        assert len(pt) == 2
        assert 34 < pt[0] < 37 and -82 < pt[1] < -79


class TestMarkerCluster:
    def test_builds_layer(self):
        layer, note = streetmap.marker_cluster_layer(_points_df(40))
        assert isinstance(layer, folium.FeatureGroup)
        assert note is None

    def test_downsample_note(self):
        layer, note = streetmap.marker_cluster_layer(_points_df(50), max_points=10)
        assert "sample" in note


class TestCFSMapLayer:
    def test_no_coords_message(self):
        df = pd.DataFrame({
            "datetime": [pd.Timestamp("2024-01-01", tz="America/New_York")],
            "call_type": ["Disturbance"],
            "address": ["100 block"],
            "division": ["D1"],
            "lat": [np.nan],
            "lon": [np.nan],
            "source_record_id": ["x"],
            "retrieved_at": [pd.Timestamp.now(tz="America/New_York")],
        })
        layer, note = streetmap.cfs_map_layer(df)
        assert layer is None
        assert "block addresses" in note
