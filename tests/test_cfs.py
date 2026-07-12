"""Tests for the calls-for-service source: field resolution, parsing, demo feed."""

import pandas as pd
import pytest

from crimemaps.config import CHARLOTTE, RALEIGH
from crimemaps.sources import cfs


class TestResolveFields:
    META = [
        {"name": "OBJECTID", "type": "esriFieldTypeOID"},
        {"name": "CALL_RECEIVED_DT", "type": "esriFieldTypeDate"},
        {"name": "EVENT_TYPE_DESC", "type": "esriFieldTypeString"},
        {"name": "BLOCK_ADDRESS", "type": "esriFieldTypeString"},
        {"name": "PATROL_DIVISION", "type": "esriFieldTypeString"},
        {"name": "LATITUDE", "type": "esriFieldTypeDouble"},
        {"name": "LONGITUDE", "type": "esriFieldTypeDouble"},
    ]

    def test_exact_match_case_insensitive(self):
        resolved = cfs.resolve_fields(
            self.META,
            {"event_datetime": "call_received_dt", "call_type": "EVENT_TYPE_DESC"},
        )
        assert resolved["event_datetime"] == "CALL_RECEIVED_DT"
        assert resolved["call_type"] == "EVENT_TYPE_DESC"

    def test_heuristic_fallback_when_configured_name_missing(self):
        resolved = cfs.resolve_fields(
            self.META,
            {
                "event_datetime": "CALENDAR_DATE",   # not in layer
                "call_type": "CALL_TYPE",            # not in layer
                "address": "ADDRESS",                # not in layer
                "division": "DIVISION",              # not in layer
                "lat": None,
                "lon": None,
            },
        )
        assert resolved["event_datetime"] == "CALL_RECEIVED_DT"
        assert resolved["call_type"] == "EVENT_TYPE_DESC"
        assert resolved["address"] == "BLOCK_ADDRESS"
        assert resolved["division"] == "PATROL_DIVISION"
        assert resolved["lat"] == "LATITUDE"
        assert resolved["lon"] == "LONGITUDE"

    def test_date_field_must_be_esri_date_type(self):
        # A string field named like a date must not be picked for event_datetime
        meta = [
            {"name": "DATE_LABEL", "type": "esriFieldTypeString"},
        ]
        resolved = cfs.resolve_fields(meta, {"event_datetime": "DATE_LABEL"})
        assert resolved["event_datetime"] is None

    def test_unresolvable_optional_field_is_none(self):
        resolved = cfs.resolve_fields(
            [{"name": "WHEN_HAPPENED", "type": "esriFieldTypeDate"}],
            {"event_datetime": "X", "division": "DIVISION"},
        )
        assert resolved["division"] is None


class TestParse:
    def test_parse_features_to_cfs_schema(self):
        source = cfs.CFSSource(CHARLOTTE)
        fields = {
            "event_datetime": "CALENDAR_DATE",
            "call_type": "CALL_TYPE",
            "address": "ADDRESS",
            "division": "DIVISION",
            "record_id": "OBJECTID",
            "lat": None,
            "lon": None,
        }
        epoch_ms = int(pd.Timestamp("2024-06-01 12:00", tz="UTC").value // 1_000_000)
        features = [
            {
                "attributes": {
                    "CALENDAR_DATE": epoch_ms,
                    "CALL_TYPE": "Disturbance",
                    "ADDRESS": "100 TRADE ST",
                    "DIVISION": "Central",
                    "OBJECTID": 1,
                },
                "geometry": {"x": -80.84, "y": 35.22},
            },
            {
                "attributes": {
                    "CALENDAR_DATE": None,
                    "CALL_TYPE": None,
                    "OBJECTID": 2,
                },
            },
        ]
        retrieved = pd.Timestamp.now(tz="America/New_York")
        df = source._parse(features, fields, retrieved)

        assert list(df.columns) == cfs.CFS_COLUMNS
        assert len(df) == 2
        row = df[df["source_record_id"] == "1"].iloc[0]
        assert row["call_type"] == "Disturbance"
        assert row["lat"] == pytest.approx(35.22)
        assert row["datetime"].tz is not None
        # Missing call type defaults to Unknown; missing geometry → NaN coords
        row2 = df[df["source_record_id"] == "2"].iloc[0]
        assert row2["call_type"] == "Unknown"
        assert pd.isna(row2["lat"])

    def test_city_without_cfs_config_raises(self):
        with pytest.raises(ValueError):
            cfs.CFSSource(RALEIGH)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeSession:
    """Serves a scripted sequence of ArcGIS JSON payloads."""

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        payload = self.payloads[self.calls]
        self.calls += 1
        return _FakeResponse(payload)


def _feature(i):
    return {"attributes": {"OBJECTID": i}, "geometry": {"x": -80.8, "y": 35.2}}


class TestPartialPagination:
    FIELDS = {
        "event_datetime": "CALL_RECEIVED_DT",
        "call_type": None,
        "address": None,
        "division": None,
        "record_id": "OBJECTID",
        "lat": None,
        "lon": None,
    }
    START = pd.Timestamp("2024-06-01", tz="America/New_York")
    END = pd.Timestamp("2024-06-02", tz="America/New_York")

    def test_mid_pagination_error_returns_partial_and_sets_flag(self, monkeypatch):
        monkeypatch.setattr(cfs, "_PAGE_SIZE", 2)
        session = _FakeSession([
            {"features": [_feature(1), _feature(2)], "exceededTransferLimit": True},
            {"error": {"code": 500, "message": "boom"}},
        ])
        source = cfs.CFSSource(CHARLOTTE, session=session)
        features = source._paginate(self.FIELDS, self.START, self.END)

        assert len(features) == 2, "page-0 results must be kept"
        assert source.partial_error is not None
        assert "page 1" in source.partial_error

    def test_mid_pagination_exception_returns_partial_and_sets_flag(self, monkeypatch):
        monkeypatch.setattr(cfs, "_PAGE_SIZE", 2)

        class ExplodingSession(_FakeSession):
            def get(self, url, params=None, timeout=None):
                if self.calls >= 1:
                    raise ConnectionError("reset by peer")
                return super().get(url, params=params, timeout=timeout)

        session = ExplodingSession([
            {"features": [_feature(1), _feature(2)], "exceededTransferLimit": True},
        ])
        source = cfs.CFSSource(CHARLOTTE, session=session)
        features = source._paginate(self.FIELDS, self.START, self.END)

        assert len(features) == 2
        assert source.partial_error is not None
        assert "reset by peer" in source.partial_error

    def test_first_page_error_raises(self, monkeypatch):
        monkeypatch.setattr(cfs, "_PAGE_SIZE", 2)
        session = _FakeSession([{"error": {"code": 403, "message": "blocked"}}])
        source = cfs.CFSSource(CHARLOTTE, session=session)
        with pytest.raises(RuntimeError, match="blocked"):
            source._paginate(self.FIELDS, self.START, self.END)

    def test_complete_pagination_leaves_flag_unset(self, monkeypatch):
        monkeypatch.setattr(cfs, "_PAGE_SIZE", 2)
        session = _FakeSession([
            {"features": [_feature(1), _feature(2)], "exceededTransferLimit": True},
            {"features": [_feature(3)]},
        ])
        source = cfs.CFSSource(CHARLOTTE, session=session)
        features = source._paginate(self.FIELDS, self.START, self.END)
        assert len(features) == 3
        assert source.partial_error is None

    def test_fetch_resets_partial_error_between_calls(self, monkeypatch):
        monkeypatch.setattr(cfs, "_PAGE_SIZE", 2)
        epoch_ms = int(pd.Timestamp("2024-06-01 12:00", tz="UTC").value // 1_000_000)
        meta = {"fields": [{"name": "CALL_RECEIVED_DT", "type": "esriFieldTypeDate"}]}
        page = {"features": [
            {"attributes": {"CALL_RECEIVED_DT": epoch_ms}, "geometry": {"x": -80.8, "y": 35.2}},
        ]}
        session = _FakeSession([meta, page])
        source = cfs.CFSSource(CHARLOTTE, session=session)
        source.partial_error = "stale flag from an earlier failed fetch"
        df = source.fetch(self.START, self.END)
        assert source.partial_error is None
        assert len(df) == 1


class TestDemoCFS:
    def test_generates_calls_within_window(self):
        start = pd.Timestamp("2024-06-01", tz="America/New_York")
        end = pd.Timestamp("2024-06-02", tz="America/New_York")
        df = cfs.DemoCFSSource(CHARLOTTE).fetch(start, end, seed=1)

        assert list(df.columns) == cfs.CFS_COLUMNS
        assert len(df) > 0
        assert (df["datetime"] >= start).all() and (df["datetime"] <= end).all()
        assert df["lat"].between(34.5, 36.0).all()
        # Sorted newest-first for the live-feed table
        assert df["datetime"].is_monotonic_decreasing

    def test_works_for_city_without_demo_clusters(self):
        import dataclasses
        bare_city = dataclasses.replace(RALEIGH, demo_clusters=())
        start = pd.Timestamp("2024-06-01", tz="America/New_York")
        end = pd.Timestamp("2024-06-01 06:00", tz="America/New_York")
        df = cfs.DemoCFSSource(bare_city).fetch(start, end, seed=2)
        assert len(df) > 0
        # Falls back to map_center; calls should land near Raleigh
        assert df["lat"].between(35.0, 36.5).all()
