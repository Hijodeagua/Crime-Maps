"""
Tests for CMPD response parsing — uses the checked-in fixture.

Covers:
- Epoch-ms → tz-aware datetime parsing
- 800-series exclusion
- Per-record fallback to report date when occurrence is null
- Schema validation
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from crimemaps import schema
from crimemaps.config import CHARLOTTE
from crimemaps.sources.cmpd import CMPDSource, TZ_LOCAL

FIXTURE = Path(__file__).parents[1] / "data" / "fixtures" / "cmpd_sample_response.json"


def load_fixture() -> list:
    with open(FIXTURE) as f:
        data = json.load(f)
    return data["features"]


def make_source() -> CMPDSource:
    return CMPDSource(CHARLOTTE)


class TestEpochParsing:
    def test_epoch_ms_converted_to_tz_aware(self):
        source = make_source()
        features = load_fixture()
        fm = CHARLOTTE.field_mapping
        retrieved_at = pd.Timestamp.now(tz=TZ_LOCAL)
        df = source._parse(features, fm.occurrence_date, fm.report_date, retrieved_at)

        assert df["datetime"].dt.tz is not None, "datetime must be tz-aware"
        assert str(df["datetime"].dt.tz) == TZ_LOCAL

        # First feature: epoch 1704067200000 ms UTC = 2024-01-01 00:00:00 UTC
        # = 2023-12-31 19:00:00 EST
        first_row = df.iloc[0]
        assert first_row["datetime"].year == 2023 or first_row["datetime"].year == 2024


class TestExclusions:
    def test_800_series_excluded(self):
        source = make_source()
        features = load_fixture()
        fm = CHARLOTTE.field_mapping
        retrieved_at = pd.Timestamp.now(tz=TZ_LOCAL)
        df = source._parse(features, fm.occurrence_date, fm.report_date, retrieved_at)

        # Fixture has one 800-series record (NIBRS "820")
        assert "820" not in df["nibrs_code"].values, "800-series should be excluded"
        # We should have 4 records (5 total - 1 excluded)
        assert len(df) == 4


class TestFallbackDate:
    def test_null_occurrence_uses_report_date(self):
        source = make_source()
        features = load_fixture()
        fm = CHARLOTTE.field_mapping
        retrieved_at = pd.Timestamp.now(tz=TZ_LOCAL)
        df = source._parse(features, fm.occurrence_date, fm.report_date, retrieved_at)

        # Feature 3 has null DATE_INCIDENT_BEGAN; its datetime should come from DATE_REPORTED
        # and should NOT be NaT
        robbery_row = df[df["nibrs_code"] == "120"]
        assert len(robbery_row) == 1
        assert not pd.isna(robbery_row.iloc[0]["datetime"])


class TestSchemaValidation:
    def test_output_conforms_to_schema(self):
        source = make_source()
        features = load_fixture()
        fm = CHARLOTTE.field_mapping
        retrieved_at = pd.Timestamp.now(tz=TZ_LOCAL)
        df = source._parse(features, fm.occurrence_date, fm.report_date, retrieved_at)

        schema.validate(df)   # raises on violation
        assert set(schema.COLUMNS).issubset(set(df.columns))

    def test_value_is_always_one(self):
        source = make_source()
        features = load_fixture()
        fm = CHARLOTTE.field_mapping
        retrieved_at = pd.Timestamp.now(tz=TZ_LOCAL)
        df = source._parse(features, fm.occurrence_date, fm.report_date, retrieved_at)
        assert (df["value"] == 1.0).all()

    def test_measure_source_slug(self):
        source = make_source()
        features = load_fixture()
        fm = CHARLOTTE.field_mapping
        retrieved_at = pd.Timestamp.now(tz=TZ_LOCAL)
        df = source._parse(features, fm.occurrence_date, fm.report_date, retrieved_at)
        assert (df["measure_source"] == "cmpd_incidents").all()


class TestZeroCoordinates:
    """A legitimate 0.0 coordinate must not be treated as missing.

    Regression for `geo.get("x") or attrs.get(fm.lon)` — falsy-`or` made a
    genuine 0.0 fall through to the attribute field (or NaN).
    """

    EPOCH_MS = 1704067200000  # 2024-01-01 00:00:00 UTC

    def _parse_one(self, geometry, extra_attrs=None):
        source = make_source()
        fm = CHARLOTTE.field_mapping
        attrs = {
            fm.occurrence_date: self.EPOCH_MS,
            fm.report_date: self.EPOCH_MS,
            fm.nibrs_code: "120",
            fm.offense: "Robbery",
        }
        if extra_attrs:
            attrs.update(extra_attrs)
        feature = {"attributes": attrs}
        if geometry is not None:
            feature["geometry"] = geometry
        retrieved_at = pd.Timestamp.now(tz=TZ_LOCAL)
        df = source._parse([feature], fm.occurrence_date, fm.report_date, retrieved_at)
        assert len(df) == 1
        return df.iloc[0]

    def test_zero_geometry_coordinates_are_kept(self):
        row = self._parse_one({"x": 0.0, "y": 0.0})
        assert row["lon"] == 0.0
        assert row["lat"] == 0.0

    def test_zero_geometry_beats_attribute_fallback(self):
        fm = CHARLOTTE.field_mapping
        row = self._parse_one(
            {"x": 0.0, "y": 0.0},
            extra_attrs={fm.lon: -80.84, fm.lat: 35.22},
        )
        # Geometry block takes precedence even when its value is 0.0
        assert row["lon"] == 0.0
        assert row["lat"] == 0.0

    def test_missing_geometry_falls_back_to_attributes(self):
        fm = CHARLOTTE.field_mapping
        row = self._parse_one(None, extra_attrs={fm.lon: -80.84, fm.lat: 35.22})
        assert row["lon"] == pytest.approx(-80.84)
        assert row["lat"] == pytest.approx(35.22)

    def test_no_coordinates_anywhere_is_nan(self):
        row = self._parse_one(None)
        assert pd.isna(row["lon"])
        assert pd.isna(row["lat"])


class TestCMPDPartialPagination:
    """Mid-pagination failures must be signalled, not silently truncated."""

    class _FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    class _FakeSession:
        def __init__(self, payloads):
            self.payloads = list(payloads)
            self.calls = 0

        def get(self, url, params=None, timeout=None):
            payload = self.payloads[self.calls]
            self.calls += 1
            return TestCMPDPartialPagination._FakeResponse(payload)

    START = pd.Timestamp("2024-01-01", tz=TZ_LOCAL)
    END = pd.Timestamp("2024-01-31", tz=TZ_LOCAL)

    @staticmethod
    def _feature(i):
        return {"attributes": {"OBJECTID": i}, "geometry": {"x": -80.8, "y": 35.2}}

    def test_mid_pagination_error_sets_partial_flag(self, monkeypatch):
        import crimemaps.sources.cmpd as cmpd_mod

        monkeypatch.setattr(cmpd_mod, "_PAGE_SIZE", 2)
        session = self._FakeSession([
            {"features": [self._feature(1), self._feature(2)],
             "exceededTransferLimit": True},
            {"error": {"code": 500, "message": "boom"}},
        ])
        source = CMPDSource(CHARLOTTE, session=session)
        fm = CHARLOTTE.field_mapping
        features = source._paginate(fm.occurrence_date, self.START, self.END)

        assert len(features) == 2, "page-0 results must be kept"
        assert source.partial_error is not None
        assert "page 1" in source.partial_error

    def test_first_page_error_still_raises(self, monkeypatch):
        import crimemaps.sources.cmpd as cmpd_mod

        monkeypatch.setattr(cmpd_mod, "_PAGE_SIZE", 2)
        session = self._FakeSession([{"error": {"code": 403, "message": "blocked"}}])
        source = CMPDSource(CHARLOTTE, session=session)
        fm = CHARLOTTE.field_mapping
        with pytest.raises(RuntimeError, match="blocked"):
            source._paginate(fm.occurrence_date, self.START, self.END)

    def test_complete_pagination_leaves_flag_unset(self, monkeypatch):
        import crimemaps.sources.cmpd as cmpd_mod

        monkeypatch.setattr(cmpd_mod, "_PAGE_SIZE", 2)
        session = self._FakeSession([
            {"features": [self._feature(1), self._feature(2)],
             "exceededTransferLimit": True},
            {"features": [self._feature(3)]},
        ])
        source = CMPDSource(CHARLOTTE, session=session)
        fm = CHARLOTTE.field_mapping
        features = source._paginate(fm.occurrence_date, self.START, self.END)
        assert len(features) == 3
        assert source.partial_error is None
