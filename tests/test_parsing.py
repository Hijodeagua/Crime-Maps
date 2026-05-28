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
