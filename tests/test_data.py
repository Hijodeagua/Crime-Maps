"""
Tests for synthetic demo source and schema contract.
"""

import pandas as pd
import pytest

from crimemaps import schema
from crimemaps.config import CHARLOTTE
from crimemaps.sources.demo import DemoSource


def make_demo():
    return DemoSource(CHARLOTTE)


class TestDemoSource:
    def test_fetch_returns_canonical_schema(self):
        demo = make_demo()
        start = pd.Timestamp("2024-01-01", tz="America/New_York")
        end = pd.Timestamp("2024-01-31", tz="America/New_York")
        df = demo.fetch(start, end)
        schema.validate(df)
        assert set(schema.COLUMNS).issubset(set(df.columns))

    def test_all_values_are_one(self):
        demo = make_demo()
        start = pd.Timestamp("2024-01-01", tz="America/New_York")
        end = pd.Timestamp("2024-01-31", tz="America/New_York")
        df = demo.fetch(start, end)
        assert (df["value"] == 1.0).all()

    def test_categories_are_valid(self):
        demo = make_demo()
        start = pd.Timestamp("2024-01-01", tz="America/New_York")
        end = pd.Timestamp("2024-01-31", tz="America/New_York")
        df = demo.fetch(start, end)
        assert df["category"].isin(schema.CATEGORIES).all()

    def test_datetimes_are_tz_aware(self):
        demo = make_demo()
        start = pd.Timestamp("2024-01-01", tz="America/New_York")
        end = pd.Timestamp("2024-01-31", tz="America/New_York")
        df = demo.fetch(start, end)
        assert df["datetime"].dt.tz is not None

    def test_geography_id_defaults_to_unassigned(self):
        demo = make_demo()
        start = pd.Timestamp("2024-01-01", tz="America/New_York")
        end = pd.Timestamp("2024-01-31", tz="America/New_York")
        df = demo.fetch(start, end)
        assert (df["geography_id"] == schema.UNASSIGNED).all()

    def test_reproducible_with_same_seed(self):
        demo = make_demo()
        start = pd.Timestamp("2024-01-01", tz="America/New_York")
        end = pd.Timestamp("2024-01-07", tz="America/New_York")
        df1 = demo.fetch(start, end, seed=99)
        df2 = demo.fetch(start, end, seed=99)
        assert len(df1) == len(df2)
        assert (df1["lat"].values == df2["lat"].values).all()

    def test_nonzero_rows(self):
        demo = make_demo()
        start = pd.Timestamp("2024-01-01", tz="America/New_York")
        end = pd.Timestamp("2024-01-07", tz="America/New_York")
        df = demo.fetch(start, end)
        assert len(df) > 0


class TestSchemaEmpty:
    def test_empty_returns_correct_columns(self):
        df = schema.empty()
        assert set(schema.COLUMNS) == set(df.columns)
        assert len(df) == 0

    def test_validate_raises_on_missing_column(self):
        df = schema.empty().drop(columns=["value"])
        with pytest.raises(ValueError, match="missing columns"):
            schema.validate(df)

    def test_validate_raises_on_negative_value(self):
        df = schema.empty().copy()
        df = pd.concat([
            df,
            pd.DataFrame([{c: None for c in schema.COLUMNS}])
        ], ignore_index=True)
        df["value"] = -1.0
        with pytest.raises(ValueError, match="negative"):
            schema.validate(df)
