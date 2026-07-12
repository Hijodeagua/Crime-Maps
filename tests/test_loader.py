"""
Tests for loader staleness surfacing: data_age_hours and the is_stale flag.
"""

import pandas as pd
import pytest

from crimemaps import loader, schema
from crimemaps.loader import STALE_AFTER_HOURS, data_age_hours


NOW = pd.Timestamp("2026-07-10 12:00", tz=loader.TZ_LOCAL)


class TestDataAgeHours:
    def test_age_of_recent_timestamp(self):
        ts = (NOW - pd.Timedelta(hours=3)).isoformat()
        assert data_age_hours(ts, now=NOW) == pytest.approx(3.0)

    def test_age_of_old_snapshot(self):
        ts = (NOW - pd.Timedelta(days=5)).isoformat()
        assert data_age_hours(ts, now=NOW) == pytest.approx(120.0)

    def test_naive_timestamp_assumed_local(self):
        ts = (NOW - pd.Timedelta(hours=6)).tz_localize(None).isoformat()
        assert data_age_hours(ts, now=NOW) == pytest.approx(6.0)

    def test_future_timestamp_clamped_to_zero(self):
        ts = (NOW + pd.Timedelta(hours=1)).isoformat()
        assert data_age_hours(ts, now=NOW) == 0.0

    def test_unparseable_returns_none(self):
        # list_snapshots falls back to the literal string "cached" when the
        # manifest is missing — age must be unknown, not an exception.
        assert data_age_hours("cached", now=NOW) is None

    def test_none_and_empty_return_none(self):
        assert data_age_hours(None, now=NOW) is None
        assert data_age_hours("", now=NOW) is None


class TestStaleFlag:
    def _finalize_at_age(self, hours):
        retrieved = (
            pd.Timestamp.now(tz=loader.TZ_LOCAL) - pd.Timedelta(hours=hours)
        ).isoformat()
        _, info = loader._finalize(schema.empty(), "snapshot", retrieved)
        return info

    def test_fresh_snapshot_not_stale(self):
        info = self._finalize_at_age(hours=2)
        assert info.is_stale is False
        assert info.age_hours == pytest.approx(2.0, abs=0.1)

    def test_old_snapshot_flagged_stale(self):
        info = self._finalize_at_age(hours=STALE_AFTER_HOURS + 24)
        assert info.is_stale is True
        assert info.age_hours > STALE_AFTER_HOURS

    def test_just_under_threshold_not_stale(self):
        info = self._finalize_at_age(hours=STALE_AFTER_HOURS - 1)
        assert info.is_stale is False

    def test_unknown_age_not_flagged(self):
        _, info = loader._finalize(schema.empty(), "snapshot", "cached")
        assert info.age_hours is None
        assert info.is_stale is False

    def test_cfs_finalize_sets_staleness(self):
        from crimemaps.sources import cfs

        retrieved = (
            pd.Timestamp.now(tz=loader.TZ_LOCAL)
            - pd.Timedelta(hours=STALE_AFTER_HOURS + 10)
        ).isoformat()
        _, info = loader._finalize_cfs(cfs.empty(), "snapshot", retrieved)
        assert info.is_stale is True

    def test_default_info_has_no_partial_error(self):
        _, info = loader._finalize(schema.empty(), "live", NOW.isoformat())
        assert info.partial_error is None
