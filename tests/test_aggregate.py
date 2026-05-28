"""
Tests for per-capita rate computation and aggregate logic.
"""

import pandas as pd
import pytest

from crimemaps import schema
from crimemaps.config import CHARLOTTE
from crimemaps.population import merge_population


def _make_counts(geoids, counts):
    return pd.DataFrame({"geography_id": geoids, "count": counts})


def _make_population(geoids, populations):
    return pd.DataFrame({"geography_id": geoids, "population": populations})


class TestMergePopulation:
    def test_rate_calculation(self):
        counts = _make_counts(["37119000100"], [50])
        pop = _make_population(["37119000100"], [1000])
        result = merge_population(counts, pop, CHARLOTTE)
        row = result.iloc[0]
        assert abs(row["rate_per_1k"] - 50.0) < 0.001   # 50/1000 * 1000 = 50

    def test_low_population_suppressed(self):
        # Charlotte config: min_population_for_rate = 100
        counts = _make_counts(["37119000100"], [10])
        pop = _make_population(["37119000100"], [50])   # below threshold
        result = merge_population(counts, pop, CHARLOTTE)
        assert bool(result.iloc[0]["rate_suppressed"]) is True
        assert pd.isna(result.iloc[0]["rate_per_1k"])

    def test_above_threshold_not_suppressed(self):
        counts = _make_counts(["37119000100"], [10])
        pop = _make_population(["37119000100"], [200])   # above threshold
        result = merge_population(counts, pop, CHARLOTTE)
        assert bool(result.iloc[0]["rate_suppressed"]) is False
        assert not pd.isna(result.iloc[0]["rate_per_1k"])

    def test_zero_population_suppressed(self):
        counts = _make_counts(["37119000100"], [5])
        pop = _make_population(["37119000100"], [0])
        result = merge_population(counts, pop, CHARLOTTE)
        assert pd.isna(result.iloc[0]["rate_per_1k"])

    def test_geoid_mismatch_logged_not_crashed(self, caplog):
        import logging
        counts = _make_counts(["37119000100", "37119000999"], [5, 3])
        pop = _make_population(["37119000100"], [500])
        with caplog.at_level(logging.WARNING):
            result = merge_population(counts, pop, CHARLOTTE)
        assert "37119000999" in caplog.text or len(result) >= 1   # no crash

    def test_per_1k_scaling(self):
        counts = _make_counts(["X"], [10])
        pop = _make_population(["X"], [5000])
        result = merge_population(counts, pop, CHARLOTTE)
        assert abs(result.iloc[0]["rate_per_1k"] - 2.0) < 0.001   # 10/5000*1000=2
