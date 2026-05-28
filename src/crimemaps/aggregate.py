"""
Aggregate canonical incidents to census-tract level.

Primary output: per-tract incident counts and per-capita rates.
Trend output: compare incident rate in recent half vs. earlier half of the date window.
"""

import logging

import pandas as pd

from crimemaps import schema
from crimemaps.config import CityConfig
from crimemaps.population import load_population, merge_population

logger = logging.getLogger(__name__)


def tract_counts(
    df: pd.DataFrame,
    city: CityConfig,
    categories: tuple = (),
) -> pd.DataFrame:
    """
    Aggregate incidents to tract level, compute per-capita rates.

    Excludes UNASSIGNED rows from rate calculations (they are retained in df).

    Returns DataFrame with columns:
        geography_id, count, population, rate_per_1k, rate_suppressed
    """
    assigned = df[df["geography_id"] != schema.UNASSIGNED].copy()

    if categories:
        assigned = assigned[assigned["category"].isin(categories)]

    if assigned.empty:
        return pd.DataFrame(columns=["geography_id", "count", "population", "rate_per_1k", "rate_suppressed"])

    counts = (
        assigned.groupby("geography_id")["value"]
        .sum()
        .reset_index()
        .rename(columns={"value": "count"})
    )

    population = load_population(city)
    merged = merge_population(counts, population, city)
    return merged


def temporal_profile(df: pd.DataFrame) -> dict:
    """
    Compute incident counts by day-of-week, hour-of-day, and month.
    Returns a dict of DataFrames keyed by 'dow', 'hour', 'month'.
    """
    result = {}
    if df.empty:
        for key in ("dow", "hour", "month"):
            result[key] = pd.DataFrame()
        return result

    dt = df["datetime"]
    dow = dt.dt.day_name()
    result["dow"] = (
        df.assign(dow=dow)
        .groupby("dow")["value"].sum()
        .reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
        .reset_index()
        .rename(columns={"value": "count"})
    )

    result["hour"] = (
        df.assign(hour=dt.dt.hour)
        .groupby("hour")["value"].sum()
        .reset_index()
        .rename(columns={"value": "count"})
    )

    result["month"] = (
        df.assign(month=dt.dt.tz_localize(None).dt.to_period("M").astype(str))
        .groupby("month")["value"].sum()
        .reset_index()
        .rename(columns={"value": "count"})
        .sort_values("month")
    )

    return result


def trend_by_tract(df: pd.DataFrame, city: CityConfig) -> pd.DataFrame:
    """
    Compare incident rates in the recent half vs. the earlier half of the date window.
    Returns [geography_id, count_recent, count_early, pct_change] for assigned tracts.
    """
    if df.empty:
        return pd.DataFrame()

    assigned = df[df["geography_id"] != schema.UNASSIGNED].copy()
    if assigned.empty:
        return pd.DataFrame()

    # Split on median datetime
    midpoint = assigned["datetime"].quantile(0.5)
    recent = assigned[assigned["datetime"] >= midpoint]
    early = assigned[assigned["datetime"] < midpoint]

    def _counts(subset):
        return (
            subset.groupby("geography_id")["value"]
            .sum()
            .rename("count")
        )

    merged = _counts(recent).rename("count_recent").to_frame().join(
        _counts(early).rename("count_early"),
        how="outer",
    ).fillna(0).reset_index()

    merged["pct_change"] = merged.apply(
        lambda r: (
            ((r["count_recent"] - r["count_early"]) / r["count_early"] * 100)
            if r["count_early"] > 0
            else float("nan")
        ),
        axis=1,
    )
    return merged
