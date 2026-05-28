"""
Data loading orchestrator with three-tier fallback:
  1. Live CMPD ArcGIS REST API
  2. Most recent local snapshot (cached parquet)
  3. Synthetic demo data

Returns the canonical DataFrame and a DataSourceInfo describing which tier was used.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from crimemaps import cache, schema
from crimemaps.config import CityConfig
from crimemaps.geography import assign_geography
from crimemaps.sources.cmpd import CMPDSource
from crimemaps.sources.demo import DemoSource

logger = logging.getLogger(__name__)

TZ_LOCAL = "America/New_York"


@dataclass
class DataSourceInfo:
    tier: str          # "live", "snapshot", or "demo"
    retrieved_at: Optional[str]
    row_count: int
    unassigned_count: int
    unassigned_pct: float
    date_field_used: Optional[str] = None
    fallback_date_count: int = 0


def load(
    city: CityConfig,
    start: pd.Timestamp,
    end: pd.Timestamp,
    categories: tuple = (),
    pinned_snapshot: Optional[str] = None,
) -> tuple[pd.DataFrame, DataSourceInfo]:
    """
    Load incidents for [start, end] for the given city and return
    (canonical_df, DataSourceInfo).

    If pinned_snapshot is set (a retrieved_at ISO prefix), load that specific snapshot.
    """
    retrieved_at = pd.Timestamp.now(tz=TZ_LOCAL)

    # --- Tier 0: pinned snapshot ---
    if pinned_snapshot:
        df = cache.load_snapshot(city.slug, "cmpd_incidents", pinned_snapshot)
        if df is not None:
            df = _filter(df, start, end, categories)
            df, info = _finalize(df, "snapshot", pinned_snapshot)
            return df, info

    # --- Tier 1: live API ---
    try:
        source = CMPDSource(city)
        df = source.fetch(start, end, retrieved_at=retrieved_at)
        if df is not None and not df.empty:
            df, boundaries = assign_geography(df, city)
            cache.save(df, city.slug, source.source_slug, retrieved_at, start, end)
            df = _filter(df, start, end, categories)
            df, info = _finalize(df, "live", retrieved_at.isoformat())
            return df, info
    except Exception as exc:
        logger.warning("Live CMPD fetch failed (%s); trying snapshot cache", exc)

    # --- Tier 2: latest snapshot ---
    df = cache.load_latest(city.slug, "cmpd_incidents")
    if df is not None and not df.empty:
        df = _filter(df, start, end, categories)
        snaps = cache.list_snapshots(city.slug, "cmpd_incidents")
        snap_ts = snaps[0]["retrieved_at"] if snaps else "cached"
        df, info = _finalize(df, "snapshot", snap_ts)
        return df, info

    # --- Tier 3: synthetic demo ---
    logger.warning("No live or cached data; using synthetic demo dataset")
    demo = DemoSource(city)
    df = demo.fetch(start, end, retrieved_at=retrieved_at)
    df, boundaries = assign_geography(df, city)
    df = _filter(df, start, end, categories)
    df, info = _finalize(df, "demo", retrieved_at.isoformat())
    return df, info


def _filter(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    categories: tuple,
) -> pd.DataFrame:
    if df.empty:
        return df
    # Normalize tz
    if df["datetime"].dt.tz is None:
        df = df.copy()
        df["datetime"] = df["datetime"].dt.tz_localize(TZ_LOCAL)
    start_tz = start.tz_localize(TZ_LOCAL) if start.tz is None else start
    end_tz = end.tz_localize(TZ_LOCAL) if end.tz is None else end
    mask = (df["datetime"] >= start_tz) & (df["datetime"] <= end_tz)
    df = df[mask]
    if categories:
        df = df[df["category"].isin(categories)]
    return df


def _finalize(
    df: pd.DataFrame,
    tier: str,
    retrieved_at: str,
) -> tuple[pd.DataFrame, DataSourceInfo]:
    n = len(df)
    if n > 0 and "geography_id" in df.columns:
        n_unassigned = (df["geography_id"] == schema.UNASSIGNED).sum()
    else:
        n_unassigned = 0
    pct = 100.0 * n_unassigned / n if n else 0.0
    info = DataSourceInfo(
        tier=tier,
        retrieved_at=retrieved_at,
        row_count=n,
        unassigned_count=int(n_unassigned),
        unassigned_pct=round(pct, 1),
    )
    return df, info
