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
from crimemaps.sources import cfs as cfs_source
from crimemaps.sources.cmpd import CMPDSource
from crimemaps.sources.demo import DemoSource

logger = logging.getLogger(__name__)

TZ_LOCAL = "America/New_York"

# Data older than this is flagged as stale in DataSourceInfo (and the app
# shows a warning banner). The nightly refresh runs daily, so anything past
# 48h means at least two consecutive refresh runs failed to land.
STALE_AFTER_HOURS = 48


@dataclass
class DataSourceInfo:
    tier: str          # "live", "snapshot", or "demo"
    retrieved_at: Optional[str]
    row_count: int
    unassigned_count: int
    unassigned_pct: float
    date_field_used: Optional[str] = None
    fallback_date_count: int = 0
    # Why the live tier was skipped (shown in the UI when on snapshot/demo tiers)
    live_error: Optional[str] = None
    # Set when a live fetch succeeded but pagination failed partway — the
    # data shown is real but incomplete (see sources/*.partial_error)
    partial_error: Optional[str] = None
    # Age of the data (hours since retrieved_at); None if unknown
    age_hours: Optional[float] = None
    is_stale: bool = False


def data_age_hours(
    retrieved_at: Optional[str],
    now: Optional[pd.Timestamp] = None,
) -> Optional[float]:
    """Hours elapsed since retrieved_at (ISO string), or None if unparseable."""
    if not retrieved_at:
        return None
    try:
        ts = pd.Timestamp(retrieved_at)
    except (ValueError, TypeError):
        return None
    if pd.isna(ts):
        return None
    if ts.tz is None:
        ts = ts.tz_localize(TZ_LOCAL)
    if now is None:
        now = pd.Timestamp.now(tz=TZ_LOCAL)
    return max((now - ts).total_seconds() / 3600.0, 0.0)


def load(
    city: CityConfig,
    start: pd.Timestamp,
    end: pd.Timestamp,
    categories: tuple = (),
    pinned_snapshot: Optional[str] = None,
    prefer_cache: bool = False,
) -> tuple[pd.DataFrame, DataSourceInfo]:
    """
    Load incidents for [start, end] for the given city and return
    (canonical_df, DataSourceInfo).

    If pinned_snapshot is set (a retrieved_at ISO prefix), load that specific snapshot.
    If prefer_cache is set, a cached snapshot covering [start, end] is served
    without hitting the live API — the fast path for multi-year ranges.
    """
    retrieved_at = pd.Timestamp.now(tz=TZ_LOCAL)
    source_slug = city.incident_source_slug

    # --- Tier 0: pinned snapshot ---
    if pinned_snapshot:
        df = cache.load_snapshot(city.slug, source_slug, pinned_snapshot)
        if df is not None:
            df = _filter(df, start, end, categories)
            df, info = _finalize(df, "snapshot", pinned_snapshot)
            return df, info

    # --- Tier 0.5: covering snapshot, when the caller prefers cache ---
    if prefer_cache:
        covering = cache.find_covering(city.slug, source_slug, start, end)
        if covering:
            df = cache.load_snapshot(city.slug, source_slug, covering["retrieved_at"])
            if df is not None:
                df = _filter(df, start, end, categories)
                df, info = _finalize(df, "snapshot", covering["retrieved_at"])
                return df, info

    # --- Tier 1: live API ---
    live_error = None
    try:
        source = CMPDSource(city)
        df = source.fetch(start, end, retrieved_at=retrieved_at)
        if df is not None and not df.empty:
            df, boundaries = assign_geography(df, city)
            if source.partial_error:
                # Don't snapshot incomplete pulls — prune_old(keep=1) would
                # replace a complete snapshot with a truncated one.
                logger.warning(
                    "Live fetch was partial; not caching: %s", source.partial_error
                )
            else:
                cache.save(df, city.slug, source.source_slug, retrieved_at, start, end)
            df = _filter(df, start, end, categories)
            df, info = _finalize(df, "live", retrieved_at.isoformat())
            info.partial_error = source.partial_error
            return df, info
        live_error = "Live API returned 0 records for the requested range"
    except Exception as exc:
        live_error = str(exc)
        logger.warning("Live CMPD fetch failed (%s); trying snapshot cache", exc)

    # --- Tier 2: latest snapshot ---
    df = cache.load_latest(city.slug, source_slug)
    if df is not None and not df.empty:
        df = _filter(df, start, end, categories)
        snaps = cache.list_snapshots(city.slug, source_slug)
        snap_ts = snaps[0]["retrieved_at"] if snaps else "cached"
        df, info = _finalize(df, "snapshot", snap_ts)
        info.live_error = live_error
        return df, info

    # --- Tier 3: synthetic demo ---
    logger.warning("No live or cached data; using synthetic demo dataset")
    demo = DemoSource(city)
    df = demo.fetch(start, end, retrieved_at=retrieved_at)
    df, boundaries = assign_geography(df, city)
    df = _filter(df, start, end, categories)
    df, info = _finalize(df, "demo", retrieved_at.isoformat())
    info.live_error = live_error
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
    age = data_age_hours(retrieved_at)
    info = DataSourceInfo(
        tier=tier,
        retrieved_at=retrieved_at,
        row_count=n,
        unassigned_count=int(n_unassigned),
        unassigned_pct=round(pct, 1),
        age_hours=age,
        is_stale=age is not None and age > STALE_AFTER_HOURS,
    )
    return df, info


def load_cfs(
    city: CityConfig,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, DataSourceInfo]:
    """
    Load calls-for-service (911 dispatch) for [start, end] with the same tiered
    fallback as incidents: live → latest snapshot → synthetic demo.

    Returns a CFS-schema DataFrame (see sources/cfs.py), NOT the canonical
    incident schema — CFS is an operational live-activity feed.
    """
    retrieved_at = pd.Timestamp.now(tz=TZ_LOCAL)
    slug = cfs_source.SOURCE_SLUG

    if city.cfs is not None:
        try:
            source = cfs_source.CFSSource(city)
            df = source.fetch(start, end, retrieved_at=retrieved_at)
            if df is not None and not df.empty:
                if source.partial_error:
                    # Don't snapshot incomplete pulls — prune_old(keep=1) would
                    # replace a complete snapshot with a truncated one.
                    logger.warning(
                        "Live CFS fetch was partial; not caching: %s",
                        source.partial_error,
                    )
                else:
                    cache.save(df, city.slug, slug, retrieved_at, start, end)
                df, info = _finalize_cfs(df, "live", retrieved_at.isoformat())
                info.partial_error = source.partial_error
                return df, info
        except Exception as exc:
            logger.warning("Live CFS fetch failed (%s); trying snapshot cache", exc)

    df = cache.load_latest(city.slug, slug)
    if df is not None and not df.empty:
        mask = (df["datetime"] >= _tz(start)) & (df["datetime"] <= _tz(end))
        df = df[mask]
        if not df.empty:
            snaps = cache.list_snapshots(city.slug, slug)
            snap_ts = snaps[0]["retrieved_at"] if snaps else "cached"
            return _finalize_cfs(df, "snapshot", snap_ts)

    logger.warning("No live or cached CFS data; using synthetic demo feed")
    df = cfs_source.DemoCFSSource(city).fetch(start, end, retrieved_at=retrieved_at)
    return _finalize_cfs(df, "demo", retrieved_at.isoformat())


def _tz(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.tz_localize(TZ_LOCAL) if ts.tz is None else ts


def _finalize_cfs(df: pd.DataFrame, tier: str, retrieved_at: str) -> tuple[pd.DataFrame, DataSourceInfo]:
    n = len(df)
    n_unmapped = int((df["lat"].isna() | (df["lat"] == 0.0)).sum()) if n else 0
    age = data_age_hours(retrieved_at)
    info = DataSourceInfo(
        tier=tier,
        retrieved_at=retrieved_at,
        row_count=n,
        unassigned_count=n_unmapped,
        unassigned_pct=round(100.0 * n_unmapped / n, 1) if n else 0.0,
        age_hours=age,
        is_stale=age is not None and age > STALE_AFTER_HOURS,
    )
    return df, info
