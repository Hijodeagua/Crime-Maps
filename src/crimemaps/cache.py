"""
Append-and-version snapshot cache.

Every pull is stored under a unique timestamped directory; nothing is ever overwritten.
A manifest.jsonl log tracks all pulls for reproducibility.

CMPD reclassifies and back-dates records, so identical queries can return different
history over time. Pinning to a specific retrieved_at timestamp lets analysis be
fully reproduced.

Directory layout
----------------
data/cache/<city>/<source>/<retrieved_at>__<start>_<end>/
    data.parquet   — canonical DataFrame
    manifest.jsonl — one-line log entry appended to data/cache/<city>/<source>/manifest.jsonl
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_BASE = Path(__file__).parents[2] / "data" / "cache"
_TS_FMT = "%Y%m%dT%H%M%S"


def _cache_dir(city_slug: str, source_slug: str) -> Path:
    return _BASE / city_slug / source_slug


def _snapshot_dir(
    city_slug: str,
    source_slug: str,
    retrieved_at: pd.Timestamp,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Path:
    ts = retrieved_at.strftime(_TS_FMT)
    date_range = f"{start.date()}_{end.date()}"
    return _cache_dir(city_slug, source_slug) / f"{ts}__{date_range}"


def save(
    df: pd.DataFrame,
    city_slug: str,
    source_slug: str,
    retrieved_at: pd.Timestamp,
    start: pd.Timestamp,
    end: pd.Timestamp,
    query_params: Optional[dict] = None,
) -> Path:
    """Persist df to a versioned snapshot and append to manifest. Returns snapshot dir."""
    snap = _snapshot_dir(city_slug, source_slug, retrieved_at, start, end)
    snap.mkdir(parents=True, exist_ok=True)

    parquet_path = snap / "data.parquet"
    df.to_parquet(parquet_path, index=False)

    manifest_path = _cache_dir(city_slug, source_slug) / "manifest.jsonl"
    entry = {
        "retrieved_at": retrieved_at.isoformat(),
        "city": city_slug,
        "source": source_slug,
        "start": start.date().isoformat(),
        "end": end.date().isoformat(),
        "row_count": len(df),
        "parquet": str(parquet_path.relative_to(_BASE)),
        "query_params": query_params or {},
    }
    with open(manifest_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

    logger.info("Cached %d rows → %s", len(df), snap)
    return snap


def list_snapshots(city_slug: str, source_slug: str) -> List[dict]:
    """Return manifest entries in reverse-chronological order."""
    manifest_path = _cache_dir(city_slug, source_slug) / "manifest.jsonl"
    if not manifest_path.exists():
        return []
    entries = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return sorted(entries, key=lambda e: e["retrieved_at"], reverse=True)


def load_latest(city_slug: str, source_slug: str) -> Optional[pd.DataFrame]:
    """Load the most recent snapshot, or None if no snapshots exist."""
    snaps = list_snapshots(city_slug, source_slug)
    if not snaps:
        return None
    return _load_parquet(snaps[0]["parquet"])


def load_snapshot(city_slug: str, source_slug: str, retrieved_at: str) -> Optional[pd.DataFrame]:
    """Load a specific snapshot by retrieved_at ISO string."""
    for snap in list_snapshots(city_slug, source_slug):
        if snap["retrieved_at"].startswith(retrieved_at):
            return _load_parquet(snap["parquet"])
    return None


def _load_parquet(rel_path: str) -> Optional[pd.DataFrame]:
    path = _BASE / rel_path
    if not path.exists():
        logger.warning("Snapshot parquet missing: %s", path)
        return None
    try:
        df = pd.read_parquet(path)
        # Restore tz-aware datetimes (parquet may strip tz info)
        for col in ("datetime", "retrieved_at"):
            if col in df.columns and hasattr(df[col], "dt"):
                if df[col].dt.tz is None:
                    df[col] = df[col].dt.tz_localize("America/New_York")
        return df
    # ArrowInvalid subclasses ValueError; OSError covers truncated/missing files
    except (OSError, ValueError, KeyError) as exc:
        logger.error("Failed to load parquet %s: %s", path, exc)
        return None
