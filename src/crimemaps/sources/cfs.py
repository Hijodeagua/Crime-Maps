"""
Calls-for-service (911 dispatch) source — ArcGIS REST API.

Why this exists
---------------
"Crime scanner" audio (Broadcastify, OpenMHz) has no structured API — it's a
live audio stream. The machine-readable counterpart is the calls-for-service
(CFS) dataset most large agencies publish: every dispatched 911/officer-initiated
call with a timestamp, call type, and (usually block-level) location. CFS is
noisier than incident reports — many calls are unfounded — but it is far closer
to real time and to "what is occurring" on the street, which makes it the right
feed for a live-activity view. Scanner-audio links are surfaced in the UI for
listening alongside.

Robust field resolution
-----------------------
City portals rename CFS fields more often than incident layers, so the
configured CFSFieldMapping is treated as a hint. On each live fetch we pull the
layer metadata and resolve each canonical field:

  1. exact configured name (case-insensitive)
  2. heuristic match on field name + esri type (date field for event_datetime,
     name containing TYPE/DESCRIPTION for call_type, ADDRESS/LOCATION for
     address, DIVISION/DISTRICT for division, LAT/LON for coordinates)

Unresolvable optional fields are simply omitted; an unresolvable event_datetime
aborts the live fetch (the loader then falls back to snapshot/demo tiers).

Output schema (NOT the canonical incident schema — CFS is an operational feed,
not an analysis measure):

    datetime, call_type, address, division, lat, lon, source_record_id, retrieved_at
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from crimemaps.config import CityConfig

logger = logging.getLogger(__name__)

SOURCE_SLUG = "calls_for_service"

CFS_COLUMNS = [
    "datetime", "call_type", "address", "division",
    "lat", "lon", "source_record_id", "retrieved_at",
]

_PAGE_SIZE = 1000
_MAX_PAGES = 200
_REQUEST_TIMEOUT = 30

TZ_LOCAL = "America/New_York"

# Heuristic name fragments per canonical field, tried in order
_NAME_HINTS: Dict[str, Tuple[str, ...]] = {
    "event_datetime": ("RECEIVED", "DISPATCH", "CALENDAR", "DATE", "TIME"),
    "call_type": ("CALL_TYPE", "CALLTYPE", "TYPE", "DESCRIPTION", "NATURE"),
    "address": ("ADDRESS", "LOCATION", "BLOCK"),
    "division": ("DIVISION", "DISTRICT", "BEAT", "ZONE"),
    "record_id": ("INCIDENT", "EVENT", "OBJECTID", "ID"),
    "lat": ("LATITUDE", "LAT", "Y"),
    "lon": ("LONGITUDE", "LON", "LNG", "X"),
}

_NUMERIC_TYPES = {"esriFieldTypeDouble", "esriFieldTypeSingle"}


def empty() -> pd.DataFrame:
    return pd.DataFrame(columns=CFS_COLUMNS)


def resolve_fields(
    meta_fields: List[Dict[str, Any]],
    configured: Dict[str, Optional[str]],
) -> Dict[str, Optional[str]]:
    """
    Resolve canonical CFS field names against live layer metadata.

    meta_fields: the "fields" list from ArcGIS layer metadata
                 (dicts with "name" and "type").
    configured:  canonical name -> configured layer field name (or None).

    Returns canonical name -> resolved layer field name (or None if absent).
    """
    by_upper = {f["name"].upper(): f["name"] for f in meta_fields}
    types = {f["name"]: f.get("type", "") for f in meta_fields}

    def type_ok(canonical: str, field_name: str) -> bool:
        ftype = types.get(field_name, "")
        if canonical == "event_datetime":
            return ftype == "esriFieldTypeDate"
        if canonical in ("lat", "lon"):
            return ftype in _NUMERIC_TYPES
        return True

    resolved: Dict[str, Optional[str]] = {}
    for canonical, cfg_name in configured.items():
        # 1. exact configured name, case-insensitive
        if cfg_name and cfg_name.upper() in by_upper:
            candidate = by_upper[cfg_name.upper()]
            if type_ok(canonical, candidate):
                resolved[canonical] = candidate
                continue
        # 2. heuristic by name fragment + type
        match = None
        for hint in _NAME_HINTS.get(canonical, ()):
            for upper_name, real_name in by_upper.items():
                if hint in upper_name and type_ok(canonical, real_name):
                    match = real_name
                    break
            if match:
                break
        resolved[canonical] = match
        if cfg_name and match and match != cfg_name:
            logger.info(
                "CFS field '%s': configured '%s' not in layer; resolved to '%s'",
                canonical, cfg_name, match,
            )
    return resolved


class CFSSource:
    """Live calls-for-service fetcher for cities with a configured CFSConfig."""

    source_slug = SOURCE_SLUG

    def __init__(self, city: CityConfig, session: Optional[requests.Session] = None):
        if city.cfs is None:
            raise ValueError(f"City '{city.slug}' has no calls-for-service endpoint configured")
        self.city = city
        self._session = session or requests.Session()

    def fetch(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        retrieved_at: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        if retrieved_at is None:
            retrieved_at = pd.Timestamp.now(tz=TZ_LOCAL)

        fields = self._resolve_live_fields()
        if fields["event_datetime"] is None:
            raise RuntimeError(
                "Could not resolve an event-datetime field on the CFS layer; "
                "check CFSFieldMapping against the live layer metadata"
            )

        features = self._paginate(fields, start, end)
        if not features:
            logger.warning("CFS fetch returned 0 features")
            return empty()
        return self._parse(features, fields, retrieved_at)

    def _resolve_live_fields(self) -> Dict[str, Optional[str]]:
        cfg = self.city.cfs
        fm = cfg.field_mapping
        configured = {
            "event_datetime": fm.event_datetime,
            "call_type": fm.call_type,
            "address": fm.address,
            "division": fm.division,
            "record_id": fm.record_id,
            "lat": fm.lat,
            "lon": fm.lon,
        }
        resp = self._session.get(cfg.layer_meta_url, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        meta = resp.json()
        if "error" in meta:
            raise RuntimeError(f"ArcGIS metadata error: {meta['error']}")
        return resolve_fields(meta.get("fields", []), configured)

    def _paginate(
        self,
        fields: Dict[str, Optional[str]],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> List[Dict[str, Any]]:
        date_field = fields["event_datetime"]
        start_ms = int(start.value // 1_000_000)
        end_ms = int(end.value // 1_000_000)
        where = f"{date_field} >= {start_ms} AND {date_field} <= {end_ms}"
        out_fields = [v for v in fields.values() if v]

        features: List[Dict[str, Any]] = []
        offset = 0
        for page_num in range(_MAX_PAGES):
            params = {
                "where": where,
                "outFields": ",".join(out_fields),
                "returnGeometry": "true",
                "outSR": "4326",
                "orderByFields": f"{date_field} DESC",
                "resultOffset": offset,
                "resultRecordCount": _PAGE_SIZE,
                "f": "json",
            }
            resp = self._session.get(
                self.city.cfs.query_url, params=params, timeout=_REQUEST_TIMEOUT
            )
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                logger.error("ArcGIS CFS error response: %s", data["error"])
                break

            page_features = data.get("features", [])
            features.extend(page_features)
            if len(page_features) < _PAGE_SIZE:
                break
            if not data.get("exceededTransferLimit", True):
                break
            offset += _PAGE_SIZE
        else:
            logger.warning("CFS: reached max pages (%d); results truncated", _MAX_PAGES)
        return features

    def _parse(
        self,
        features: List[Dict[str, Any]],
        fields: Dict[str, Optional[str]],
        retrieved_at: pd.Timestamp,
    ) -> pd.DataFrame:
        rows = []
        for feat in features:
            attrs = feat.get("attributes", {})
            geo = feat.get("geometry") or {}

            epoch_ms = attrs.get(fields["event_datetime"])
            dt = (
                pd.Timestamp(epoch_ms, unit="ms", tz="UTC").tz_convert(TZ_LOCAL)
                if epoch_ms is not None else pd.NaT
            )

            lat = geo.get("y")
            lon = geo.get("x")
            if lat is None and fields["lat"]:
                lat = attrs.get(fields["lat"])
            if lon is None and fields["lon"]:
                lon = attrs.get(fields["lon"])

            def attr(canonical):
                name = fields.get(canonical)
                val = attrs.get(name) if name else None
                return str(val) if val is not None else None

            rows.append({
                "datetime": dt,
                "call_type": attr("call_type") or "Unknown",
                "address": attr("address"),
                "division": attr("division"),
                "lat": float(lat) if lat is not None else float("nan"),
                "lon": float(lon) if lon is not None else float("nan"),
                "source_record_id": attr("record_id"),
                "retrieved_at": retrieved_at,
            })

        df = pd.DataFrame(rows, columns=CFS_COLUMNS)
        return df.sort_values("datetime", ascending=False).reset_index(drop=True)


_DEMO_CALL_TYPES = [
    ("Disturbance", 0.16),
    ("Suspicious Person/Vehicle", 0.14),
    ("Traffic Stop", 0.13),
    ("Alarm — Burglary", 0.10),
    ("Domestic Disturbance", 0.09),
    ("Larceny In Progress", 0.08),
    ("Accident With Injury", 0.08),
    ("Noise Complaint", 0.07),
    ("Assault In Progress", 0.05),
    ("Shots Fired", 0.04),
    ("Robbery In Progress", 0.03),
    ("Missing Person", 0.03),
]


class DemoCFSSource:
    """Synthetic calls-for-service feed for cities/networks without live CFS."""

    source_slug = SOURCE_SLUG

    def __init__(self, city: CityConfig):
        self.city = city

    def fetch(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        calls_per_hour: float = 9.0,
        seed: Optional[int] = None,
        retrieved_at: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        logger.warning("Using SYNTHETIC calls-for-service data — not a real dispatch feed")
        if retrieved_at is None:
            retrieved_at = pd.Timestamp.now(tz=TZ_LOCAL)
        rng = np.random.default_rng(seed)

        hours = max((end - start).total_seconds() / 3600.0, 1.0)
        n = max(1, int(round(calls_per_hour * hours)))

        clusters = list(self.city.demo_clusters) or [(*self.city.map_center, 1.0)]
        weights = np.array([c[2] for c in clusters])
        weights = weights / weights.sum()
        chosen = rng.choice(len(clusters), size=n, p=weights)
        lats = np.array([clusters[i][0] for i in chosen]) + rng.normal(0, 0.012, n)
        lons = np.array([clusters[i][1] for i in chosen]) + rng.normal(0, 0.015, n)

        type_labels = [t for t, _ in _DEMO_CALL_TYPES]
        type_weights = np.array([w for _, w in _DEMO_CALL_TYPES])
        call_types = rng.choice(type_labels, size=n, p=type_weights / type_weights.sum())

        ts = rng.integers(start.value, end.value, size=n)
        datetimes = pd.to_datetime(ts).tz_localize("UTC").tz_convert(TZ_LOCAL)

        df = pd.DataFrame({
            "datetime": datetimes,
            "call_type": call_types,
            "address": [f"{int(b)}00 block (synthetic)" for b in rng.integers(1, 99, n)],
            "division": [f"Division {d}" for d in rng.integers(1, 9, n)],
            "lat": lats,
            "lon": lons,
            "source_record_id": [f"demo-cfs-{i}" for i in range(n)],
            "retrieved_at": retrieved_at,
        })
        return df.sort_values("datetime", ascending=False).reset_index(drop=True)
