"""
CMPD Incidents source — ArcGIS REST API.

Date semantics
--------------
ArcGIS date fields are epoch milliseconds, UTC. We convert to America/New_York
tz-aware datetimes. The preferred field is DATE_INCIDENT_BEGAN (occurrence); when
that is null for a record, we fall back to DATE_REPORTED. Both assumptions are
documented in provenance and logged.

Field validation
----------------
On each live fetch we query the layer metadata (?f=json) to confirm which date
fields exist and pick the authoritative one based on CityConfig.date_field_preference.
This guards against upstream schema changes.

Pagination
----------
We do NOT trust the advertised maxRecordCount. We page with resultOffset /
resultRecordCount and stop when:
  - A response returns fewer rows than requested (last page)
  - exceededTransferLimit is absent or false
  - A hard max-pages safety cap is reached
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from crimemaps import schema
from crimemaps.config import CityConfig
from crimemaps.sources.base import MeasureSource

logger = logging.getLogger(__name__)

_PAGE_SIZE = 1000
_MAX_PAGES = 500
_REQUEST_TIMEOUT = 30

TZ_LOCAL = "America/New_York"


class CMPDSource(MeasureSource):
    source_slug = "cmpd_incidents"

    def __init__(self, city: CityConfig, session: Optional[requests.Session] = None):
        super().__init__(city)
        self._session = session or requests.Session()
        self._validated_fields: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        retrieved_at: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Fetch CMPD incidents in [start, end] and return a canonical DataFrame."""
        if retrieved_at is None:
            retrieved_at = pd.Timestamp.now(tz=TZ_LOCAL)

        date_field, fallback_field = self._resolve_date_fields()
        logger.info(
            "CMPD fetch: %s → %s using date field '%s'",
            start.date(), end.date(), date_field,
        )

        raw_features = self._paginate(date_field, start, end)
        if not raw_features:
            logger.warning("CMPD fetch returned 0 features")
            return schema.empty()

        return self._parse(raw_features, date_field, fallback_field, retrieved_at)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_date_fields(self) -> Tuple[str, str]:
        """
        Query layer metadata to confirm authoritative date field.
        Returns (primary_field, fallback_field).
        Falls back to CityConfig defaults if metadata is unreachable.
        """
        fm = self.city.field_mapping
        default = (fm.occurrence_date, fm.report_date)

        try:
            resp = self._session.get(
                self.city.layer_meta_url, timeout=_REQUEST_TIMEOUT
            )
            resp.raise_for_status()
            meta = resp.json()
        except Exception as exc:
            logger.warning(
                "Could not fetch layer metadata (%s); using default field mapping", exc
            )
            return default

        date_fields = {
            f["name"]
            for f in meta.get("fields", [])
            if f.get("type") == "esriFieldTypeDate"
        }

        occurrence = fm.occurrence_date
        report = fm.report_date
        preference = self.city.date_field_preference

        if preference == "occurrence" and occurrence in date_fields:
            primary, fallback = occurrence, report
        elif report in date_fields:
            primary, fallback = report, occurrence
            if preference == "occurrence":
                logger.warning(
                    "Preferred date field '%s' not found in layer; "
                    "falling back to '%s'",
                    occurrence, report,
                )
        else:
            logger.warning(
                "Neither '%s' nor '%s' found in layer date fields %s; "
                "using config defaults",
                occurrence, report, date_fields,
            )
            primary, fallback = default

        self._validated_fields = meta
        return primary, fallback

    def _paginate(
        self,
        date_field: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> List[Dict[str, Any]]:
        """Page through the ArcGIS REST query endpoint and return all features."""
        fm = self.city.field_mapping
        # ArcGIS wants epoch ms for date comparisons in WHERE clauses
        start_ms = int(start.value // 1_000_000)
        end_ms = int(end.value // 1_000_000)

        where = f"{date_field} >= {start_ms} AND {date_field} <= {end_ms}"

        out_fields = [
            fm.occurrence_date, fm.report_date,
            fm.lat, fm.lon, fm.offense, fm.nibrs_code,
        ]
        if fm.division:
            out_fields.append(fm.division)
        if fm.record_id:
            out_fields.append(fm.record_id)

        features: List[Dict[str, Any]] = []
        offset = 0

        for page_num in range(_MAX_PAGES):
            params = {
                "where": where,
                "outFields": ",".join(out_fields),
                "returnGeometry": "true",
                "outSR": "4326",
                "resultOffset": offset,
                "resultRecordCount": _PAGE_SIZE,
                "f": "json",
            }
            try:
                resp = self._session.get(
                    self.city.query_url,
                    params=params,
                    timeout=_REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                logger.error("CMPD page %d fetch error: %s", page_num, exc)
                break

            if "error" in data:
                logger.error("ArcGIS error response: %s", data["error"])
                break

            page_features = data.get("features", [])
            features.extend(page_features)
            logger.debug("Page %d: got %d features (total %d)", page_num, len(page_features), len(features))

            # Stop conditions
            if len(page_features) < _PAGE_SIZE:
                break
            if not data.get("exceededTransferLimit", True):
                break

            offset += _PAGE_SIZE

        else:
            logger.warning("Reached max pages (%d); results may be truncated", _MAX_PAGES)

        return features

    def _parse(
        self,
        features: List[Dict[str, Any]],
        date_field: str,
        fallback_field: str,
        retrieved_at: pd.Timestamp,
    ) -> pd.DataFrame:
        """Convert raw ArcGIS features to the canonical schema DataFrame."""
        fm = self.city.field_mapping
        rows = []
        fallback_date_count = 0

        for feat in features:
            attrs = feat.get("attributes", {})
            geo = feat.get("geometry") or {}

            # Coordinates: prefer geometry block, fall back to lat/lon attribute fields
            lon = geo.get("x") or attrs.get(fm.lon)
            lat = geo.get("y") or attrs.get(fm.lat)

            # Date: occurrence preferred, fall back to report per-record
            epoch_ms = attrs.get(date_field)
            if epoch_ms is None:
                epoch_ms = attrs.get(fallback_field)
                if epoch_ms is not None:
                    fallback_date_count += 1

            if epoch_ms is not None:
                dt = pd.Timestamp(epoch_ms, unit="ms", tz="UTC").tz_convert(TZ_LOCAL)
            else:
                dt = pd.NaT

            nibrs_code = str(attrs.get(fm.nibrs_code) or "")
            offense = str(attrs.get(fm.offense) or "")

            # Exclude 800-series (non-criminal) if configured
            if self.city.exclude_800_series and nibrs_code.startswith("8"):
                continue

            category = self._classify(nibrs_code)

            record_id = str(attrs.get(fm.record_id, "")) if fm.record_id else None

            rows.append({
                "measure_source": self.source_slug,
                "datetime": dt,
                "geography_id": schema.UNASSIGNED,  # filled by geography.assign_geography
                "lat": float(lat) if lat is not None else float("nan"),
                "lon": float(lon) if lon is not None else float("nan"),
                "category": category,
                "nibrs_code": nibrs_code or None,
                "value": 1.0,
                "source_record_id": record_id,
                "retrieved_at": retrieved_at,
            })

        if fallback_date_count:
            logger.info(
                "%d/%d records used fallback date field '%s' (occurrence was null)",
                fallback_date_count, len(rows), fallback_field,
            )

        df = pd.DataFrame(rows)
        if df.empty:
            return schema.empty()

        return self._finalize(df)

    def _classify(self, nibrs_code: str) -> str:
        for category, prefixes in self.city.nibrs_groups.items():
            if category == "Other":
                continue
            if any(nibrs_code.startswith(p) for p in prefixes):
                return category
        if nibrs_code:
            return "Other"
        return "Unknown"
