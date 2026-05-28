"""
Synthetic demo data source — dev/CI fallback when live API and snapshots are unavailable.

Generates realistic-looking incidents clustered around actual Charlotte neighborhoods,
emitting the canonical schema so all downstream code runs without modification.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from crimemaps import schema
from crimemaps.config import CityConfig
from crimemaps.sources.base import MeasureSource

logger = logging.getLogger(__name__)

TZ_LOCAL = "America/New_York"

# Rough cluster centers: (lat, lon, weight) for Charlotte neighborhoods
_CHARLOTTE_CLUSTERS = [
    (35.2271, -80.8431, 0.15),   # Uptown
    (35.1988, -80.8273, 0.10),   # South End / Dilworth
    (35.2560, -80.8070, 0.08),   # NoDa / Plaza Midwood
    (35.1720, -80.8700, 0.08),   # South Park
    (35.3140, -80.7510, 0.07),   # University City
    (35.2090, -80.9150, 0.07),   # Steele Creek / Berewick
    (35.2750, -80.8950, 0.07),   # Toringdon / Ballantyne W
    (35.1540, -80.8020, 0.07),   # Ballantyne
    (35.2840, -80.7940, 0.06),   # Concord Mills area
    (35.2430, -80.7640, 0.06),   # Mint Hill
    (35.3500, -80.8600, 0.06),   # Huntersville
    (35.1050, -80.8700, 0.06),   # Pineville
    (35.2300, -80.9400, 0.07),   # Airport / Berewick
]

_CATEGORIES = ["Violent", "Property", "Property", "Property", "Other"]
_NIBRS_BY_CATEGORY = {
    "Violent": ["13A", "13B", "120", "09A", "11A"],
    "Property": ["23A", "23B", "23F", "220", "290", "240"],
    "Other": ["90A", "90Z", "35A", "90D"],
}


class DemoSource(MeasureSource):
    source_slug = "cmpd_incidents"

    def fetch(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        n_per_day: float = 40.0,
        seed: Optional[int] = 42,
        retrieved_at: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Generate synthetic incidents for the given date range."""
        logger.warning("Using SYNTHETIC demo data — not real CMPD incidents")
        if retrieved_at is None:
            retrieved_at = pd.Timestamp.now(tz=TZ_LOCAL)

        rng = np.random.default_rng(seed)
        days = (end - start).days + 1
        n = max(1, int(round(n_per_day * days)))

        lats, lons = self._sample_locations(rng, n)
        datetimes = self._sample_datetimes(rng, start, end, n)
        categories = rng.choice(_CATEGORIES, size=n)
        nibrs_codes = [
            rng.choice(_NIBRS_BY_CATEGORY[c]) for c in categories
        ]

        df = pd.DataFrame({
            "measure_source": self.source_slug,
            "datetime": datetimes,
            "geography_id": schema.UNASSIGNED,
            "lat": lats,
            "lon": lons,
            "category": categories,
            "nibrs_code": nibrs_codes,
            "value": 1.0,
            "source_record_id": [f"demo-{i}" for i in range(n)],
            "retrieved_at": retrieved_at,
        })
        return self._finalize(df)

    def _sample_locations(self, rng: np.random.Generator, n: int):
        clusters = _CHARLOTTE_CLUSTERS
        weights = np.array([c[2] for c in clusters])
        weights /= weights.sum()
        chosen = rng.choice(len(clusters), size=n, p=weights)

        lats, lons = [], []
        for idx in chosen:
            clat, clon, _ = clusters[idx]
            lats.append(clat + rng.normal(0, 0.015))
            lons.append(clon + rng.normal(0, 0.018))
        return np.array(lats), np.array(lons)

    def _sample_datetimes(
        self,
        rng: np.random.Generator,
        start: pd.Timestamp,
        end: pd.Timestamp,
        n: int,
    ) -> pd.DatetimeTZDtype:
        start_ts = start.value
        end_ts = end.value
        random_ns = rng.integers(start_ts, end_ts, size=n)
        return pd.to_datetime(random_ns).tz_localize("UTC").tz_convert(TZ_LOCAL)
