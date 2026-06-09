"""
Synthetic demo data source — dev/CI fallback when live API and snapshots are unavailable.

Generates realistic-looking incidents clustered around the city's configured
demo_clusters (or synthesized clusters around map_center), emitting the canonical
schema so all downstream code runs without modification.
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

_CATEGORIES = ["Violent", "Property", "Property", "Property", "Other"]
_NIBRS_BY_CATEGORY = {
    "Violent": ["13A", "13B", "120", "09A", "11A"],
    "Property": ["23A", "23B", "23F", "220", "290", "240"],
    "Other": ["90A", "90Z", "35A", "90D"],
}


class DemoSource(MeasureSource):
    # Class-level default shadows the abstract property; overridden per-city in __init__
    source_slug = "police_incidents"

    def __init__(self, city: CityConfig):
        super().__init__(city)
        self.source_slug = city.incident_source_slug

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
        clusters = list(self.city.demo_clusters)
        if not clusters:
            # No configured clusters — synthesize a ring of clusters around map_center
            clat, clon = self.city.map_center
            clusters = [(clat, clon, 0.3)] + [
                (clat + 0.06 * np.cos(a), clon + 0.07 * np.sin(a), 0.1)
                for a in np.linspace(0, 2 * np.pi, 7, endpoint=False)
            ]
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
