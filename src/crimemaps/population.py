"""
ACS population denominators at the census-tract level.

Pulls ACS 5-year total population (B01003_001E) for a county via the Census Bureau API.
Vintage and ACS release are pinned in CityConfig (tract_vintage, acs_release) so that
the population GEOIDs align with the boundary file and incident assignments.

Fallback hierarchy
------------------
1. Local cache (data/cache/<city>/population/acs_<release>.parquet)
2. Live Census API (api.census.gov — works keyless at reasonable volume;
   set CENSUS_API_KEY env var if rate-limited)
3. Bundled static tract table (data/fixtures/population_sample.parquet) — dev/CI only

GEOIDs present in one side but not the other (incidents-without-population or
population-without-incidents) are logged rather than silently producing NaN rates.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from crimemaps.config import CityConfig

logger = logging.getLogger(__name__)

_BASE = Path(__file__).parents[2]
_CENSUS_BASE = "https://api.census.gov/data"
_REQUEST_TIMEOUT = 30
_FIXTURE_PATH = _BASE / "data" / "fixtures" / "population_sample.parquet"


def population_cache_path(city: CityConfig) -> Path:
    release = city.acs_release.replace("-", "_")
    return _BASE / "data" / "cache" / city.slug / "population" / f"acs_{release}.parquet"


def load_population(city: CityConfig) -> pd.DataFrame:
    """
    Return a DataFrame with columns [geography_id, population].
    geography_id is the 11-digit census tract GEOID.
    """
    # 1. Cache
    cached = population_cache_path(city)
    if cached.exists():
        logger.debug("Loading population from cache: %s", cached)
        return pd.read_parquet(cached)

    # 2. Live Census API
    pop_df = _fetch_acs(city)
    if pop_df is not None:
        cached.parent.mkdir(parents=True, exist_ok=True)
        pop_df.to_parquet(cached, index=False)
        return pop_df

    # 3. Fixture fallback
    if _FIXTURE_PATH.exists():
        logger.warning("Using bundled population fixture — not real ACS data")
        return pd.read_parquet(_FIXTURE_PATH)

    logger.error(
        "No population data available. App will run without per-capita rates."
    )
    return pd.DataFrame(columns=["geography_id", "population"])


def _fetch_acs(city: CityConfig) -> Optional[pd.DataFrame]:
    """Fetch ACS 5-year B01003_001E for all tracts in the county."""
    # ACS release "2018-2022" → endpoint year is 2022
    release_year = city.acs_release.split("-")[-1]
    url = f"{_CENSUS_BASE}/{release_year}/acs/acs5"
    params = {
        "get": "GEO_ID,B01003_001E",
        "for": "tract:*",
        "in": f"state:{city.census_state_fips} county:{city.census_county_fips}",
    }
    api_key = os.environ.get("CENSUS_API_KEY")
    if api_key:
        params["key"] = api_key

    try:
        resp = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("ACS API fetch failed: %s", exc)
        return None

    if len(data) < 2:
        logger.warning("ACS API returned empty response")
        return None

    headers, *rows = data
    df = pd.DataFrame(rows, columns=headers)

    # Build 11-digit GEOID from state + county + tract components
    df["geography_id"] = df["state"] + df["county"] + df["tract"]
    df["population"] = pd.to_numeric(df["B01003_001E"], errors="coerce").fillna(0).astype(int)

    result = df[["geography_id", "population"]].copy()
    logger.info(
        "ACS %s: loaded population for %d tracts (county %s%s)",
        city.acs_release, len(result), city.census_state_fips, city.census_county_fips,
    )
    return result


def merge_population(
    tract_counts: pd.DataFrame,
    population: pd.DataFrame,
    city: CityConfig,
) -> pd.DataFrame:
    """
    Left-join tract_counts onto population and compute per-capita rates.

    tract_counts must have columns [geography_id, count].
    Returns columns [geography_id, count, population, rate_per_1k, rate_suppressed].

    GEOIDs in one side only are logged as data-quality warnings.
    Tracts below city.min_population_for_rate have rate_suppressed=True.
    """
    pop = population.set_index("geography_id")
    counts = tract_counts.set_index("geography_id")

    # Log mismatches
    incidents_only = counts.index.difference(pop.index)
    population_only = pop.index.difference(counts.index)
    if len(incidents_only):
        logger.warning(
            "%d GEOIDs have incidents but no population: %s%s",
            len(incidents_only),
            list(incidents_only[:5]),
            " ..." if len(incidents_only) > 5 else "",
        )
    if len(population_only):
        logger.info(
            "%d GEOIDs have population but no incidents (expected for quiet tracts)",
            len(population_only),
        )

    merged = counts.join(pop, how="left").reset_index()
    merged["population"] = merged["population"].fillna(0).astype(int)
    merged["count"] = merged.get("count", merged.get("value", 0)).fillna(0)

    merged["rate_suppressed"] = merged["population"] < city.min_population_for_rate

    # Rate per 1,000 residents; NaN for suppressed / zero-pop tracts
    denom = merged["population"].where(~merged["rate_suppressed"], other=0)
    merged["rate_per_1k"] = merged.apply(
        lambda r: (r["count"] / r["population"] * 1000)
        if (not r["rate_suppressed"] and r["population"] > 0) else float("nan"),
        axis=1,
    )

    return merged
