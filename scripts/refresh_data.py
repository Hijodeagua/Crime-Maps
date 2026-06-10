"""
Scheduled data refresh — fetches real data and writes it into data/cache/.

Run by .github/workflows/refresh-data.yml (nightly + manual) and committable
to the repo, so deployments that cannot reach the city APIs directly (e.g.
Streamlit Community Cloud behind an IP-blocking WAF) still serve real data
through the loader's snapshot tier.

Usage:
    python scripts/refresh_data.py [--city SLUG] [--days N] [--no-prune]

Exit status: 0 if incidents were fetched live for at least one city,
1 if every city failed (so CI surfaces the breakage).
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pandas as pd

from crimemaps import cache
from crimemaps.config import CITIES
from crimemaps.geography import assign_geography, load_boundaries
from crimemaps.population import load_population
from crimemaps.sources import cfs as cfs_source
from crimemaps.sources.cmpd import CMPDSource

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("refresh_data")

TZ_LOCAL = "America/New_York"
CFS_LOOKBACK_DAYS = 7


def refresh_city(slug: str, days: int, prune: bool) -> bool:
    """Fetch incidents (+ boundaries, population, CFS) for one city.
    Returns True if the incident fetch succeeded live."""
    city = CITIES[slug]
    now = pd.Timestamp.now(tz=TZ_LOCAL)
    start = (now - pd.Timedelta(days=days)).normalize()

    ok = False

    # Boundaries and population cache themselves to data/cache on success
    try:
        gdf = load_boundaries(city)
        logger.info("[%s] boundaries: %d tracts", slug, len(gdf))
    except Exception as exc:
        logger.error("[%s] boundaries failed: %s", slug, exc)
    try:
        pop = load_population(city)
        logger.info("[%s] population: %d tracts", slug, len(pop))
    except Exception as exc:
        logger.error("[%s] population failed: %s", slug, exc)

    # Incidents
    try:
        source = CMPDSource(city)
        df = source.fetch(start, now, retrieved_at=now)
        if df.empty:
            raise RuntimeError("live fetch returned 0 records")
        df, _ = assign_geography(df, city)
        cache.save(df, city.slug, source.source_slug, now, start, now)
        logger.info("[%s] incidents: %d rows (%s → %s)", slug, len(df), start.date(), now.date())
        ok = True
        if prune:
            n = cache.prune_old(city.slug, source.source_slug, keep=1)
            if n:
                logger.info("[%s] pruned %d old incident snapshots", slug, n)
    except Exception as exc:
        logger.error("[%s] incidents failed: %s", slug, exc)

    # Calls for service (when configured)
    if city.cfs is not None:
        try:
            cfs_start = now - pd.Timedelta(days=CFS_LOOKBACK_DAYS)
            cdf = cfs_source.CFSSource(city).fetch(cfs_start, now, retrieved_at=now)
            if not cdf.empty:
                cache.save(cdf, city.slug, cfs_source.SOURCE_SLUG, now, cfs_start, now)
                logger.info("[%s] calls-for-service: %d rows", slug, len(cdf))
                if prune:
                    cache.prune_old(city.slug, cfs_source.SOURCE_SLUG, keep=1)
            else:
                logger.warning("[%s] calls-for-service returned 0 rows", slug)
        except Exception as exc:
            logger.error("[%s] calls-for-service failed: %s", slug, exc)

    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", choices=list(CITIES.keys()), help="single city slug (default: all)")
    parser.add_argument("--days", type=int, default=None,
                        help="history window in days (default: each city's max_history_days)")
    parser.add_argument("--no-prune", action="store_true", help="keep old snapshots")
    args = parser.parse_args()

    slugs = [args.city] if args.city else list(CITIES.keys())
    results = {}
    for slug in slugs:
        days = args.days or CITIES[slug].max_history_days
        results[slug] = refresh_city(slug, days, prune=not args.no_prune)

    logger.info("Refresh summary: %s", results)
    return 0 if any(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
