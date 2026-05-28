# Crime Maps

An open, measure-agnostic platform for exploring recorded police incidents alongside
other measures of crime — starting with Charlotte, NC (CMPD incidents).

---

## Design principles

**Police incident records are one measure source, not "crime."**
This tool models CMPD reports as one source in a tidy canonical schema that future
sources (victimization surveys, ED admissions, 911/311 calls, etc.) can populate
without schema changes. The goal is to support comparison across sources, not to
treat any single source as ground truth.

**Per-capita rates, not raw counts, are the headline.**
All hotspot outputs are expressible as incidents per 1,000 residents using ACS
population at the census-tract level. Raw counts are a secondary toggle.

**Census tract is the stored unit of analysis.**
Tracts (~4,000 residents) give stable per-capita rates with manageable ACS margins
of error. Block-group estimates are too noisy for reliable rate maps.

---

## Data sources and caveats

| Source | URL | Vintage |
|--------|-----|---------|
| CMPD Incidents | [data.charlottenc.gov](https://data.charlottenc.gov/datasets/cmpd-incidents-1) | Live ArcGIS REST |
| Census tract boundaries | TIGERweb (2020 tracts) | 2020 |
| ACS population | [api.census.gov](https://api.census.gov) (B01003_001E) | 2018–2022 ACS 5-year |

**Important caveats:**
- Incident reports ≠ convictions. Records include all reported incidents regardless of outcome.
- NIBRS 800-series (non-criminal) records are excluded from analysis.
- CMPD periodically reclassifies and back-dates incidents. Every pull is snapshotted with a
  retrieval timestamp; results may differ across pulls of identical date ranges.
- **Denominator caveat:** population denominators use *residential* census-tract population
  from ACS. Tracts dominated by commercial, industrial, or institutional land use (e.g. airport,
  university, hospitals) have daytime populations far exceeding their residential count.
  Per-capita rates for such tracts are artificially inflated and should be interpreted with
  caution. Tracts below the configured minimum population threshold are suppressed on the
  rate map. Daytime-vs-residential adjustment is not implemented in v1.
- **Unassigned incidents:** points with null or zero coordinates, or that fall outside all
  census-tract boundaries, are retained in the dataset but excluded from per-capita rate maps.
  The unassigned count is displayed as a data-quality indicator in the UI.

**Census vintage alignment:** tract GEOIDs and ACS population data use the same vintage
(`tract_vintage = 2020`, `acs_release = "2018-2022"`) so geographic identifiers align.
Mismatched GEOIDs between the incidents and population tables are logged as warnings.

---

## Date semantics

ArcGIS date fields are epoch milliseconds, UTC. We convert to `America/New_York`
(Eastern time). The preferred date field is `DATE_INCIDENT_BEGAN` (occurrence date);
when that is null for a record, we fall back to `DATE_REPORTED`. Both field choices are
validated against the live layer metadata at fetch time.

---

## "Recent-Intensity Projection" tab

The third tab is **not a forecast.** It uses a recency-weighted kernel density estimator
(KDE) to visualize where incidents have been concentrated recently — older incidents are
exponentially down-weighted. This is a smoothed summary of inertia in historical data.
Coordinates are projected to UTM 17N (EPSG:32617) before fitting the KDE, so the
bandwidth slider is in meaningful meters. Bandwidth and recency half-life are tunable.

---

## Quick start

```bash
# Clone and install
git clone https://github.com/hijodeagua/crime-maps
cd crime-maps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the app
streamlit run src/crimemaps/app.py

# Run tests
pytest tests/
```

**Environment variables:**
- `CENSUS_API_KEY` — optional Census Bureau API key (app works without it at low volume)

---

## Data fallback tiers

When the live CMPD API is unreachable (network policy, maintenance), the app falls
back gracefully:

1. **Live** — ArcGIS REST API, paginated, snapshotted to `data/cache/`
2. **Snapshot** — most recent cached parquet (or a pinned retrieval timestamp)
3. **Demo** — synthetic incidents clustered around real Charlotte neighborhoods

The active tier and unassigned-incident rate are displayed in the UI header.

---

## Adding a city

1. Add a `CityConfig` entry to `src/crimemaps/config.py` with the city's ArcGIS
   endpoint, census FIPS codes, planar CRS, and NIBRS groups.
2. Register it in `CITIES`.
3. The city selector in the sidebar will pick it up automatically.

---

## Attribution

Crime data: City of Charlotte / CMPD. Available under the City of Charlotte open data
license. See [data.charlottenc.gov](https://data.charlottenc.gov) for terms.

Population data: U.S. Census Bureau, American Community Survey 5-Year Estimates.
