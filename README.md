# Crime Maps

An open, measure-agnostic platform for exploring recorded police incidents alongside
other measures of crime — starting with Charlotte, NC (CMPD incidents), with
Raleigh, NC as a second (experimental) city.

**Views:** street-level heat maps (static + time-slider animation over up to 3 years),
per-capita census-tract choropleths, temporal trends, a recency-weighted intensity
projection, and a live-activity feed of 911 calls for service with scanner-audio links.

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

## "Street-Level Heat Map" tab

Renders the raw incident point cloud directly (folium HeatMap), so hot blocks and
corridors are visible at street zoom. Three sub-views:

- **Static heat map** over the selected period
- **Animated over time** — `HeatMapWithTime` with one frame per week (ranges ≤ ~6
  months) or per month (longer ranges), with a play/scrub slider covering up to the
  full 3-year window
- **Individual incidents** — clustered markers with offense/date popups, color-coded
  by category

Caveats: raw points are **not population-adjusted** — dense areas glow partly because
more people are there (the per-capita choropleth remains the analysis-grade view).
Agencies geocode to block midpoints, so apparent address precision is approximate.
Very large pulls are randomly downsampled for rendering (noted in the UI).

### Why the date range is capped at 3 years

Three years is supported and is the cap. The Charlotte dataset goes back further,
but (a) multi-year pulls are large (~100k+ records/year) and (b) upstream
reclassification/back-dating makes older records progressively less comparable
across pulls. A 3-year range is fetched once, snapshotted to `data/cache/`, and
subsequent requests inside that range can be served from disk via the
**"Prefer cached data"** sidebar toggle (the loader picks any snapshot whose stored
range covers the requested one).

---

## "Live Activity" tab — calls for service + scanner audio

Police **scanner audio** (Broadcastify, OpenMHz) is a live audio stream with no
structured API, so it can't be ingested into maps directly. The machine-readable
counterpart is the **calls-for-service (CFS)** dataset: every dispatched 911 /
officer-initiated call with timestamp, call type, and (usually block-level) location.
The Live Activity tab shows:

- recent calls on a map (when the layer publishes coordinates) and as a table
- top call types over a configurable lookback window (6 h – 7 days)
- links to scanner-audio directories for listening alongside

**Calls ≠ crimes:** many calls are unfounded, duplicated, or reclassified after
investigation. This is a "what is being dispatched right now" view, not a crime measure.

CFS field names drift across portals, so the configured `CFSFieldMapping` is treated
as a hint: the source resolves each canonical field against live layer metadata
(exact match first, then name/type heuristics) and falls back to snapshot → synthetic
demo tiers when the endpoint is unreachable. If the live CMPD CFS layer's endpoint or
fields change, fixing it is a one-line edit in `config.py`.

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

## Deployment & keeping the app awake

The app is deployed on **Streamlit Community Cloud** at
<https://tre-crime-maps.streamlit.app> and linked from the whosyurgoat hub.

**Real data on a blocked host.** Streamlit Cloud (and some city ArcGIS WAFs)
can't always reach the live endpoints, so a scheduled GitHub Action
(`.github/workflows/refresh-data.yml`) fetches real data from GitHub's runners
and commits snapshots into `data/cache/`. The deployed app clones the repo and
serves those snapshots via the **Snapshot** tier — the repo acts as a data relay.
Run it on demand from **Actions → "Refresh data snapshots" → Run workflow**, or
let the nightly schedule handle it. (Set a `CENSUS_API_KEY` repo secret to avoid
ACS rate limits.)

**Sleep / cold starts.** Free Streamlit Cloud apps sleep after a stretch of no
traffic and show a "waking up" screen on the next visit. Two ways to keep it warm:

1. **`.github/workflows/keep-warm.yml`** (in this repo) pings the app every ~10
   minutes. Zero setup, but GitHub cron is best-effort and auto-disables after 60
   days of repo inactivity.
2. **UptimeRobot** (recommended for reliability) — a free HTTP monitor on the
   same URL at a 5-minute interval keeps the app awake and alerts on downtime.

Truly always-on (no sleep at all) requires a non-free host — e.g. Render,
Railway, Fly.io, or a small VPS running `streamlit run`.

---

## Adding a city

1. Add a `CityConfig` entry to `src/crimemaps/config.py` with the city's ArcGIS
   endpoint, census FIPS codes, planar CRS, and NIBRS groups. Optionally add a
   `CFSConfig` (calls-for-service endpoint), `scanner_feeds` links, and
   `demo_clusters` for the synthetic fallback source.
2. Register it in `CITIES`.
3. The city selector in the sidebar will pick it up automatically.

The ArcGIS incident source (`sources/cmpd.py`) is generic — all city specifics come
from `CityConfig.field_mapping`, and date fields are validated against live layer
metadata at fetch time.

**Raleigh, NC** ships as a second city and is marked experimental: its endpoint and
field names are best-effort from the Raleigh open data portal and validated at
runtime. If the live layer disagrees, the app falls back to snapshot/demo tiers and
the data-source banner says so.

---

## Attribution

Crime data: City of Charlotte / CMPD. Available under the City of Charlotte open data
license. See [data.charlottenc.gov](https://data.charlottenc.gov) for terms.

Population data: U.S. Census Bureau, American Community Survey 5-Year Estimates.
