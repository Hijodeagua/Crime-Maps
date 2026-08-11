# REPO-EVAL: MiroFish and open-source alternatives for Charlotte crime prediction

**Date:** 2026-08-11
**Scope:** Read-only evaluation. No installs run, no `.env` created, no API keys used, no
existing code modified. Target stack: Next.js 14 on Vercel, FastAPI on Render,
Postgres on Neon, Python, WSL.
**Method note:** GitHub metadata was gathered via public repo pages, commit Atom feeds, raw
manifest files, and the npm/PyPI registries (the session proxy blocks anonymous
`api.github.com`). Star counts are as displayed by GitHub (some rounded). Maintenance
cutoff: anything last pushed before **2025-02-11** (>18 months stale) is discarded.

---

## TL;DR

**MiroFish: REJECT.** It is an AGPL-3.0, Flask + Vue, LLM multi-agent *social simulation*
engine with **zero geospatial code** — no coordinates, polygons, rasters, gridding, map
layers, or time-indexed spatial events anywhere in the tree. It solves a different problem
(narrative/opinion prediction via LLM agent swarms), requires two paid external services
(any OpenAI-compatible LLM + Zep Cloud), and its AGPL license is incompatible with a
closed hosted app if you modify or link its code.

**Trial instead (ranked):**
1. **`tick` Hawkes baseline + hand-rolled PAI/PEI** — the actual predictive core.
2. **`h3-py` v4 + Neon's `h3`/`h3_postgis` Postgres extensions** — spatial gridding.
3. **deck.gl + MapLibre (`react-map-gl/maplibre`)** — heat/hex rendering in Next.js.

---

## TASK 1 — MiroFish recon (read-only)

### What it actually is

A "swarm intelligence prediction engine": upload seed documents (PDF/MD/TXT) → chunk text
→ build an entity/relationship knowledge graph in **Zep Cloud** → LLM-generate agent
personas → run **OASIS** (`camel-oasis`) Twitter/Reddit social simulations as subprocesses
→ an LLM ReportAgent writes a prediction report → chat with the simulated agents.
Incubated by Shanda Group; simulation engine is CAMEL-AI's OASIS (per README
acknowledgments).

### As-built architecture (differs from my stack on every layer)

| Layer | MiroFish | My stack |
|---|---|---|
| Backend | **Flask** 3.x, port 5001 (`backend/run.py`, `backend/app/__init__.py`) | FastAPI on Render |
| Frontend | **Vue 3 + Vite** (`frontend/package.json`: vue, vue-router, vue-i18n, axios, d3) | Next.js 14 / React 18 on Vercel |
| Storage | **No database.** Local filesystem: `backend/uploads/` + JSON task/project state in `backend/app/models/` (docker-compose volume-mounts `./backend/uploads`) | Postgres on Neon |
| Python | >=3.11,<3.13 (`backend/pyproject.toml`) | 3.11+ |
| Key deps | `zep-cloud==3.25.0`, `camel-oasis==0.2.5`, `camel-ai==0.2.78`, `openai`, `PyMuPDF`, Flask | — |
| Docker | Single container running **both dev servers** via `npm run dev`/concurrently (Dockerfile CMD) — not a production build | — |

Backend service modules (`backend/app/services/`): `graph_builder.py` (Zep standalone
graphs), `ontology_generator.py`, `simulation_config_generator.py`, `simulation_runner.py`
(subprocess + threading + IPC), `simulation_ipc.py`, `report_agent.py`, plus five modules
of Zep plumbing (`zep_tools.py`, `zep_graph_memory_updater.py`, `zep_entity_reader.py`,
utils `zep.py`, `zep_lifecycle.py`, `zep_paging.py`). API surface: three Flask blueprints
(`backend/app/api/graph.py`, `simulation.py`, `report.py`).

### Geospatial audit: nothing

Case-insensitive grep of the whole repo for `latitude|longitude|coord|polygon|raster|
geojson|spatial|geohash|h3|leaflet|mapbox|shapely|geopandas` yields only false positives:

- `scripts/star_history.py:906–915` — x/y **chart** coordinates for the README's
  star-history SVG (`x_coord`, `y_coord`, `line_coordinates`).
- `frontend/src/components/Step5Interaction.vue:194,308` and
  `frontend/src/components/Step4Report.vue:238` — SVG **icon** `<polygon>` elements.
- `<h3>` HTML heading tags throughout the Vue components.
- `backend/app/utils/zep_lifecycle.py:1` — "lifecycle **coordination**" in a docstring.

No geo dependencies in `backend/pyproject.toml` or `frontend/package.json`. No map
components, no tiles, no projections, no gridding.

**Time-indexed spatial events:** none. Simulation time is a synthetic round clock —
`minutes_per_round: int = 60` and `TimeSimulationConfig` in
`backend/app/services/simulation_config_generator.py:91,157`; agent actions are logged per
simulated round (`backend/scripts/action_logger.py`). These are timestamps inside an
artificial social world, never attached to a location.

### License: AGPL-3.0 (repo `LICENSE`, `package.json`, `backend/pyproject.toml`)

For a closed hosted app this is the strictest mainstream copyleft: AGPL §13's network-use
clause means that if you run **modified** MiroFish code (or code linking/importing it)
server-side, every network user must be offered the complete corresponding source of your
modified version — which would pull the crime-map service's backend under AGPL. Practical
options are: (a) run it only as a completely separate, unmodified service (disclosure
obligation limited to unmodified MiroFish itself), or (b) don't use the code. Copying
snippets into the FastAPI app is not an option without open-sourcing that app.

### External services it phones home to (runtime)

1. **Zep Cloud** — `https://api.getzep.com/api/v2` (`backend/app/utils/zep.py`). Hard
   requirement (`ZEP_API_KEY` in `.env.example`); SDK pinned `zep-cloud==3.25.0`; no
   self-host path in the code.
2. **An OpenAI-compatible LLM endpoint** — configurable `LLM_BASE_URL`, default
   `https://api.openai.com/v1` (`backend/app/config.py`); README recommends Alibaba
   Bailian/dashscope `qwen-plus`. Hard requirement, plus an optional second "boost"
   endpoint (`LLM_BOOST_*`).
3. **ghcr.io** — `ghcr.io/666ghj/mirofish:latest` image pull (`docker-compose.yml`).
4. CI-only: `api.github.com` (star-history workflow). Nothing else at runtime.

Cost note: the README itself warns simulations are token-hungry ("try fewer than 40
rounds first") — thousands of agents × rounds × LLM calls per prediction.

### Verdict for the crime map

Nothing to extract. Even the generic parts (Flask blueprints, Zep GraphRAG plumbing, Vue
wizard) target a different framework on every layer, and AGPL blocks code reuse. The only
conceivable use — LLM-agent simulation of neighborhood dynamics as a *narrative* companion
to the epidemiology framing — would be an expensive research toy, not a predictive
baseline, and would still run afoul of validation (no ground truth, no metrics).

---

## TASK 2 — Survey of alternatives

### (a) Spatiotemporal self-exciting point processes / Hawkes

| Repo | Stars | Last push | License | Python | Runnable example | Notes |
|---|---|---|---|---|---|---|
| [x-datainitiative/tick](https://github.com/x-datainitiative/tick) | 549 | 2026-05-04 (v0.8.0.2) | BSD-3 | **>=3.11** (PyPI; classifiers 3.11–3.14) | Yes — docs + example gallery | C++-accelerated multivariate temporal Hawkes (sim + parametric/nonparametric fit). No native continuous-space kernel — grid space into a multivariate process (standard Mohler-style approach). Recently revived. |
| [lmizrahi/etas](https://github.com/lmizrahi/etas) | 99 | 2026-04-10 | MIT | unclear | Yes — scripts + example catalogs | The only living **true spatiotemporal** ETAS (EM inversion + simulation); earthquake semantics (magnitude, completeness) need adapting to crime marks. |
| [ant-research/EasyTemporalPointProcess](https://github.com/ant-research/EasyTemporalPointProcess) | 351 | 2026-07-13 (v0.3.0) | Apache-2.0 | 3.9+ | Yes — 5 Colab notebooks | Neural TPP benchmark suite (NHP, THP, …). Marked-temporal only; no spatial kernel. |
| [pysal/pointpats](https://github.com/pysal/pointpats) | 93 | 2026-08-08 | BSD-3 | >=3.12 | Yes — docs + notebooks | Point-pattern *statistics* (K/L functions, quadrat, **Knox space-time interaction test**) — diagnostics complement, not a forecaster. |
| [stmorse/hawkes](https://github.com/stmorse/hawkes) | 178 | 2025-08-27 (cosmetic) | MIT | unclear | demo.ipynb | Teaching-grade MAP-EM multivariate Hawkes class; effectively dormant. |
| [slinderman/pyhawkes](https://github.com/slinderman/pyhawkes) | 249 | 2020-05-10 | MIT | unclear | Yes | **DISCARDED — unmaintained** (>18 mo). |
| [canerturkmen/hawkeslib](https://github.com/canerturkmen/hawkeslib) | 72 | 2022-06-03 | MIT | 2.7/3.6 | Yes | **DISCARDED — unmaintained**, author-deprecated. |
| [QuantCrimAtLeeds/PredictCode](https://github.com/QuantCrimAtLeeds/PredictCode) (open_cp) | 21 | 2018-02-03 | MIT | pre-3.7 era | Extensive notebooks | **DISCARDED — unmaintained.** Painful, because it's the one crime-specific SEPP/ETAS library ever built; useful as *reading material* for algorithm structure only. |
| [omitakahiro/Hawkes](https://github.com/omitakahiro/Hawkes) | 72 | 2021-04-01 | MIT | unclear | Colab | **DISCARDED — unmaintained.** |
| [Pat-Laub/hawkesbook](https://github.com/Pat-Laub/hawkesbook) | 32 | 2023-09-28 | MIT | unclear | README examples | **DISCARDED — unmaintained** (textbook companion). |

**Category bottom line:** no turnkey maintained spatiotemporal-Hawkes-for-crime library
exists. The viable maintained path is `tick` for the Hawkes core over H3/grid cells
(multivariate, Mohler-style), with `pointpats` for space-time diagnostics; `lmizrahi/etas`
is the fallback if a continuous-space kernel proves necessary.

### (b) Spatiotemporal GNNs on tract/grid adjacency

| Repo | Stars | Last push | License | Python | Runnable example | Notes |
|---|---|---|---|---|---|---|
| [benedekrozemberczki/pytorch_geometric_temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal) | ~3,000 | 2026-05-30 | MIT | >=3.6 declared; works on 3.11 w/ modern torch/PyG | Yes — examples/ + notebooks | DCRNN, TGCN, A3TGCN, EvolveGCN etc. `StaticGraphTemporalSignal` maps directly onto a fixed tract-adjacency graph. |
| [TorchSpatiotemporal/tsl](https://github.com/TorchSpatiotemporal/tsl) | 384 | 2026-06-18 | MIT | >=3.8 | Yes — Lightning-based examples | Cleaner data pipelines (DCRNN, Graph WaveNet, STGCN built in) but drags in PyTorch Lightning + Hydra. |
| [LibCity/Bigscity-LibCity](https://github.com/LibCity/Bigscity-LibCity) | ~1,200 | 2024-12-18 | Apache-2.0 | 3.7+ | Yes | **DISCARDED — unmaintained** (>18 mo). |
| [liyaguang/DCRNN](https://github.com/liyaguang/DCRNN) | ~1,400 | 2024-12-09 | MIT | TF 1.x era | Yes | **DISCARDED — unmaintained** reference impl. |
| [chnsh/DCRNN_PyTorch](https://github.com/chnsh/DCRNN_PyTorch) | 553 | 2019-11-04 | MIT | unclear | Yes | **DISCARDED — unmaintained.** |
| [nnzhan/Graph-WaveNet](https://github.com/nnzhan/Graph-WaveNet) | 810 | 2019-12-22 | MIT | "Python 3" | Yes | **DISCARDED — unmaintained.** |
| [akaxlh/ST-SHN](https://github.com/akaxlh/ST-SHN) (and sibling STHSL) | 22 | 2021-01-31 | none | unclear | Single research script | **DISCARDED — unmaintained, unlicensed** academic one-off; typical of every crime-specific ST-GNN repo found. |

**Category bottom line:** the crime-specific ST-GNN literature has produced no maintained
library — only stale paper dumps. PyTorch Geometric Temporal is the adoption target if/when
a neural model is warranted, with one hard caveat: torch + torch-geometric is a ~2 GB
install, impractical inside a Render FastAPI service. Train offline (WSL/worker), serve
exported predictions from Postgres; never import the training stack at serve time.

### (c) Crime forecasting evaluation metrics (PAI, PEI, hit rate)

| Repo | Stars | Last push | License | Lang | Notes |
|---|---|---|---|---|---|
| [apwheele/ptools](https://github.com/apwheele/ptools) | 5 | 2023-10-18 | MIT | **R** (CRAN) | The only *packaged* implementation found anywhere: `pai()` returns PAI, PEI, RRI. Stable but stale, and it's R. |
| [apwheele/crimepy](https://github.com/apwheele/crimepy) | 18 | 2026-04-10 | MIT | Python | Wheeler's maintained crime-analysis toolkit (aoristic, hotspot DBScan, WDD test) — **no PAI/PEI/hit-rate functions**. |
| [apwheele/Blog_Code](https://github.com/apwheele/Blog_Code) | 28 | 2026-07-16 | none | mixed | PAI/predictive-crime-curve snippets exist as blog companions, not a library. |
| [boldten/GraphTrace](https://github.com/boldten/GraphTrace) | 1 | 2025-09-22 | MIT | Python 3.8+ | Computes PAI internally to benchmark itself; not a reusable metrics API. |
| [adaj/predspot](https://github.com/adaj/predspot) | 1 | 2024-11-13 | BSD-3 | Python 3.8 | **DISCARDED — unmaintained**, self-described not production-ready. |
| [MichaelChirico/portland](https://github.com/MichaelChirico/portland) | 8 | 2017-11-10 | none | R/Python | **DISCARDED — unmaintained** NIJ challenge winner code; competition-specific scoring. |

Also ruled out after checking: `apwheele/retenmod` (police retention, unrelated),
`pysal/esda` (hotspot *detection* stats, not forecast evaluation), `srai`,
`scikit-mobility`, Stanford `openpolicing` — none package PAI/PEI/hit-rate.

**Category bottom line:** **no maintained Python library for PAI/PEI/hit-rate exists.**
These metrics are universally hand-rolled (~20–30 lines of pandas/numpy: rank cells by
predicted intensity, cumulative crime share vs. cumulative area share). The right move is
to port the well-documented logic of R `ptools::pai()` (MIT) into a small module in the
crime-map codebase — this is a half-day task, not a dependency decision.

### (d) H3 / geohash spatial gridding for Python

| Repo | Stars | Last push | License | Python | Runnable example | Notes |
|---|---|---|---|---|---|---|
| [uber/h3-py](https://github.com/uber/h3-py) | ~1.0k | 2026-08-03 (v4.5.0) | Apache-2.0 | >=3.10 (3.10–3.14) | Yes | Core H3 bindings. v4 renamed the whole API (`geo_to_h3`→`latlng_to_cell`) — ignore v3-era tutorials. |
| [postgis/h3-pg](https://github.com/postgis/h3-pg) | 381 | 2026-06-30 | Apache-2.0 | n/a (PG extension) | Yes (SQL) | H3 in Postgres; adopted by the PostGIS org late 2025. **Both `h3` and `h3_postgis` are on Neon's supported-extensions list** — hex aggregation can be plain SQL (`h3_lat_lng_to_cell(geom, res)` + GROUP BY). |
| [pysal/tobler](https://github.com/pysal/tobler) | 169 | 2026-08-09 | BSD-3 | **>=3.12** (v0.14.0; geopandas>=1, numpy>=2) | Yes | Areal interpolation / tract↔hex crosswalks (area-weighted, dasymetric). Caveat: current releases exclude 3.11 — run 3.12 or pin 0.12.x. |
| [nmandery/h3ronpy](https://github.com/nmandery/h3ronpy) | 117 | 2026-08-09 | MIT | >=3.9, Arrow-native | Yes | Rust-vectorized bulk point→cell; commits current but maintainer self-describes as no longer working much with H3 — maintenance-mode risk. |
| [kraina-ai/srai](https://github.com/kraina-ai/srai) | 385 | 2025-11-04 | Apache-2.0 | 3.9–3.12 | Yes | H3/S2/Voronoi regionalizers + OSM embedders; heavy tree (osmnx, polars, torch extras) — overkill for binning. |
| [wdm0006/pygeohash](https://github.com/wdm0006/pygeohash) | 177 | 2026-08-10 | MIT (≥3.0.0) | >=3.8 | Yes | Maintained geohash pick — only relevant if square/string-prefix cells are specifically wanted. |
| [hkwi/python-geohash](https://github.com/hkwi/python-geohash) | 326 | 2026-06-12 | tri-licensed | Py3 | Minimal | Alive but mostly dependabot churn. |
| [DahnJ/H3-Pandas](https://github.com/DahnJ/H3-Pandas) | 224 | 2025-03-02 | MIT | >=3.9 | Yes | Technically inside the 18-mo window by 3 weeks, but one cosmetic commit in ~17 months — **treat as dormant; skip** (a groupby on h3-py output replaces it). |

**Category bottom line:** `h3-py` v4 for Python-side assignment; push aggregation into
Neon via the PostGIS-org `h3`/`h3_postgis` extensions and keep the API thin; `tobler` for
hex↔tract crosswalks (mind the Python >=3.12 floor). Geohash only if hexes are rejected.

### (e) Heat map rendering layers for Next.js

| Repo | Stars | Last push | License | npm | Next.js notes | Notes |
|---|---|---|---|---|---|---|
| [visgl/deck.gl](https://github.com/visgl/deck.gl) | ~14.4k | 2026-08-11 | MIT | `deck.gl` / `@deck.gl/react` 9.3.10 | `next/dynamic` `ssr:false` required | HeatmapLayer + HexagonLayer (`@deck.gl/aggregation-layers`) + **H3HexagonLayer** (`@deck.gl/geo-layers`, pairs with `h3-js`) — the only stack covering all three natively. |
| [maplibre/maplibre-gl-js](https://github.com/maplibre/maplibre-gl-js) | ~11.3k | 2026-08-11 | **BSD-3** | `maplibre-gl` 6.3.0 | Client-only import | Native zoom-responsive `heatmap` paint layer + `fill` choropleths; no token, no vendor cost (vs mapbox-gl v2+ proprietary). |
| [visgl/react-map-gl](https://github.com/visgl/react-map-gl) | ~8.5k | 2026-08-06 | MIT | `react-map-gl` 8.1.2 | `'use client'` / dynamic import | Import from `react-map-gl/maplibre` for token-free MapLibre; official heatmap example. |
| [keplergl/kepler.gl](https://github.com/keplergl/kepler.gl) | ~12k | 2026-08-11 | MIT | 3.3.0-alpha.6 | Heavy; Redux required | Full geo-analytics UI on deck.gl — overkill for embedded layers; 3.x line still alpha-tagged. |
| [PaulLeCam/react-leaflet](https://github.com/PaulLeCam/react-leaflet) | ~5.6k | 2025-06-21 | Hippocratic-2.1 | v5 needs **React 19**; React 18 pins v4.2.1 | `ssr:false` mandatory | Slow cadence; nonstandard license worth noting; no WebGL aggregation. |
| [Leaflet/Leaflet.heat](https://github.com/Leaflet/Leaflet.heat) | ~1.7k | 2024-06-28 | BSD-2 | `leaflet.heat` 0.2.0 (npm last published 2015) | — | **DISCARDED — unmaintained** (last real release era 2015; 2024 commit was a dep bump). |
| [geoql/maplibre-gl-interpolate-heatmap](https://github.com/geoql/maplibre-gl-interpolate-heatmap) | 12 | 2026-03-08 | MIT | same | Client-only | IDW value-interpolation heatmap; 12★ single-maintainer — caution. |

**Category bottom line:** deck.gl 9 over MapLibre via `react-map-gl/maplibre` — all
pushed within days of this evaluation, MIT/BSD, zero token cost, and H3HexagonLayer means
the rendering grid can literally be the same H3 cells the model predicts on. For a
lighter footprint, MapLibre's built-in heatmap/fill paint layers alone suffice.

---

## TASK 3 — Scoring

Scale 1–5, **higher = better on every axis** (so "integration" = ease, "dep weight" =
lightness). License axis is scored against a **closed hosted app**.

| Candidate | Fit (ST crime pred.) | Integration ease | License compat | Maintenance | Dep lightness | Total /25 | Verdict |
|---|---|---|---|---|---|---|---|
| **h3-py v4 (+ Neon `h3_postgis`)** | 5 | 5 | 5 (Apache-2.0) | 5 | 5 | **25** | **ADOPT** |
| **deck.gl + MapLibre + react-map-gl** | 5 | 4 (ssr:false wrappers) | 5 (MIT/BSD-3) | 5 | 4 | **23** | **ADOPT** |
| **Hand-rolled PAI/PEI (port of `ptools::pai()` logic)** | 5 | 5 | 5 (MIT source, ~30 LOC) | n/a (own code) | 5 | — | **ADOPT** (build, not install) |
| **tick (Hawkes core)** | 4 (temporal Hawkes; space via H3 gridding) | 3 (C++ wheels, 3.11+ ok) | 5 (BSD-3) | 3 (revived 2026 after dormancy) | 3 | **18** | **TRIAL** |
| **pysal/pointpats (diagnostics)** | 3 (Knox test, K-functions — validation, not prediction) | 4 | 5 (BSD-3) | 4 | 4 | **20** | **ADOPT** (support role) |
| **pysal/tobler (tract↔hex crosswalk)** | 4 | 3 (needs Py≥3.12 or pin 0.12.x; geopandas≥1) | 5 (BSD-3) | 4 | 3 | **19** | **TRIAL** (adopt when on 3.12) |
| **PyTorch Geometric Temporal** | 4 (DCRNN/TGCN on tract adjacency) | 2 (offline training only; never on Render) | 5 (MIT) | 3 (single-maintainer history) | 1 (~2 GB torch+PyG) | **15** | **TRIAL** — phase 2, only after the Hawkes baseline sets a bar to beat |
| **TorchSpatiotemporal/tsl** | 4 | 2 | 5 (MIT) | 3 | 1 (adds Lightning+Hydra) | **15** | REJECT in favor of PGT (equivalent fit, more deps) |
| **lmizrahi/etas** | 4 (true ST kernel; earthquake semantics) | 2 | 5 (MIT) | 3 | 4 | **18** | REJECT for now — fallback if grid-Hawkes proves insufficient |
| **EasyTPP** | 2 (temporal-only neural TPP) | 2 | 5 (Apache-2.0) | 4 | 2 | **15** | REJECT |
| **apwheele/crimepy** | 2 (adjacent utilities; no PAI/PEI) | 4 | 5 (MIT) | 4 | 4 | **19** | REJECT as dependency; keep as reference |
| **srai** | 3 | 2 | 5 (Apache-2.0) | 4 | 1 | **15** | REJECT (overkill for binning) |
| **h3ronpy** | 4 | 4 | 5 (MIT) | 3 (self-declared low priority) | 4 | **20** | REJECT for now — revisit only if h3-py+SQL is too slow at CMPD data volume (it won't be) |
| **kepler.gl** | 3 | 2 (Redux, alpha 3.x) | 5 (MIT) | 4 | 1 | **15** | REJECT |
| **react-leaflet + Leaflet.heat** | 3 | 3 | 3 (Hippocratic license is nonstandard; heat plugin BSD) | 1 (heat dead; v5 needs React 19) | 3 | **13** | REJECT |
| **MiroFish** | **1** (zero geospatial/temporal-event capability) | **1** (Flask+Vue+filesystem vs FastAPI+Next+Postgres; every layer mismatched) | **1** (AGPL-3.0 network copyleft vs closed hosted app) | 3 (active, but young, single-org, CN-cloud-default) | **1** (camel-ai+OASIS+Zep Cloud+LLM spend per run) | **7** | **REJECT** |

Discarded outright for staleness (>18 months, per cutoff 2025-02-11): pyhawkes,
hawkeslib, open_cp/PredictCode, omitakahiro/Hawkes, hawkesbook, LibCity, DCRNN (both
impls), Graph-WaveNet, ST-SHN/STHSL, predspot, MichaelChirico/portland, Leaflet.heat.
Effectively dormant despite squeaking past the cutoff: H3-Pandas, stmorse/hawkes.

---

## Ranked shortlist — at most 3 to actually trial

1. **`tick` grid-Hawkes baseline + hand-rolled PAI/PEI** *(the decision-maker)*.
   Aggregate CMPD incidents to H3 cells (res ~8/9), fit a multivariate temporal Hawkes
   with `tick` (Mohler-style near-repeat structure — which is also the social-epidemiology
   story: contagion of events), forecast next-period intensity per cell, and score
   PAI/PEI/hit-rate against a naive "yesterday's hotspots" baseline. If grid-Hawkes can't
   beat naive persistence on PAI, nothing fancier is justified. Risk to watch: tick's
   2026 revival follows years of dormancy — pin the version and keep the model API
   swappable.

2. **`h3-py` v4 + Neon `h3`/`h3_postgis` extensions** *(the plumbing)*. Near-zero-risk
   adoption: `CREATE EXTENSION h3; CREATE EXTENSION h3_postgis;` on Neon, store cell IDs
   alongside the existing tract assignment, aggregate in SQL. Use `tobler` for hex↔tract
   crosswalks so per-capita (ACS tract denominators) and hex intensity views stay
   consistent — noting tobler's current releases want Python ≥3.12.

3. **deck.gl `H3HexagonLayer` + MapLibre in the Next.js hub** *(the rendering)*.
   `@deck.gl/react` + `@deck.gl/geo-layers` + `react-map-gl/maplibre` + `maplibre-gl`,
   wrapped in `next/dynamic({ ssr: false })` on Vercel. Predictions come out of Postgres
   keyed by the same H3 cell IDs the model was fit on — no re-projection layer in between.

Explicitly deferred: **PyTorch Geometric Temporal** (phase 2, offline-trained, only if the
Hawkes baseline's PAI plateaus and more expressiveness is warranted). Explicitly rejected:
**MiroFish** — wrong problem, wrong stack, wrong license.
