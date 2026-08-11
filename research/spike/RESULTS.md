# Crime model spike — RESULTS

**Status: BLOCKED at Step 1 (data pull) by this session's network egress
policy. No model has seen real data; nothing below is a performance result.**
Steps 0 and the data-independent parts of Steps 2–5 are built and validated;
the pipeline is ready to run end-to-end the moment the data hosts are
reachable (from WSL, or from a session whose environment allows them).

## Step 0 — tick vs fallback: FALLBACK TAKEN

- `pip install tick` on Python 3.11.15: **succeeded in 37 s** (prebuilt
  manylinux wheel, tick 0.8.0.2, no C++ toolchain involved).
- The C++ simulation core works (`SimuHawkesExpKernels` simulated a 3-node
  multivariate Hawkes fine: `smoke_test_tick.py`).
- **Every inference learner is broken in this build**: `HawkesExpKern`,
  `HawkesADM4`, `HawkesEM`, `HawkesSumExpKern` all crash in the constructor
  or during `fit` with internal `AttributeError`s from tick's strict
  `_attrinfos` attribute whitelist (e.g. `'HawkesExpKern' object has no
  settable attribute 'events'`, `'History' object has no settable attribute
  '_minimum_col_width'`) on Python 3.11 + numpy 2.4. One cheap runtime patch
  of the attribute guard was attempted and did not take (the guard is
  generated per class); per the no-yak-shaving rule, stopped there.
- **Path taken: hand-rolled discrete-time exponential-kernel Hawkes MLE in
  numpy** (`hawkes_numpy.py`, ~85 lines): Mohler-style per-cell background
  (long-run share) + shared self- and ring-1-neighbor excitation, Poisson
  likelihood, L-BFGS-B. Validated by simulating from the generative model on
  a 100-cell × 2-year grid and recovering all four parameters within
  tolerance (`test_hawkes_numpy.py`).
- Eval-note: this is a *second* strike on tick beyond the maintenance-risk 3/5
  in REPO-EVAL.md — installable but unusable for inference in this exact
  target environment (py3.11/numpy2). Recommend demoting tick from TRIAL
  unless pinning an old numpy is acceptable.

## Step 1 — data: BLOCKED (environment, not the data)

The sandbox's egress policy denies CONNECT to every required host (gateway
403, confirmed for both shell tools and the WebFetch tool):

| Host | Needed for |
|---|---|
| `gis.charlottenc.gov` | CMPD incidents feed + city boundary layer |
| `data.charlottenc.gov` | portal metadata |
| `tigerweb.geo.census.gov` | tract boundaries (crosswalk, later) |
| `api.census.gov` | ACS denominators (later) |

Per the proxy's own policy ("do not route around, report the blocked host"),
no workaround was attempted. **Fix options:** (1) add these hosts to the
Claude Code environment's network allowlist and rerun
`data_pull.py` + `gridding.py` here, or (2) run those two scripts on WSL
(`pip install -r requirements.txt`, then `python data_pull.py`), which caches
raw pulls under `data/` (gitignored) and prints the full Step 1 report.

What `data_pull.py` will report when it runs (implemented, untested against
the live feed): row count, occurred/reported date ranges, null rates,
geocoding-failure rate (null or ~0,0 coordinates), reported-vs-occurred gap
at p50/p90/p99 and the share of gaps > 30 days, and rows lost to the
backfill guard.

- **Feed semantics:** the source is CMPD **reported incidents**
  (NIBRS-classified records: `DATE_INCIDENT_BEGAN`, `DATE_REPORTED`,
  `HIGHEST_NIBRS_CODE`) — NOT calls for service. The models therefore
  estimate *where incidents get reported and recorded*, which under-counts
  unreported crime and reflects reporting/patrol practice; the CFS feed
  (separate layer) would estimate demand for police response instead.
- **Backfill guard:** the most recent **30 days** (by report date) are
  dropped from all training and evaluation (`BACKFILL_DROP_DAYS` in
  `data_pull.py`), because CMPD back-dates and reclassifies records. Cost:
  the evaluation can never include the freshest month, so the spike measures
  performance as of "one month ago" and any very-short-lived hotspot dynamics
  inside that window are invisible; in production the same guard would delay
  model refresh by up to 30 days.

## Step 2 — units: FIXED IN CODE, INVENTORY PENDING DATA ACCESS

H3 **resolution 9** primary, clipped to the Charlotte city boundary
(`gridding.py`). The rule "write total cell count and cell area into
RESULTS.md before any model runs" is enforced mechanically:
`walkforward.py` refuses to start unless `gridding.py` has written the cell
inventory, and the inventory (n_cells, mean cell area ~0.105 km², total
area) must be pasted here before a real run.

> **Cell inventory: NOT YET COMPUTED — boundary layer unreachable (blocked
> host above). This section must be filled in before the first model run.**

Strata (no pooling; full pipeline per stratum), by `HIGHEST_NIBRS_CODE`:
burglary (220), robbery (120), aggravated assault / gun violence (13A),
motor vehicle theft (240).

## Step 3 — split: IMPLEMENTED

Walk-forward only (`walkforward.py`): 3-year rolling training window,
forecast the next 7 days, roll weekly; refuses to run with fewer than 24
folds; per-fold scores are written to `data/fold_scores.csv`.

## Step 4 — models: IMPLEMENTED, VALIDATED ON SYNTHETIC ONLY

Identical grid/horizon/stratum inputs for all four (`models.py`):
(a) long-run 24-month cumulative hotspot — the real baseline; (b) recency-
weighted counts (60-day half-life); (c) numpy grid-Hawkes (above);
(d) LightGBM 4.7, Poisson objective, 13 features (lags 1/2/3/7/14/28,
ring-1 neighbor lags 1/7/28, day-of-week, day-of-year sin/cos, long-run
cell rate). End-to-end harness dry run on labeled synthetic data:
`test_pipeline_dryrun.py` (all four models produce finite, correctly-shaped,
better-than-uniform scores on a persistent-hotspot toy world).

## Step 5 — scoring: IMPLEMENTED + UNIT-TESTED

`metrics.py` ports the logic of R `ptools::pai()`: hit rate, PAI, oracle
PAI*, PEI, whole-cell flagging with floor(a·n) cells and deterministic
tie-breaks. Unit tests against a fully hand-computed 4-cell toy example in
`test_metrics.py` (e.g. coverage 25%: PAI 1.2, PAI* 1.6, PEI 0.75).
Reported at 1%, 2%, 5% coverage; `walkforward.py` prints per-fold spreads
and fold-win counts vs baseline (a), per stratum — a model only "beats"
baseline by winning a majority of folds.

## Step 6 — output: PENDING DATA

The metric table, per-fold chart, LightGBM feature importances, and the
single recommendation cannot honestly be produced without real data. To
finish: unblock the hosts (or run on WSL) →
`python data_pull.py && python gridding.py && python walkforward.py`,
paste the Step 1 report and Step 2 inventory here, then fill the tables
from `data/fold_scores.csv`.

## Test suite

`pytest -q` → **12 passed** (metrics toy example ×7, Hawkes recovery ×3,
pipeline dry run ×2). Venv: Python 3.11.15, ~700 MB total (largest single
contributor: tick's transitive scipy/sklearn/matplotlib/pandas set, ~600 MB
combined — noted against the 500 MB flag rule; no torch, nothing GPU).
