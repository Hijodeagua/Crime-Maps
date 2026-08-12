"""Step 1: pull CMPD incidents (ArcGIS REST), cache raw pages, build parquet,
and print the data-quality report.

Feed: CMPD **reported incidents** (NIBRS-classified records), NOT calls for
service — the target variable is therefore "recorded incident reports", with
all the reporting/recording biases that implies (the model estimates where
incidents get REPORTED, not where crime occurs).

Usage:
  .venv/bin/python data_pull.py            # precheck + pull + report
  .venv/bin/python data_pull.py --report   # report from cache only
  .venv/bin/python data_pull.py --precheck # layer metadata + counts only
  .venv/bin/python data_pull.py --tzcheck  # 500-row hour-of-day timezone probe

Raw pages land in data/raw/ (gitignored — never commit incident data);
the assembled table in data/incidents.parquet.

Timezone: esriFieldTypeDate values arrive as epoch milliseconds. Whether
they represent true UTC instants or local wall time stored as-if-UTC is
verified empirically by --tzcheck (hour-of-day distribution of
DATE_REPORTED); the adopted interpretation is applied in _to_datetime and
recorded in RESULTS.md.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = ("https://gis.charlottenc.gov/arcgis/rest/services/CMPD/"
            "CMPDIncidents/MapServer/0")
QUERY_URL = f"{BASE_URL}/query"
LAYER_META_URL = f"{BASE_URL}?f=json"

LAT, LON = "LATITUDE_PUBLIC", "LONGITUDE_PUBLIC"

# Coordinates, occurred-from, and the stratum key are load-bearing: without
# any of them the spike is pointless, so their absence aborts the pull.
REQUIRED_FIELDS = ["OBJECTID", "DATE_INCIDENT_BEGAN", LAT, LON,
                   "HIGHEST_NIBRS_CODE"]
# These degrade with a warning: END feeds the censoring audit, the
# descriptive fields feed the address-granularity audit.
OPTIONAL_FIELDS = ["DATE_INCIDENT_END", "DATE_REPORTED",
                   "HIGHEST_NIBRS_DESCRIPTION", "DIVISION_ID",
                   "ADDRESS_DESCRIPTION", "LOCATION_TYPE_DESCRIPTION", "NPA"]
FIELDS = REQUIRED_FIELDS + OPTIONAL_FIELDS

PAGE = 2000
DATA = Path(__file__).parent / "data"
RAW = DATA / "raw"
PARQUET = DATA / "incidents.parquet"

# Backfill guard: drop this many most-recent days from ALL train/eval use.
BACKFILL_DROP_DAYS = 30

# Adopted after the --tzcheck probe (see RESULTS.md): epoch-ms values decode
# to plausible local wall time directly, so no timezone shift is applied.
# Flip to "utc_to_eastern" only if the probe shows a 4-5 h shifted profile.
TZ_INTERPRETATION = "naive_local"

STRATA = {  # NIBRS code -> stratum
    "220": "burglary",
    "120": "robbery",
    "13A": "agg_assault_gun",
    "240": "mv_theft",
}


def _get(params: dict) -> dict:
    r = requests.get(QUERY_URL, params={"f": "json", **params}, timeout=120)
    r.raise_for_status()
    payload = r.json()
    if "error" in payload:
        raise RuntimeError(f"ArcGIS error: {payload['error']}")
    return payload


def _to_datetime(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, unit="ms", errors="coerce")
    if TZ_INTERPRETATION == "utc_to_eastern":
        ts = (ts.dt.tz_localize("UTC").dt.tz_convert("America/New_York")
                .dt.tz_localize(None))
    return ts


def partition_fields(live_fields: set[str]) -> list[str]:
    """Degradation policy: abort on missing required fields, warn on the rest."""
    missing_req = [f for f in REQUIRED_FIELDS if f not in live_fields]
    if missing_req:
        raise RuntimeError(
            f"Layer is missing REQUIRED fields {missing_req}; aborting pull. "
            f"Live schema must provide {REQUIRED_FIELDS}.")
    missing_opt = [f for f in OPTIONAL_FIELDS if f not in live_fields]
    if missing_opt:
        print(f"WARNING: layer lacks optional fields {missing_opt}; "
              "pulling without them")
    return [f for f in FIELDS if f in live_fields]


def layer_meta() -> dict:
    r = requests.get(LAYER_META_URL, timeout=60)
    r.raise_for_status()
    return r.json()


def precheck() -> dict:
    """Layer capabilities + full-coverage numbers, before any multi-year pull."""
    meta = layer_meta()
    adv = meta.get("advancedQueryCapabilities", {})
    total = _get({"where": "1=1", "returnCountOnly": "true"})["count"]
    stats = _get({
        "where": "1=1", "outStatistics": json.dumps([
            {"statisticType": "min", "onStatisticField": "DATE_INCIDENT_BEGAN",
             "outStatisticFieldName": "occ_min"},
            {"statisticType": "max", "onStatisticField": "DATE_INCIDENT_BEGAN",
             "outStatisticFieldName": "occ_max"},
        ])})["features"][0]["attributes"]
    info = {
        "maxRecordCount": meta.get("maxRecordCount"),
        "supportsPagination": adv.get("supportsPagination"),
        "supportsStatistics": meta.get("supportsStatistics"),
        "total_rows": total,
        "occ_min": str(pd.to_datetime(stats["occ_min"], unit="ms")),
        "occ_max": str(pd.to_datetime(stats["occ_max"], unit="ms")),
    }
    return info


def tzcheck(n: int = 500) -> dict:
    """Hour-of-day distribution of DATE_REPORTED under both interpretations.

    Incident reporting to police concentrates in waking hours (~08-22 local)
    and bottoms out ~03-05. If the raw epoch-ms decode already shows that
    shape, values are local wall time stored as-if-UTC; if the shape appears
    only after shifting -4/-5 h, values are true UTC.
    """
    payload = _get({"where": "DATE_REPORTED IS NOT NULL",
                    "outFields": "DATE_REPORTED", "returnGeometry": "false",
                    "orderByFields": "OBJECTID DESC",
                    "resultRecordCount": n})
    ms = pd.Series([f["attributes"]["DATE_REPORTED"]
                    for f in payload["features"]])
    raw = pd.to_datetime(ms, unit="ms")
    eastern = (raw.dt.tz_localize("UTC").dt.tz_convert("America/New_York")
                  .dt.tz_localize(None))
    out = {"n": len(ms)}
    for name, ts in (("as_stored", raw), ("shifted_to_eastern", eastern)):
        hours = ts.dt.hour.value_counts().sort_index()
        out[name] = {
            "hour_hist": hours.reindex(range(24), fill_value=0).tolist(),
            "night_share_0305": round(float(ts.dt.hour.isin([3, 4, 5]).mean()), 4),
            "day_share_0818": round(float(ts.dt.hour.between(8, 18).mean()), 4),
        }
    return out


def pull() -> pd.DataFrame:
    RAW.mkdir(parents=True, exist_ok=True)
    meta = layer_meta()
    fields = partition_fields({f["name"] for f in meta.get("fields", [])})
    paginated = meta.get("advancedQueryCapabilities", {}).get(
        "supportsPagination", False)
    page_size = min(PAGE, meta.get("maxRecordCount") or PAGE)
    frames, offset, last_oid, page_no = [], 0, -1, 0
    while True:
        cache = RAW / f"page_{page_no:07d}.json"
        if cache.exists():
            payload = json.loads(cache.read_text())
        else:
            params = {"outFields": ",".join(fields),
                      "returnGeometry": "false",
                      "orderByFields": "OBJECTID",
                      "resultRecordCount": page_size}
            if paginated:
                params.update(where="1=1", resultOffset=offset)
            else:  # keyset paging by OBJECTID ranges
                params.update(where=f"OBJECTID > {last_oid}")
            payload = _get(params)
            cache.write_text(json.dumps(payload))
            time.sleep(0.2)
        feats = payload.get("features", [])
        if not feats:
            break
        frames.append(pd.DataFrame([f["attributes"] for f in feats]))
        last_oid = feats[-1]["attributes"]["OBJECTID"]
        offset += len(feats)
        page_no += 1
        if not payload.get("exceededTransferLimit") and len(feats) < page_size:
            break
    df = pd.concat(frames, ignore_index=True).drop_duplicates("OBJECTID")
    for col in ("DATE_INCIDENT_BEGAN", "DATE_INCIDENT_END", "DATE_REPORTED"):
        if col in df:
            df[col] = _to_datetime(df[col])
        else:
            df[col] = pd.NaT
    DATA.mkdir(exist_ok=True)
    df.to_parquet(PARQUET)
    return df


def quality_report(df: pd.DataFrame) -> dict:
    occ, rep = df["DATE_INCIDENT_BEGAN"], df["DATE_REPORTED"]
    geocode_fail = (df[LAT].isna() | df[LON].isna() | (df[LAT].abs() < 1e-6))
    gap_days = (rep - occ).dt.total_seconds() / 86400
    cutoff = rep.max() - pd.Timedelta(days=BACKFILL_DROP_DAYS)
    rpt = {
        "rows": len(df),
        "tz_interpretation": TZ_INTERPRETATION,
        "date_min_occurred": str(occ.min()),
        "date_max_occurred": str(occ.max()),
        "date_max_reported": str(rep.max()),
        "null_occurred_pct": round(100 * occ.isna().mean(), 2),
        "null_reported_pct": round(100 * rep.isna().mean(), 2),
        "geocode_fail_pct": round(100 * geocode_fail.mean(), 2),
        "gap_days_p50": round(gap_days.quantile(0.50), 2),
        "gap_days_p90": round(gap_days.quantile(0.90), 2),
        "gap_days_p99": round(gap_days.quantile(0.99), 2),
        "gap_over_30d_pct": round(100 * (gap_days > 30).mean(), 2),
        "backfill_cutoff": str(cutoff),
        "rows_dropped_by_cutoff": int((occ > cutoff).sum()),
        "stratum_counts": df["HIGHEST_NIBRS_CODE"].map(STRATA).value_counts()
                            .to_dict(),
    }
    return rpt


def load_usable() -> pd.DataFrame:
    """Cached incidents, geocoded, strata only, minus the backfill window."""
    df = pd.read_parquet(PARQUET)
    cutoff = df["DATE_REPORTED"].max() - pd.Timedelta(days=BACKFILL_DROP_DAYS)
    df = df[df["DATE_INCIDENT_BEGAN"].notna()
            & df[LAT].notna() & df[LON].notna()
            & (df[LAT].abs() > 1e-6)
            & (df["DATE_INCIDENT_BEGAN"] <= cutoff)]
    df = df[df["HIGHEST_NIBRS_CODE"].isin(STRATA)]
    return df.assign(stratum=df["HIGHEST_NIBRS_CODE"].map(STRATA))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="report from cache")
    ap.add_argument("--precheck", action="store_true")
    ap.add_argument("--tzcheck", action="store_true")
    args = ap.parse_args()
    if args.precheck:
        print(json.dumps(precheck(), indent=2))
    elif args.tzcheck:
        print(json.dumps(tzcheck(), indent=2))
    else:
        frame = (pd.read_parquet(PARQUET)
                 if args.report and PARQUET.exists() else pull())
        print(json.dumps(quality_report(frame), indent=2, default=str))
