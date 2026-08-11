"""Step 1: pull CMPD incidents (ArcGIS REST), cache raw pages, build parquet,
and print the data-quality report.

Feed: CMPD **reported incidents** (NIBRS-classified records), NOT calls for
service — the target variable is therefore "recorded incident reports", with
all the reporting/recording biases that implies (the model estimates where
incidents get REPORTED, not where crime occurs).

Usage:
  .venv/bin/python data_pull.py            # incremental pull + report
  .venv/bin/python data_pull.py --report   # report from cache only

Raw pages land in data/raw/ (gitignored — never commit incident data);
the assembled table in data/incidents.parquet.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

QUERY_URL = ("https://gis.charlottenc.gov/arcgis/rest/services/CMPD/"
             "CMPDIncidents/MapServer/0/query")
FIELDS = ["OBJECTID", "DATE_INCIDENT_BEGAN", "DATE_REPORTED", "Latitude",
          "Longitude", "HIGHEST_NIBRS_CODE", "HIGHEST_NIBRS_DESCRIPTION",
          "DIVISION_ID"]
PAGE = 2000
DATA = Path(__file__).parent / "data"
RAW = DATA / "raw"
PARQUET = DATA / "incidents.parquet"

# Backfill guard: drop this many most-recent days from ALL train/eval use.
BACKFILL_DROP_DAYS = 30

STRATA = {  # NIBRS code -> stratum
    "220": "burglary",
    "120": "robbery",
    "13A": "agg_assault_gun",
    "240": "mv_theft",
}


def pull() -> pd.DataFrame:
    RAW.mkdir(parents=True, exist_ok=True)
    offset, frames = 0, []
    while True:
        cache = RAW / f"page_{offset:07d}.json"
        if cache.exists():
            payload = json.loads(cache.read_text())
        else:
            r = requests.get(QUERY_URL, params={
                "where": "1=1", "outFields": ",".join(FIELDS),
                "returnGeometry": "false", "f": "json",
                "orderByFields": "OBJECTID",
                "resultOffset": offset, "resultRecordCount": PAGE,
            }, timeout=120)
            r.raise_for_status()
            payload = r.json()
            if "error" in payload:
                raise RuntimeError(f"ArcGIS error at offset {offset}: {payload['error']}")
            cache.write_text(json.dumps(payload))
            time.sleep(0.3)
        feats = payload.get("features", [])
        if not feats:
            break
        frames.append(pd.DataFrame([f["attributes"] for f in feats]))
        if not payload.get("exceededTransferLimit") and len(feats) < PAGE:
            break
        offset += len(feats)
    df = pd.concat(frames, ignore_index=True).drop_duplicates("OBJECTID")
    for col in ("DATE_INCIDENT_BEGAN", "DATE_REPORTED"):
        df[col] = pd.to_datetime(df[col], unit="ms", errors="coerce")
    DATA.mkdir(exist_ok=True)
    df.to_parquet(PARQUET)
    return df


def quality_report(df: pd.DataFrame) -> dict:
    occ, rep = df["DATE_INCIDENT_BEGAN"], df["DATE_REPORTED"]
    geocode_fail = (df["Latitude"].isna() | df["Longitude"].isna()
                    | (df["Latitude"].abs() < 1e-6))
    gap_days = (rep - occ).dt.total_seconds() / 86400
    cutoff = rep.max() - pd.Timedelta(days=BACKFILL_DROP_DAYS)
    rpt = {
        "rows": len(df),
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
            & df["Latitude"].notna() & df["Longitude"].notna()
            & (df["Latitude"].abs() > 1e-6)
            & (df["DATE_INCIDENT_BEGAN"] <= cutoff)]
    df = df[df["HIGHEST_NIBRS_CODE"].isin(STRATA)]
    return df.assign(stratum=df["HIGHEST_NIBRS_CODE"].map(STRATA))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="report from cache")
    args = ap.parse_args()
    frame = pd.read_parquet(PARQUET) if args.report and PARQUET.exists() else pull()
    print(json.dumps(quality_report(frame), indent=2, default=str))
