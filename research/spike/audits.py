"""Pre-model DATA QUALITY audits. Both must run (and their JSON must exist)
before walkforward will start; their output belongs in RESULTS.md ahead of
any metrics table.

1. Coordinate precision audit — CMPD geocodes to block level, so raw
   coordinates are heavily reused. If the data's effective precision is
   coarser than an H3 res-9 cell, res 9 would just shatter block centroids
   across arbitrary cell boundaries; the audit decides the primary
   resolution mechanically:
       res 9 stays primary iff median nearest-neighbor distance between
       distinct coordinate pairs < res-9 mean edge length (~174 m)
   otherwise res 8 becomes primary and res 9 is demoted to a labeled
   sensitivity check.

2. Temporal censoring audit — burglary/MVT are discovered, not observed, so
   the occurred-from -> occurred-to window can span days. Any stratum with
   > 15% of windows over 6 h gets aoristic weighting: each incident's unit
   weight is spread uniformly over the days its window touches
   (day-overlap-proportional), and the weighted counts feed the model cube.

Usage: .venv/bin/python audits.py   (after data_pull.py; writes
data/audit_coordinates.json, data/audit_temporal.json, and prints the
RESULTS.md-ready markdown to stdout)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from data_pull import STRATA, load_usable

DATA = Path(__file__).parent / "data"
COORD_JSON = DATA / "audit_coordinates.json"
TEMP_JSON = DATA / "audit_temporal.json"

RES9_EDGE_M = 174.4   # mean hex edge length at H3 res 9
AORISTIC_WINDOW_HOURS = 6.0
AORISTIC_STRATUM_PCT = 15.0


def _meters(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Local equirectangular projection (fine at city scale)."""
    lat0 = np.deg2rad(lat.mean())
    return np.c_[np.deg2rad(lon) * 6371000 * np.cos(lat0),
                 np.deg2rad(lat) * 6371000]


def coordinate_audit(df: pd.DataFrame) -> dict:
    out = {"per_stratum": {}}
    pairs_all = df[["Latitude", "Longitude"]].round(6)
    for stratum, g in df.groupby("stratum"):
        p = g[["Latitude", "Longitude"]].round(6)
        out["per_stratum"][stratum] = {
            "incidents": len(g),
            "distinct_coords": int(len(p.drop_duplicates())),
            "distinct_pct": round(100 * len(p.drop_duplicates()) / len(g), 2),
        }
    top = (pairs_all.groupby(["Latitude", "Longitude"]).size()
           .sort_values(ascending=False).head(20))
    out["top20"] = [{"lat": la, "lon": lo, "count": int(c),
                     "share_pct": round(100 * c / len(df), 3)}
                    for (la, lo), c in top.items()]
    distinct = pairs_all.drop_duplicates().to_numpy()
    xy = _meters(distinct[:, 0], distinct[:, 1])
    dist, _ = cKDTree(xy).query(xy, k=2)
    med_nn = float(np.median(dist[:, 1]))
    out["distinct_coords_total"] = int(len(distinct))
    out["incidents_total"] = int(len(df))
    out["median_nn_distance_m"] = round(med_nn, 1)
    out["res9_edge_m"] = RES9_EDGE_M
    out["res9_finer_than_data"] = med_nn >= RES9_EDGE_M
    out["primary_resolution"] = 8 if out["res9_finer_than_data"] else 9
    DATA.mkdir(exist_ok=True)
    COORD_JSON.write_text(json.dumps(out, indent=2))
    return out


def temporal_audit(df: pd.DataFrame) -> dict:
    out = {"per_stratum": {}, "aoristic_strata": []}
    hours = ((df["DATE_INCIDENT_END"] - df["DATE_INCIDENT_BEGAN"])
             .dt.total_seconds() / 3600)
    hours = hours.where(hours > 0, 0.0)  # missing/inverted end -> point event
    for stratum, g in df.assign(win_h=hours).groupby("stratum"):
        w = g["win_h"]
        rec = {
            "incidents": len(g),
            "end_missing_pct": round(100 * g["DATE_INCIDENT_END"].isna().mean(), 2),
            "win_h_p50": round(float(w.quantile(0.5)), 2),
            "win_h_p90": round(float(w.quantile(0.9)), 2),
            "win_h_p99": round(float(w.quantile(0.99)), 2),
            "over_6h_pct": round(100 * float((w > 6).mean()), 2),
            "over_24h_pct": round(100 * float((w > 24).mean()), 2),
        }
        rec["aoristic"] = rec["over_6h_pct"] > AORISTIC_STRATUM_PCT
        if rec["aoristic"]:
            out["aoristic_strata"].append(stratum)
        out["per_stratum"][stratum] = rec
    TEMP_JSON.write_text(json.dumps(out, indent=2))
    return out


def aoristic_daily_weights(begin: pd.Series, end: pd.Series):
    """Spread each incident's unit weight uniformly across the days its
    [begin, end] window touches, proportional to overlap with each day.

    Returns (day_normalized: list of DatetimeIndex-able arrays, weights),
    flattened: one (day, weight) row per incident-day. Point events (no/zero
    window) put weight 1.0 on the begin day.
    """
    end = end.where(end.notna() & (end > begin), begin)
    days_list, weights, idx = [], [], []
    for i, (b, e) in enumerate(zip(begin, end)):
        d0, d1 = b.normalize(), e.normalize()
        n_days = (d1 - d0).days + 1
        if n_days == 1:
            days_list.append(d0); weights.append(1.0); idx.append(i)
            continue
        total = (e - b).total_seconds()
        for k in range(n_days):
            day = d0 + pd.Timedelta(days=k)
            seg0 = max(b, day)
            seg1 = min(e, day + pd.Timedelta(days=1))
            ov = (seg1 - seg0).total_seconds()
            if ov > 0:
                days_list.append(day); weights.append(ov / total); idx.append(i)
    return (pd.DatetimeIndex(days_list), np.array(weights), np.array(idx))


def load_audits() -> tuple[dict, dict]:
    if not (COORD_JSON.exists() and TEMP_JSON.exists()):
        raise SystemExit("Run audits.py first: DATA QUALITY audits must be "
                         "written to RESULTS.md before any model runs.")
    return (json.loads(COORD_JSON.read_text()), json.loads(TEMP_JSON.read_text()))


def markdown_report(coord: dict, temp: dict) -> str:
    lines = ["## DATA QUALITY (pre-model audits)", "",
             "### 1. Coordinate precision", "",
             "| stratum | incidents | distinct coords | distinct % |",
             "|---|---|---|---|"]
    for s, r in sorted(coord["per_stratum"].items()):
        lines.append(f"| {s} | {r['incidents']} | {r['distinct_coords']} "
                     f"| {r['distinct_pct']} |")
    lines += ["", f"Distinct pairs overall: {coord['distinct_coords_total']} "
              f"of {coord['incidents_total']} incidents. "
              f"Median NN distance between distinct points: "
              f"**{coord['median_nn_distance_m']} m** vs res-9 edge "
              f"{coord['res9_edge_m']} m.",
              "", f"**Res 9 finer than data precision: "
              f"{coord['res9_finer_than_data']} -> primary resolution "
              f"{coord['primary_resolution']}**"
              + (" (res 9 kept only as labeled sensitivity check)"
                 if coord["res9_finer_than_data"] else ""), "",
              "Top 20 most frequent coordinates:", "",
              "| lat | lon | count | share % |", "|---|---|---|---|"]
    for t in coord["top20"]:
        lines.append(f"| {t['lat']} | {t['lon']} | {t['count']} | {t['share_pct']} |")
    lines += ["", "### 2. Temporal censoring", "",
              "| stratum | end missing % | p50 h | p90 h | p99 h | >6h % | >24h % | aoristic |",
              "|---|---|---|---|---|---|---|---|"]
    for s, r in sorted(temp["per_stratum"].items()):
        lines.append(f"| {s} | {r['end_missing_pct']} | {r['win_h_p50']} "
                     f"| {r['win_h_p90']} | {r['win_h_p99']} | {r['over_6h_pct']} "
                     f"| {r['over_24h_pct']} | {'YES' if r['aoristic'] else 'no'} |")
    lines += ["", f"Aoristic weighting applied to: "
              f"{', '.join(temp['aoristic_strata']) or 'none'} "
              f"(rule: >{AORISTIC_STRATUM_PCT}% of windows over "
              f"{AORISTIC_WINDOW_HOURS} h)."]
    return "\n".join(lines)


if __name__ == "__main__":
    df = load_usable()
    print(markdown_report(coordinate_audit(df), temporal_audit(df)))
