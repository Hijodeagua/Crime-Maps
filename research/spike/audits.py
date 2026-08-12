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

from data_pull import LAT, LON, STRATA, load_usable

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


def quantization_check(values: np.ndarray, axis_scale_m: float) -> dict:
    """Detect whether published coordinates sit on a discrete lattice.

    Takes one axis's values, estimates the lattice pitch as the modal gap
    between consecutive distinct values (rounded to 1e-6 deg), and reports
    what share of gaps equal that pitch. A smooth (non-generalized) spatial
    distribution has no dominant gap; block-centroid/segment-midpoint
    generalization shows a dominant pitch. Fires when the modal gap holds
    >= 30% of gap mass AND the pitch is >= half the res-9 edge (~87 m) —
    a lattice finer than that cannot degrade a res-9 grid.
    """
    vals = np.unique(np.round(values, 6))
    gaps = np.round(np.diff(vals), 6)
    gaps = gaps[gaps > 0]
    if len(gaps) < 10:
        return {"lattice_detected": False, "note": "too few distinct values"}
    uniq, counts = np.unique(gaps, return_counts=True)
    modal_gap = float(uniq[counts.argmax()])
    modal_share = float(counts.max() / len(gaps))
    pitch_m = modal_gap * axis_scale_m
    frac = values - np.floor(values)
    hist, _ = np.histogram(frac, bins=1000, range=(0, 1))
    occupied_millis = int((hist > 0).sum())  # of 1000 possible 1e-3 bins
    return {
        "modal_gap_deg": modal_gap,
        "modal_gap_share": round(modal_share, 3),
        "pitch_m": round(pitch_m, 1),
        "frac_bins_occupied_of_1000": occupied_millis,
        "lattice_detected": bool(modal_share >= 0.30
                                 and pitch_m >= RES9_EDGE_M / 2),
    }


def address_granularity(df: pd.DataFrame, n: int = 50, seed: int = 7) -> dict:
    """Sample ADDRESS_DESCRIPTION: block ranges vs specific addresses."""
    if "ADDRESS_DESCRIPTION" not in df or df["ADDRESS_DESCRIPTION"].isna().all():
        return {"available": False}
    addr = df["ADDRESS_DESCRIPTION"].dropna().astype(str)
    sample = addr.sample(min(n, len(addr)), random_state=seed)
    block_pat = sample.str.contains(r"\bBLK\b|\bBLOCK\b|^\s*\d+00\b",
                                    case=False, regex=True)
    return {
        "available": True,
        "sample_n": int(len(sample)),
        "block_range_pct": round(100 * float(block_pat.mean()), 1),
        "examples": sample.head(10).tolist(),
    }


def coordinate_audit(df: pd.DataFrame) -> dict:
    out = {"per_stratum": {}}
    pairs_all = df[[LAT, LON]].round(6)
    for stratum, g in df.groupby("stratum"):
        p = g[[LAT, LON]].round(6)
        out["per_stratum"][stratum] = {
            "incidents": len(g),
            "distinct_coords": int(len(p.drop_duplicates())),
            "distinct_pct": round(100 * len(p.drop_duplicates()) / len(g), 2),
        }
    top = (pairs_all.groupby([LAT, LON]).size()
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

    lat0 = np.deg2rad(df[LAT].mean())
    out["quantization_lat"] = quantization_check(
        df[LAT].to_numpy(), 111320.0)
    out["quantization_lon"] = quantization_check(
        df[LON].to_numpy(), 111320.0 * np.cos(lat0))
    out["address_granularity"] = address_granularity(df)

    quant_fired = (out["quantization_lat"]["lattice_detected"]
                   or out["quantization_lon"]["lattice_detected"])
    out["nn_rule_fired"] = med_nn >= RES9_EDGE_M
    out["quantization_rule_fired"] = bool(quant_fired)
    out["res9_finer_than_data"] = out["nn_rule_fired"] or quant_fired
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
    ql, qo = coord["quantization_lat"], coord["quantization_lon"]
    ag = coord["address_granularity"]
    lines += ["", f"Distinct pairs overall: {coord['distinct_coords_total']} "
              f"of {coord['incidents_total']} incidents. "
              f"Median NN distance between distinct points: "
              f"**{coord['median_nn_distance_m']} m** vs res-9 edge "
              f"{coord['res9_edge_m']} m (NN rule fired: "
              f"{coord['nn_rule_fired']}).",
              "", f"Quantization: lat modal gap {ql.get('modal_gap_deg')} deg "
              f"(pitch {ql.get('pitch_m')} m, share {ql.get('modal_gap_share')}); "
              f"lon modal gap {qo.get('modal_gap_deg')} deg "
              f"(pitch {qo.get('pitch_m')} m, share {qo.get('modal_gap_share')}). "
              f"Lattice detected: lat={ql['lattice_detected']}, "
              f"lon={qo['lattice_detected']} (quantization rule fired: "
              f"{coord['quantization_rule_fired']}).",
              "", (f"Address granularity ({ag.get('sample_n', 0)} sampled): "
                   f"{ag.get('block_range_pct', 'n/a')}% block ranges. "
                   f"Examples: {'; '.join(ag.get('examples', [])[:5])}"
                   if ag.get("available") else
                   "Address granularity: ADDRESS_DESCRIPTION unavailable."),
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
