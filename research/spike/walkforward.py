"""Steps 3+5: walk-forward evaluation. Train on rolling history, forecast the
next 7 days, roll weekly across >= 2 years (>= 24 folds enforced), score
PAI/PEI/hit-rate at 1/2/5% coverage per fold, per stratum, per model.

Usage (after data_pull.py and gridding.py have both been run):
  .venv/bin/python walkforward.py            # full run, writes results tables
  .venv/bin/python walkforward.py --folds 4  # smoke run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import models
from audits import aoristic_daily_weights, load_audits
from data_pull import STRATA, load_usable
from gridding import assign, load_cells
from metrics import score_coverages

DATA = Path(__file__).parent / "data"
OUT = DATA / "fold_scores.csv"
HORIZON = 7
MIN_FOLDS = 24
TRAIN_YEARS = 3
COVERAGES = (0.01, 0.02, 0.05)

MODELS = {
    "a_longrun": lambda c, n, d: models.longrun_hotspot(c),
    "b_recency": lambda c, n, d: models.recency_weighted(c),
    "c_hawkes": lambda c, n, d: models.grid_hawkes(c, n, HORIZON),
    "d_lgbm": lambda c, n, d: models.lgbm(c, n, d, HORIZON),
}


def daily_cube(df: pd.DataFrame, cells: list[str], neighbors: dict,
               aoristic: bool = False):
    """(n_cells, n_days) count cube + neighbor cube + date index.

    aoristic=True spreads each incident's unit weight uniformly across the
    days its occurred-from -> occurred-to window touches (audits.py rule);
    otherwise all weight lands on the begin day.
    """
    idx = {c: i for i, c in enumerate(cells)}
    if aoristic:
        event_days, weights, src = aoristic_daily_weights(
            df["DATE_INCIDENT_BEGAN"].reset_index(drop=True),
            df["DATE_INCIDENT_END"].reset_index(drop=True))
        cells_per_row = df["cell"].reset_index(drop=True).to_numpy()[src]
    else:
        event_days = pd.DatetimeIndex(df["DATE_INCIDENT_BEGAN"].dt.normalize())
        weights = np.ones(len(df))
        cells_per_row = df["cell"].to_numpy()
    days = pd.date_range(event_days.min(), event_days.max(), freq="D")
    day_pos = {d: i for i, d in enumerate(days)}
    counts = np.zeros((len(cells), len(days)))
    for cell, day, w in zip(cells_per_row, event_days, weights):
        counts[idx[cell], day_pos[day]] += w
    nbr = np.zeros_like(counts)
    for c, ns in neighbors.items():
        if ns:
            nbr[idx[c]] = counts[[idx[n] for n in ns]].sum(axis=0)
    return counts, nbr, days


def folds(days: pd.DatetimeIndex, n_folds: int):
    """Weekly-rolling folds, newest history last; each = (train_slice, eval_slice)."""
    train_len = TRAIN_YEARS * 365
    usable = len(days) - train_len - HORIZON
    if usable < (MIN_FOLDS - 1) * 7:
        raise SystemExit(f"Not enough history: need >= {MIN_FOLDS} weekly folds "
                         f"after a {TRAIN_YEARS}y training window.")
    starts = range(train_len, len(days) - HORIZON + 1, 7)
    starts = list(starts)[-n_folds:] if n_folds else list(starts)
    return [(slice(s - train_len, s), slice(s, s + HORIZON)) for s in starts]


def run(n_folds: int | None, res: int | None = None, tag: str = "primary"):
    coord_audit, temp_audit = load_audits()   # hard gate: audits first
    if res is None:
        res = coord_audit["primary_resolution"]
    aoristic_strata = set(temp_audit["aoristic_strata"])
    inv = load_cells(res)
    cells, neighbors = inv["cells"], inv["neighbors"]
    print(f"[{tag}] grid fixed: {inv['n_cells']} cells, "
          f"{inv['total_area_km2']} km^2 (resolution {inv['h3_resolution']})")
    df = load_usable()
    df, outside_pct = assign(df, cells, res)
    print(f"incidents outside boundary dropped: {outside_pct}%")

    records = []
    for stratum in sorted(set(STRATA.values())):
        sub = df[df["stratum"] == stratum]
        aoristic = stratum in aoristic_strata
        counts, nbr, days = daily_cube(sub, cells, neighbors, aoristic)
        for k, (tr, ev) in enumerate(folds(days, n_folds)):
            observed = counts[:, ev].sum(axis=1)
            for name, fn in MODELS.items():
                pred = fn(counts[:, tr], nbr[:, tr], days[tr])
                for cov, s in score_coverages(pred, observed, COVERAGES).items():
                    records.append(dict(stratum=stratum, fold=k, model=name,
                                        resolution=res, run=tag,
                                        aoristic=aoristic, coverage=cov,
                                        hit_rate=s.hit_rate, pai=s.pai,
                                        pei=s.pei,
                                        n_events=round(float(observed.sum()), 1)))
            print(f"{stratum} fold {k}: done ({observed.sum():.0f} events"
                  f"{', aoristic' if aoristic else ''})")
    out = pd.DataFrame(records)
    path = OUT if tag == "primary" else OUT.with_name(f"fold_scores_{tag}.csv")
    out.to_csv(path, index=False)
    summarize(out)


def summarize(out: pd.DataFrame):
    """Per-stratum medians + fold-win counts vs baseline (a)."""
    for (stratum, cov), g in out.groupby(["stratum", "coverage"]):
        print(f"\n== {stratum} @ {cov:.0%} coverage ==")
        med = g.groupby("model")[["pai", "pei", "hit_rate"]].median().round(3)
        print(med)
        base = g[g.model == "a_longrun"].set_index("fold")["pai"]
        for m in sorted(g.model.unique()):
            if m == "a_longrun":
                continue
            mine = g[g.model == m].set_index("fold")["pai"]
            wins = int((mine > base.reindex(mine.index)).sum())
            print(f"  {m} beats a_longrun on {wins}/{len(mine)} folds")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=None,
                    help="limit number of folds (default: all, min 24)")
    ap.add_argument("--res", type=int, default=None,
                    help="override H3 resolution (default: audit decision)")
    ap.add_argument("--tag", default="primary",
                    help="run label; use e.g. res9-sensitivity for the "
                         "labeled sensitivity check when res 8 is primary")
    args = ap.parse_args()
    run(args.folds, args.res, args.tag)
