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


def daily_cube(df: pd.DataFrame, cells: list[str], neighbors: dict):
    """(n_cells, n_days) count cube + neighbor cube + date index."""
    idx = {c: i for i, c in enumerate(cells)}
    days = pd.date_range(df["DATE_INCIDENT_BEGAN"].min().normalize(),
                         df["DATE_INCIDENT_BEGAN"].max().normalize(), freq="D")
    day_pos = {d: i for i, d in enumerate(days)}
    counts = np.zeros((len(cells), len(days)))
    for cell, day in zip(df["cell"], df["DATE_INCIDENT_BEGAN"].dt.normalize()):
        counts[idx[cell], day_pos[day]] += 1
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


def run(n_folds: int | None):
    inv = load_cells()
    cells, neighbors = inv["cells"], inv["neighbors"]
    print(f"grid fixed: {inv['n_cells']} cells, "
          f"{inv['total_area_km2']} km^2 (resolution {inv['h3_resolution']})")
    df = load_usable()
    df, outside_pct = assign(df, cells)
    print(f"incidents outside boundary dropped: {outside_pct}%")

    records = []
    for stratum in sorted(set(STRATA.values())):
        sub = df[df["stratum"] == stratum]
        counts, nbr, days = daily_cube(sub, cells, neighbors)
        for k, (tr, ev) in enumerate(folds(days, n_folds)):
            observed = counts[:, ev].sum(axis=1)
            for name, fn in MODELS.items():
                pred = fn(counts[:, tr], nbr[:, tr], days[tr])
                for cov, s in score_coverages(pred, observed, COVERAGES).items():
                    records.append(dict(stratum=stratum, fold=k, model=name,
                                        coverage=cov, hit_rate=s.hit_rate,
                                        pai=s.pai, pei=s.pei,
                                        n_events=int(observed.sum())))
            print(f"{stratum} fold {k}: done ({int(observed.sum())} events)")
    out = pd.DataFrame(records)
    out.to_csv(OUT, index=False)
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
    args = ap.parse_args()
    run(args.folds)
