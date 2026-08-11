"""Step 4: the four models. Every model consumes the same inputs and returns
one score per cell for the next `horizon` days.

Inputs (per stratum, per fold):
  counts     (n_cells, n_train_days) daily incident counts, cell-indexed
  nbr_counts same shape, ring-1 neighbor sums
  dates      DatetimeIndex aligned to the day axis
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import hawkes_numpy

EPS = 1e-12


def longrun_hotspot(counts: np.ndarray, months: int = 24) -> np.ndarray:
    """(a) Rank by prior `months` cumulative counts. The real baseline."""
    days = min(counts.shape[1], months * 30)
    return counts[:, -days:].sum(axis=1).astype(float)


def recency_weighted(counts: np.ndarray, half_life_days: float = 60.0) -> np.ndarray:
    """(b) Exponentially decayed counts."""
    n_days = counts.shape[1]
    w = 0.5 ** ((n_days - 1 - np.arange(n_days)) / half_life_days)
    return counts @ w


def grid_hawkes(counts: np.ndarray, nbr_counts: np.ndarray,
                horizon: int = 7) -> np.ndarray:
    """(c) numpy fallback grid-Hawkes (see hawkes_numpy.py + RESULTS.md for
    why tick is not used despite being the planned path)."""
    bg = counts.sum(axis=1).astype(float)
    bg = (bg + 0.1) / (bg + 0.1).sum()  # smoothed long-run share
    params = hawkes_numpy.fit(counts, nbr_counts, bg)
    return hawkes_numpy.forecast(params, counts, nbr_counts, bg, horizon)


def _lgbm_frame(counts: np.ndarray, nbr_counts: np.ndarray,
                dates: pd.DatetimeIndex, horizon: int):
    """Cell-by-day design matrix: lags, DOW, seasonality, neighbor lags.
    Target: next-`horizon`-day count per cell."""
    n_cells, n_days = counts.shape
    lags = [1, 2, 3, 7, 14, 28]
    rows = {f"lag_{l}": [] for l in lags}
    rows.update({f"nbrlag_{l}": [] for l in (1, 7, 28)})
    rows.update(dow=[], doy_sin=[], doy_cos=[], cell_rate=[], y=[])
    start = max(lags)
    cum = counts.cumsum(axis=1)
    for d in range(start, n_days - horizon + 1):
        for l in lags:
            rows[f"lag_{l}"].append(counts[:, d - l])
        for l in (1, 7, 28):
            rows[f"nbrlag_{l}"].append(nbr_counts[:, d - l])
        doy = dates[d].dayofyear
        rows["dow"].append(np.full(n_cells, dates[d].dayofweek))
        rows["doy_sin"].append(np.full(n_cells, np.sin(2 * np.pi * doy / 365)))
        rows["doy_cos"].append(np.full(n_cells, np.cos(2 * np.pi * doy / 365)))
        rows["cell_rate"].append(cum[:, d - 1] / d)
        rows["y"].append(counts[:, d:d + horizon].sum(axis=1))
    return {k: np.concatenate(v) for k, v in rows.items()}


def lgbm(counts: np.ndarray, nbr_counts: np.ndarray, dates: pd.DatetimeIndex,
         horizon: int = 7, return_model: bool = False):
    """(d) LightGBM (Poisson objective) on cell-by-day features."""
    import lightgbm as lgb

    data = _lgbm_frame(counts, nbr_counts, dates, horizon)
    y = data.pop("y")
    X = pd.DataFrame(data)
    model = lgb.LGBMRegressor(objective="poisson", n_estimators=300,
                              learning_rate=0.05, num_leaves=31,
                              min_child_samples=50, subsample=0.8,
                              colsample_bytree=0.8, random_state=7,
                              verbose=-1)
    model.fit(X, y)
    # score "today": features as of the last training day
    n_cells, n_days = counts.shape
    last = {f"lag_{l}": counts[:, n_days - l] for l in (1, 2, 3, 7, 14, 28)}
    last.update({f"nbrlag_{l}": nbr_counts[:, n_days - l] for l in (1, 7, 28)})
    doy = dates[-1].dayofyear
    last["dow"] = np.full(n_cells, (dates[-1].dayofweek + 1) % 7)
    last["doy_sin"] = np.full(n_cells, np.sin(2 * np.pi * doy / 365))
    last["doy_cos"] = np.full(n_cells, np.cos(2 * np.pi * doy / 365))
    last["cell_rate"] = counts.sum(axis=1) / n_days
    pred = model.predict(pd.DataFrame(last)[X.columns])
    return (pred, model) if return_model else pred
