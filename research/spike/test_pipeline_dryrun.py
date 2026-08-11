"""End-to-end harness validation on SYNTHETIC data (not CMPD; produces no
reportable results). Builds a real H3 res-9 grid around uptown Charlotte,
simulates hotspot-clustered incidents, and runs the full walk-forward loop
with all four models on a reduced fold count.
"""
import numpy as np
import pandas as pd
import pytest
import h3

import models
from metrics import score_coverages

RNG = np.random.default_rng(11)


@pytest.fixture(scope="module")
def grid():
    center = h3.latlng_to_cell(35.2271, -80.8431, 9)
    cells = sorted(h3.grid_disk(center, 8))          # 217 cells
    cell_set = set(cells)
    neighbors = {c: [n for n in h3.grid_ring(c, 1) if n in cell_set]
                 for c in cells}
    return cells, neighbors


@pytest.fixture(scope="module")
def cube(grid):
    cells, neighbors = grid
    n_cells, n_days = len(cells), 3 * 365 + 60
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    hot = RNG.dirichlet(np.full(n_cells, 0.5))       # persistent hotspots
    dow = 1 + 0.3 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    counts = RNG.poisson(np.outer(hot * 40, dow))
    idx = {c: i for i, c in enumerate(cells)}
    nbr = np.zeros_like(counts, dtype=float)
    for c, ns in neighbors.items():
        if ns:
            nbr[idx[c]] = counts[[idx[n] for n in ns]].sum(axis=0)
    return counts.astype(float), nbr, dates


def test_all_models_run_and_score(cube):
    counts, nbr, dates = cube
    tr = slice(0, 3 * 365)
    ev = slice(3 * 365, 3 * 365 + 7)
    observed = counts[:, ev].sum(axis=1)
    preds = {
        "a_longrun": models.longrun_hotspot(counts[:, tr]),
        "b_recency": models.recency_weighted(counts[:, tr]),
        "c_hawkes": models.grid_hawkes(counts[:, tr], nbr[:, tr]),
        "d_lgbm": models.lgbm(counts[:, tr], nbr[:, tr], dates[tr]),
    }
    for name, pred in preds.items():
        assert pred.shape == observed.shape, name
        assert np.isfinite(pred).all(), name
        scores = score_coverages(pred, observed)
        # persistent-hotspot world: any sane model beats uniform (PAI > 1)
        assert scores[0.05].pai > 1.0, f"{name} PAI at 5%: {scores[0.05].pai}"


def test_lgbm_feature_importances(cube):
    counts, nbr, dates = cube
    tr = slice(0, 3 * 365)
    _, model = models.lgbm(counts[:, tr], nbr[:, tr], dates[tr],
                           return_model=True)
    imp = sorted(zip(model.feature_name_, model.feature_importances_),
                 key=lambda t: -t[1])
    assert len(imp) == 13  # 6 lags + 3 nbr lags + dow + doy_sin/cos + cell_rate
    assert imp[0][1] > 0
