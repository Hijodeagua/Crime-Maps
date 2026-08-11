"""Validation of the numpy discrete-Hawkes fallback.

1) Parameter recovery: simulate from the model's own generative process on a
   toy 100-cell grid (ring-of-cells adjacency) and check the MLE recovers
   (mu_scale, theta_self, theta_nbr, omega) within tolerance.
2) Forecast sanity: expected counts are nonnegative and concentrate on the
   cells that were recently active.
"""
import numpy as np
import pytest

from hawkes_numpy import excitation, fit, forecast

RNG = np.random.default_rng(7)
N_CELLS, N_DAYS = 100, 730
TRUE = dict(mu_scale=8.0, theta_self=0.35, theta_nbr=0.10, omega=0.60)


def ring_neighbors(i, n=N_CELLS):
    return [(i - 1) % n, (i + 1) % n]


def simulate(true=TRUE, n_cells=N_CELLS, n_days=N_DAYS, rng=RNG):
    bg = rng.dirichlet(np.full(n_cells, 2.0))
    counts = np.zeros((n_cells, n_days))
    nbrs = np.zeros((n_cells, n_days))
    decay = np.exp(-true["omega"])
    state = np.zeros(n_cells)
    for d in range(n_days):
        lam = true["mu_scale"] * bg + true["omega"] * state
        counts[:, d] = rng.poisson(lam)
        nbrs[:, d] = sum(np.roll(counts[:, d], s) for s in (-1, 1))
        state = decay * (state + true["theta_self"] * counts[:, d]
                         + true["theta_nbr"] * nbrs[:, d])
    return counts, nbrs, bg


def test_parameter_recovery():
    counts, nbrs, bg = simulate()
    est = fit(counts, nbrs, bg)
    assert est["converged"]
    assert est["mu_scale"] == pytest.approx(TRUE["mu_scale"], rel=0.15)
    assert est["theta_self"] == pytest.approx(TRUE["theta_self"], abs=0.06)
    assert est["theta_nbr"] == pytest.approx(TRUE["theta_nbr"], abs=0.06)
    assert est["omega"] == pytest.approx(TRUE["omega"], rel=0.30)


def test_excitation_matches_bruteforce():
    counts = RNG.poisson(1.0, size=(5, 40)).astype(float)
    nbrs = RNG.poisson(2.0, size=(5, 40)).astype(float)
    ts, tn, om = 0.4, 0.1, 0.7
    exc = excitation(counts, nbrs, ts, tn, om)
    # brute force: sum over all past days
    d = 17
    brute = sum(om * np.exp(-om * (d - dp)) * (ts * counts[:, dp] + tn * nbrs[:, dp])
                for dp in range(d))
    assert np.allclose(exc[:, d], brute)


def test_forecast_shape_and_mass():
    counts, nbrs, bg = simulate(n_days=200)
    est = fit(counts, nbrs, bg)
    f = forecast(est, counts, nbrs, bg, horizon=7)
    assert f.shape == (N_CELLS,)
    assert (f >= 0).all()
    # recently-hot cells should outrank the coldest cells on average
    recent = counts[:, -14:].sum(axis=1)
    hot = f[np.argsort(-recent)[:10]].mean()
    cold = f[np.argsort(recent)[:10]].mean()
    assert hot > cold
