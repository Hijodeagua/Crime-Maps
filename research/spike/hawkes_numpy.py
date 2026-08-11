"""Hand-rolled discrete-time (daily) grid-Hawkes with exponential kernel.

Fallback engaged per Step 0: tick 0.8.0.2 installs but every inference
learner crashes on py3.11/numpy2 (see RESULTS.md). This is the ~80-line
numpy MLE deemed adequate for a baseline.

Model (Mohler-style SEPP collapsed to shared excitation parameters so the
parameter count stays at 4 regardless of grid size):

  lambda[i, d] = mu_scale * bg[i]
               + sum_{d' < d} omega * exp(-omega * (d - d'))
                 * (theta_self * n[i, d'] + theta_nbr * m[i, d'])

  n[i, d]  events in cell i on day d
  m[i, d]  events in ring-1 H3 neighbors of cell i on day d
  bg[i]    fixed background shape (long-run share per cell, sums to 1)

Fit (mu_scale, theta_self, theta_nbr, omega) by maximizing the Poisson
log-likelihood over all cells/days; the exponential kernel makes the
excitation state a one-step recursion, so evaluation is O(cells * days).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

EPS = 1e-12


def excitation(counts: np.ndarray, nbr_counts: np.ndarray,
               theta_self: float, theta_nbr: float, omega: float) -> np.ndarray:
    """Excitation term of lambda for each (cell, day); day axis last."""
    n_cells, n_days = counts.shape
    exc = np.zeros((n_cells, n_days))
    decay = np.exp(-omega)
    state = np.zeros(n_cells)
    for d in range(1, n_days):
        state = decay * (state + theta_self * counts[:, d - 1]
                         + theta_nbr * nbr_counts[:, d - 1])
        exc[:, d] = omega * state
    return exc


def intensity(params, counts, nbr_counts, bg):
    mu_scale, theta_self, theta_nbr, omega = params
    exc = excitation(counts, nbr_counts, theta_self, theta_nbr, omega)
    return mu_scale * bg[:, None] + exc


def neg_loglik(params, counts, nbr_counts, bg):
    lam = intensity(params, counts, nbr_counts, bg)
    # Poisson LL up to the constant log(n!): sum(n * log(lam) - lam)
    return -(counts * np.log(lam + EPS) - lam).sum()


def fit(counts: np.ndarray, nbr_counts: np.ndarray, bg: np.ndarray,
        x0=(1.0, 0.3, 0.05, 0.5)) -> dict:
    """MLE via L-BFGS-B. counts/nbr_counts: (n_cells, n_days); bg sums to 1."""
    bounds = [(1e-6, None), (0.0, 5.0), (0.0, 5.0), (1e-3, 5.0)]
    res = minimize(neg_loglik, x0=np.asarray(x0, float),
                   args=(counts, nbr_counts, bg),
                   method="L-BFGS-B", bounds=bounds)
    mu_scale, theta_self, theta_nbr, omega = res.x
    return {"mu_scale": mu_scale, "theta_self": theta_self,
            "theta_nbr": theta_nbr, "omega": omega,
            "neg_loglik": res.fun, "converged": bool(res.success)}


def forecast(params: dict, counts: np.ndarray, nbr_counts: np.ndarray,
             bg: np.ndarray, horizon: int = 7) -> np.ndarray:
    """Expected count per cell over the next `horizon` days, conditioning on
    history and propagating expected intensities forward (no simulation)."""
    p = (params["mu_scale"], params["theta_self"], params["theta_nbr"],
         params["omega"])
    mu_scale, theta_self, theta_nbr, omega = p
    decay = np.exp(-omega)
    state = np.zeros(counts.shape[0])
    for d in range(counts.shape[1]):
        state = decay * (state + theta_self * counts[:, d]
                         + theta_nbr * nbr_counts[:, d])
    # nbr expectation needs the cell->neighbor mapping; callers pass a closure
    # via nbr_of; keep it simple here: neighbor feedback during the horizon is
    # approximated with the same realized nbr/self ratio as the history.
    total_hist = counts.sum() + EPS
    nbr_ratio = nbr_counts.sum() / total_hist
    expected = np.zeros(counts.shape[0])
    lam_prev_state = state
    for _ in range(horizon):
        lam = mu_scale * bg + omega * lam_prev_state
        expected += lam
        lam_prev_state = decay * (lam_prev_state
                                  + theta_self * lam + theta_nbr * nbr_ratio * lam)
    return expected
