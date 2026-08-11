"""Crime-forecast evaluation metrics: PAI, PEI, hit rate.

Port of the logic of R ptools::pai() (Andrew Wheeler, MIT):
  hit rate  h(a) = crimes captured in flagged cells / total crimes
  PAI(a)    = h(a) / a,  a = flagged area / total study area
  PAI*(a)   = same but flagging cells by *observed* counts (oracle ceiling)
  PEI(a)    = hits(a) / hits*(a)  (equivalently PAI / PAI*)

Cells are flagged whole, in descending score order, until floor(a * n_cells)
cells are selected (equal-area H3 cells assumed; pass `cell_area` weights
otherwise). Ties broken by stable sort on (-score, cell_id) for determinism.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HotspotScore:
    coverage: float   # requested area fraction a
    n_flagged: int
    hit_rate: float
    pai: float
    pai_star: float
    pei: float


def _flag_top(values: np.ndarray, cell_ids: np.ndarray, n_flag: int) -> np.ndarray:
    order = np.lexsort((cell_ids, -values))
    return order[:n_flag]


def hotspot_metrics(
    predicted: np.ndarray,
    observed: np.ndarray,
    coverage: float,
    cell_ids: np.ndarray | None = None,
) -> HotspotScore:
    """Score one forecast window at one area-coverage fraction.

    predicted: score per cell (any monotone intensity ranking)
    observed:  actual crime counts per cell in the evaluation window
    coverage:  target area fraction in (0, 1)
    """
    predicted = np.asarray(predicted, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if predicted.shape != observed.shape:
        raise ValueError("predicted and observed must have the same shape")
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be in (0, 1)")
    n = predicted.size
    if cell_ids is None:
        cell_ids = np.arange(n)
    n_flag = int(np.floor(coverage * n))
    if n_flag == 0:
        raise ValueError(f"coverage {coverage} flags zero of {n} cells")

    total = observed.sum()
    a_realized = n_flag / n
    if total == 0:
        # No crime in window: hit rate undefined -> report zeros, PEI of 1
        # (nothing to find, nothing missed); callers should also track totals.
        return HotspotScore(coverage, n_flag, 0.0, 0.0, 0.0, 1.0)

    hits = observed[_flag_top(predicted, cell_ids, n_flag)].sum()
    hits_star = observed[_flag_top(observed, cell_ids, n_flag)].sum()

    hit_rate = hits / total
    pai = hit_rate / a_realized
    pai_star = (hits_star / total) / a_realized
    pei = hits / hits_star if hits_star > 0 else 1.0
    return HotspotScore(coverage, n_flag, hit_rate, pai, pai_star, pei)


def score_coverages(predicted, observed, coverages=(0.01, 0.02, 0.05)):
    return {c: hotspot_metrics(predicted, observed, c) for c in coverages}
