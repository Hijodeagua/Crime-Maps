"""Unit tests for metrics.py against a hand-computed toy example.

Toy world: 4 equal-area cells A,B,C,D (ids 0..3), one eval window.
Model ranks A > B > C > D. Observed counts: A=3, B=1, C=4, D=2 (total 10).

Coverage 25% -> flag 1 cell (A):
  hit rate = 3/10 = 0.30, PAI = 0.30/0.25 = 1.2
  oracle flags C (4): PAI* = 0.40/0.25 = 1.6, PEI = 3/4 = 0.75
Coverage 50% -> flag 2 cells (A,B):
  hits = 4, hit rate 0.40, PAI = 0.8
  oracle flags C,A (7): PAI* = 1.4, PEI = 4/7
"""
import numpy as np
import pytest

from metrics import hotspot_metrics

PRED = np.array([4.0, 3.0, 2.0, 1.0])   # A > B > C > D
OBS = np.array([3.0, 1.0, 4.0, 2.0])


def test_quarter_coverage():
    s = hotspot_metrics(PRED, OBS, 0.25)
    assert s.n_flagged == 1
    assert s.hit_rate == pytest.approx(0.30)
    assert s.pai == pytest.approx(1.2)
    assert s.pai_star == pytest.approx(1.6)
    assert s.pei == pytest.approx(0.75)


def test_half_coverage():
    s = hotspot_metrics(PRED, OBS, 0.50)
    assert s.n_flagged == 2
    assert s.hit_rate == pytest.approx(0.40)
    assert s.pai == pytest.approx(0.8)
    assert s.pai_star == pytest.approx(1.4)
    assert s.pei == pytest.approx(4.0 / 7.0)


def test_perfect_prediction_gives_pei_1():
    s = hotspot_metrics(OBS, OBS, 0.25)
    assert s.pei == pytest.approx(1.0)
    assert s.pai == pytest.approx(s.pai_star)


def test_floor_not_round():
    # 30% of 4 cells floors to 1 flagged cell, realized coverage 0.25
    s = hotspot_metrics(PRED, OBS, 0.30)
    assert s.n_flagged == 1
    # PAI uses realized area fraction (1/4), not requested 0.30
    assert s.pai == pytest.approx(0.30 / 0.25)


def test_deterministic_tie_break():
    pred_tied = np.array([1.0, 1.0, 1.0, 1.0])
    s1 = hotspot_metrics(pred_tied, OBS, 0.25)
    s2 = hotspot_metrics(pred_tied, OBS, 0.25)
    assert s1 == s2          # stable under repetition
    assert s1.hit_rate == pytest.approx(0.30)  # lowest cell_id (A) wins ties


def test_empty_window():
    s = hotspot_metrics(PRED, np.zeros(4), 0.25)
    assert s.hit_rate == 0.0 and s.pai == 0.0 and s.pei == 1.0


def test_input_validation():
    with pytest.raises(ValueError):
        hotspot_metrics(PRED, OBS[:3], 0.25)
    with pytest.raises(ValueError):
        hotspot_metrics(PRED, OBS, 0.0)
    with pytest.raises(ValueError):
        hotspot_metrics(PRED, OBS, 0.10)  # floors to zero cells of 4
