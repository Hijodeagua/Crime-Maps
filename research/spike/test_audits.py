"""Unit tests for audits.py and the pull degradation policy, with
hand-computed expectations."""
import numpy as np
import pandas as pd
import pytest

from audits import (aoristic_daily_weights, coordinate_audit,
                    quantization_check, temporal_audit, RES9_EDGE_M)
from data_pull import (partition_fields, LAT, LON, REQUIRED_FIELDS,
                       OPTIONAL_FIELDS)


def _df(lats, lons, strata, begins, ends):
    return pd.DataFrame({
        LAT: lats, LON: lons, "stratum": strata,
        "DATE_INCIDENT_BEGAN": pd.to_datetime(begins),
        "DATE_INCIDENT_END": pd.to_datetime(ends),
    })


def test_partition_fields_aborts_on_missing_required():
    live = set(REQUIRED_FIELDS + OPTIONAL_FIELDS) - {LAT}
    with pytest.raises(RuntimeError, match="REQUIRED.*LATITUDE_PUBLIC"):
        partition_fields(live)
    live = set(REQUIRED_FIELDS + OPTIONAL_FIELDS) - {"HIGHEST_NIBRS_CODE"}
    with pytest.raises(RuntimeError, match="REQUIRED"):
        partition_fields(live)


def test_partition_fields_degrades_on_missing_optional(capsys):
    live = set(REQUIRED_FIELDS + OPTIONAL_FIELDS) - {"DATE_INCIDENT_END", "NPA"}
    got = partition_fields(live)
    assert "DATE_INCIDENT_END" not in got and "NPA" not in got
    assert all(f in got for f in REQUIRED_FIELDS)
    assert "WARNING" in capsys.readouterr().out


def test_quantization_detects_lattice():
    rng = np.random.default_rng(5)
    # coarse lattice: every value on a 0.002-deg grid (~222 m in lat)
    coarse = 35.2 + 0.002 * rng.integers(0, 200, 4000)
    q = quantization_check(coarse, 111320.0)
    assert q["lattice_detected"] is True
    assert q["pitch_m"] == pytest.approx(222.6, abs=1.0)
    # smooth scatter: no dominant gap at meaningful pitch
    smooth = 35.2 + rng.uniform(0, 0.4, 4000)
    q2 = quantization_check(smooth, 111320.0)
    assert q2["lattice_detected"] is False


def test_coordinate_audit_counts_and_decision(tmp_path, monkeypatch):
    monkeypatch.setattr("audits.DATA", tmp_path)
    monkeypatch.setattr("audits.COORD_JSON", tmp_path / "c.json")
    # 6 incidents, 3 distinct coords; the two burglary coords are ~555 m
    # apart (0.005 deg lat), far coarser than res-9 precision.
    df = _df(
        lats=[35.2000, 35.2000, 35.2000, 35.2050, 35.2050, 35.3000],
        lons=[-80.8000] * 5 + [-80.9000],
        strata=["burglary"] * 5 + ["robbery"],
        begins=["2024-01-01"] * 6, ends=[None] * 6)
    out = coordinate_audit(df)
    assert out["per_stratum"]["burglary"] == {
        "incidents": 5, "distinct_coords": 2, "distinct_pct": 40.0}
    assert out["distinct_coords_total"] == 3
    assert out["top20"][0]["count"] == 3
    assert out["top20"][0]["share_pct"] == 50.0
    # median NN among 3 distinct points: ~555m, ~555m, ~9km -> median 555m
    assert out["median_nn_distance_m"] == pytest.approx(555, rel=0.02)
    assert out["median_nn_distance_m"] >= RES9_EDGE_M
    assert out["res9_finer_than_data"] is True
    assert out["primary_resolution"] == 8


def test_coordinate_audit_fine_data_keeps_res9(tmp_path, monkeypatch):
    monkeypatch.setattr("audits.DATA", tmp_path)
    monkeypatch.setattr("audits.COORD_JSON", tmp_path / "c.json")
    rng = np.random.default_rng(3)
    n = 400  # dense scatter ~ tens of meters apart -> res 9 stays primary
    df = _df(lats=35.20 + rng.uniform(0, 0.01, n),
             lons=-80.80 + rng.uniform(0, 0.01, n),
             strata=["robbery"] * n,
             begins=["2024-01-01"] * n, ends=[None] * n)
    out = coordinate_audit(df)
    assert out["res9_finer_than_data"] is False
    assert out["primary_resolution"] == 9


def test_temporal_audit_flags_stratum(tmp_path, monkeypatch):
    monkeypatch.setattr("audits.TEMP_JSON", tmp_path / "t.json")
    # burglary: 2 of 10 windows over 6h = 20% -> aoristic
    # robbery: all point events -> not aoristic
    b_begin = ["2024-01-01 08:00"] * 10
    b_end = ["2024-01-01 09:00"] * 8 + ["2024-01-01 20:00", "2024-01-03 08:00"]
    df = _df(lats=[35.2] * 12, lons=[-80.8] * 12,
             strata=["burglary"] * 10 + ["robbery"] * 2,
             begins=b_begin + ["2024-01-05 12:00"] * 2,
             ends=b_end + [None, None])
    out = temporal_audit(df)
    burg = out["per_stratum"]["burglary"]
    assert burg["over_6h_pct"] == 20.0
    assert burg["over_24h_pct"] == 10.0
    assert burg["aoristic"] is True
    rob = out["per_stratum"]["robbery"]
    assert rob["end_missing_pct"] == 100.0
    assert rob["aoristic"] is False
    assert out["aoristic_strata"] == ["burglary"]


def test_aoristic_weights_hand_computed():
    # Window 2024-01-01 18:00 -> 2024-01-03 06:00 (36h) touches 3 days:
    #   day1 6h/36h = 1/6, day2 24h/36h = 2/3, day3 6h/36h = 1/6
    begin = pd.Series(pd.to_datetime(["2024-01-01 18:00", "2024-01-05 10:00"]))
    end = pd.Series(pd.to_datetime(["2024-01-03 06:00", pd.NaT]))
    days, w, idx = aoristic_daily_weights(begin, end)
    first = w[idx == 0]
    assert first == pytest.approx([1 / 6, 2 / 3, 1 / 6])
    assert list(days[idx == 0].strftime("%Y-%m-%d")) == [
        "2024-01-01", "2024-01-02", "2024-01-03"]
    # point event: single day, weight 1
    assert w[idx == 1] == pytest.approx([1.0])
    # total mass conserved: one unit per incident
    assert w.sum() == pytest.approx(2.0)


def test_aoristic_inverted_end_treated_as_point():
    begin = pd.Series(pd.to_datetime(["2024-01-02 12:00"]))
    end = pd.Series(pd.to_datetime(["2024-01-01 12:00"]))  # end < begin
    days, w, idx = aoristic_daily_weights(begin, end)
    assert w.tolist() == [1.0]
    assert days[0] == pd.Timestamp("2024-01-02")
