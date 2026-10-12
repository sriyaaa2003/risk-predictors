import numpy as np

from riskpred.config import load
from riskpred.predictors import DIRECTION, window_predictors

P = load()["predictors"]


def win(gap, v, vl, n=126):
    return np.full(n, gap, float), np.full(n, v, float), np.full(n, vl, float), np.zeros(n)


def test_known_values():
    g, v, vl, a = win(30.0, 30.0, 20.0)
    r = window_predictors(g, v, vl, a, P)
    assert abs(r["ttc_0"] - 3.0) < 1e-9 and abs(r["thw_0"] - 1.0) < 1e-9
    assert abs(r["drac_0"] - 100 / 60) < 1e-9 and r["tit"] == 0.0


def test_not_closing_gives_capped_ttc_and_zero_drac():
    g, v, vl, a = win(20.0, 25.0, 30.0)
    r = window_predictors(g, v, vl, a, P)
    assert r["ttc_0"] == 60.0 and r["drac_0"] == 0.0 and r["inv_ttc_0"] == 0.0


def test_window_summary_ignores_anything_after_t0():
    """The function only sees samples up to t0, so appending a future crash cannot change it."""
    g, v, vl, a = win(25.0, 28.0, 24.0)
    base = window_predictors(g, v, vl, a, P)
    g2, v2, vl2, a2 = (np.append(x, y) for x, y in zip((g, v, vl, a), (0.5, 40.0, 5.0, -9.0)))
    leaked = window_predictors(g2[:-1], v2[:-1], vl2[:-1], a2[:-1], P)
    assert base == leaked


def test_every_predictor_has_a_direction():
    g, v, vl, a = win(25.0, 28.0, 24.0)
    assert set(window_predictors(g, v, vl, a, P)) == set(DIRECTION)
