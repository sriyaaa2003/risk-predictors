import numpy as np

from riskpred.config import load
from riskpred.stress import simulate

P = load()["stress"]
KW = dict(lead_decel=7.0, lead_brake=3.0, aeb_ttc=1.4, sensor_delay=0.1, max_brake=9.0)


def run(gap, v, vl, seed=0, **over):
    kw = {**KW, **over}
    return simulate(np.array([gap]), np.array([v]), np.array([vl]), p=P, rng=np.random.default_rng(seed), **kw)


def test_large_gap_never_crashes():
    c, _ = run(150.0, 30.0, 30.0)
    assert not c[0]


def test_tiny_gap_with_fast_closing_crashes_with_impact_speed():
    c, dv = run(2.0, 30.0, 10.0)
    assert c[0] and dv[0] > 5.0


def test_deterministic_given_seed():
    a = run(20.0, 30.0, 25.0, seed=3)
    b = run(20.0, 30.0, 25.0, seed=3)
    assert a[0][0] == b[0][0] and a[1][0] == b[1][0]


def test_more_gap_never_hurts_in_aggregate():
    rng = np.random.default_rng(1)
    v = rng.uniform(22, 35, 4000)
    vl = v - rng.uniform(0, 8, 4000)
    small = simulate(np.full(4000, 15.0), v, vl, p=P, rng=np.random.default_rng(2), **KW)[0].mean()
    big = simulate(np.full(4000, 40.0), v, vl, p=P, rng=np.random.default_rng(2), **KW)[0].mean()
    assert big <= small


def test_harder_lead_braking_never_helps_in_aggregate():
    rng = np.random.default_rng(1)
    v = rng.uniform(22, 35, 4000)
    g = rng.uniform(10, 40, 4000)
    soft = simulate(g, v, v, p=P, rng=np.random.default_rng(2), **{**KW, "lead_decel": 3.0})[0].mean()
    hard = simulate(g, v, v, p=P, rng=np.random.default_rng(2), **{**KW, "lead_decel": 7.0})[0].mean()
    assert hard >= soft


def test_contact_requires_closing_speed_and_no_contact_means_no_impact():
    rng = np.random.default_rng(5)
    g = rng.uniform(3, 25, 3000)
    v = rng.uniform(20, 35, 3000)
    c, dv = simulate(g, v, v * 0.7, p=P, rng=rng, **KW)
    assert (dv[c] >= 0).all() and (dv[~c] == 0).all()
