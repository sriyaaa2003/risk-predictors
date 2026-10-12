"""Counterfactual braking stress test of an ACC + AEB system, vectorised over decisions.

Starting from the REAL state (gap, ego speed, lead speed) at the decision time, the lead vehicle
brakes at a constant deceleration for a fixed time (or until it stops), and the system under test (SUT) has to cope.
The SUT is a constant time-gap ACC with an autonomous emergency brake (AEB) that triggers on the
time to collision it *measures* (delayed and noisy), then brakes through a first-order actuator.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .config import Config, load


def simulate(gap0, v0, vl0, *, lead_decel, lead_brake, aeb_ttc, sensor_delay, max_brake, p, rng):
    """Returns (crashed: bool[n], impact_dv: float[n] closing speed at first contact, 0 if none)."""
    dt, n = p["dt_s"], len(gap0)
    steps = int(round(p["horizon_s"] / dt))
    D = max(1, int(round(sensor_delay / dt)))
    gap, v, vl = (np.asarray(x, dtype=np.float64).copy() for x in (gap0, v0, vl0))
    h_gap, h_v, h_vl = (np.tile(x, (D + 1, 1)) for x in (gap, v, vl))     # ring buffer for delayed sensing
    v_set = np.asarray(v0, dtype=np.float64).copy()       # set speed = the speed the driver had at t0
    a = np.zeros(n)
    trig = np.zeros(n, bool)
    crashed = np.zeros(n, bool)
    impact = np.zeros(n)
    k1, k2, th, d0 = p["acc_k1"], p["acc_k2"], p["acc_time_gap_s"], p["acc_standstill_m"]
    for k in range(steps):
        idx = k % (D + 1)                    # the slot about to be overwritten holds the state D steps ago
        g_m = h_gap[idx] + rng.normal(0, p["sigma_gap_m"], n)
        v_m = h_v[idx] + rng.normal(0, p["sigma_speed_ms"], n)
        vl_m = h_vl[idx] + rng.normal(0, p["sigma_speed_ms"], n)
        rel = v_m - vl_m
        ttc_m = np.where(rel > 0.1, g_m / np.maximum(rel, 0.1), np.inf)
        trig |= ttc_m < aeb_ttc
        a_follow = k1 * (g_m - d0 - th * v_m) + k2 * (vl_m - v_m)
        a_acc = np.clip(np.minimum(a_follow, p["acc_speed_gain"] * (v_set - v_m)), -p["acc_max_decel"], p["acc_max_accel"])
        a_cmd = np.where(trig, -max_brake, a_acc)
        a = np.clip(a + (a_cmd - a) * dt / p["actuator_tau_s"], -max_brake, p["acc_max_accel"])
        h_gap[idx], h_v[idx], h_vl[idx] = gap, v, vl
        vl_n = np.maximum(vl - (lead_decel * dt if k * dt < lead_brake else 0.0), 0.0)
        v_n = np.maximum(v + a * dt, 0.0)
        gap_n = gap + 0.5 * (vl + vl_n) * dt - 0.5 * (v + v_n) * dt
        hit = (gap_n <= 0) & ~crashed
        impact[hit] = np.maximum(v_n - vl_n, 0.0)[hit]
        crashed |= hit
        gap, v, vl = gap_n, v_n, vl_n
    return crashed, impact


def configs(p) -> list[dict]:
    keys = ["lead_decel_ms2", "lead_brake_s", "aeb_ttc_s", "sensor_delay_s", "max_brake_ms2"]
    names = ["lead_decel", "lead_brake", "aeb_ttc", "sensor_delay", "max_brake"]
    return [dict(zip(names, c)) for c in itertools.product(*[p[k] for k in keys])]


def run(cfg: Config | None = None, df: pd.DataFrame | None = None, chunk: int = 20000) -> pd.DataFrame:
    cfg = cfg or load()
    p = cfg["stress"]
    df = df if df is not None else pd.read_parquet(cfg.out_dir / "episodes.parquet")
    cs = configs(p)
    rng = np.random.default_rng(p["seed"])
    crash = np.zeros((len(df), len(cs)), bool)
    dv = np.zeros((len(df), len(cs)))
    g0, v0, vl0 = df.gap0.values, df.v0.values, df.vl0.values
    for j, c in enumerate(cs):
        for s in range(0, len(df), chunk):
            sl = slice(s, s + chunk)
            crash[sl, j], dv[sl, j] = simulate(g0[sl], v0[sl], vl0[sl], p=p, rng=rng, **c)
        print(f"stress config {j + 1}/{len(cs)} {c}: crash rate {crash[:, j].mean():.4f}", flush=True)
    out = df[["rec", "loc", "ego", "lead", "frame0"]].copy()
    out["risk"] = crash.mean(1)                                  # share of stress cases that end in contact
    out["severity"] = (crash * dv ** 2).mean(1)                  # mean of (closing speed at contact)^2
    for j in range(len(cs)):
        out[f"crash_{j}"] = crash[:, j]
    out.to_parquet(cfg.out_dir / "stress.parquet", index=False)
    pd.DataFrame(cs).to_csv(cfg.out_dir / "stress_configs.csv", index=False)
    print(f"risk > 0 for {(out.risk > 0).mean():.4f} of {len(out)} decisions")
    return out


if __name__ == "__main__":
    run()
