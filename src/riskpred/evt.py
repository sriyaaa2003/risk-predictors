"""How much data does it take to estimate the frequency of rare close calls? Empirical vs extreme-value (GPD)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import genpareto

from .config import Config


def tail_prob_gpd(y: np.ndarray, y0: float, q: float, min_exceed: int = 15) -> float:
    """P(Y > y0) with a generalised Pareto tail above the q-quantile of the positive part of Y.
    Y is zero when the car is not closing in, so the model is p_pos * P(Y > y0 | Y > 0)."""
    pos = y[y > 0]
    if len(pos) == 0:
        return float("nan")
    p_pos = len(pos) / len(y)
    u = float(np.quantile(pos, q))
    exc = pos[pos > u] - u
    if len(exc) < min_exceed:
        return float("nan")
    if y0 <= u:
        return float((y > y0).mean())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xi, _, sigma = genpareto.fit(exc, floc=0.0)
    if not np.isfinite(xi) or sigma <= 0:
        return float("nan")
    z = (y0 - u) / sigma
    surv = np.exp(-z) if abs(xi) < 1e-6 else max(1.0 + xi * z, 0.0) ** (-1.0 / xi)
    return float(p_pos * (len(exc) / len(pos)) * surv)


def extrapolation_study(e: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    a = cfg["analysis"]
    rng = np.random.default_rng(a["seed"] + 2)
    y = e.inv_ttc_max.to_numpy(float)              # 1 / (smallest time to collision in the 5 s window)
    rows = []
    for theta in a["evt_thetas_s"]:
        y0 = 1.0 / theta
        truth = float((y > y0).mean())
        n_events = int((y > y0).sum())
        for n in a["evt_sizes"]:
            emp, gpd = [], []
            for _ in range(a["evt_repeats"]):
                s = y[rng.choice(len(y), n, replace=False)]
                emp.append(float((s > y0).mean()))
                gpd.append(tail_prob_gpd(s, y0, a["evt_tail_quantile"]))
            emp, gpd = np.array(emp), np.array(gpd)
            within = lambda est: float(np.mean((est > 0) & (est / truth > 0.5) & (est / truth < 2.0)))
            ok = np.isfinite(gpd)
            rows.append({
                "ttc_threshold_s": theta, "true_events_in_full_data": n_events, "true_rate": truth, "sample_size": n,
                "empirical_within_2x": within(emp), "empirical_zero_events": float((emp == 0).mean()),
                "gpd_within_2x": float(np.mean(ok & (gpd / truth > 0.5) & (gpd / truth < 2.0))),
                "gpd_fit_failed": float((~ok).mean()),
                "gpd_median_ratio": float(np.nanmedian(gpd / truth)) if ok.any() else float("nan"),
            })
    return pd.DataFrame(rows)
