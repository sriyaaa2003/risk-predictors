"""Can a safety metric replace simulation at a NEW site? Estimate a site's risk from metrics alone,
calibrated on other sites, and compare with simulating a few of the new site's own decisions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

from .analysis import oriented
from .config import Config
from .predictors import INSTANT


def transfer_study(e: pd.DataFrame, cfg: Config, best_metric: str) -> pd.DataFrame:
    a = cfg["analysis"]
    rng = np.random.default_rng(a["seed"] + 3)
    S = oriented(e)
    risk = e.risk.to_numpy()
    loc = e["loc"].to_numpy()
    rows = []
    sizes = [500, 2000, 10000, 40000]
    reps = 30
    for L in np.unique(loc):
        tgt, src = np.flatnonzero(loc == L), np.flatnonzero(loc != L)
        truth = risk[tgt].mean()
        for n in sizes:
            errs = {"source-site mean": [], f"isotonic on {best_metric}": [], "gradient boosting on 9 state metrics": []}
            for _ in range(reps):
                tr = rng.choice(src, min(n, len(src)), replace=False)
                errs["source-site mean"].append(risk[tr].mean())
                iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(S[best_metric][tr], risk[tr])
                errs[f"isotonic on {best_metric}"].append(iso.predict(S[best_metric][tgt]).mean())
                gb = HistGradientBoostingRegressor(max_iter=100, random_state=0).fit(e.iloc[tr][INSTANT], risk[tr])
                errs["gradient boosting on 9 state metrics"].append(np.clip(gb.predict(e.iloc[tgt][INSTANT]), 0, 1).mean())
            for name, v in errs.items():
                v = np.array(v)
                rows.append({"target_location": int(L), "target_decisions": len(tgt), "true_mean_risk": truth,
                             "estimator": name, "calibration_decisions": n,
                             "median_rel_error": float(np.median(np.abs(v - truth) / truth))})
        # the alternative: simulate m of the target site's own decisions and average them
        for m in [10, 30, 100, 300, 1000, 3000]:
            if m > len(tgt):
                continue
            v = np.array([risk[rng.choice(tgt, m, replace=False)].mean() for _ in range(300)])
            rows.append({"target_location": int(L), "target_decisions": len(tgt), "true_mean_risk": truth,
                         "estimator": "simulate m target decisions", "calibration_decisions": m,
                         "median_rel_error": float(np.median(np.abs(v - truth) / truth))})
    return pd.DataFrame(rows)
