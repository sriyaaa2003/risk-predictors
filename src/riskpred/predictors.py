"""Surrogate safety measures computed from the observation window only (causal)."""
from __future__ import annotations

import numpy as np

# name -> +1 if a LARGER value means MORE risk, -1 if a SMALLER value means more risk
DIRECTION = {
    # state at the decision time t0
    "gap_0": -1, "speed_0": +1, "closing_0": +1, "thw_0": -1, "ttc_0": -1,
    "inv_ttc_0": +1, "drac_0": +1, "psd_0": -1, "rss_ratio_0": -1,
    # summaries over the window that ends at t0
    "ttc_min": -1, "inv_ttc_max": +1, "thw_min": -1, "thw_mean": -1, "drac_max": +1,
    "tit": +1, "closing_max": +1, "gap_cv": +1, "brake_max": +1,
}
INSTANT = [k for k in DIRECTION if k.endswith("_0")]
WINDOW = [k for k in DIRECTION if not k.endswith("_0")]
TTC_CAP = 60.0


def window_predictors(gap, v, vl, a, p) -> dict:
    """gap, v, vl, a: 1-D arrays over the window; the last element is the decision time t0.
    v is ego speed, vl lead speed, a ego acceleration along the driving direction."""
    rel = v - vl
    closing = rel > p["ttc_eps_ms"]
    ttc = np.where(closing, gap / np.where(closing, rel, 1.0), TTC_CAP)
    ttc = np.minimum(ttc, TTC_CAP)
    inv_ttc = np.where(closing, rel / gap, 0.0)
    thw = gap / v
    drac = np.where(closing, rel ** 2 / (2.0 * gap), 0.0)
    g0, v0, vl0 = gap[-1], v[-1], vl[-1]
    d_rss = max(0.0, v0 * p["rss_rho_s"] + v0 ** 2 / (2 * p["rss_b_min"]) - vl0 ** 2 / (2 * p["rss_b_max"]))
    return {
        "gap_0": g0, "speed_0": v0, "closing_0": rel[-1], "thw_0": thw[-1], "ttc_0": ttc[-1],
        "inv_ttc_0": inv_ttc[-1], "drac_0": drac[-1],
        "psd_0": g0 / (v0 ** 2 / (2 * p["psd_brake_ms2"])),
        "rss_ratio_0": g0 / d_rss if d_rss > 0 else 1e3,
        "ttc_min": ttc.min(), "inv_ttc_max": inv_ttc.max(), "thw_min": thw.min(), "thw_mean": thw.mean(),
        "drac_max": drac.max(), "tit": float((ttc < p["ttc_critical_s"]).mean()),
        "closing_max": rel.max(), "gap_cv": float(gap.std() / max(gap.mean(), 1e-6)),
        "brake_max": float(max(0.0, -a.min())),
    }
