"""Predictor-vs-risk analyses: ranking with recording-level bootstrap, history value, data-size stability."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from .config import Config, load
from .predictors import DIRECTION, INSTANT, WINDOW


def load_all(cfg: Config) -> pd.DataFrame:
    e = pd.read_parquet(cfg.out_dir / "episodes.parquet")
    s = pd.read_parquet(cfg.out_dir / "stress.parquet")
    assert (e.rec.values == s.rec.values).all() and (e.frame0.values == s.frame0.values).all() \
        and (e.ego.values == s.ego.values).all(), "episodes and stress rows are not aligned"
    e["risk"] = s.risk.values
    e["severity"] = s.severity.values
    return e


def oriented(e: pd.DataFrame, names=None) -> dict[str, np.ndarray]:
    """Scores where a LARGER value always means MORE predicted risk."""
    return {k: DIRECTION[k] * e[k].to_numpy(float) for k in (names or DIRECTION)}


def spearman_ranks(rs: np.ndarray, rr: np.ndarray) -> float:
    a, b = rs - rs.mean(), rr - rr.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def auc_from_ranks(rs: np.ndarray, y: np.ndarray) -> float:
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((rs[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def metrics(S: dict[str, np.ndarray], risk: np.ndarray, idx: np.ndarray, thresholds) -> dict[str, np.ndarray]:
    rr = rankdata(risk[idx])
    ys = [risk[idx] > t if t == 0 else risk[idx] >= t for t in thresholds]
    out = {}
    for k, s in S.items():
        rs = rankdata(s[idx])
        out[k] = np.array([spearman_ranks(rs, rr)] + [auc_from_ranks(rs, y) for y in ys])
    return out


def ranking_table(e: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Per-predictor Spearman and AUROC against the stress-test risk, bootstrap over whole recordings."""
    a = cfg["analysis"]
    rng = np.random.default_rng(a["seed"])
    S, risk = oriented(e), e.risk.to_numpy()
    recs = e.rec.to_numpy()
    by_rec = {r: np.flatnonzero(recs == r) for r in np.unique(recs)}
    keys = list(by_rec)
    th = a["risk_thresholds"]
    full = metrics(S, risk, np.arange(len(e)), th)
    boots = {k: [] for k in S}
    for _ in range(a["bootstrap"]):
        idx = np.concatenate([by_rec[r] for r in rng.choice(keys, len(keys), replace=True)])
        m = metrics(S, risk, idx, th)
        for k in S:
            boots[k].append(m[k])
    cols = ["spearman"] + [f"auroc_risk{'>0' if t == 0 else f'>={t}'}" for t in th]
    rows = []
    for k in S:
        b = np.array(boots[k])
        row = {"predictor": k, "group": "window" if k in WINDOW else "instant"}
        for j, c in enumerate(cols):
            row[c] = full[k][j]
            row[c + "_lo"], row[c + "_hi"] = np.nanpercentile(b[:, j], [2.5, 97.5])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("spearman", ascending=False).reset_index(drop=True)


def history_value(e: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Does a combined model, or the 5 s of history, beat the best single named metric? Grouped CV by recording."""
    th = [t for t in cfg["analysis"]["risk_thresholds"] if t > 0 or True]
    sets = {"best single metric": None, "instant metrics (9)": INSTANT, "window metrics (9)": WINDOW,
            "all metrics (18)": INSTANT + WINDOW}
    S = oriented(e)
    risk = e.risk.to_numpy()
    rows = []
    for t in th:
        y = (risk > t) if t == 0 else (risk >= t)
        if y.sum() < 50:
            continue
        # best single metric chosen inside each training fold would be cleaner, but a single score has no fit
        best = max(S, key=lambda k: roc_auc_score(y, S[k]))
        for name, feats in sets.items():
            aucs, aps = [], []
            for tr, te in GroupKFold(5).split(e, y, e.rec):
                if feats is None:
                    p = S[best][te]
                else:
                    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0)
                    m.fit(e.iloc[tr][feats], y[tr])
                    p = m.predict_proba(e.iloc[te][feats])[:, 1]
                aucs.append(roc_auc_score(y[te], p))
                aps.append(average_precision_score(y[te], p))
            rows.append({"target": f"risk{'>0' if t == 0 else f'>={t}'}", "features": name + (f" = {best}" if feats is None else ""),
                         "auroc": np.mean(aucs), "auroc_sd": np.std(aucs), "avg_precision": np.mean(aps),
                         "base_rate": y.mean()})
    return pd.DataFrame(rows)


def data_size_stability(e: pd.DataFrame, cfg: Config, table: pd.DataFrame) -> pd.DataFrame:
    """With k recordings of data, how often is the best metric (and the order of the top two) recovered?"""
    a = cfg["analysis"]
    rng = np.random.default_rng(a["seed"] + 1)
    S, risk, recs = oriented(e), e.risk.to_numpy(), e.rec.to_numpy()
    by_rec = {r: np.flatnonzero(recs == r) for r in np.unique(recs)}
    keys = list(by_rec)
    top = table.predictor.tolist()
    best, second = top[0], top[1]
    fam_ttc = ["ttc_min", "ttc_0", "inv_ttc_max", "inv_ttc_0", "drac_max", "drac_0"]
    fam_thw = ["thw_0", "thw_min", "thw_mean"]
    rows = []
    for k in a["data_sizes_recordings"]:
        wins, top3, order_ok, rho_best, n_ep, fam_ok = 0, 0, 0, [], [], 0
        full_sign = np.sign(table.spearman[0] - table.spearman[1])
        for _ in range(a["size_repeats"]):
            idx = np.concatenate([by_rec[r] for r in rng.choice(keys, k, replace=False)])
            rr = rankdata(risk[idx])
            rho = {p: spearman_ranks(rankdata(S[p][idx]), rr) for p in dict.fromkeys(top[:8] + fam_ttc + fam_thw)}
            fam_ok += max(rho[p] for p in fam_ttc) > max(rho[p] for p in fam_thw)
            ranked = sorted((p for p in rho if p in top[:8]), key=rho.get, reverse=True)
            wins += ranked[0] == best
            top3 += best in ranked[:3]
            order_ok += np.sign(rho[best] - rho[second]) == full_sign
            rho_best.append(rho[best])
            n_ep.append(len(idx))
        lo, hi = np.percentile(rho_best, [2.5, 97.5])
        rows.append({"recordings": k, "decisions_median": int(np.median(n_ep)), "p_best_found": wins / a["size_repeats"],
                     "p_best_in_top3": top3 / a["size_repeats"], "p_top2_order_right": order_ok / a["size_repeats"],
                     "p_ttc_family_beats_thw_family": fam_ok / a["size_repeats"],
                     "best_spearman_lo": lo, "best_spearman_hi": hi, "best_spearman_width": hi - lo})
    return pd.DataFrame(rows)
