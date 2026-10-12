"""Four figures from the CSVs in results/."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ROOT

BLUE, ORANGE, GREY, INK = "#0072B2", "#D55E00", "#8a8f98", "#222222"
FAMILY = {"ttc": BLUE, "thw": ORANGE, "other": GREY}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="both", color="#e6e8eb", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=9)


def family(name: str) -> str:
    if name.startswith(("ttc", "inv_ttc", "drac")):
        return "ttc"
    return "thw" if name.startswith("thw") else "other"


def fig_ranking(res, out):
    t = pd.read_csv(res / "ranking.csv").sort_values("spearman")
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    y = np.arange(len(t))
    for i, r in enumerate(t.itertuples()):
        c = FAMILY[family(r.predictor)]
        ax.plot([r.spearman_lo, r.spearman_hi], [i, i], color=c, lw=1.6, solid_capstyle="round")
        ax.plot(r.spearman, i, "o", color=c, ms=7, mec="white", mew=1.2)
    ax.set_yticks(y, t.predictor)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("Spearman correlation with stress-test risk (95% CI, bootstrap over recordings)", fontsize=9)
    ax.set_title("Which safety metric tracks the braking stress test?", loc="left", fontsize=11, color=INK)
    ax.text(0.26, 3.0, "TTC / DRAC family", color=BLUE, fontsize=9, weight="bold")
    ax.text(0.26, 2.0, "time-headway family", color=ORANGE, fontsize=9, weight="bold")
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "ranking.png", dpi=170)


def fig_size(res, out):
    d = pd.read_csv(res / "data_size.csv")
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.8))
    ax = axs[0]
    ax.plot(d.decisions_median, d.p_ttc_family_beats_thw_family, color=BLUE, lw=2, marker="o", ms=5)
    ax.plot(d.decisions_median, d.p_best_found, color=ORANGE, lw=2, marker="o", ms=5)
    ax.text(d.decisions_median.iloc[0], 1.03, "TTC family beats headway family", color=BLUE, fontsize=8.5, weight="bold")
    ax.text(d.decisions_median.iloc[0] * 1.5, 0.18, "single best metric picked", color=ORANGE, fontsize=8.5, weight="bold")
    ax.set_xscale("log"); ax.set_ylim(0, 1.1)
    ax.set_xlabel("decisions available (median)", fontsize=9); ax.set_ylabel("share of 200 random subsets that get it right", fontsize=9)
    ax.set_title("What a conclusion costs in data", loc="left", fontsize=10.5, color=INK)
    style(ax)
    ax = axs[1]
    ax.fill_between(d.decisions_median, d.best_spearman_lo, d.best_spearman_hi, color=BLUE, alpha=0.18, lw=0)
    ax.plot(d.decisions_median, (d.best_spearman_lo + d.best_spearman_hi) / 2, color=BLUE, lw=2, marker="o", ms=5)
    ax.set_xscale("log")
    ax.set_xlabel("decisions available (median)", fontsize=9); ax.set_ylabel("Spearman of the best metric (95% range)", fontsize=9)
    ax.set_title("How tight the estimate gets", loc="left", fontsize=10.5, color=INK)
    style(ax)
    fig.tight_layout(); fig.savefig(out / "data_size.png", dpi=170)


def fig_evt(res, out):
    e = pd.read_csv(res / "evt.csv")
    fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.5), sharey=True)
    for ax, (th, g) in zip(axs, e.groupby("ttc_threshold_s", sort=False)):
        ax.plot(g.sample_size, g.gpd_within_2x, color=BLUE, lw=2, marker="o", ms=5)
        ax.plot(g.sample_size, g.empirical_within_2x, color=ORANGE, lw=2, marker="o", ms=5)
        ax.set_xscale("log"); ax.set_ylim(-0.02, 1.05)
        ax.set_title(f"TTC < {th:g} s ({int(g.true_events_in_full_data.iloc[0])} events in 139k)", loc="left", fontsize=9.5, color=INK)
        ax.set_xlabel("decisions in the sample", fontsize=9)
        style(ax)
    axs[0].set_ylabel("estimates within a factor 2 of the truth", fontsize=9)
    axs[0].text(520, 0.12, "counting", color=ORANGE, fontsize=9, weight="bold")
    axs[0].text(520, 0.62, "extreme-value tail", color=BLUE, fontsize=9, weight="bold")
    fig.tight_layout(); fig.savefig(out / "extrapolation.png", dpi=170)


def fig_transfer(res, out):
    t = pd.read_csv(res / "transfer.csv")
    g = t.groupby(["estimator", "calibration_decisions"]).median_rel_error.median().reset_index()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    spec = {"simulate m target decisions": (ORANGE, "simulate m decisions at the new site"),
            "gradient boosting on 9 state metrics": (BLUE, "metrics model calibrated on other sites"),
            "source-site mean": (GREY, "just reuse other sites' average")}
    for est, (c, label) in spec.items():
        d = g[g.estimator == est]
        ax.plot(d.calibration_decisions, d.median_rel_error, color=c, lw=2, marker="o", ms=5, label=label)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("decisions used (simulated at the new site, or used for calibration elsewhere)", fontsize=9)
    ax.set_ylabel("median error of the site's mean risk", fontsize=9)
    ax.set_title("Assessing a new site without simulating it", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8.5)
    style(ax)
    fig.tight_layout(); fig.savefig(out / "transfer.png", dpi=170)


def run():
    res = ROOT / "results"
    out = res / "figures"
    out.mkdir(exist_ok=True)
    fig_ranking(res, out); fig_size(res, out); fig_evt(res, out); fig_transfer(res, out)


if __name__ == "__main__":
    run()
