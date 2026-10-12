"""python -m riskpred.cli {ingest,stress,analyze,all}"""
from __future__ import annotations

import sys

from . import analysis, evt, ingest, stress, transfer
from .config import ROOT, load


def analyze() -> None:
    cfg = load()
    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    e = analysis.load_all(cfg)
    print(f"{len(e)} decisions, {e.rec.nunique()} recordings; risk>0 for {(e.risk > 0).mean():.3%}", flush=True)
    t = analysis.ranking_table(e, cfg)
    t.to_csv(res / "ranking.csv", index=False)
    print(t[["predictor", "group", "spearman", "spearman_lo", "spearman_hi"]].to_string(), flush=True)
    analysis.history_value(e, cfg).to_csv(res / "history_value.csv", index=False)
    analysis.data_size_stability(e, cfg, t).to_csv(res / "data_size.csv", index=False)
    print("data size done", flush=True)
    evt.extrapolation_study(e, cfg).to_csv(res / "evt.csv", index=False)
    print("evt done", flush=True)
    transfer.transfer_study(e, cfg, t.predictor[0]).to_csv(res / "transfer.csv", index=False)
    print("transfer done", flush=True)


if __name__ == "__main__":
    {"ingest": ingest.run, "stress": stress.run, "analyze": analyze}.get(
        sys.argv[1] if len(sys.argv) > 1 else "analyze", analyze)()
