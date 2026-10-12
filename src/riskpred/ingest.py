"""highD recordings -> one row per car-following decision, with causal predictors and the t0 state."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, load
from .predictors import window_predictors

COLS = ["frame", "id", "x", "width", "xVelocity", "xAcceleration", "precedingId", "laneId", "dhw"]


def recordings(root) -> list[int]:
    return sorted(int(p.name[:2]) for p in root.glob("[0-9][0-9]_tracks.csv"))


def read_recording(root, rid: int) -> tuple[pd.DataFrame, int, float]:
    pre = root / f"{rid:02d}"
    rm = pd.read_csv(f"{pre}_recordingMeta.csv").iloc[0]
    tm = pd.read_csv(f"{pre}_tracksMeta.csv")[["id", "class", "drivingDirection"]]
    t = pd.read_csv(f"{pre}_tracks.csv", usecols=COLS).merge(tm, on="id", how="left")
    lead = t[["frame", "id", "x", "width", "xVelocity", "laneId"]].rename(columns={
        "id": "precedingId", "x": "xl", "width": "wl", "xVelocity": "vl_raw", "laneId": "lane_l"})
    t = t.merge(lead, on=["frame", "precedingId"], how="left")
    t = t.sort_values(["id", "frame"], kind="stable").reset_index(drop=True)
    sgn = np.where(t.drivingDirection == 2, 1.0, -1.0)   # drivingDirection 2 drives toward +x
    t["v"] = sgn * t.xVelocity
    t["vl"] = sgn * t.vl_raw
    t["a"] = sgn * t.xAcceleration
    # bumper-to-bumper gap from positions (x is the left edge, width is the vehicle length)
    t["gap"] = np.where(t.drivingDirection == 2, t.xl - (t.x + t.width), t.x - (t.xl + t.wl))
    return t, int(rm.locationId), float(rm.frameRate)


def episodes(t: pd.DataFrame, rid: int, loc: int, fps: float, cfg: Config) -> list[dict]:
    d, p = cfg["data"], cfg["predictors"]
    n_obs = int(round(d["obs_s"] * fps))
    ok = ((t["class"] == d["ego_class"]) & (t.precedingId > 0) & t.gap.notna() & (t.lane_l == t.laneId)
          & (t.gap > 0) & (t.gap <= d["max_gap_m"]) & (t.v >= d["min_speed_ms"]) & (t.vl >= 0)).values
    ids, fr, lead_id = t.id.values, t.frame.values, t.precedingId.values
    brk = np.ones(len(t), bool)
    brk[1:] = (ids[1:] != ids[:-1]) | (fr[1:] != fr[:-1] + 1) | (lead_id[1:] != lead_id[:-1]) | ~ok[1:] | ~ok[:-1]
    run = np.cumsum(brk)
    run[~ok] = -1
    g, v, vl, a, dhw = t.gap.values, t.v.values, t.vl.values, t.a.values, t.dhw.values
    out = []
    for s in np.flatnonzero(brk & ok):
        e = s
        while e + 1 < len(t) and run[e + 1] == run[s]:
            e += 1
        for w in range((e - s + 1) // (n_obs + 1)):        # non-overlapping windows, decision at the last frame
            i0 = s + w * (n_obs + 1)
            sl = slice(i0, i0 + n_obs + 1)
            row = window_predictors(g[sl], v[sl], vl[sl], a[sl], p)
            row.update(rec=rid, loc=loc, ego=int(ids[s]), lead=int(lead_id[s]), frame0=int(fr[sl][-1]),
                       gap0=float(g[sl][-1]), v0=float(v[sl][-1]), vl0=float(vl[sl][-1]),
                       dhw_err=float(dhw[sl][-1] - g[sl][-1]))
            out.append(row)
    return out


def run(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load()
    root = cfg.raw_dir
    rows = []
    for rid in recordings(root):
        t, loc, fps = read_recording(root, rid)
        r = episodes(t, rid, loc, fps, cfg)
        rows += r
        print(f"recording {rid:02d} location {loc} -> {len(r)} decisions", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(cfg.out_dir / "episodes.parquet", index=False)
    print(f"{len(df)} decisions from {df.rec.nunique()} recordings, {df.ego.nunique()} ego vehicles")
    return df


if __name__ == "__main__":
    run()
