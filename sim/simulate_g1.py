#!/usr/bin/env python
"""Simulator G1 (sequence + XOR, no loops) with one zero-gap mechanism.

Writes per log into --out:
  <name>.csv        CaseID, Activity, Timestamp   (observed log, input of prepare_uq4ppm.py)
  <name>_truth.csv  CaseID, Activity, true_ts, obs_ts, true_rank, cause
Mechanisms: none; date (date-only activities lose their time of day);
batch (an event copies its predecessor's timestamp; --batch_acts limits it to the listed activities,
log name batchR). Each applies to an event
with probability q, calibrated by bisection to hit the target zero-gap share.
Rows inside a tie block are written in random order.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

START = pd.Timestamp("2026-01-01", tz="UTC")
ARRIVAL_MEAN_H = 2.0
MEDIAN_H = {"Check_A": 20, "Check_B": 40, "Decide": 30,          # median gap (h) ending at activity
            "Approve": 1, "Reject": 0.5, "Notify": 2, "Archive": 3}
SIGMA = 0.8
DATE_ONLY = {"Decide", "Approve", "Reject", "Notify", "Archive"}


def trace(rng) -> list[str]:
    """Draw one activity sequence; the decision depends on the check branch."""
    check = "Check_A" if rng.random() < 0.6 else "Check_B"
    end = "Approve" if rng.random() < (0.8 if check == "Check_A" else 0.4) else "Reject"
    return ["Register", check, "Decide", end, "Notify", "Archive"]


def true_log(n_cases: int, rng) -> pd.DataFrame:
    """Fine-grained log: Poisson arrivals, LogNormal gaps, no zero gaps."""
    starts = START + pd.to_timedelta(np.cumsum(rng.exponential(ARRIVAL_MEAN_H, n_cases)), unit="h")
    rows = []
    for c, t in enumerate(starts):
        for rank, act in enumerate(trace(rng)):
            if rank:
                t += pd.Timedelta(hours=float(rng.lognormal(np.log(MEDIAN_H[act]), SIGMA)))
            rows.append((f"c{c}", act, t, rank))
    return pd.DataFrame(rows, columns=["CaseID", "Activity", "true_ts", "true_rank"])


def coarsen(df: pd.DataFrame, mech: str, q: float, u: np.ndarray, batch_acts=None) -> pd.DataFrame:
    """Coarsen every eligible event with u < q (batch: only batch_acts if given); record which events were coarsened."""
    out = df.assign(obs_ts=df["true_ts"], cause="none")
    if mech == "date":
        hit = out["Activity"].isin(DATE_ONLY).to_numpy() & (u < q)
        out.loc[hit, "obs_ts"] = out.loc[hit, "true_ts"].dt.floor("D")
    elif mech == "batch":
        hit = (out["true_rank"] > 0).to_numpy() & (u < q)
        if batch_acts:
            hit &= out["Activity"].isin(batch_acts).to_numpy()
        obs = out["obs_ts"].to_numpy().copy()
        for i in np.flatnonzero(hit):  # left to right, so runs chain
            obs[i] = obs[i - 1]
        out["obs_ts"] = pd.to_datetime(obs, utc=True)
    else:
        return out
    out.loc[hit, "cause"] = mech
    return out


def zero_share(df: pd.DataFrame) -> float:
    """Share of within-case gaps equal to 0 in the observed times."""
    gaps = df.sort_values(["CaseID", "obs_ts"]).groupby("CaseID")["obs_ts"].diff().dropna()
    return float((gaps == pd.Timedelta(0)).mean())


def calibrate(df: pd.DataFrame, mech: str, share: float, u: np.ndarray, batch_acts=None) -> float:
    """Bisection on q until the zero-gap share reaches the target."""
    top = zero_share(coarsen(df, mech, 1.0, u, batch_acts))
    if top < share:
        raise SystemExit(f"{mech}: share {share:.2f} unreachable (max {top:.3f})")
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if zero_share(coarsen(df, mech, mid, u, batch_acts)) < share else (lo, mid)
    return hi


def observe(df: pd.DataFrame, rng) -> pd.DataFrame:
    """Observed row order: by observed time, random inside a tie block."""
    return (df.assign(_r=rng.random(len(df)))
              .sort_values(["CaseID", "obs_ts", "_r"]).drop(columns="_r"))


def main() -> None:
    """Generate one coarsened G1 log and its truth file."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mech", choices=["none", "date", "batch"], required=True)
    ap.add_argument("--share", type=float, default=0.0, help="target zero-gap share of gaps")
    ap.add_argument("--cases", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0, help="generation seed")
    ap.add_argument("--out", type=Path, default=Path("sim/out"))
    ap.add_argument("--batch_acts", nargs="+", help="batch only: activities that may copy their predecessor's time")
    a = ap.parse_args()
    if a.batch_acts:
        assert a.mech == "batch", "--batch_acts needs --mech batch"
        assert set(a.batch_acts) <= set(MEDIAN_H), f"unknown or first activity in {a.batch_acts}"

    rng = np.random.default_rng(a.seed)
    df = true_log(a.cases, rng)
    u = rng.random(len(df))
    q = calibrate(df, a.mech, a.share, u, a.batch_acts) if a.mech != "none" else 0.0
    obs = observe(coarsen(df, a.mech, q, u, a.batch_acts), rng)

    scale = "" if a.cases == 2000 else f"_{a.cases // 1000}k"   # 2k logs keep their old names
    name = f"g1_{a.mech}{'R' if a.batch_acts else ''}{'' if a.mech == 'none' else round(100 * a.share)}{scale}_s{a.seed}"
    a.out.mkdir(parents=True, exist_ok=True)
    obs.rename(columns={"obs_ts": "Timestamp"})[["CaseID", "Activity", "Timestamp"]] \
       .to_csv(a.out / f"{name}.csv", index=False)
    obs[["CaseID", "Activity", "true_ts", "obs_ts", "true_rank", "cause"]] \
       .to_csv(a.out / f"{name}_truth.csv", index=False)
    print(f"{name}: q={q:.3f}  zero-gap share={zero_share(obs):.3f}  events={len(obs)}")


if __name__ == "__main__":
    main()
