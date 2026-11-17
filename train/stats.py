"""Paired case bootstrap for plan_exp.md claim rows 1-4 (checks 2.1, 2.2): gain, 95% CI and pass flag per row.

  python -m train.stats runs g1_date30_s0 [g1_date30_s1 g1_date30_s2]      # logs of one config (generation seeds)
Per log, B resamples of the test cases (all targets of a case together); every method is scored on the same
resamples; per resample a metric is averaged over training seeds, the gain over logs (plan_exp.md §6).
Gain = improvement in the metric's good direction. better: CI lower bound > 0; not worse: CI lower bound > −δ,
δ = 0.01 for accuracy / macro-F1, else 5% of the baseline's value. A (check, method, baseline) passes iff all
its rows pass; "report" rows (2.2 accuracy, macro-F1, NLL) get CIs but no pass flag. 2.2 passes on brier_true.
Real logs: python -m train.stats runs_real BPIC15_1 --out results_real ("real" rows: A3C vs baselines, report only).
Writes <out>/stats_<config>.csv.
"""
import argparse
import re
from pathlib import Path

import pandas as pd
import torch

from train.evaluate import HIGHER, metrics, run_targets, subsets

REPO = Path(__file__).resolve().parent.parent
B, SEED, LEVEL = 2000, 0, 0.95
# 2.1: part 1 brier_zero (HA = constant π̄₀), part 2 crps; all targets; A3C (same time head as A3) vs H-A3
TIME_PAIRS = [("A1", "HA1"), ("A3", "HA3"), ("A3C", "HA3")]
ACT_PAIRS = [("A2", "A0"), ("A3", "A1")]                     # 2.2
# (check, method, baseline, subset, metric, condition)
# 2.2 pass rule on brier_true only (2026-10-27, post hoc); accuracy, macro-F1, NLL kept as "report" rows (no pass)
ROWS = ([("2.1", a, b, "all", m, "better") for a, b in TIME_PAIRS for m in ("brier_zero", "crps")]
        + [("2.2", a, b, "ambiguous", "brier_true", "better") for a, b in ACT_PAIRS]
        + [("2.2", a, b, "unambiguous", "brier_true", "not_worse") for a, b in ACT_PAIRS]
        + [("2.2", a, b, "ambiguous", m, "report") for a, b in ACT_PAIRS
           for m in ("acc_true", "macro_f1_true", "nll_true")]
        + [("2.2", a, b, "unambiguous", m, "report") for a, b in ACT_PAIRS for m in ("acc_true", "macro_f1_true")])
# Exploratory extension; these comparisons do not change the paper's C2 gate.
ROWS += [("joint", "A3J", "A3", s, m, c) for s, m, c in (
    ("all", "brier_zero", "better"), ("all", "crps", "better"),
    ("ambiguous", "brier_true", "better"), ("unambiguous", "brier_true", "not_worse"),
    ("ambiguous", "acc_true", "report"), ("ambiguous", "macro_f1_true", "report"),
    ("ambiguous", "nll_true", "report"), ("unambiguous", "acc_true", "report"),
    ("unambiguous", "macro_f1_true", "report"))]
# Check 2.10 (plan_exp.md, post hoc 2026-10-29) for A3C: part 1 vs A1 (2.2 rows), part 3 vs A3, part 4 = 2.4 vs A0 on M-none
ROWS += ([("2.10", "A3C", "A1", "ambiguous", "brier_true", "better"),
          ("2.10", "A3C", "A1", "unambiguous", "brier_true", "not_worse")]
         + [("2.10", "A3C", "A3", s, "brier_true", "not_worse") for s in ("ambiguous", "unambiguous")]
         + [("2.10", "A3C", "A0", "all", m, "not_worse") for m in ("acc_true", "macro_f1_true", "mae_pos", "crps")])
# RQ1 (2026-11-03, post hoc): A3C keeps THP-M's time head (A1) on the time
ROWS += [("rq1", "A3C", "A1", "all", m, "not_worse") for m in ("brier_zero", "crps")]
# Real logs (BPIC15, 2026-10-31, post hoc): A3C vs baselines, CIs only; brier_zero only where the baseline has P(Δt=0)
REAL_BASELINES = ("ALL0", "SEPLSTM", "PT", "SUTRAN", "A0")
ROWS += ([("real", "A3C", b, "all", m, "report") for b in REAL_BASELINES
          for m in ("acc_true", "macro_f1_true", "nll_true", "brier_true", "mae_pos", "crps")]
         + [("real", "A3C", b, "all", "brier_zero", "report") for b in ("ALL0", "A0")])


def case_weights(cases: list, n_boot: int, seed: int) -> torch.Tensor:
    """[1 + n_boot, N] target weights: row 0 = the test set, then case resamples (a case's count on all its targets)."""
    ids = {c: i for i, c in enumerate(dict.fromkeys(cases))}
    idx = torch.tensor([ids[c] for c in cases])
    draw = torch.randint(len(ids), (n_boot, len(ids)), generator=torch.Generator().manual_seed(seed))
    counts = torch.zeros(n_boot, len(ids), dtype=torch.float64).scatter_add_(1, draw, torch.ones_like(draw, dtype=torch.float64))
    return torch.cat([torch.ones(1, len(cases), dtype=torch.float64), counts[:, idx]])


def log_scores(root: Path, log: str, seed: int) -> tuple:
    """(variant, subset, metric) -> [1 + B] values averaged over training seeds, on shared resamples; runs per variant."""
    runs = {}
    for f in sorted((root / log).glob("*/test_pred.pt")):
        t = run_targets(f.parent)
        runs.setdefault(t["cfg"]["variant"], []).append(t)
    assert runs, f"no finished run under {root / log}"
    ref = next(iter(runs.values()))[0]
    keys = list(zip(ref["case"], ref["pos"]))
    w = case_weights(ref["case"], B, seed)
    out = {}
    for v, ts in runs.items():
        assert all(list(zip(t["case"], t["pos"])) == keys for t in ts), f"{log} {v}: test targets differ"
        for s in subsets(ref):
            per_run = [metrics(t, w * subsets(t)[s].double()) for t in ts]
            for m in per_run[0]:
                out[(v, s, m)] = torch.stack([r[m] for r in per_run]).mean(0)
    return out, {v: len(ts) for v, ts in runs.items()}


def main():
    """Gain, CI and pass flag of every ROWS entry for the logs of one config; writes results/stats_<config>.csv."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--out", type=Path, default=REPO / "results", help="directory for paired comparison tables")
    args = parser.parse_args()
    configs = {re.sub(r"_s\d+$", "", log) for log in args.logs}
    assert len(configs) == 1, f"logs from more than one config: {configs}"
    per_log = [log_scores(args.root, log, SEED + i) for i, log in enumerate(args.logs)]
    q = torch.tensor([(1 - LEVEL) / 2, (1 + LEVEL) / 2], dtype=torch.float64)
    rows = []
    for check, a, b, s, m, cond in ROWS:
        if not all((a, s, m) in sc and (b, s, m) in sc for sc, _ in per_log):
            print(f"skip {check} {a} vs {b} {s} {m}: variant not run on every log")
            continue
        sign = 1 if m in HIGHER else -1
        gain = torch.stack([sign * (sc[(a, s, m)] - sc[(b, s, m)]) for sc, _ in per_log]).mean(0)   # [1 + B]
        base = float(torch.stack([sc[(b, s, m)][0] for sc, _ in per_log]).mean())
        lo, hi = torch.nanquantile(gain[1:], q).tolist()
        delta = 0.01 if m in HIGHER else 0.05 * abs(base)
        rows.append({"check": check, "method": a, "baseline": b, "subset": s, "metric": m, "condition": cond,
                     "baseline_value": base, "gain": float(gain[0]), "ci_lo": lo, "ci_hi": hi, "delta": delta,
                     "pass": (bool(lo > 0) if cond == "better" else bool(lo > -delta) if cond == "not_worse"
                              else None),
                     "runs_method": sum(n[a] for _, n in per_log), "runs_baseline": sum(n[b] for _, n in per_log)})
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No requested comparison has both variants; include the baseline runs in root")
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"stats_{configs.pop()}.csv"
    df.to_csv(out, index=False, float_format="%.4f")
    for (check, a, b), g in df[df["condition"] != "report"].groupby(["check", "method", "baseline"], sort=False):
        g = g.assign(**{"pass": g["pass"].astype(bool)})
        print(f"{check} {a} vs {b}: {'PASS' if g['pass'].all() else 'FAIL'} ({int(g['pass'].sum())}/{len(g)} rows)")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
