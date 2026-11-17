"""Test metrics per run from test_pred.pt and sim/out/<log>_truth.csv; one CSV per log.

  python -m train.evaluate runs g1_none_s0 g1_date30_s0 g1_date50_s0
Writes results/<log>/metrics.csv, one row per (variant, seed, subset); subset = all / ambiguous (|R| > 1) /
unambiguous / zero_gap (target Δt = 0) / positive_gap (Δt > 0). acc, macro_f1: activity vs the logged label;
acc_true, macro_f1_true, nll_true, brier_true: vs the true next activity (the not-yet-seen event with the lowest
true_rank); brier_true = Σ_a (p_act[a] − 1[a = true next])², multi-class Brier in [0, 2];
out_R: share of targets whose true next activity is not in R (a property of the log, same for every variant).
Time metrics vs the observed Δt, in days. mae_pos: Δt > 0 targets; point = Δt prediction (A0, A2) or mixture
median (A1, A3, A3eps). crps: point forecast → |Δt̂ − Δt|; mixture → Monte Carlo (CRPS_SAMPLES, fixed seed).
brier_zero: P(Δt=0) = π₀ (A1, A3), P(Δt ≤ ε) (A3eps), P(Δt ≤ τ) from the THP intensity (A0, A2).
A3J stores time [N,T,1+3K]; all time metrics use its activity-marginal distribution.
HA1, HA3 (hurdle H-M, plan_exp.md 2.1): π₀ := constant π̄₀ = train zero share (config pi0_bar), so their
brier_zero is the constant baseline's and their crps scores π̄₀·δ₀ + (1 − π̄₀)·f_M⁺.
ALL0 (train/all0.py): point = 0, crps = |Δt|, P(Δt=0) = 1; p_act = train-target frequency.
Real logs (no sim/out/<log>_truth.csv): true next = logged label, so *_true = logged-label metrics.
run_targets and metrics are shared with train/stats.py (bootstrap = metrics under case-resample weights).
"""
import argparse
import json
import math
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from model.losses import _softplus

REPO = Path(__file__).resolve().parent.parent
POINT_HEAD = {"A0", "A2"}          # baseline_act: time[:, 0] = Δt prediction
MIXTURE = {"A1", "A3", "A3eps", "HA1", "HA3", "A3J", "A3M", "A3B", "A3C"}
NO_ATOM = {"A3eps"}                # tie logit untrained: no zero atom, π₀ = 0
HURDLE = {"HA1", "HA3"}            # tie logit trained on Δt > 0 only, unused: π₀ = constant π̄₀
ALL0 = {"ALL0"}                    # train/all0.py: time[:, 0] = Δt̂ = 0, P(Δt=0) = 1
POINT_ONLY = {"SUTRAN", "SEPLSTM", "PT"} # baselines/: time[:, 0] = Δt̂, no P(Δt=0) (brier_zero NaN)
CRPS_SAMPLES, CRPS_SEED = 1000, 0
TAU = 1e-6 / 86400                 # days; one timestamp tick (µs); A0/A2 P(Δt=0) := P(Δt ≤ τ) (plan_exp.md Metrics)
HIGHER = ("acc", "macro_f1", "acc_true", "macro_f1_true")   # higher is better; every other metric: lower


def _mixture(params: torch.Tensor, K: int) -> tuple:
    """Weights, means and scales (as in model/losses.py) of the LogNormal mixture, in double precision."""
    params = params.double()
    return F.softmax(params[:, :K], -1), params[:, K:2 * K], F.softplus(params[:, 2 * K:]) + 1e-4


def joint_time_marginal(p_act: torch.Tensor, time: torch.Tensor, K: int) -> tuple:
    """A3J [N,T,1+3K] -> zero mass and a T*K positive mixture, marginalized over all activities."""
    logp = p_act.double().log()
    logp = logp - torch.logsumexp(logp, -1, keepdim=True)
    tie, params = time[..., 0].double(), time[..., 1:].double()
    pi0 = torch.logsumexp(logp + F.logsigmoid(tie), -1).exp().clamp(0, 1)
    # Positive component weights include P(a) * P(dt>0|a) before normalization.
    log_w = (logp + F.logsigmoid(-tie)).unsqueeze(-1) + F.log_softmax(params[..., :K], -1)
    flat = torch.cat((log_w.flatten(-2), params[..., K:2*K].flatten(-2),
                      params[..., 2*K:].flatten(-2)), -1)
    return pi0, flat


def mixture_cdf(params: torch.Tensor, K: int, x: float) -> torch.Tensor:
    """CDF of Σ w_k LogNormal(μ_k, σ_k) at x > 0, per target."""
    w, mu, sigma = _mixture(params, K)
    return (w * 0.5 * (1 + torch.erf((math.log(x) - mu) / (sigma * math.sqrt(2))))).sum(-1).float()


def thp_p_zero(time: torch.Tensor, alpha: float, beta: float, tau: float = TAU, n: int = 16) -> torch.Tensor:
    """A0/A2: P(Δt ≤ τ) = 1 − exp(−∫₀^τ Σ_k λ_k(s) ds), λ_k(s) = softplus_β(linear_k(h) + α s / (t + 1)) (THP; trapezoid)."""
    lin, t = time[:, 1:-1].double(), time[:, -1].double()            # intensity pre-activation, prefix end time
    s = torch.linspace(0, tau, n, dtype=torch.float64)
    lam = _softplus(lin.unsqueeze(-1) + alpha * s / (t + 1).view(-1, 1, 1), beta).sum(1)   # [N, n]
    return (1 - torch.exp(-torch.trapezoid(lam, s, dim=-1))).float()


def crps_mixture(pi0: torch.Tensor, params: torch.Tensor, K: int, y: torch.Tensor) -> torch.Tensor:
    """CRPS of π₀·δ₀ + (1 − π₀)·mixture at y, per target: E|X − y| − ½ E|X − X'| from sorted samples."""
    g = torch.Generator().manual_seed(CRPS_SEED)
    w, mu, sigma = _mixture(params, K)
    k = torch.multinomial(w, CRPS_SAMPLES, replacement=True, generator=g)          # [N, M] component
    z = torch.randn(k.shape, generator=g, dtype=torch.float64)
    x = torch.exp(mu.gather(1, k) + sigma.gather(1, k) * z)
    x = torch.where(torch.rand(k.shape, generator=g, dtype=torch.float64) < pi0.double().unsqueeze(1), 0.0, x)
    x = x.sort(1).values
    i = torch.arange(1, CRPS_SAMPLES + 1, dtype=torch.float64)
    spread = 2 * ((2 * i - CRPS_SAMPLES - 1) * x).sum(1) / CRPS_SAMPLES ** 2     # mean |X_i − X_j|
    return ((x - y.double().unsqueeze(1)).abs().mean(1) - 0.5 * spread).float()


def true_next(log: str, case: list, pos: list, logged: torch.Tensor, vocab: dict) -> torch.Tensor:
    """True next activity (0-based) per target: the event at or after the target position with the lowest true_rank; logged label without a truth file (real logs)."""
    path = REPO / "sim" / "out" / f"{log}_truth.csv"
    if not path.exists():
        assert not log.startswith("g1_"), f"{log}: {path} missing; regenerate it (README Guide 2, step 1)"
        return logged.clone()
    t = pd.read_csv(path)                                                 # rows in file order = prefix positions
    rank = t.groupby("CaseID", sort=False)["true_rank"].apply(list).to_dict()
    act = t.groupby("CaseID", sort=False)["Activity"].apply(list).to_dict()
    out = []
    for c, j, y in zip(case, pos, logged.tolist()):
        assert vocab[act[c][j]] - 1 == y, f"{log} {c} position {j}: truth file differs from the logged label"
        r = rank[c][j:]
        out.append(vocab[act[c][j + r.index(min(r))]] - 1)
    return torch.tensor(out)


def mixture_median(pi0: torch.Tensor, params: torch.Tensor, K: int, iters: int = 60) -> torch.Tensor:
    """Median of π₀·δ₀ + (1 − π₀)·Σ w_k LogNormal(μ_k, σ_k): 0 if π₀ ≥ 0.5, else bisection on log Δt."""
    w, mu, sigma = _mixture(params, K)
    q = ((0.5 - pi0.double()) / (1 - pi0.double())).clamp(min=0)     # quantile of the positive part
    lo, hi = (mu - 10 * sigma).min(-1).values, (mu + 10 * sigma).max(-1).values
    for _ in range(iters):
        mid = (lo + hi) / 2
        cdf = (w * 0.5 * (1 + torch.erf((mid.unsqueeze(-1) - mu) / (sigma * math.sqrt(2))))).sum(-1)
        lo, hi = torch.where(cdf < q, mid, lo), torch.where(cdf < q, hi, mid)
    return torch.where(pi0 >= 0.5, 0.0, torch.exp((lo + hi) / 2)).float()


def run_targets(folder: Path) -> dict:
    """Per-target labels, predictions and scores of one finished run (tensors of length N), with its config."""
    cfg = json.loads((folder / "config.json").read_text())
    p = torch.load(folder / "test_pred.pt", weights_only=True)
    v = cfg["variant"]
    assert v in POINT_HEAD | MIXTURE | ALL0 | POINT_ONLY, f"no time metrics for {v}"
    true, dt = torch.tensor(p["type"]) - 1, torch.tensor(p["dt"])     # 0-based activity, as p_act columns
    tn = true_next(cfg["log"], p["case"], p["pos"], true, cfg["vocab"])
    K, n = int(cfg["n_mix"]), len(dt)
    if v in POINT_HEAD | ALL0 | POINT_ONLY:
        point = p["time"][:, 0]
        crps = (point - dt).abs()
        p_zero = (torch.ones(n) if v in ALL0 else torch.full((n,), float("nan")) if v in POINT_ONLY
                  else thp_p_zero(p["time"], p["alpha"], p["beta"]))
    else:
        if v == "A3J":
            pi0, params = joint_time_marginal(p["p_act"], p["time"], K)
            K *= p["p_act"].size(-1)
        else:
            params = p["time"][:, 1:]
            pi0 = (torch.full((n,), float(cfg["pi0_bar"])) if v in HURDLE
                   else torch.zeros(n) if v in NO_ATOM else torch.sigmoid(p["time"][:, 0]))
        point = mixture_median(pi0, params, K)
        crps = crps_mixture(pi0, params, K, dt)
        p_zero = mixture_cdf(params, K, float(cfg["eps"])) if v in NO_ATOM else pi0
    return {"cfg": cfg, "case": list(p["case"]), "pos": list(p["pos"]), "n_types": p["p_act"].size(1),
            "true": true, "true_next": tn, "pred": p["p_act"].argmax(-1),
            "nll_true": -torch.log(p["p_act"][torch.arange(n), tn].clamp(min=1e-12)),
            "brier_true": ((p["p_act"].double() - F.one_hot(tn, p["p_act"].size(1)).double()) ** 2).sum(-1),
            "out_R": (~p["R"][torch.arange(n), tn]).float(), "amb": p["R"].sum(-1) > 1,
            "dt": dt, "point": point, "crps": crps, "p_zero": p_zero}


def subsets(t: dict) -> dict:
    """Boolean target masks: all, ambiguous / unambiguous (|R|), zero_gap / positive_gap (observed Δt)."""
    amb, dt = t["amb"], t["dt"]
    return {"all": torch.ones_like(amb), "ambiguous": amb, "unambiguous": ~amb,
            "zero_gap": dt == 0, "positive_gap": dt > 0}


def _wmean(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Weighted mean of x [N] per weight row of w [B, N]; NaN for a row without weight."""
    return (w @ x.double()) / w.sum(1)


def _wmacro_f1(pred: torch.Tensor, label: torch.Tensor, w: torch.Tensor, T: int) -> torch.Tensor:
    """Macro-F1 per weight row, over activities with weighted label or prediction (unweighted: as before)."""
    P, Y = F.one_hot(pred, T).double(), F.one_hot(label, T).double()
    tp, fp, fn = w @ (P * Y), w @ (P * (1 - Y)), w @ ((1 - P) * Y)
    present = (tp + fp + fn) > 0
    f1 = torch.where(present, 2 * tp / (2 * tp + fp + fn).clamp(min=1e-12), torch.zeros_like(tp))
    return f1.sum(1) / present.sum(1)


def metrics(t: dict, w: torch.Tensor) -> dict:
    """Metrics per weight row of w [B, N] (0 = target left out; a 0/1 row = a plain subset) -> name: [B]."""
    w = w.double()
    zero = (t["dt"] == 0).double()
    return {"acc": _wmean((t["pred"] == t["true"]).double(), w),
            "macro_f1": _wmacro_f1(t["pred"], t["true"], w, t["n_types"]),
            "acc_true": _wmean((t["pred"] == t["true_next"]).double(), w),
            "macro_f1_true": _wmacro_f1(t["pred"], t["true_next"], w, t["n_types"]),
            "nll_true": _wmean(t["nll_true"], w), "brier_true": _wmean(t["brier_true"], w), "out_R": _wmean(t["out_R"], w),
            "mae_pos": _wmean((t["point"] - t["dt"]).abs(), w * (1 - zero)),
            "crps": _wmean(t["crps"], w), "brier_zero": _wmean((t["p_zero"].double() - zero) ** 2, w)}


def run_rows(folder: Path) -> list:
    """Metric rows (all / ambiguous / unambiguous / zero_gap / positive_gap) of one finished run."""
    t = run_targets(folder)
    rows = []
    for subset, m in subsets(t).items():
        r = {k: float(v[0]) for k, v in metrics(t, m.double().unsqueeze(0)).items()}
        rows.append({"log": t["cfg"]["log"], "variant": t["cfg"]["variant"], "seed": int(t["cfg"]["seed"]),
                     "subset": subset, "n": int(m.sum()), "n_pos": int((m & (t["dt"] > 0)).sum()), **r})
    return rows


def main():
    """Evaluate every finished run of each log under root; write results/<log>/metrics.csv."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--out", type=Path, default=REPO / "results", help="directory for metrics; use a separate root for new experiments")
    args = parser.parse_args()
    for log in args.logs:
        folders = sorted(f for f in (args.root / log).glob("*/") if (f / "test_pred.pt").exists())
        if not folders:
            print(f"skip {log}: no finished run under {args.root}")
            continue
        df = pd.DataFrame([r for f in folders for r in run_rows(f)])
        out = args.out / log / "metrics.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, float_format="%.4f")
        print(f"{log}: {len(folders)} runs -> {out}")


if __name__ == "__main__":
    main()
