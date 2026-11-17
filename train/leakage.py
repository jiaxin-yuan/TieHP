#!/usr/bin/env python
"""T2 (hypothesis H2): where A1 (CE) and A3 (set likelihood) put probability on ambiguous targets (|R| > 1).

Per ambiguous test target: p_true = p(true next); L_succ = p(member of R that truly follows the true next);
L_far = p(other members of R); out = p(outside R). Printed: mean ± sd over training seeds per (log, variant).
Collapse line: share of ambiguous targets with p_true < COLLAPSE per model, and the mean Brier change A3 − A1
split into A3-collapsed targets + the rest (paired by training seed, averaged over seeds).
--worst N: per log, the N targets with the largest NLL increase A3 vs A1 (training seed 0).
G1 has no loops, so each activity has one true rank per case (asserted).

  python -m train.leakage runs_v3 g1_batch30_10k_s0 g1_date30_10k_s0 --worst 10
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from train.evaluate import REPO, true_next

PARTS = ("p_true", "L_succ", "L_far", "out")
COLLAPSE = 0.01                    # p_true below this = collapsed target


def load(root: Path, log: str, variant: str, seed: int) -> dict:
    """test_pred.pt and vocabulary of one run, plus the true next activity per target (0-based)."""
    folder = root / log / f"{variant}_s{seed}"
    vocab = json.loads((folder / "config.json").read_text())["vocab"]
    p = torch.load(folder / "test_pred.pt", weights_only=True)
    p["true_next"] = true_next(log, p["case"], p["pos"], torch.tensor(p["type"]) - 1, vocab)
    p["vocab"] = vocab
    return p


def truth(log: str) -> tuple:
    """Per case: activities in file order, and (case, true_rank) -> activity."""
    t = pd.read_csv(REPO / "sim" / "out" / f"{log}_truth.csv")
    assert not t.duplicated(["CaseID", "Activity"]).any(), f"{log}: repeated activity in a case (loops)"
    acts = t.groupby("CaseID", sort=False)["Activity"].apply(list).to_dict()
    by_rank = dict(zip(zip(t["CaseID"], t["true_rank"]), t["Activity"]))
    return acts, by_rank


def successor(p: dict, by_rank: dict) -> torch.Tensor:
    """0-based activity that truly follows the true next in its case (-1 at the case end)."""
    inv = {v - 1: a for a, v in p["vocab"].items()}
    rank = {(c, a): r for (c, r), a in by_rank.items()}
    out = []
    for c, y in zip(p["case"], p["true_next"].tolist()):
        nxt = by_rank.get((c, rank[(c, inv[y])] + 1))
        out.append(p["vocab"][nxt] - 1 if nxt is not None else -1)
    return torch.tensor(out)


def split(p: dict, succ: torch.Tensor) -> tuple:
    """Ambiguous mask and per-target probability parts p_true, L_succ, L_far, out."""
    P, R, y = p["p_act"].double(), p["R"], p["true_next"]
    idx = torch.arange(len(y))
    p_true, in_R = P[idx, y], (P * R).sum(-1)
    s = succ.clamp(min=0)
    s_in = (succ >= 0) & R[idx, s]
    l_succ = torch.where(s_in, P[idx, s], torch.zeros_like(p_true))
    l_far = in_R - torch.where(R[idx, y], p_true, torch.zeros_like(p_true)) - l_succ
    return R.sum(-1) > 1, dict(zip(PARTS, (p_true, l_succ, l_far, 1 - in_R)))


def collapse(log: str, runs: dict, seeds: list) -> None:
    """Print the share of ambiguous targets with p_true < COLLAPSE and their part of the Brier change A1 -> A3."""
    rows = []
    for s in seeds:
        a1, a3 = runs[("A1", s)], runs[("A3", s)]
        y, amb = a1["true_next"], a1["R"].sum(-1) > 1
        idx = torch.arange(len(y))
        onehot = F.one_hot(y, a1["p_act"].size(1)).double()
        d = (((a3["p_act"].double() - onehot) ** 2).sum(-1) - ((a1["p_act"].double() - onehot) ** 2).sum(-1))[amb]
        c3 = (a3["p_act"][idx, y] < COLLAPSE)[amb]
        rows.append(torch.tensor([float((a1["p_act"][idx, y] < COLLAPSE)[amb].double().mean()), float(c3.double().mean()),
                                  float(d.mean()), float((d * c3).sum() / len(d)), float((d * ~c3).sum() / len(d))]))
    m = torch.stack(rows).mean(0)
    print(f"{log:22s} p_true<{COLLAPSE}: A1 {m[0]:.3f}  A3 {m[1]:.3f}   "
          f"Brier A3-A1 {m[2]:+.4f} = A3-collapsed {m[3]:+.4f} + rest {m[4]:+.4f}  (mean over {len(seeds)} seeds)")


def worst(log: str, a1: dict, a3: dict, acts: dict, n: int) -> None:
    """Print the n ambiguous targets with the largest NLL increase from A1 to A3."""
    inv = {v - 1: a for a, v in a1["vocab"].items()}
    idx = torch.arange(len(a1["true_next"]))
    y = a1["true_next"]
    amb = a1["R"].sum(-1) > 1
    d = torch.log(a1["p_act"][idx, y].clamp(min=1e-12)) - torch.log(a3["p_act"][idx, y].clamp(min=1e-12))
    d[~amb] = float("-inf")
    print(f"\n{log}: {n} ambiguous targets with the largest NLL increase A3 vs A1 (seed 0)")
    for i in d.topk(n).indices.tolist():
        c, j = a1["case"][i], a1["pos"][i]
        R = [inv[a] for a in a1["R"][i].nonzero().flatten().tolist()]
        print(f"  {c} prefix={'>'.join(acts[c][:j])}  R={R}  true={inv[int(y[i])]}  "
              f"p_true A1={a1['p_act'][i, y[i]]:.3f} A3={a3['p_act'][i, y[i]]:.3f}  "
              f"A3 top={inv[int(a3['p_act'][i].argmax())]}")


def main() -> None:
    """Probability split per (log, variant) over training seeds; optional worst-target listing."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--worst", type=int, default=0)
    ap.add_argument("--method", default="A3", help="variant compared with A1, shown as A3 (e.g. A3M)")
    a = ap.parse_args()
    print(f"A3 = {a.method}")

    print(f"{'log':22s} var  n_amb  " + "  ".join(f"{k:>15s}" for k in PARTS))
    for log in a.logs:
        acts, by_rank = truth(log)
        runs = {(v, s): load(a.root, log, a.method if v == "A3" else v, s) for v in ("A1", "A3") for s in a.seeds}
        ref = runs[("A1", a.seeds[0])]
        for p in runs.values():
            assert p["case"] == ref["case"] and p["pos"] == ref["pos"], f"{log}: target order differs"
        succ = successor(ref, by_rank)
        for v in ("A1", "A3"):
            means = []
            for s in a.seeds:
                amb, parts = split(runs[(v, s)], succ)
                means.append(torch.stack([parts[k][amb].mean() for k in PARTS]))
            m = torch.stack(means)
            cells = "  ".join(f"{mu:.3f} ± {sd:.3f}".rjust(15) for mu, sd in zip(m.mean(0), m.std(0)))
            print(f"{log:22s} {v}  {int(amb.sum()):5d}  {cells}")
        collapse(log, runs, a.seeds)
        if a.worst:
            worst(log, runs[("A1", a.seeds[0])], runs[("A3", a.seeds[0])], acts, a.worst)


if __name__ == "__main__":
    main()
