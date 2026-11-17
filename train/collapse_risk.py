#!/usr/bin/env python
"""Collapse test without outcome selection: Brier change A3 − A1 on ambiguous targets, split by a train-fixed stratum.

Context = observed activity sequence before a target (as eda/context_ambiguity.py). A context is at risk when
every train target there is ambiguous (|R| > 1) and one activity is in every R: all probability on that
activity then gives zero set loss whatever the truth. A test target is at risk if its context is.
Per log: at-risk share of ambiguous targets, A3 collapsed share (p_true < COLLAPSE) inside / outside the
stratum, and mean Brier change A3 − A1 = at-risk part + rest (paired by training seed, averaged).

  python -m train.collapse_risk runs_v3 g1_batch30_10k_s0 g1_date30_10k_s0
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from train.dataset import check_hashes, load_split, vocabulary
from train.leakage import COLLAPSE, load


def targets(log: str, split: str, vocab: dict) -> list:
    """(context key, R mask, case, position) per target of one split, in test_pred.pt order."""
    d = load_split(log, split, vocab)
    return [(tuple(d["types"][i, :k].tolist()), d["R"][i, k], case, k)
            for i, case in enumerate(d["case"]) for k in d["target_mask"][i].nonzero().flatten().tolist()]


def risky(train: list) -> set:
    """Contexts whose train targets are all ambiguous and share one activity in every R."""
    common, all_amb = {}, {}
    for key, R, _, _ in train:
        common[key] = common.get(key, torch.ones_like(R)) & R
        all_amb[key] = all_amb.get(key, True) and bool(R.sum() > 1)
    return {k for k in common if all_amb[k] and bool(common[k].any())}


def brier(p: dict) -> torch.Tensor:
    """Multi-class Brier score per target against the true next activity."""
    P = p["p_act"].double()
    return ((P - F.one_hot(p["true_next"], P.size(1)).double()) ** 2).sum(-1)


def summary(root: Path, log: str, seeds: list, method: str = "A3") -> str:
    """One result line for a log."""
    check_hashes(log)
    vocab = vocabulary(log)
    keys = risky(targets(log, "train", vocab))
    test = targets(log, "test", vocab)
    rows = []
    for s in seeds:
        a1, a3 = load(root, log, "A1", s), load(root, log, method, s)
        for p in (a1, a3):
            assert p["case"] == [t[2] for t in test] and p["pos"] == [t[3] for t in test], f"{log}: target order differs"
        amb = a1["R"].sum(-1) > 1
        risk = torch.tensor([t[0] in keys for t in test])[amb]
        d = (brier(a3) - brier(a1))[amb]
        col = (a3["p_act"][torch.arange(len(amb)), a3["true_next"]] < COLLAPSE)[amb]
        rows.append(torch.tensor([float(risk.double().mean()),
                                  float(col[risk].double().mean()) if risk.any() else float("nan"),
                                  float(col[~risk].double().mean()),
                                  float(d.mean()), float((d * risk).sum() / len(d)), float((d * ~risk).sum() / len(d))]))
    m = torch.stack(rows).mean(0)
    return (f"{log:22s} at-risk {m[0]:.3f} of ambiguous | A3 p_true<{COLLAPSE}: at-risk {m[1]:.3f} rest {m[2]:.3f} | "
            f"Brier A3-A1 {m[3]:+.4f} = at-risk {m[4]:+.4f} + rest {m[5]:+.4f}  ({len(seeds)} seeds)")


def main() -> None:
    """Print one line per log."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--method", default="A3", help="variant compared with A1, shown as A3 (e.g. A3M)")
    a = ap.parse_args()
    print(f"A3 = {a.method}")
    for log in a.logs:
        print(summary(a.root, log, a.seeds, a.method))


if __name__ == "__main__":
    main()
