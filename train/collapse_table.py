#!/usr/bin/env python
"""Table a3c: per config (gen seeds 3-5 x init seeds 0-2), at-risk share of ambiguous test prefixes and collapse share (%) of A1 / A3 / A3C.

  python -m train.collapse_table [config ...]   -> results/a3c_10k/collapse_table.txt (stdout)
"""
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent           # repo root, wherever it is cloned
sys.path.insert(0, str(REPO))
from train.collapse_risk import risky, targets
from train.dataset import vocabulary
from train.leakage import COLLAPSE, load

ROOT, VARIANTS = REPO / "runs_a3c_10k", ("A1", "A3", "A3C")

for c in sys.argv[1:] or ("batch30", "batch50", "date30", "date50"):
    risk_sh, col = [], {v: [] for v in VARIANTS}
    for g in (3, 4, 5):
        log = f"g1_{c}_10k_s{g}"
        vocab = vocabulary(log)
        keys, test = risky(targets(log, "train", vocab)), targets(log, "test", vocab)
        for s in (0, 1, 2):
            runs = {v: load(ROOT, log, v, s) for v in VARIANTS}
            amb = runs["A1"]["R"].sum(-1) > 1
            risk = torch.tensor([t[0] in keys for t in test]) & amb
            risk_sh.append(float(risk.sum() / amb.sum()))
            for v, p in runs.items():
                pt = p["p_act"][torch.arange(len(amb)), p["true_next"]]
                col[v].append(float((pt[risk] < COLLAPSE).double().mean()))
    mean = lambda x: 100 * torch.tensor(x).nanmean().item()   # seeds without at-risk prefixes skipped
    print(c, f"at-risk {mean(risk_sh):.2f}", f"runs with at-risk {sum(r > 0 for r in risk_sh)}/9", *(f"{v} {mean(col[v]):.1f}" for v in VARIANTS))
