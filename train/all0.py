"""All-0 baseline: Δt̂ = 0 and P(Δt=0) = 1 for every test target; activity = train-target frequency p(a).

  python -m train.all0 --out runs_real BPIC15_1 BPIC15_2
p(a) = (count of a among train targets + 1) / (N + T) (add-one, so no activity gets p = 0 and NLL stays finite).
Writes <out>/<log>/ALL0_s0/{config.json, test_pred.pt} in train.py's format (targets in predict() order), so
train/evaluate.py and train/stats.py score it like any run. Deterministic: one seed.
"""
import argparse
import json
from pathlib import Path

import torch

from train.dataset import OUT_DIR, load_log


def target_rows(test: dict) -> dict:
    """Labels per test target (case, pos, type, dt, R, cause) in train.py predict() order; shared by all baselines."""
    rows = {k: [] for k in ("case", "pos", "type", "dt", "R", "cause")}
    for i, case in enumerate(test["case"]):
        for j in test["target_mask"][i].nonzero().flatten().tolist():
            rows["case"].append(case)
            rows["pos"].append(j)
            rows["type"].append(int(test["types"][i, j]))
            rows["dt"].append(float(test["dt"][i, j]))
            rows["R"].append(test["R"][i, j])
            rows["cause"].append(int(test["cause"][i, j]))
    rows["R"] = torch.stack(rows["R"])
    return rows


def write_run(log: str, out: Path) -> None:
    """Build and save the all-0 predictions for one log."""
    train, _, test, vocab = load_log(log)                                 # hash check, train vocab
    T = len(vocab)
    counts = torch.bincount(train["types"][train["target_mask"]] - 1, minlength=T).double()
    p_act = ((counts + 1) / (counts.sum() + T)).float()

    rows = target_rows(test)
    n = len(rows["case"])
    rows["p_act"] = p_act.expand(n, T).clone()
    rows["time"] = torch.zeros(n, 1)                                      # time[:, 0] = Δt̂ = 0

    folder = out / log / "ALL0_s0"
    folder.mkdir(parents=True, exist_ok=True)
    cfg = {"variant": "ALL0", "log": log, "seed": "0", "n_mix": 0, "vocab": vocab,
           "split_hashes": json.loads((OUT_DIR / "hashes.json").read_text())[log]}
    (folder / "config.json").write_text(json.dumps(cfg, indent=1))
    torch.save(rows, folder / "test_pred.pt")
    print(f"{log}: {n} test targets, {T} activities -> {folder}")


def main():
    """Write the all-0 run of each log."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--out", type=Path, default=Path("runs_real"))
    args = parser.parse_args()
    for log in args.logs:
        write_run(log, args.out)


if __name__ == "__main__":
    main()
