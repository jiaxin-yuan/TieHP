"""Turn the prefix files in data/out/ into padded per-case tensors for model/ (option a: each target once).

Each source case is one sequence: its longest prefix in the split holds every event any of its prefixes
needs, and the causal mask makes position k see only events 0..k. target_mask[k] = True for every prefix
k present in the split (k = 2..n-1 in train/val; test keeps only gated prefixes). Activities are 1-based
(0 = PAD), vocabulary from train only, plus one OOV index when val/test hold unseen activities (BPIC15; never on
simulated logs). Times in days. The files' hashes are checked against hashes.json first.
"""
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "data" / "out"
TIME_FMT = "%Y/%m/%d %H:%M:%S.%f"
CAUSE = {"date": 0, "batch": 1, "positive": 2, "none": -1}   # A4 label; none = first event, never a target
SEP = "|"
OOV = "<OOV>"                                                # unseen val/test activity (as SuTraN's preprocessing)


def split_path(log: str, split: str) -> Path:
    """Path of one split file of a log."""
    return OUT_DIR / f"{split}_fold0_variation0_{log}.csv"


def check_hashes(log: str) -> None:
    """Stop if any split file of the log differs from data/out/hashes.json."""
    expected = json.loads((OUT_DIR / "hashes.json").read_text())[log]
    for name, h in expected.items():
        got = hashlib.sha256((OUT_DIR / name).read_bytes()).hexdigest()
        assert got == h, f"{name}: hash mismatch, re-run data/prepare_data.py and data/tie_meta.py"


def vocabulary(log: str) -> dict:
    """Activity name -> 1-based index, from the train split; one last OOV index only if val/test have unseen activities."""
    acts = sorted(pd.read_csv(split_path(log, "train"), usecols=["Activity"])["Activity"].unique())
    vocab = {a: i + 1 for i, a in enumerate(acts)}
    other = set().union(*(pd.read_csv(split_path(log, s), usecols=["Activity"])["Activity"] for s in ("val", "test")))
    if other - set(vocab):                                                  # real logs (BPIC15); simulated logs: none
        vocab[OOV] = len(vocab) + 1
    return vocab


def load_split(log: str, split: str, vocab: dict) -> dict:
    """One split as padded tensors [N, S] (R: [N, S, T]) plus the source case id and target positions; unseen activities -> OOV."""
    df = pd.read_csv(split_path(log, split))
    unseen = set(df["Activity"]) - set(vocab)
    assert not unseen or OOV in vocab, f"{split}: activities not in train: {unseen}"
    df["Activity"] = df["Activity"].where(df["Activity"].isin(vocab.keys()), OOV)
    df["src"] = df["CaseID"].str.rsplit("_p", n=1).str[0]
    df["k"] = df["CaseID"].str.rsplit("_p", n=1).str[1].astype(int)
    df["pos"] = df.groupby("CaseID", sort=False).cumcount()

    targets = df.groupby("src", sort=False)["k"].apply(sorted)               # prefix k -> target at position k
    longest = df[df["k"] == df.groupby("src", sort=False)["k"].transform("max")]
    cases = list(targets.index)
    N, S, T = len(cases), int(longest.groupby("src").size().max()), len(vocab)

    out = {"types": torch.zeros(N, S, dtype=torch.long), "event_time": torch.zeros(N, S),
           "dt": torch.zeros(N, S), "R": torch.zeros(N, S, T, dtype=torch.bool),
           "cause": torch.full((N, S), -1, dtype=torch.long), "target_mask": torch.zeros(N, S, dtype=torch.bool)}
    for i, (src, g) in enumerate(longest.groupby("src", sort=False)):
        assert src == cases[i]
        n = len(g)
        ts = pd.to_datetime(g["Timestamp"], format=TIME_FMT)
        out["types"][i, :n] = torch.tensor(g["Activity"].map(vocab).tolist())
        out["event_time"][i, :n] = torch.tensor(((ts - ts.iloc[0]).dt.total_seconds() / 86400).tolist())
        out["dt"][i, 1:n] = torch.tensor((ts.diff().dt.total_seconds() / 86400).tolist()[1:])
        out["cause"][i, :n] = torch.tensor(g["zero_gap_cause"].map(CAUSE).tolist())
        for j, members in enumerate(g["remaining_block_members"]):
            out["R"][i, j, [vocab.get(a, vocab.get(OOV)) - 1 for a in members.split(SEP)]] = True
        out["target_mask"][i, [int(k) for k in targets[src]]] = True
    out["case"] = cases
    return out


def load_log(log: str) -> tuple:
    """Hash check, then (train, val, test, vocab) of one log."""
    check_hashes(log)
    vocab = vocabulary(log)
    return (*(load_split(log, s, vocab) for s in ("train", "val", "test")), vocab)


if __name__ == "__main__":
    import sys
    tr, va, te, vocab = load_log(sys.argv[1] if len(sys.argv) > 1 else "g1_batch30_s0")
    for name, d in (("train", tr), ("val", va), ("test", te)):
        m = d["target_mask"]
        print(f"{name:5s} cases={len(d['case']):5d} max_len={m.size(1)} targets={int(m.sum()):5d} "
              f"ambiguous={int((d['R'].sum(-1) > 1)[m].sum()):5d} zero dt={int((d['dt'] == 0)[m].sum()):5d}")
    print("vocab", vocab)
