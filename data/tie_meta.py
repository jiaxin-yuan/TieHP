"""Add tie metadata columns to the prefix files in data/out/ and write their hashes.

Columns per event (position j of its case):
  block_id                 run of consecutive equal timestamps in the case (0, 1, ...)
  remaining_block_members  activities of that block from position j on, joined by "|" (R on a target row)
  ambiguous                more than one remaining member (|R| > 1)
  zero_gap_cause           gap before the event: positive / date (0 and midnight UTC) / batch (0 otherwise) / none (first event)
Blocks come from the full case in sim/out/, since a block can continue past the target of a prefix.
Hashes (SHA-256 per file) go to data/out/hashes.json; every method checks them before training.
"""
import hashlib
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SIM_DIR = REPO / "sim" / "out"
OUT_DIR = REPO / "data" / "out"
SPLITS = ("train", "val", "test", "full_train")
META = ["block_id", "remaining_block_members", "ambiguous", "zero_gap_cause"]
SEP = "|"


def case_events(log: str) -> pd.DataFrame:
    """Full cases of one sim/out log in prepare_data.py's order, with metadata per event; asserts file order inside ties."""
    df = pd.read_csv(SIM_DIR / f"{log}.csv")
    df["row"] = range(len(df))
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="mixed", utc=True)
    df = df.sort_values(["CaseID", "Timestamp"]).reset_index(drop=True)   # same sort as prepare_data.py
    assert (df.groupby(["CaseID", "Timestamp"])["row"].diff().dropna() > 0).all(), \
        f"{log}: sort changed the file order inside a tie block"

    df["pos"] = df.groupby("CaseID").cumcount()
    gap = df.groupby("CaseID")["Timestamp"].diff()
    df["block_id"] = (gap != pd.Timedelta(0)).astype(int).groupby(df["CaseID"]).cumsum() - 1

    rem = []                                                                  # members from j to the block end
    for _, acts in df.groupby(["CaseID", "block_id"], sort=False)["Activity"]:
        a = list(acts)
        rem += [SEP.join(a[i:]) for i in range(len(a))]
    df["remaining_block_members"] = rem
    df["ambiguous"] = df["remaining_block_members"].str.contains(SEP, regex=False)

    midnight = df["Timestamp"] == df["Timestamp"].dt.floor("D")              # simulator truncates to UTC days
    df["zero_gap_cause"] = "positive"
    df.loc[gap == pd.Timedelta(0), "zero_gap_cause"] = "batch"
    df.loc[(gap == pd.Timedelta(0)) & midnight, "zero_gap_cause"] = "date"
    df.loc[gap.isna(), "zero_gap_cause"] = "none"
    return df


def add_meta(prefixes: pd.DataFrame, events: pd.DataFrame, name: str) -> pd.DataFrame:
    """Join the event metadata onto the prefix rows by (source case, position); checks each prefix matches its case."""
    p = prefixes.drop(columns=[c for c in META if c in prefixes.columns])     # re-run safe
    p["src"] = p["CaseID"].str.rsplit("_p", n=1).str[0]
    p["pos"] = p.groupby("CaseID", sort=False).cumcount()
    m = p.merge(events[["CaseID", "pos", "Activity", "Timestamp"] + META]
                .rename(columns={"CaseID": "src", "Activity": "src_act", "Timestamp": "src_ts"}),
                on=["src", "pos"], how="left", validate="many_to_one")
    assert m["src_act"].notna().all(), f"{name}: prefix row without a source event"
    assert (m["Activity"] == m["src_act"]).all(), f"{name}: prefix activity differs from its case"
    ts = pd.to_datetime(m["Timestamp"], format="%Y/%m/%d %H:%M:%S.%f", utc=True)
    assert (ts == m["src_ts"].dt.floor("us")).all(), f"{name}: prefix timestamp differs from its case"

    last = m.groupby("CaseID", sort=False)["pos"].transform("max") == m["pos"]
    tgt = m[last]
    assert all(a in r.split(SEP) for a, r in zip(tgt["Activity"], tgt["remaining_block_members"])), \
        f"{name}: a target is not in its R"
    return m[list(prefixes.columns.drop(META, errors="ignore")) + META]


def sha256(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    """Add metadata to every log's four split files, then write data/out/hashes.json."""
    logs = sorted(p.name.removeprefix("train_fold0_variation0_").removesuffix(".csv")
                  for p in OUT_DIR.glob("train_fold0_variation0_*.csv"))
    hashes = {}
    for log in logs:
        events = case_events(log)
        hashes[log] = {}
        for split in SPLITS:
            path = OUT_DIR / f"{split}_fold0_variation0_{log}.csv"
            out = add_meta(pd.read_csv(path), events, path.name)
            out.to_csv(path, index=False)
            hashes[log][path.name] = sha256(path)
        tr = pd.read_csv(OUT_DIR / f"train_fold0_variation0_{log}.csv")
        tgt = tr[tr.groupby("CaseID", sort=False).cumcount(ascending=False) == 0]
        print(f"{log:15s} train targets={len(tgt):5d}  ambiguous={int(tgt['ambiguous'].sum()):5d}  "
              f"causes={tgt['zero_gap_cause'].value_counts().to_dict()}")
    (OUT_DIR / "hashes.json").write_text(json.dumps(hashes, indent=1))
    print(f"wrote {(OUT_DIR / 'hashes.json').relative_to(REPO)}")


if __name__ == "__main__":
    main()
