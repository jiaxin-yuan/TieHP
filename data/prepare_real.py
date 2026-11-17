"""Prepare the real BPIC15_{1..5} logs: XES -> CSV, then the same split (prepare_data.py) and tie metadata (tie_meta.py).

Timestamps are kept as local wall clock (offset dropped, then read as UTC), so local midnight stays midnight
for zero_gap_cause; gaps across a DST switch are off by +-1 h.
Writes data/out/real/{log}.csv (full cases), the four prefix files per log in data/out/,
and merges the logs' hashes into data/out/hashes.json (simulated logs untouched).
"""
import json
import xml.etree.ElementTree as ET

import pandas as pd

import prepare_data
import tie_meta

RAW_DIR = prepare_data.REPO / "raw_datasets"
CSV_DIR = prepare_data.OUTPUT_DIR / "real"
LOGS = [f"BPIC15_{i}" for i in range(1, 6)]


def xes_to_csv(log: str) -> None:
    """Write one XES log as CaseID, Activity, Timestamp (wall clock) in file order, exact duplicates dropped."""
    rows, events, case = [], [], None
    for _, el in ET.iterparse(RAW_DIR / f"{log}.xes", events=("end",)):
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "event":
            a = {c.get("key"): c.get("value") for c in el}
            events.append((a["concept:name"], a["time:timestamp"][:19]))   # drop "+hh:mm"
            el.clear()
        elif tag == "trace":
            case = next(c.get("value") for c in el if c.get("key") == "concept:name")
            rows += [(case, act, ts) for act, ts in events]
            events = []
            el.clear()
    df = pd.DataFrame(rows, columns=["CaseID", "Activity", "Timestamp"]).drop_duplicates()
    df.to_csv(CSV_DIR / f"{log}.csv", index=False)
    print(f"{log}: {df['CaseID'].nunique()} cases, {len(df)} events ({len(rows) - len(df)} duplicates dropped)")


def main():
    """Convert, split and annotate every BPIC15 log, then merge its hashes."""
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    prepare_data.INPUT_DIR = CSV_DIR          # both scripts read full cases from here
    tie_meta.SIM_DIR = CSV_DIR
    hashes_path = tie_meta.OUT_DIR / "hashes.json"
    hashes = json.loads(hashes_path.read_text())

    for log in LOGS:
        xes_to_csv(log)
        prepare_data.prepare_dataset(log, log)
        events = tie_meta.case_events(log)
        events["CaseID"] = events["CaseID"].astype(str)   # numeric ids; prefix ids "<id>_p{k}" are strings
        hashes[log] = {}
        for split in tie_meta.SPLITS:
            path = tie_meta.OUT_DIR / f"{split}_fold0_variation0_{log}.csv"
            tie_meta.add_meta(pd.read_csv(path), events, path.name).to_csv(path, index=False)
            hashes[log][path.name] = tie_meta.sha256(path)
        te = pd.read_csv(tie_meta.OUT_DIR / f"test_fold0_variation0_{log}.csv")
        tgt = te[te.groupby("CaseID", sort=False).cumcount(ascending=False) == 0]
        print(f"    test targets={len(tgt)}  ambiguous={int(tgt['ambiguous'].sum())}  "
              f"causes={tgt['zero_gap_cause'].value_counts().to_dict()}\n")

    hashes_path.write_text(json.dumps(hashes, indent=1))
    print(f"merged {len(LOGS)} logs into {hashes_path.relative_to(prepare_data.REPO)}")


if __name__ == "__main__":
    main()
