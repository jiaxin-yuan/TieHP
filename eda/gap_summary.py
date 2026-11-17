"""Cross-log gap summary: how many inter-event gaps are zero, and at what tick.

The notebook profiles one log at a time. This script answers the question that
needs all of them at once: are zero-second gaps a quirk of one log or a
property of the collection? Every number comes from ``eventlog.py``, so the
notebook and this table cannot disagree.

    python eda/gap_summary.py          # every log in raw_datasets/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import eventlog as el


def summarise(path: Path) -> dict:
    """Gap composition and timestamp granularity for one log."""
    raw = el.read_any(path)
    spec = el.detect_spec(raw)
    log = el.prepare(raw, spec)
    gaps = el.inter_event_times(log, spec)
    comp = el.gap_composition(gaps)
    gran = el.gap_granularity(gaps)
    tick = el.gap_tick(gaps)
    return {
        "log": path.name.split(".")[0],
        "events": len(log),
        "cases": log[spec.case].nunique(),
        "gaps": len(gaps),
        "pct_zero": comp.loc["zero", "pct_of_gaps"],
        "pct_negative": comp.loc["negative", "pct_of_gaps"],
        "gcd_seconds": tick["gcd_seconds"],
        "pct_min_multiples": gran.loc["1 min", "pct_of_nonzero_gaps"],
        "pct_hour_multiples": gran.loc["1 h", "pct_of_nonzero_gaps"],
        "pct_day_multiples": gran.loc["1 day", "pct_of_nonzero_gaps"],
        **el.clock_time_mode(log, spec).to_dict(),
        "pct_local_midnight": el.local_midnight_share(log, spec),
    }


def main() -> None:
    """Print the cross-log table; --csv also writes it, for the paper to cite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=None, help="also write the table here")
    args = parser.parse_args()

    el.run_table(summarise, csv=args.csv, sort_by="pct_zero")


if __name__ == "__main__":
    main()
