"""Cross-log zero-gap causes: is a zero gap lost date resolution, or simultaneity?

``gap_summary.py`` counts how many inter-event gaps are zero. It cannot say why.
Two mechanisms produce identical zeros: an event recorded as a bare date (stored
as local midnight, so a whole day collapses onto one timestamp), and two events
genuinely written at the same instant. Which one it is decides whether the zero
mass means the same thing in different logs.

    python eda/zero_gap_causes.py     # every log in raw_datasets/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import eventlog as el


def summarise(path: Path) -> dict:
    """Zero-gap count and its cause split for one log."""
    raw = el.read_any(path)
    spec = el.detect_spec(raw)
    log = el.prepare(raw, spec)
    gaps = el.inter_event_times(log, spec)
    comp = el.gap_composition(gaps)
    causes = el.zero_gap_causes(log, spec)
    return {
        "log": path.name.split(".")[0],
        "gaps": len(gaps),
        "zero_gaps": int(causes["zero_gaps"].sum()),
        "pct_zero": comp.loc["zero", "pct_of_gaps"],
        "pct_date_only": causes.loc["date_only", "pct_of_zero_gaps"],
        "pct_same_instant": causes.loc["same_instant", "pct_of_zero_gaps"],
        "pct_local_midnight": el.local_midnight_share(log, spec),
    }


def main() -> None:
    """Print the cause split per log; --csv writes the table, --fig the stacked bar."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=None, help="also write the table here")
    parser.add_argument("--fig", type=Path, default=None, help="also write the stacked bar here")
    args = parser.parse_args()

    table = el.run_table(summarise, csv=args.csv, sort_by="pct_zero")
    if args.fig and not table.empty:
        el.use_style()
        ax = el.plot_zero_gap_causes(table)
        args.fig.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(args.fig)
        print(f"wrote {args.fig}")


if __name__ == "__main__":
    main()
