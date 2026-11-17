#!/usr/bin/env python
"""Paper figure (intro): date-only vs batch share of zero gaps, for logs with ties.

Reads the tables written by `make zero-causes` and `make tie-blocks`; it does not
re-read raw logs. Logs below --min-zero % zero gaps are dropped (this also drops
the BPIC20 alias duplicates, which all have < 1.5% zero gaps).

    python eda/fig_intro_zero_gaps.py --fig benchmark-results/fig_intro_zero_gaps.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eventlog as el


def main() -> None:
    """Merge the two tables, keep logs with ties, draw the paper-sized stacked bar."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causes", type=Path, default=Path("benchmark-results/t1_zero_gap_causes.csv"))
    parser.add_argument("--ties", type=Path, default=Path("benchmark-results/t5_tie_blocks.csv"))
    parser.add_argument("--min-zero", type=float, default=2.0, help="min %% zero gaps to show a log")
    parser.add_argument("--fig", type=Path, required=True)
    parser.add_argument("--highlight-edits", action="store_true", help="show revised legend labels in red")
    args = parser.parse_args()

    causes = pd.read_csv(args.causes)
    ties = pd.read_csv(args.ties)[["log", "pct_events_tied"]]
    data = causes.merge(ties, on="log").query("pct_zero >= @args.min_zero")
    data = data.sort_values("pct_zero").set_index("log")  # barh: largest on top
    print(data[["pct_zero", "pct_date_only", "pct_same_instant", "pct_events_tied"]])

    el.use_style()
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8})
    fig, ax = plt.subplots(figsize=(3.7, 2.2))
    date_only = data["pct_date_only"].to_numpy()
    batch = data["pct_same_instant"].to_numpy()
    gap = np.where((date_only >= 1) & (batch >= 1), 1.5, 0.0)  # boundary only between visible segments
    ax.barh(data.index, date_only, color=el.ACCENT, height=0.65, label="Date-only")
    ax.barh(data.index, batch - gap, left=date_only + gap, color=el.ACCENT_2, height=0.65, label="Batch")
    for y, row in enumerate(data.itertuples()):
        ax.text(102, y, f"{row.pct_zero:.0f}% zero · {row.pct_events_tied:.0f}% tied",
                va="center", fontsize=7, color=el.INK_MUTED)

    ax.set_xlabel("% of the log's zero gaps")
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", visible=False)
    legend = ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, borderaxespad=0.2)
    for label in legend.get_texts():
        label.set_color("red" if args.highlight_edits else el.INK_MUTED)
    args.fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.fig}")


if __name__ == "__main__":
    main()
