#!/usr/bin/env python
"""Cross-log identifiability of tied timestamps (S4a).

A *tie block* is a maximal run of consecutive events inside one case that share a
timestamp. Within a block the log records no order, so the question this script
answers is: can the order be recovered from any other recorded field, and if not,
how much of the log sits in an unresolvable block?

A block is *unresolvable* when it holds at least two distinct activities and its
`lifecycle:transition` values are not known-and-distinct. Both halves matter:
permuting a block of identical activities is a no-op, and `lifecycle:transition`
is the one field carrying a semantic order (start before complete). Lifecycle is
credited generously -- distinct ranks count as resolving, even though lifecycle
does not really order two different activities -- so the unresolvable share is a
lower bound. `pct_events_unresolvable` counts all m events of such a block, not
the m-1 zero gaps it produces, because a permutation moves every member.

Re-run `make tie-blocks` instead of quoting a stale table or figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import eventlog as el


def summarise(path: Path) -> dict:
    """Count tie blocks in one log and split them into resolvable and unresolvable."""
    df = el.read_any(path)
    spec = el.detect_spec(df)
    df = el.prepare(df, spec)

    df = df.assign(_block=el.block_ids(df, spec))
    sizes = df.groupby("_block").size()
    tie_ids = sizes[sizes > 1].index
    tied = df[df["_block"].isin(tie_ids)]

    n_events = len(df)
    n_blocks = len(tie_ids)
    row = {
        "log": path.stem,
        "events": n_events,
        # A block of size m contributes m-1 zero gaps; cross-checks gap_composition.
        "zero_gaps": int(sizes[sizes > 1].sub(1).sum()),
        "blocks": n_blocks,
        "max_block": int(sizes.max()) if n_blocks else 0,
        "pct_events_tied": round(100 * len(tied) / n_events, 2),
    }

    if n_blocks == 0:
        row.update(
            blocks_multi_act=0,
            pct_blocks_multi_act=0.0,
            resolvable_lifecycle=0,
            blocks_unresolvable=0,
            pct_events_unresolvable=0.0,
        )
        return row

    # Ambiguity needs two distinct activities; a uniform block permutes to itself.
    distinct_act = tied.groupby("_block")[spec.activity].nunique()
    multi = distinct_act[distinct_act > 1].index
    multi_frames = tied[tied["_block"].isin(multi)]

    resolvable = 0
    unresolvable_ids = []
    for block_id, frame in multi_frames.groupby("_block"):
        if el.lifecycle_resolves(frame):
            resolvable += 1
        else:
            unresolvable_ids.append(block_id)

    # Every member of an unresolvable block moves under a permutation, so all m
    # events count, not the m-1 zero gaps the block contributes.
    unresolvable_events = int(sizes.loc[unresolvable_ids].sum()) if unresolvable_ids else 0
    row.update(
        blocks_multi_act=len(multi),
        pct_blocks_multi_act=round(100 * len(multi) / n_blocks, 2),
        resolvable_lifecycle=resolvable,
        blocks_unresolvable=len(unresolvable_ids),
        pct_events_unresolvable=round(100 * unresolvable_events / n_events, 2),
    )
    return row


def main() -> None:
    """Print one row per log; --csv also writes the table, --fig the tied-share bar."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=None, help="also write the table here")
    parser.add_argument("--fig", type=Path, default=None, help="also write the tied-share bar here")
    args = parser.parse_args()

    table = el.run_table(summarise, csv=args.csv, index="log")
    if args.fig and not table.empty:
        el.use_style()
        ax = el.plot_tie_blocks(table)
        args.fig.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(args.fig)
        print(f"wrote {args.fig}")


if __name__ == "__main__":
    main()
