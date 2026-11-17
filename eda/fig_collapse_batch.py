#!/usr/bin/env python
"""Paper figure (method): how batch coarsening creates an at-risk context and collapses the set likelihood.

(a) all train targets of the context, their remaining sets R and I_c (the case itself is Fig. 2a, tikz/case.tex);
(b) true next activity vs the optimum of the set likelihood (support in I_c).
Reads the train split (data/out) and truth file (sim/out) of one simulated log; pandas only.

    python eda/fig_collapse_batch.py --fig benchmark-results/fig_collapse_batch.pdf
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

import eventlog as el

ORDER = ["Register", "Check_A", "Check_B", "Decide", "Approve", "Reject", "Notify", "Archive"]  # G1 order


def context_targets(log: str, context: list) -> pd.DataFrame:
    """Train targets whose observed prefix equals `context`: case, R, true next, recorded label."""
    df = pd.read_csv(f"data/out/train_fold0_variation0_{log}.csv")
    df["src"] = df["CaseID"].str.rsplit("_p", n=1).str[0]
    df["k"] = df["CaseID"].str.rsplit("_p", n=1).str[1].astype(int)
    longest = df[df["k"] == df.groupby("src")["k"].transform("max")]
    truth = pd.read_csv(f"sim/out/{log}_truth.csv")
    rows, k = [], len(context)
    for src, g in longest.groupby("src", sort=False):
        acts = g["Activity"].tolist()
        if acts[:k] != context or len(acts) <= k:
            continue
        t = truth[truth["CaseID"] == src]
        rank = t["true_rank"].tolist()[k:]
        R = sorted(g["remaining_block_members"].iloc[k].split("|"), key=ORDER.index)
        rows.append({"case": src, "R": tuple(R), "true": t["Activity"].iloc[k + rank.index(min(rank))],
                     "recorded": acts[k]})
    return pd.DataFrame(rows)


def chip(ax, x, y, text, w=0.17, h=0.1, fc="white", ec=el.GRID, lw=0.8):
    """One activity box centred at (x, y) in axes coordinates."""
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.005,rounding_size=0.015",
                                fc=fc, ec=ec, lw=lw, transform=ax.transAxes))
    ax.text(x, y, text, ha="center", va="center", fontsize=6.5, transform=ax.transAxes,
            color="white" if fc == el.ACCENT else el.INK)


def panel_sets(ax, rows: pd.DataFrame, Ic: set):
    """(a) Distinct R sets of the context with counts; I_c filled, the true next outlined."""
    groups = Counter(zip(rows["R"], rows["true"]))
    cols = {"Check_A": 0.15, "Check_B": 0.15, "Decide": 0.45, "Notify": 0.75}
    for j, ((R, z), n) in enumerate(groups.most_common()):
        y = 0.85 - 0.19 * j
        for a in R:
            chip(ax, cols.get(a, 0.75), y, a, w=0.28, h=0.13, fc=el.ACCENT if a in Ic else "white",
                 ec=el.ACCENT_2 if a == z else el.GRID, lw=1.4 if a == z else 0.8)
        ax.text(0.92, y, f"×{n}", va="center", fontsize=6.5, color=el.INK_MUTED, transform=ax.transAxes)
    ax.text(0.45, 0.0, "$I_c$ = {" + ", ".join(sorted(Ic)) + "}", ha="center", fontsize=7, transform=ax.transAxes)
    ax.set_title(f"(a) $R_i$ of all {len(rows)} training prefixes at $c$", fontsize=7, loc="left", pad=14)


def panel_optima(ax, rows: pd.DataFrame, Ic: set, highlight_edits: bool = False):
    """(b) Share per activity with positive share: true next and set-likelihood optimum."""
    series = {"true next": rows["true"].value_counts(normalize=True),
              "set likelihood": pd.Series({a: 1 / len(Ic) for a in Ic})}
    assert not set(series["true next"].index) & Ic, "true next in I_c: no collapse to show"  # bars do not overlap
    acts = sorted(set(series["true next"].index) | Ic, key=ORDER.index)
    for (name, share), c in zip(series.items(), [el.ACCENT_2, el.ACCENT]):
        ax.barh(range(len(acts)), [share.get(a, 0) for a in acts], height=0.6, color=c, label=name)
    ax.set_yticks(range(len(acts)), acts)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.grid(axis="y", visible=False)
    legend = ax.legend(fontsize=6, loc="lower left", bbox_to_anchor=(-0.05, 1.0), ncol=2,
                       frameon=False, handlelength=1, columnspacing=0.8)
    if highlight_edits:
        for label in legend.get_texts():
            label.set_color("red")
    ax.set_title("(b) Share at $c$", fontsize=7, loc="left", pad=14,
                 color="red" if highlight_edits else el.INK)


def main() -> None:
    """Load the context, draw the two panels, save the figure."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", default="g1_batch30_s0")
    ap.add_argument("--context", nargs="+", default=["Register", "Reject"])
    ap.add_argument("--fig", type=Path, required=True)
    ap.add_argument("--highlight-edits", action="store_true", help="show revised figure labels in red")
    a = ap.parse_args()

    rows = context_targets(a.log, a.context)
    Ic = set.intersection(*map(set, rows["R"]))
    print(rows.to_string(index=False), f"\nI_c = {Ic}")
    assert len(Ic) > 0 and all(len(R) > 1 for R in rows["R"]), "context is not at risk"

    el.use_style()
    plt.rcParams.update({"font.size": 7, "xtick.labelsize": 6, "ytick.labelsize": 6.5})
    fig = plt.figure(figsize=(4.2, 1.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.55)
    ax_b, ax_c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_b.axis("off")
    panel_sets(ax_b, rows, Ic)
    panel_optima(ax_c, rows, Ic, highlight_edits=a.highlight_edits)
    a.fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.fig, bbox_inches="tight", facecolor="white")
    print(f"wrote {a.fig}")


if __name__ == "__main__":
    main()
