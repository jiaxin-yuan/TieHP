#!/usr/bin/env python
"""Test metrics of A0-A3, A3-ε, H-A1 / H-A3 and A3C on the G1 logs: mean +/- std per config over runs
(generation × training seeds; config = log name without _s<seed>, as in train/stats.py).

Reads results/<log>/metrics.csv (written by train/evaluate.py), prints the
aggregated table (incl. the number of seeds per cell) and draws only the panels in PANELS:
the (metric, subset) pairs of the plan_exp.md claim table (2.1, 2.2) and the report-only time panels,
each titled with its check; x = log, one dot per variant with a std bar; H-A1 / H-A3 in time panels only.
A note box under the legend lists what is not computed or unclear (NOTES) and the share of targets
whose true next activity is outside R (out_R, from metrics.csv).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

LOGS = ["g1_none_s0", "g1_batch30_s0", "g1_batch50_s0", "g1_date30_s0", "g1_date50_s0"]  # expected drop order
VARIANTS = ["A0", "A1", "A2", "A3", "A3eps", "HA1", "HA3", "A3J", "A3M", "A3B", "A3C"]
LABELS = {  # plan.md §2 C2 variant table: role | time head, time loss | activity loss
    "A0": "A0  THP baseline (original THP) | point Δt̂, THP intensity NLL + squared error | CE on the logged label",
    "A1": "A1  ablation (THP-M without LSTM) | zero atom π₀ + LogNormal mixture, zero-atom NLL | CE on the logged label",
    "A2": "A2  ablation | point Δt̂, THP intensity NLL + squared error | set likelihood over R",
    "A3": "A3  ablation (no at-risk CE; collapses on batch) | zero atom π₀ + LogNormal mixture, zero-atom NLL | set likelihood over R",
    "A3eps": "A3-ε  ablation (isolates the zero atom) | LogNormal mixture, NLL on max(Δt, ε); P(Δt=0) := P(Δt ≤ ε) "
             "| set likelihood over R",
    "HA1": "H-A1  hurdle baseline of A1 (2.1) | π̄₀ = train zero share; A1's mixture trained on Δt > 0 targets only "
           "| as A1, Δt > 0 targets only",
    "HA3": "H-A3  hurdle baseline of A3 (2.1) | π̄₀ = train zero share; A3's mixture trained on Δt > 0 targets only "
           "| as A3, Δt > 0 targets only",
    "A3J": "A3J  exploratory joint variant | activity-conditioned zero atom + LogNormal mixture "
           "| joint set likelihood; time scores marginalize all activities",
    "A3M": "A3M  failed 2.10 (A3 + tie-block design b′) | as A3 | set likelihood over R on Δt > 0 targets only",
    "A3B": "A3B  failed 2.10 (A3 + tie-block design b) | as A3 | set likelihood over R from the state before "
           "the tie block; test: π₀-mixture of in-block and new-block predictions",
    "A3C": "A3C  ours (A3 + CE at at-risk contexts) | as A3 | CE on the logged label at "
           "at-risk contexts (train-fixed), set likelihood over R elsewhere",
}
SETTING = ("G1 simulated logs, {scale} cases, generation seed(s) {gens} (x axis: zero-gap mechanism + share of gaps); "
           "same THP encoder, no LSTM; mean ± std over {n} runs (generation × training seeds);\ntest targets: *_true vs the true next "
           "activity (truth file); time metrics vs the observed Δt; {comparison}")
NOTES = [  # plan_exp.md §2 parts not computed or not understood; shown in the figure
    "Not computed: true P(Δt=0) per prefix (no closed form under M-date; needs per-prefix re-simulation), so "
    "brier_zero and crps are scored against the observed Δt, not the true P(Δt=0).",
    "Partly unclear: share of targets whose true next activity is outside R: {out_r}. Cause (checked): a "
    "day-truncated event sorted before its same-day timestamped predecessor; no variant can recover these. "
    "Why the share is higher at date30 than date50 is unchecked.",
]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#7a5cd6", "#6b6b6b", "#168b8b", "#b0413e", "#8c6d1f", "#d62728"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "p", "d"]
SUBSETS = {"all": "all", "ambiguous": "ambiguous (|R| > 1)", "unambiguous": "unambiguous (|R| = 1)",
           "zero_gap": "zero-gap (Δt = 0)", "positive_gap": "positive-gap (Δt > 0)"}
DIRECTION = {"acc_true": "higher", "macro_f1_true": "higher", "nll_true": "lower", "brier_true": "lower",
             "mae_pos": "lower", "crps": "lower", "brier_zero": "lower"}
PANELS = [  # (metric, subset, role): plan_exp.md claim table rows, then report-only time panels; nothing else drawn
    ("brier_zero", "all", "2.1 part 1"), ("crps", "all", "2.1 part 2"),
    ("brier_true", "ambiguous", "2.2"), ("brier_true", "unambiguous", "2.2"),
    ("crps", "zero_gap", "report-only"), ("crps", "positive_gap", "report-only"),
    ("mae_pos", "positive_gap", "report-only"),
    ("acc_true", "ambiguous", "report-only"), ("macro_f1_true", "ambiguous", "report-only"),
    ("nll_true", "ambiguous", "report-only"),
    ("acc_true", "unambiguous", "report-only"), ("macro_f1_true", "unambiguous", "report-only"),
]
PAPER_PANELS = [  # --paper: (metric, subset, panel title), drawn 2 × 2
    ("brier_zero", "all", r"(a) Time: Brier of $\hat{P}(\Delta_i=0)$, all prefixes"), ("crps", "all", "(b) Time: CRPS (days), all prefixes"),
    ("brier_true", "ambiguous", "(c) Activity: Brier, ambiguous prefixes"),
    ("brier_true", "unambiguous", "(d) Activity: Brier, unambiguous prefixes"),
]
PAPER_LABELS = {"A0": "THP", "A1": "THP-M (no LSTM)", "A2": "A2", "A3": "THP-M+Set",  # paper names (Sect. sim Variants)
                "A3eps": "A3-ε", "HA1": "H-A1 (hurdle)", "HA3": "Hurdle+Set", "A3J": "A3J", "A3M": "A3M",
                "A3B": "A3B", "A3C": "TieHP (ours)"}
TIME = {"brier_zero", "crps", "mae_pos"}   # H-A1 / H-A3 are time baselines only (2.1)
NCOLS = 6
MECH_ORDER = ["none", "batch", "date"]   # x order: none, then each mechanism by share


def config_key(c: str) -> tuple:
    """Sort key of a config: mechanism in MECH_ORDER, then share."""
    mech, share = re.match(r"g1_([a-z]+)(\d*)", c).groups()
    return MECH_ORDER.index(mech), int(share or 0)


def label(c: str) -> str:
    """Tick / note label of a config: no g1_ prefix, no scale tag (the scale is in the subplot titles)."""
    return re.sub(r"_\d+k$", "", c.removeprefix("g1_"))


def load(root: Path, logs: list[str]) -> pd.DataFrame:
    """Concatenate metrics.csv of the given logs."""
    return pd.concat([pd.read_csv(root / log / "metrics.csv") for log in logs], ignore_index=True)


def aggregate(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Mean, std and seed count per (metric, subset, log, variant); missing values dropped."""
    long = df.melt(id_vars=["log", "variant", "seed", "subset"], value_vars=[m for m in metrics if m in df],
                   var_name="metric").dropna(subset=["value"])
    out = (long.groupby(["metric", "subset", "log", "variant"])["value"]
           .agg(mean="mean", std="std", n_seeds="count").reset_index())
    return out.round({"mean": 4, "std": 4})


def out_r_text(df: pd.DataFrame, logs: list[str]) -> str:
    """Share of all targets with the true next activity outside R, per log (from out_R; 'n/a' before the rerun)."""
    if "out_R" not in df:
        return "n/a (rerun train/evaluate.py)"
    s = df[df["subset"] == "all"].groupby("log")["out_R"].mean().reindex(logs)
    return ", ".join(f"{label(log)} {v:.0%}" for log, v in s.items())


def plot(agg: pd.DataFrame, logs: list[str], path: Path, note: str, comparison: str = "claims", panels=PANELS, **title) -> None:
    """One panel per panels entry, titled with its check; variants offset around each log's x position; notes below."""
    ncols = min(NCOLS, len(panels))
    nrows = -(-len(panels) // ncols)
    foot = 2.6                                   # inches below the panels for legend + notes
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.4 * nrows + foot), squeeze=False)
    handles = {}
    for ax, (metric, subset, role) in zip(axes.flat, panels):
        if comparison == "joint" and role != "report-only":
            role = "joint"
        part = agg[(agg["metric"] == metric) & (agg["subset"] == subset)]
        variants = [v for v in VARIANTS if v in set(part["variant"]) and (metric in TIME or not v.startswith("HA"))]
        for k, v in enumerate(variants):
            col, m = COLORS[VARIANTS.index(v)], MARKERS[VARIANTS.index(v)]   # colour follows the variant
            rows = part[part["variant"] == v].set_index("log").reindex(logs)
            x = [i + (k - (len(variants) - 1) / 2) * 0.12 for i in range(len(logs))]
            handles[v] = ax.errorbar(x, rows["mean"], yerr=rows["std"], fmt=m, color=col, ms=6,
                                     capsize=3, lw=1.5, label=LABELS[v])
        ax.set_xticks(range(len(logs)), [label(log) for log in logs])
        ax.set_ylabel(f"{metric} ({DIRECTION.get(metric, '?')} is better)")
        ax.set_title(f"{role}: {SUBSETS[subset]} targets, {title['scale']} cases", color="0.45" if role == "report-only" else "black")
        ax.grid(axis="y", color="0.9")
        ax.set_axisbelow(True)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    lo, hi = int(agg["n_seeds"].min()), int(agg["n_seeds"].max())
    order = [v for v in VARIANTS if v in handles]
    fig.legend([handles[v] for v in order], [LABELS[v] for v in order], loc="lower center",
               bbox_to_anchor=(0.5, 0.6 / fig.get_figheight()), ncol=1, fontsize=9, frameon=False)
    fig.text(0.01, 0.003, note, fontsize=8.5, va="bottom", ha="left", wrap=True,
             bbox=dict(boxstyle="round", fc="#fff6e5", ec="#eda100"))
    context = "exploratory A3J vs A3" if comparison == "joint" else "panel title = plan_exp.md check or report-only"
    fig.suptitle(SETTING.format(n=hi if lo == hi else f"{lo}–{hi}", comparison=context, **title), fontsize=10)
    fig.tight_layout(rect=(0, foot / fig.get_figheight(), 1, 0.96))   # room for legend + notes and the title
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_paper(agg: pd.DataFrame, logs: list[str], path: Path) -> None:
    """Paper figure: PAPER_PANELS in 2 × 2 at LNCS text width, one-row legend, no setting title or notes."""
    plt.rcParams.update({"font.size": 8})
    fig, axes = plt.subplots(2, 2, figsize=(6.0, 4.6))
    handles = {}
    for ax, (metric, subset, title) in zip(axes.flat, PAPER_PANELS):
        part = agg[(agg["metric"] == metric) & (agg["subset"] == subset)]
        variants = [v for v in VARIANTS if v in set(part["variant"]) and (metric in TIME or not v.startswith("HA"))]
        for k, v in enumerate(variants):
            rows = part[part["variant"] == v].set_index("log").reindex(logs)
            x = [i + (k - (len(variants) - 1) / 2) * 0.14 for i in range(len(logs))]
            handles[v] = ax.errorbar(x, rows["mean"], yerr=rows["std"], fmt=MARKERS[VARIANTS.index(v)],
                                     color=COLORS[VARIANTS.index(v)], ms=4, capsize=2, lw=1)
        ax.set_xticks(range(len(logs)), [label(log) for log in logs], rotation=35, ha="right")
        ax.set_title(title, fontsize=8.5)
        ax.grid(axis="y", color="0.9")
        ax.set_axisbelow(True)
    order = [v for v in VARIANTS if v in handles]
    fig.legend([handles[v] for v in order], [PAPER_LABELS[v] for v in order], loc="lower center",
               ncol=len(order), fontsize=7.5, frameon=False, handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    """Load, print the aggregated table, write the figure."""
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("results"))
    p.add_argument("--logs", nargs="+", default=LOGS)
    p.add_argument("--fig", type=Path, default=Path("results/fig_g1_metrics.png"))
    p.add_argument("--comparison", choices=("claims", "joint"), default="claims")
    p.add_argument("--variants", nargs="+", choices=VARIANTS, help="draw only these variants (default: all present)")
    p.add_argument("--claims", action="store_true", help="draw only the claim panels (no report-only panels)")
    p.add_argument("--paper", action="store_true", help="paper figure: claim panels 2 × 2, short legend, no notes")
    a = p.parse_args()

    df = load(a.root, a.logs)
    if a.comparison == "joint":
        df = df[df["variant"].isin(("A3", "A3J"))]
        if set(df["variant"]) != {"A3", "A3J"}:
            p.error("joint comparison needs both A3 and A3J results")
    if a.variants:
        df = df[df["variant"].isin(a.variants)]
    df["log"] = df["log"].str.replace(r"_s\d+$", "", regex=True)          # pool generation seeds per config
    configs = sorted({re.sub(r"_s\d+$", "", log) for log in a.logs}, key=config_key)
    scales = {m.group(1) if (m := re.search(r"_(\d+k)$", c)) else "2k" for c in configs}
    gens = ", ".join(sorted({log.rsplit("_s", 1)[1] for log in a.logs}, key=int))
    agg = aggregate(df, list(DIRECTION))
    print(agg.to_string(index=False))
    note = "\n".join(n.format(out_r=out_r_text(df, configs)) for n in NOTES)
    if a.comparison != "joint" and "A3eps" not in set(agg["variant"]):
        note += "\nNot in this grid: A3-ε (variant A3eps; §3 ablation)."
    print(note)
    if a.paper:
        plot_paper(agg, configs, a.fig)
        print(f"wrote {a.fig}")
        return
    panels =[p_ for p_ in PANELS if p_[2] != "report-only"] if a.claims else PANELS
    plot(agg, configs, a.fig, note, comparison=a.comparison, panels=panels, scale="/".join(sorted(scales)), gens=gens)
    print(f"wrote {a.fig}")


if __name__ == "__main__":
    main()
