#!/usr/bin/env python
"""BPIC15 time table for the paper: gain of A3C per log over each baseline (rows "real" of train/stats.py).

Bold: 95% CI above 0 (A3C better); italic: below 0 (worse); plain: contains 0.
  python -m train.table_real_gains   ->  results_real/table_real_gains.tex
"""
from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parent.parent / "results_real"
LOGS = [f"BPIC15_{i}" for i in range(1, 6)]
NAMES = {"A0": "THP", "SEPLSTM": "SEP-LSTM", "PT": "ProcessTransformer", "SUTRAN": "SuTraN", "ALL0": "All-0"}
BLOCKS = [("crps", "CRPS (days)", ("A0", "SEPLSTM", "PT", "SUTRAN", "ALL0"), 2),
          ("brier_zero", r"Brier of $\hat P(\Delta_i=0)$", ("A0", "ALL0"), 3)]


def cell(r: pd.Series, digits: int) -> str:
    """Signed gain, bold if the CI is above 0, italic if below 0."""
    g = f"{r.gain:+.{digits}f}"
    return rf"$\mathbf{{{g}}}$" if r.ci_lo > 0 else rf"$\mathit{{{g}}}$" if r.ci_hi < 0 else f"${g}$"


def main() -> None:
    """Write the LaTeX tabular body (one row per metric x baseline, one column per log)."""
    s = pd.concat(pd.read_csv(RES / f"stats_{log}.csv").assign(log=log) for log in LOGS)
    s = s[(s.check == "real") & (s.method == "A3C") & (s.subset == "all")]
    out = [r"\begin{tabular}{llrrrrr}\toprule",
           r" & & \multicolumn{5}{c}{BPIC15 log}\\\cmidrule(lr){3-7}",
           r"Metric & Baseline & " + " & ".join(str(i) for i in range(1, 6)) + r"\\\midrule"]
    for k, (metric, label, bases, digits) in enumerate(BLOCKS):
        for j, b in enumerate(bases):
            rows = s[(s.metric == metric) & (s.baseline == b)].set_index("log")
            out.append((label if j == 0 else "") + f" & {NAMES[b]} & "
                       + " & ".join(cell(rows.loc[log], digits) for log in LOGS) + r"\\")
        out.append(r"\midrule" if k < len(BLOCKS) - 1 else r"\bottomrule")
    out.append(r"\end{tabular}")
    (RES / "table_real_gains.tex").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
