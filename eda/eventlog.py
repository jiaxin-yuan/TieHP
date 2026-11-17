"""Loading + distribution helpers for the raw event logs in ``raw_datasets/``.

The notebooks are deliberately thin: everything reusable lives here so the two
of us can read and review the logic in one place, outside of notebook JSON.

Vocabulary used throughout:

* **case**    one trace / process instance (a single ``case_id``).
* **event**   one row of the log: a case did an activity at a timestamp.
* **case length**       number of events in a case.
* **inter-event time**  gap between two consecutive events *within the same
  case*, in seconds. Gaps across case boundaries are never computed.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "raw_datasets"
LINKS_FILE = RAW_DIR / "links.txt"
FIGURE_DIR = PROJECT_ROOT / "eda" / "figures"

TABULAR_SUFFIXES = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json", ".parquet"}
XES_SUFFIXES = {".xes", ".gz", ".zip"}


# --------------------------------------------------------------------------
# Plot style — one accent hue, recessive grid, no chartjunk.
# Colors are the validated reference palette (slots 1-3 only, which clear the
# all-pairs colorblind gate); single-series charts use ACCENT alone.
# --------------------------------------------------------------------------

ACCENT = "#2a78d6"          # blue, categorical slot 1 / sequential base
ACCENT_2 = "#eb6834"        # orange, slot 2
ACCENT_3 = "#1baf7a"        # aqua, slot 3
SERIES = (ACCENT, ACCENT_2, ACCENT_3)
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"


def use_style() -> None:
    """Apply the shared chart style. Call once per notebook."""
    mpl.rcParams.update(
        {
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_MUTED,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlecolor": INK,
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "text.color": INK,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "figure.dpi": 110,
            "savefig.bbox": "tight",
        }
    )


# --------------------------------------------------------------------------
# Discovery & reading
# --------------------------------------------------------------------------


def list_data_files(raw_dir: Path = RAW_DIR) -> list[Path]:
    """Every readable data file under ``raw_dir`` (links.txt excluded)."""
    return sorted(
        p
        for p in raw_dir.rglob("*")
        if p.is_file()
        and p.name != "links.txt"
        and (p.suffix.lower() in TABULAR_SUFFIXES or ".xes" in p.name.lower())
    )


def unique_logs(paths: list[Path]) -> list[Path]:
    """One path per log, preferring the .csv cache when both formats exist."""
    chosen: dict[str, Path] = {}
    for path in paths:
        stem = path.name.split(".")[0]
        if stem not in chosen or path.suffix.lower() == ".csv":
            chosen[stem] = path
    return [chosen[k] for k in sorted(chosen)]


def run_table(
    summarise: Callable[[Path], dict],
    csv: Path | None = None,
    sort_by: str | None = None,
    index: str | None = None,
) -> pd.DataFrame:
    """Run ``summarise`` on every log, print one row each, optionally write a CSV."""
    warnings.filterwarnings("ignore", message=".*r4pm.*")
    paths = unique_logs(list_data_files())
    if not paths:
        print("No data files found — run `python eda/fetch_data.py` from the repo root.")
        return pd.DataFrame()

    rows = []
    for path in paths:
        print(f"{path.name} ...", flush=True)
        try:
            rows.append(summarise(path))
        except Exception as exc:  # one unreadable log must not sink the table
            print(f"  skipped: {exc}")

    table = pd.DataFrame(rows)
    if sort_by and sort_by in table.columns:
        table = table.sort_values(sort_by, ascending=False)
    if index and index in table.columns:
        table = table.set_index(index)
    pd.set_option("display.width", 200, "display.max_columns", 50)
    print()
    print(table.to_string(index=bool(index)))

    if csv is not None:
        csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(csv, index=bool(index))
        print(f"\nwrote {csv}")
    return table


def read_any(path: Path, **kwargs) -> pd.DataFrame:
    """Read one log file into a DataFrame, dispatching on extension."""
    name = path.name.lower()
    if ".xes" in name:
        return _read_xes(path)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, **kwargs)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", **kwargs)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, **kwargs)
    if suffix == ".json":
        return pd.read_json(path, **kwargs)
    if suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    raise ValueError(f"unsupported file type: {path.name}")


def _read_xes(path: Path) -> pd.DataFrame:
    """Read an XES log via pm4py, without the progress bar the notebook would store."""
    try:
        import pm4py
        from pm4py.util import constants
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            f"{path.name} is an XES log and pm4py is missing from this env. "
            "Build the env with `make env` — a bare `pip install pm4py` fails, "
            "because its cvxopt dependency has no PyPI wheel."
        ) from exc
    constants.SHOW_PROGRESS_BAR = False
    with warnings.catch_warnings():
        # pm4py advertises its optional `r4pm` backend on every read; the
        # fallback is correct, just slower, so the nudge is noise here.
        warnings.filterwarnings("ignore", message=".*r4pm.*")
        return pm4py.convert_to_dataframe(pm4py.read_xes(str(path)))


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

# Ordered by preference; XES/pm4py names first, then common CSV spellings.
_CASE_CANDIDATES = ("case:concept:name", "case_id", "caseid", "case", "trace_id", "id")
_ACTIVITY_CANDIDATES = ("concept:name", "activity", "activity_name", "event", "task", "action")
_TIMESTAMP_CANDIDATES = ("time:timestamp", "timestamp", "time", "datetime", "start_time", "event_time")


@dataclass(frozen=True)
class LogSpec:
    """Which columns carry the case id, the activity and the timestamp."""

    case: str
    activity: str
    timestamp: str

    def missing_from(self, df: pd.DataFrame) -> list[str]:
        return [c for c in (self.case, self.activity, self.timestamp) if c not in df.columns]


def _first_match(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {c.lower().replace(" ", "_"): c for c in columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    # fall back to a substring hit, e.g. "Case ID (anon)"
    for cand in candidates:
        for key, original in lowered.items():
            if cand in key:
                return original
    return None


def detect_spec(df: pd.DataFrame) -> LogSpec:
    """Guess the three key columns. Always print the result and sanity-check it."""
    cols = list(df.columns)
    case = _first_match(cols, _CASE_CANDIDATES)
    activity = _first_match(cols, _ACTIVITY_CANDIDATES)
    timestamp = _first_match(cols, _TIMESTAMP_CANDIDATES)
    missing = [
        label
        for label, value in (("case", case), ("activity", activity), ("timestamp", timestamp))
        if value is None
    ]
    if missing:
        raise ValueError(
            f"could not detect {', '.join(missing)} column(s) among {cols}. "
            "Pass an explicit LogSpec(case=..., activity=..., timestamp=...)."
        )
    return LogSpec(case=case, activity=activity, timestamp=timestamp)


def prepare(df: pd.DataFrame, spec: LogSpec) -> pd.DataFrame:
    """Normalise to the three key columns: parsed timestamps, sorted, typed.

    Rows with an unparseable timestamp are dropped (and reported by
    :func:`describe_log`, which recomputes counts on the prepared frame).
    """
    missing = spec.missing_from(df)
    if missing:
        raise KeyError(f"columns not in the log: {missing}")

    out = df.copy()
    out[spec.timestamp] = pd.to_datetime(out[spec.timestamp], errors="coerce", format="mixed")
    dropped = int(out[spec.timestamp].isna().sum())
    if dropped:
        print(f"  dropped {dropped:,} row(s) with an unparseable timestamp")
    out = out.dropna(subset=[spec.timestamp])
    out[spec.case] = out[spec.case].astype(str)
    out[spec.activity] = out[spec.activity].astype(str)
    # Stable sort keeps the original row order for ties within a case, which is
    # the only sensible ordering when a log has second-granularity timestamps.
    return out.sort_values([spec.case, spec.timestamp], kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def describe_log(df: pd.DataFrame, spec: LogSpec) -> pd.Series:
    """Headline numbers for a prepared log."""
    ts = df[spec.timestamp]
    return pd.Series(
        {
            "events": len(df),
            "cases": df[spec.case].nunique(),
            "activities": df[spec.activity].nunique(),
            "events_per_case_mean": round(len(df) / max(df[spec.case].nunique(), 1), 2),
            "first_event": ts.min(),
            "last_event": ts.max(),
        }
    )


def activity_stats(df: pd.DataFrame, spec: LogSpec) -> pd.DataFrame:
    """Per activity: event count, share of events, and case coverage."""
    n_cases = df[spec.case].nunique()
    events = df[spec.activity].value_counts().rename("events")
    cases = (
        df.groupby(spec.activity, observed=True)[spec.case]
        .nunique()
        .rename("cases")
    )
    out = pd.concat([events, cases], axis=1)
    out["pct_of_events"] = (out["events"] / len(df) * 100).round(2)
    out["pct_of_cases"] = (out["cases"] / max(n_cases, 1) * 100).round(2)
    out["events_per_case_seen"] = (out["events"] / out["cases"]).round(2)
    return out.sort_values("events", ascending=False)


def case_stats(df: pd.DataFrame, spec: LogSpec) -> pd.DataFrame:
    """Per case: length, start/end, duration, and number of distinct activities."""
    grouped = df.groupby(spec.case, observed=True)
    out = pd.DataFrame(
        {
            "length": grouped.size(),
            "distinct_activities": grouped[spec.activity].nunique(),
            "start": grouped[spec.timestamp].min(),
            "end": grouped[spec.timestamp].max(),
        }
    )
    out["duration_hours"] = (out["end"] - out["start"]).dt.total_seconds() / 3600
    return out.sort_values("length", ascending=False)


def inter_event_times(df: pd.DataFrame, spec: LogSpec) -> pd.DataFrame:
    """Gap to the previous event **within the same case**, in seconds.

    The first event of every case has no predecessor and is excluded, so the
    result has ``len(df) - n_cases`` rows.
    """
    delta = df.groupby(spec.case, observed=True)[spec.timestamp].diff()
    out = pd.DataFrame(
        {
            spec.case: df[spec.case],
            "from_activity": df.groupby(spec.case, observed=True)[spec.activity].shift(),
            "to_activity": df[spec.activity],
            "seconds": delta.dt.total_seconds(),
        }
    ).dropna(subset=["seconds"])
    out["hours"] = out["seconds"] / 3600
    out["days"] = out["seconds"] / 86400
    return out.reset_index(drop=True)


def quantile_table(values: pd.Series, name: str = "value") -> pd.DataFrame:
    """The spread of a distribution at the quantiles worth arguing about."""
    qs = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    clean = pd.Series(values).dropna()
    table = clean.quantile(qs).to_frame(name)
    table.index = [f"p{q * 100:g}" for q in qs]
    table.loc["mean"] = clean.mean()
    table.loc["std"] = clean.std()
    table.loc["n"] = len(clean)
    return table.round(3)


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------


def _label_bars(ax, bars, values, fmt="{:,.0f}") -> None:
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(value),
            va="center",
            ha="left",
            fontsize=9,
            color=INK_MUTED,
        )


def plot_activity_frequency(stats: pd.DataFrame, top: int = 25, ax=None):
    """How often each activity occurs. Sorted, labelled, one measure per chart."""
    data = stats.head(top).iloc[::-1]
    if ax is None:
        _, ax = plt.subplots(figsize=(9, max(3.0, 0.32 * len(data))))
    bars = ax.barh(data.index, data["events"], color=ACCENT, height=0.68)
    _label_bars(ax, bars, data["events"].to_numpy())
    shown = f" (top {top} of {len(stats)})" if len(stats) > top else ""
    ax.set_title(f"Events per activity{shown}")
    ax.set_xlabel("events")
    ax.set_xlim(0, data["events"].max() * 1.12)
    ax.grid(axis="y", visible=False)
    return ax


def plot_activity_case_coverage(stats: pd.DataFrame, top: int = 25, ax=None):
    """Share of cases that contain each activity — a different question than
    raw frequency, so it gets its own chart rather than a second y-axis."""
    data = stats.head(top).sort_values("pct_of_cases").iloc[-top:]
    if ax is None:
        _, ax = plt.subplots(figsize=(9, max(3.0, 0.32 * len(data))))
    bars = ax.barh(data.index, data["pct_of_cases"], color=ACCENT, height=0.68)
    _label_bars(ax, bars, data["pct_of_cases"].to_numpy(), fmt="{:.1f}%")
    ax.set_title("Share of cases containing the activity")
    ax.set_xlabel("% of cases")
    ax.set_xlim(0, 108)
    ax.grid(axis="y", visible=False)
    return ax


def plot_case_length_distribution(cases: pd.DataFrame, bins: int = 40, clip_q: float = 0.99):
    """Histogram + ECDF of events per case.

    The histogram is clipped at ``clip_q`` so a handful of very long cases does
    not flatten the body of the distribution; the ECDF beside it shows the full
    range untouched, so nothing is hidden.
    """
    lengths = cases["length"]
    cutoff = lengths.quantile(clip_q)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    body = lengths[lengths <= cutoff]
    # Case length is a count: use integer-aligned bins when the range is small,
    # otherwise float bins put two integers in one bar and none in the next.
    span = int(body.max() - body.min()) + 1
    edges = np.arange(body.min() - 0.5, body.max() + 1.5) if span <= bins else bins
    axes[0].hist(body, bins=edges, color=ACCENT, edgecolor="#fcfcfb", linewidth=0.6)
    axes[0].axvline(lengths.median(), color=ACCENT_2, linewidth=2, linestyle="--")
    axes[0].text(
        lengths.median(), axes[0].get_ylim()[1] * 0.94,
        f"  median {lengths.median():,.0f}", color=ACCENT_2, fontsize=9, va="top",
    )
    axes[0].set_title(f"Case length (events per case, clipped at p{clip_q * 100:g} = {cutoff:,.0f})")
    axes[0].set_xlabel("events in case")
    axes[0].set_ylabel("cases")
    axes[0].grid(axis="x", visible=False)

    _ecdf(axes[1], lengths)
    axes[1].set_title("Case length — ECDF (full range)")
    axes[1].set_xlabel("events in case")
    axes[1].set_ylabel("share of cases <= x")
    fig.tight_layout()
    return fig, axes


def plot_interevent_distribution(gaps: pd.DataFrame, bins: int = 50):
    """Histogram (log-spaced) + ECDF of within-case inter-event times.

    Inter-event times span seconds to months, so the histogram uses log-spaced
    bins on a log x-axis; a linear histogram of this data is unreadable.
    Zero-second gaps (simultaneous events) cannot sit on a log axis, so they are
    excluded from the histogram and reported separately in the title.
    """
    seconds = gaps["seconds"]
    positive = seconds[seconds > 0]
    n_zero = int((seconds == 0).sum())
    n_negative = int((seconds < 0).sum())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    edges = np.logspace(np.log10(positive.min()), np.log10(positive.max()), bins)
    axes[0].hist(positive, bins=edges, color=ACCENT, edgecolor="#fcfcfb", linewidth=0.6)
    axes[0].set_xscale("log")
    for secs, label in ((60, "1 min"), (3600, "1 h"), (86400, "1 day"), (604800, "1 week")):
        if positive.min() <= secs <= positive.max():
            axes[0].axvline(secs, color=GRID, linewidth=1.2, zorder=0)
            axes[0].text(secs, axes[0].get_ylim()[1] * 0.98, f" {label}",
                         fontsize=8, color=INK_MUTED, va="top")
    zero_note = f"\n{n_zero:,} zero-second gap(s) excluded from this chart" if n_zero else ""
    axes[0].set_title("Inter-event time (log scale)")
    axes[0].set_xlabel(f"seconds between consecutive events in a case{zero_note}")
    axes[0].set_ylabel("events")
    axes[0].grid(axis="x", visible=False)

    _ecdf(axes[1], positive, logx=True)
    axes[1].set_title("Inter-event time — ECDF")
    axes[1].set_xlabel("seconds (log scale)")
    axes[1].set_ylabel("share of gaps <= x")

    if n_negative:
        print(f"  warning: {n_negative:,} negative gap(s) — timestamps out of order")
    fig.tight_layout()
    return fig, axes


def plot_interevent_by_activity(gaps: pd.DataFrame, top: int = 12, ax=None):
    """Where the waiting happens: gap distribution by the activity being waited
    *for*, as a box plot on a log axis, ordered by median gap."""
    positive = gaps[gaps["seconds"] > 0]
    order = (
        positive.groupby("to_activity")["seconds"]
        .agg(["median", "size"])
        .sort_values("median", ascending=False)
        .head(top)
    )
    data = [positive.loc[positive["to_activity"] == a, "seconds"].to_numpy() for a in order.index]
    if ax is None:
        _, ax = plt.subplots(figsize=(9, max(3.0, 0.42 * len(order))))
    box = ax.boxplot(
        data[::-1],
        vert=False,
        labels=[f"{a}  (n={int(order.loc[a, 'size']):,})" for a in order.index][::-1],
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#fcfcfb", "linewidth": 1.6},
    )
    for patch in box["boxes"]:
        patch.set_facecolor(ACCENT)
        patch.set_edgecolor(ACCENT)
    for whisker in box["whiskers"] + box["caps"]:
        whisker.set_color(INK_MUTED)
    ax.set_xscale("log")
    ax.set_title(f"Inter-event time by target activity (top {len(order)} by median wait)")
    ax.set_xlabel("seconds (log scale)")
    ax.grid(axis="y", visible=False)
    return ax


def _ecdf(ax, values: pd.Series, logx: bool = False):
    clean = np.sort(pd.Series(values).dropna().to_numpy())
    y = np.arange(1, len(clean) + 1) / len(clean)
    ax.plot(clean, y, color=ACCENT, linewidth=2)
    if logx:
        ax.set_xscale("log")
    for q, label in ((0.5, "p50"), (0.9, "p90")):
        xq = float(np.quantile(clean, q))
        ax.axhline(q, color=GRID, linewidth=1.2, zorder=0)
        ax.text(clean[0], q, f" {label} = {xq:,.0f}", fontsize=8, color=INK_MUTED, va="bottom")
    ax.set_ylim(0, 1.02)
    return ax


# --------------------------------------------------------------------------
# Gap composition + granularity
# --------------------------------------------------------------------------

# The logs in raw_datasets/ are Dutch (BPIC / Sepsis), so a bare date is stored
# as midnight in Amsterdam, i.e. 22:00 or 23:00 UTC depending on the season.
TZ = "Europe/Amsterdam"

GAP_UNITS: tuple[tuple[str, int], ...] = (
    ("1 s", 1),
    ("10 s", 10),
    ("1 min", 60),
    ("5 min", 300),
    ("1 h", 3600),
    ("1 day", 86400),
)


def gap_composition(gaps: pd.DataFrame) -> pd.DataFrame:
    """How the gaps split into zero / positive / negative, in counts and shares."""
    seconds = gaps["seconds"]
    counts = {
        "zero": int((seconds == 0).sum()),
        "positive": int((seconds > 0).sum()),
        "negative": int((seconds < 0).sum()),
    }
    out = pd.DataFrame({"gaps": pd.Series(counts)})
    out["pct_of_gaps"] = (out["gaps"] / len(seconds) * 100).round(2)
    return out


def gap_granularity(gaps: pd.DataFrame, units: tuple[tuple[str, int], ...] = GAP_UNITS) -> pd.DataFrame:
    """Share of non-zero gaps that are exact multiples of each time unit.

    Zero gaps are excluded: zero is a multiple of every unit and would inflate
    every row. A log recorded at day resolution has ~100% at "1 day"; one with
    true second resolution drops off immediately after "1 s".
    """
    seconds = gaps["seconds"].abs()
    seconds = seconds[seconds > 0]
    rows = {}
    for label, step in units:
        rows[label] = (seconds % step == 0).mean() * 100
    out = pd.DataFrame({"pct_of_nonzero_gaps": pd.Series(rows).round(2)})
    out.index.name = "multiple of"
    return out


def gap_tick(gaps: pd.DataFrame) -> pd.Series:
    """The coarsest unit every non-zero gap is a multiple of (their GCD), plus context."""
    seconds = gaps["seconds"].abs()
    seconds = seconds[seconds > 0]
    integral = bool(np.all(seconds % 1 == 0))
    tick = int(np.gcd.reduce(seconds.astype("int64").to_numpy())) if integral else 0
    return pd.Series(
        {
            "n_nonzero_gaps": len(seconds),
            "n_distinct_values": int(seconds.nunique()),
            "min_seconds": float(seconds.min()),
            "gcd_seconds": tick if integral else float("nan"),
            "sub_second_values": not integral,
        }
    )


def is_local_midnight(df: pd.DataFrame, spec: LogSpec, tz: str = TZ) -> pd.Series:
    """Per-event flag: is this timestamp midnight in ``tz``? — the date-only signature."""
    ts = df[spec.timestamp]
    local = ts.dt.tz_convert(tz) if ts.dt.tz is not None else ts.dt.tz_localize("UTC").dt.tz_convert(tz)
    return local.dt.normalize() == local


def local_midnight_share(df: pd.DataFrame, spec: LogSpec, tz: str = TZ) -> float:
    """Share of events at midnight in ``tz`` — the signature of date-only timestamps.

    A timestamp recorded as a bare date is stored as local midnight, which in
    UTC is 22:00 or 23:00 depending on daylight saving, so reading the clock in
    UTC hides day granularity entirely.
    """
    return round(float(is_local_midnight(df, spec, tz).mean() * 100), 2)


def zero_gap_causes(df: pd.DataFrame, spec: LogSpec, tz: str = TZ) -> pd.DataFrame:
    """Why each zero inter-event gap is zero: lost date resolution, or simultaneity.

    A zero gap means the two events share one timestamp, so both sit at local
    midnight or neither does — there is no mixed case. Midnight means the date
    was recorded without a time of day; anything else is a genuine same-instant
    record.
    """
    delta = df.groupby(spec.case, observed=True)[spec.timestamp].diff().dt.total_seconds()
    at_midnight = is_local_midnight(df, spec, tz)[delta == 0]
    counts = {"date_only": int(at_midnight.sum()), "same_instant": int((~at_midnight).sum())}
    out = pd.DataFrame({"zero_gaps": pd.Series(counts)})
    out["pct_of_zero_gaps"] = (
        (out["zero_gaps"] / len(at_midnight) * 100).round(2) if len(at_midnight) else float("nan")
    )
    return out


def clock_time_mode(df: pd.DataFrame, spec: LogSpec) -> pd.Series:
    """The most common time of day in the log, and its share of all events.

    A large share at one clock time means events are written by a batch at a
    fixed hour, which is what makes gaps land on exact multiples of a day.
    """
    counts = df[spec.timestamp].dt.time.value_counts()
    return pd.Series(
        {
            "top_clock_time": str(counts.index[0]),
            "pct_at_top_clock_time": round(counts.iloc[0] / len(df) * 100, 2),
        }
    )


def plot_gap_composition(comp: pd.DataFrame, ax=None):
    """One bar per gap sign: how much of the log is simultaneous or out of order."""
    data = comp.loc[[i for i in ("negative", "zero", "positive") if i in comp.index]]
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 2.6))
    bars = ax.barh(data.index, data["pct_of_gaps"], color=ACCENT, height=0.6)
    _label_bars(ax, bars, data["gaps"].to_numpy(), fmt="{:,.0f} gaps")
    ax.set_title("Gap composition — zero-second gaps as a share of all gaps")
    ax.set_xlabel("% of gaps")
    ax.set_xlim(0, 112)
    ax.grid(axis="y", visible=False)
    return ax


def plot_gap_granularity(gran: pd.DataFrame, ax=None):
    """Timestamp resolution: how many non-zero gaps land exactly on each unit."""
    data = gran.iloc[::-1]
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3.0))
    bars = ax.barh(data.index, data["pct_of_nonzero_gaps"], color=ACCENT, height=0.6)
    _label_bars(ax, bars, data["pct_of_nonzero_gaps"].to_numpy(), fmt="{:.1f}%")
    ax.set_title("Inter-event time granularity — gaps that are exact multiples of a unit")
    ax.set_xlabel("% of non-zero gaps")
    ax.set_xlim(0, 112)
    ax.grid(axis="y", visible=False)
    return ax


def plot_zero_gap_causes(table: pd.DataFrame, ax=None):
    """One 100% bar per log: how much of its zero mass is date-only vs same-instant."""
    data = table.set_index("log") if "log" in table.columns else table
    data = data.iloc[::-1]  # highest pct_zero on top once barh flips the axis
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 0.42 * len(data) + 1.8))

    date_only = data["pct_date_only"].to_numpy()
    same = data["pct_same_instant"].to_numpy()
    # A 2-point spacer reads the boundary without a stroke; it comes out of the
    # second segment, so a stacked bar still ends at 100.
    gap = np.where((date_only > 0) & (same > 0), 2.0, 0.0)
    ax.barh(data.index, date_only, color=ACCENT, height=0.6, label="date-only")
    ax.barh(data.index, same - gap, left=date_only + gap, color=ACCENT_2, height=0.6, label="same instant")

    for y, (share, pct_zero) in enumerate(zip(date_only, data["pct_zero"])):
        if share >= 12:  # only label a segment wide enough to hold the text
            ax.text(share - 1.5, y, f"{share:.0f}%", va="center", ha="right", fontsize=8, color="#fcfcfb")
        ax.text(104, y, f"{pct_zero:.0f}% zero", va="center", fontsize=8, color=INK_MUTED)

    ax.set_title("Zero-gap causes by log")
    ax.set_xlabel("% of the log's zero gaps")
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
    return ax


# --------------------------------------------------------------------------
# Tie blocks
# --------------------------------------------------------------------------

# Known XES lifecycle order; unlisted values rank as missing and never resolve.
LIFECYCLE_RANK = {
    "schedule": 0,
    "assign": 1,
    "reassign": 2,
    "start": 3,
    "suspend": 4,
    "resume": 5,
    "complete": 6,
    "autoskip": 7,
    "manualskip": 8,
    "withdraw": 9,
    "ate_abort": 10,
    "pi_abort": 11,
}

LIFECYCLE_COL = "lifecycle:transition"


def block_ids(df: pd.DataFrame, spec: LogSpec) -> pd.Series:
    """Label each event with a tie-block id: a run of equal timestamps in one case."""
    case_changed = df[spec.case].ne(df[spec.case].shift())
    time_changed = df[spec.timestamp].ne(df[spec.timestamp].shift())
    return (case_changed | time_changed).cumsum()


def lifecycle_resolves(frame: pd.DataFrame) -> bool:
    """True when lifecycle values are known and distinct, so they totally order the block."""
    if LIFECYCLE_COL not in frame.columns:
        return False
    ranks = frame[LIFECYCLE_COL].astype(str).str.lower().map(LIFECYCLE_RANK)
    return bool(ranks.notna().all() and ranks.is_unique)


def plot_tie_blocks(table: pd.DataFrame, ax=None):
    """One bar per log: share of events inside a tie block, and how much of it is unresolvable."""
    data = table.set_index("log") if "log" in table.columns else table
    data = data.sort_values("pct_events_tied")  # barh flips the axis: largest on top

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 0.42 * len(data) + 1.8))

    tied = data["pct_events_tied"].to_numpy()
    unresolvable = data["pct_events_unresolvable"].to_numpy()
    ax.barh(data.index, tied, color=GRID, height=0.6, label="in a tie block")
    ax.barh(data.index, unresolvable, color=ACCENT, height=0.6, label="unresolvable")

    for y, (share, longest) in enumerate(zip(unresolvable, data["max_block"])):
        ax.text(share + 1.5, y, f"{share:.1f}%", va="center", fontsize=8, color=INK_MUTED)
        ax.text(86, y, f"longest {longest:,}", va="center", fontsize=8, color=INK_MUTED)

    ax.set_title("Tied events, and how many are unresolvable", pad=30)
    ax.set_xlabel("% of the log's events")
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75])
    ax.grid(axis="y", visible=False)
    # Its own line under the title: every row's right edge is taken by a label.
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
    return ax
