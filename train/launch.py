"""Run a grid of train.train runs (logs × variants × seeds), each logged to its own file; finished runs are skipped.

  python -m train.launch --logs g1_batch30_s0 --variants A0 A3 --seeds 0 1 2 --jobs 2 -- --sched plateau
Arguments after `--` go to every train.train call. A run counts as finished when test_pred.pt exists;
an interrupted run resumes from its last.pt. Output per run: <out>/<log>/<variant>_s<seed>/train.log.
"""
import argparse
import itertools
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def run_one(log: str, variant: str, seed: int, out: Path, extra: list) -> str:
    """Train one run unless finished; stdout and stderr go to its train.log. Returns a status line."""
    folder = out / log / f"{variant}_s{seed}"
    if (folder / "test_pred.pt").exists():
        return f"skip {folder}"
    folder.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "train.train", "--log", log, "--variant", variant, "--seed", str(seed),
           "--out", str(out), *extra]
    with (folder / "train.log").open("a") as f:
        code = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT).returncode
    return f"{'done' if code == 0 else f'FAIL ({code})'} {folder}"


def main():
    """Parse the grid and run it with --jobs parallel processes."""
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--variants", nargs="+", default=["A0", "A1", "A2", "A3"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", type=Path, default=REPO / "runs")
    args = ap.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
    grid = list(itertools.product(args.logs, args.variants, args.seeds))
    failures = 0
    with ThreadPoolExecutor(args.jobs) as pool:
        for status in pool.map(lambda g: run_one(*g, args.out.resolve(), extra), grid):
            print(status, flush=True)
            failures += status.startswith("FAIL")
    if failures:
        raise SystemExit(f"{failures}/{len(grid)} runs failed; inspect their train.log files")


if __name__ == "__main__":
    main()
