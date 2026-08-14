"""
Standalone integrity audit of a written RQ3 stage directory. Computes nothing
new and touches nothing -- it only re-reads `rq3_results_*.csv` /
`rq3_profile_*.csv` and reports the conditions that are easy to miss when
reading those files by eye, each of which has silently distorted a conclusion
at least once:

  1 degeneracy      constant-target cells, and whether `degenerate` agrees with
                    what the metrics imply
  2 ties            cells with an exactly tied optimum -- these make a naive
                    `argmax` win count depend on file load order, not on data
  3 profile cover   cells that were labelled but never actually fitted
  4 fold coverage   cells fitted in some folds but not all (skews a per-task
                    mean without ever showing up as an absent task)
  5 timing outliers folds far from their own cell's median (external
                    interference rather than a property of the workload)
  6 sanity          ran_end_to_end, fold counts, ragged per-cell coverage

Run it after a stage finishes and before drawing any conclusion from it; see
"Reproducing this study on another stage" in a stage's ANALYSIS.md for how the
findings feed the analysis.

Exit status is 1 when a PROBLEM is found (something that should be fixed or
explained), 0 when only INFO is reported (ties and degeneracy are properties of
the data, not defects).

Run:
    python -m evaluation.audit_stage --log-group predictcollab --scope full
    python -m evaluation.audit_stage --all
    python -m evaluation.audit_stage --results-dir data/results/bpi2013_full
"""
import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .config import ExperimentConfig

# A fold whose fit time exceeds its own cell's median by this factor is
# reported. Chosen well above observed fold-to-fold variation (~1.3x at worst
# on a quiet machine) so it flags interference, not normal jitter.
OUTLIER_RATIO = 3.0


def _load(results_dir: Path, kind: str) -> Dict[str, pd.DataFrame]:
    """One frame per predictor. Suffixed patch files (`__TASK`) are skipped:
    they are deliberately partial and would look like ragged coverage."""
    out: Dict[str, pd.DataFrame] = {}
    for f in sorted(glob.glob(str(results_dir / f"rq3_{kind}_*.csv"))):
        if "__" in os.path.basename(f):
            continue
        frame = pd.read_csv(f)
        if len(frame):
            out[frame["predictor"].iloc[0]] = frame
    return out


def audit(results_dir: Path, n_folds: int, outlier_ratio: float = OUTLIER_RATIO
          ) -> List[str]:
    """Print the report; return the list of PROBLEM lines (empty == clean)."""
    problems: List[str] = []

    def problem(msg: str) -> None:
        problems.append(msg)
        print(f"  !! {msg}")

    print("=" * 78)
    print(results_dir)
    print("=" * 78)

    results, profiles = _load(results_dir, "results"), _load(results_dir, "profile")
    if not results:
        problem(f"no rq3_results_*.csv under {results_dir}")
        return problems

    df = pd.concat(results.values(), ignore_index=True)
    models = sorted(results)
    print(f"\n[0] {len(models)} predictors {models} | "
          f"{df.groupby(['log', 'task']).ngroups} (log,task) cells | {len(df)} rows")

    # 1 degeneracy -----------------------------------------------------------
    print("\n[1] degeneracy")
    # Checked per file, not on the concatenation: when only SOME predictors
    # were run with a version that writes the column, concat silently fills the
    # rest with NaN, so the column is "present" while half its values are
    # missing -- and a boolean mask over it raises. Normalise before any use.
    without = sorted(p for p, d in results.items() if "degenerate" not in d.columns)
    if without:
        problem(f"`degenerate` column missing for {without} -- written before "
                f"the column existed; those rows cannot be distinguished from "
                f"genuinely perfect ones (back-fill them or re-run)")
    if "degenerate" in df.columns:
        df["degenerate"] = df["degenerate"].fillna(False).astype(bool)
    else:
        df["degenerate"] = False

    if not without:
        deg = df[df["degenerate"]].groupby(["log", "task"]).size()
        print(f"  degenerate cells: {len(deg)} {list(deg.index)}")
        for lg, tk in deg.index:
            g = df[(df["log"] == lg) & (df["task"] == tk)]
            off = g[g["metric_mean"] != g["baseline_mean"]]
            if len(off):
                problem(f"{lg}/{tk} flagged degenerate but metric != baseline "
                        f"for {off['predictor'].tolist()}")
        # the inverse: everyone ties the baseline, yet it is not flagged
        for (lg, tk), g in df.groupby(["log", "task"]):
            if (g["metric_mean"] == g["baseline_mean"]).all() and not g["degenerate"].iloc[0]:
                print(f"  ?  {lg}/{tk}: every model scores exactly its baseline "
                      f"but the cell is not degenerate (n_labels="
                      f"{g['n_labels'].iloc[0] if 'n_labels' in g else '?'}) "
                      f"-- a saturated target, not a constant one")

    # 2 ties -----------------------------------------------------------------
    print("\n[2] tied optima")
    ties, decided = [], []
    for (lg, tk), g in df.groupby(["log", "task"]):
        ascending = g["metric_name"].iloc[0] == "mae"
        best = g["metric_mean"].min() if ascending else g["metric_mean"].max()
        top = g[g["metric_mean"] == best]
        degen = bool(g["degenerate"].iloc[0]) if "degenerate" in g else False
        if len(top) > 1:
            ties.append((lg, tk, sorted(top["predictor"]), degen))
        else:
            decided.append((lg, tk, top["predictor"].iloc[0]))
    print(f"  decided: {len(decided)}   tied: {len(ties)}")
    for lg, tk, tied_models, degen in ties:
        print(f"    {lg}/{tk}: {len(tied_models)} tied {tied_models}"
              f"{'  [degenerate]' if degen else ''}")
    if ties:
        print("  -> count wins over the decided cells only; awarding a tie to "
              "whichever model sorts first makes the table depend on load order")
    if decided:
        wins = pd.Series([w for _, _, w in decided]).value_counts()
        print(f"  outright wins (of {len(decided)}): {wins.to_dict()}")

    # 3 profile coverage -----------------------------------------------------
    print("\n[3] profile coverage")
    if not profiles:
        print("  no rq3_profile_*.csv (profiling is opt-in via --profile)")
    for pred, prof in sorted(profiles.items()):
        real = prof[prof["stage"] == "fit"]
        if "note" in prof.columns:      # tagged rows were entered but did no work
            real = real[real["note"].isna() | (real["note"] == "")]
        unfitted = set()
        for log, g in prof.groupby("log"):
            fitted = set(real[real["log"] == log]["task"].dropna())
            labelled = set(g[g["stage"] == "labeling"]["task"].dropna())
            unfitted |= {(log, t) for t in labelled - fitted}
        marker = "tagged" if "note" in prof.columns else "no `note` column"
        print(f"  {pred:<14} cells never fitted: "
              f"{sorted(unfitted) if unfitted else 'none'}  ({marker})")
    if profiles:
        print("  -> exclude these before comparing summed fit time across models: "
              "predictors without the short-circuit do pay to fit them")

    # 4 fold coverage --------------------------------------------------------
    print("\n[4] fold coverage")
    ragged = False
    for pred, prof in sorted(profiles.items()):
        counts = prof[prof["stage"] == "fit"].groupby(["log", "task"])["fold"].nunique()
        odd = counts[(counts > 0) & (counts < n_folds)]
        if len(odd):
            ragged = True
            problem(f"{pred}: fitted in some folds but not all -> {odd.to_dict()}")
    if profiles and not ragged:
        print(f"  every fitted cell has all {n_folds} folds")

    # 5 timing outliers ------------------------------------------------------
    print(f"\n[5] fit-time outliers (>= {outlier_ratio}x the cell's own median)")
    for pred, prof in sorted(profiles.items()):
        fit = prof[prof["stage"] == "fit"]
        if fit.empty:
            continue
        median = fit.groupby(["log", "task"])["seconds"].transform("median")
        ratio = (fit["seconds"] / median.replace(0, pd.NA)).astype(float)
        worst = ratio.max()
        i = ratio.idxmax()
        line = (f"  {pred:<14} max {worst:5.2f}x  {fit.loc[i, 'log']}/"
                f"{fit.loc[i, 'task']} fold {fit.loc[i, 'fold']}: "
                f"{fit.loc[i, 'seconds']:.1f}s vs median {median.loc[i]:.1f}s")
        if worst >= outlier_ratio:
            problem(line.strip() + "  -- re-run just this cell with --tasks/--logs")
        else:
            print(line)

    # 6 sanity ---------------------------------------------------------------
    print("\n[6] sanity")
    if not df["ran_end_to_end"].all():
        problem("rows with ran_end_to_end=False: "
                f"{df[~df['ran_end_to_end']][['predictor', 'log', 'task']].values.tolist()}")
    else:
        print("  all rows ran_end_to_end=True")
    folds = df["folds"].value_counts().to_dict()
    if set(folds) != {n_folds}:
        problem(f"not every row used {n_folds} folds: {folds}")
    else:
        print(f"  every row used {n_folds} folds")
    per_cell = sorted(df.groupby(["log", "task"]).size().unique().tolist())
    if per_cell != [len(models)]:
        problem(f"ragged coverage -- rows per cell {per_cell}, expected "
                f"{len(models)} (one per predictor)")
    else:
        print(f"  every cell has {len(models)} rows, one per predictor")

    return problems


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Audit a written RQ3 stage directory for the conditions "
                    "that silently distort its analysis.")
    p.add_argument("--log-group", default="predictcollab",
                   help="log group of the stage to audit (default: predictcollab)")
    p.add_argument("--scope", default="full", choices=("partial", "full"),
                   help="task-catalog scope of the stage (default: full)")
    p.add_argument("--results-dir", type=Path, default=None,
                   help="audit this directory directly, ignoring "
                        "--log-group/--scope")
    p.add_argument("--all", action="store_true",
                   help="audit every {log_group}_{scope} directory that exists "
                        "under the results root")
    p.add_argument("--results-root", type=Path, default=None,
                   help="root holding the stage directories "
                        "(default: ExperimentConfig.out_dir)")
    p.add_argument("--outlier-ratio", type=float, default=OUTLIER_RATIO,
                   help=f"flag a fold this many times its cell's median fit "
                        f"time (default: {OUTLIER_RATIO})")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    cfg = ExperimentConfig()
    root = args.results_root or Path(cfg.out_dir)

    if args.results_dir is not None:
        stages = [args.results_dir]
    elif args.all:
        stages = sorted(d for d in root.iterdir()
                        if d.is_dir() and list(d.glob("rq3_results_*.csv")))
        if not stages:
            print(f"no stage directories with rq3_results_*.csv under {root}")
            return 1
    else:
        stages = [root / f"{args.log_group}_{args.scope}"]

    problems = {}
    for stage in stages:
        if not stage.is_dir():
            print(f"!! not a directory: {stage}")
            problems[str(stage)] = ["missing directory"]
            continue
        found = audit(stage, cfg.n_folds, args.outlier_ratio)
        if found:
            problems[stage.name] = found
        print()

    print("=" * 78)
    if problems:
        total = sum(len(v) for v in problems.values())
        print(f"{total} PROBLEM(S) across {len(problems)} stage(s):")
        for name, found in problems.items():
            for f in found:
                print(f"  [{name}] {f}")
        return 1
    print(f"{len(stages)} stage(s) audited, no problems found.")
    print("Ties and degenerate cells above are properties of the data, not "
          "defects -- but they must be excluded from win counts and cost totals.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
