"""
Evaluation orchestrator. Runs the stages owned by the experimentation tool:
  * RQ2 — label fidelity (equivalence: R1 reads the source XES directly via the
    source accessors, R2 reads the OCEL 2.0 SQLite; see rq2_fidelity.py),
  * RQ3 — end-to-end feasibility + descriptive metrics on the representative subset,

RQ1 (XES->OCEL transformation, properties P1.1-P1.6 incl. the P1.2b order check,
OCEL 2.0 schema validation) is produced by the CONVERTER (a separate tool) and is
therefore out of scope here.

Four independent axes select what runs: `rqs` (RQ2, RQ3), `log_groups`
(predictcollab, bpi2013), `rq3_scopes` (partial = the 6-task representative
subset, one task per anchor x problem-type combination; full = all 14 tasks,
supplementary coverage -- RQ3 only, ignored for RQ2) and `predictors` (any
subset of `predictors.dispatch.PREDICTOR_REGISTRY`, e.g. random_forest,
xgboost, lstm, lstm_torch, transformer, gnn -- RQ3 only; RQ2 has no predictor axis). Output
filenames are derived from the combination, e.g. `rq3_results_random_forest_predictcollab.csv`
(predictcollab, partial, random_forest), `rq3_results_gnn_predictcollab_full.csv`
(predictcollab, full, gnn), `rq2_fidelity_bpi2013.csv` (RQ2 has no predictor
axis), `rq3_results_xgboost_bpi2013_full.csv` (bpi2013, full, xgboost). The
log group always appears in the name, so files stay self-describing as more
predictors are added. RQ3 files additionally land in a per-(log_group, scope)
stage subdirectory of `cfg.out_dir`, e.g. `data/results/predictcollab_full/`
-- see `plot_rq3_metrics.py`'s matching `--output-dir` default. See `main()`
below.

Some (predictor, log_group) combinations are slow (BPI2013 in particular, see
below), so each axis can be restricted independently to run the evaluation in
parts -- e.g. one predictor on one log group at a time -- either by calling
`main()` with a subset of each iterable, or via the CLI:

    python -m evaluation.run_evaluation                                    # default: RQ2+RQ3 partial, Predict-Collab, random_forest
    python -m evaluation.run_evaluation --rqs RQ3 --predictors gnn xgboost  # RQ3 only, two predictors, Predict-Collab
    python -m evaluation.run_evaluation --log-groups bpi2013               # BPI2013 only
    python -m evaluation.run_evaluation --rq3-scopes full --out-dir /tmp/x  # RQ3 full catalog, custom output dir
    python -m evaluation.run_evaluation --help                              # full flag list

Log paths/hyperparameters are adjusted in config.py; which stages run is
adjusted via the CLI flags above (see `_parse_args` below) or by calling
`main()` directly.

Reproducibility note: RandomForest draws features and bootstrap rows BY
POSITION, so a reordered column set or row order moves the metrics at a fixed
random_state even with every feature value, label and sample count unchanged.
Both orders are pinned at their source, not by the hash seed: the per-activity
feature columns come from a sorted vocabulary (features.ocpa.build_feature_set)
and the table is sorted by (case_id, event_id) before training
(features.ocpa.extract_feature_table) -- see the comments there for why the
upstream order is not usable as-is.

The __main__ guard below still re-execs once with PYTHONHASHSEED=0 if unset,
as a guard against hash-order dependence inside OCPA's own leading-type
extraction, which we do not control. That is defensive only: with the two
orders above pinned, running this module under different hash seeds reproduces
every metric to within the last representable digit -- the same spread as two
runs at one seed -- so no reported figure depends on the variable being set.
"""
import argparse
import os
import re
import sys
from dataclasses import replace
from typing import Iterable, Literal, Optional
from tasks.catalog import EQUIVALENCE_TASKS, RQ3_SUBSET, TASKS
from .config import ExperimentConfig, predictcollab_ocel_logs, real_world_ocel_logs
from .rq2_fidelity import run_rq2
from .rq3_pipeline import run_rq3
from predictors.dispatch import PREDICTOR_REGISTRY

RQ = Literal["RQ2", "RQ3"]
LogGroup = Literal["predictcollab", "bpi2013"]
Scope = Literal["partial", "full"]

# BPI2013 is ~29.5x larger than the largest study log by events (69,584 OCEL
# events vs. Artificial5's 2,360), across 7,554 cases -- 65,533 source events
# plus 4,051 synthesized SendTask events; OCPA feature extraction + RandomForest
# fitting time is untested at this scale, so it is opt-in, not a default.
_LOG_GROUPS = {
    "predictcollab": predictcollab_ocel_logs,
    "bpi2013": real_world_ocel_logs,
}
# RQ3-only: "full" (all 14 tasks, EQUIVALENCE_TASKS) roughly doubles the OCPA
# feature extraction + RandomForest fitting cost of "partial" (RQ3_SUBSET, the
# 6-task representative subset -- one task per anchor x problem-type combination).
_RQ3_SCOPES = {
    "partial": lambda: list(RQ3_SUBSET),
    "full": lambda: list(EQUIVALENCE_TASKS),
}


def _subset_suffix(tasks: Optional[Iterable[str]], logs: Optional[Iterable[str]]) -> str:
    """Filename marker for a narrowed run (see main()'s `tasks`/`logs`).

    Non-empty only when the run is a strict subset of its stage, so a
    single-task patch run can never silently overwrite the stage's canonical
    14-task CSV -- which would discard hours of computation. The suffix is
    appended to the results/profile/progress basenames, e.g.
    rq3_results_transformer_bpi2013_full__OB-M.csv.
    """
    parts = []
    if logs:
        parts.extend(str(x) for x in logs)
    if tasks:
        parts.extend(str(x) for x in tasks)
    if not parts:
        return ""
    return "__" + "-".join(re.sub(r"[^A-Za-z0-9-]+", "", p) for p in parts)


def main(cfg: Optional[ExperimentConfig] = None,
        rqs: Iterable[RQ] = ("RQ2", "RQ3"),
        log_groups: Iterable[LogGroup] = ("predictcollab",),
        rq3_scopes: Iterable[Scope] = ("partial",),
        predictors: Iterable[str] = ("random_forest",),
        tasks: Optional[Iterable[str]] = None,
        logs: Optional[Iterable[str]] = None):
    """Run RQ2/RQ3 over the requested (rq, log_group, rq3_scope, predictor) combinations.

    rq3_scopes and predictors are only meaningful when "RQ3" is in rqs; both
    are ignored for RQ2 (RQ2 always evaluates all 14 tasks via
    EQUIVALENCE_TASKS, unconditionally, and has no predictor). RQ2 therefore
    runs once per log_group regardless of how many predictors are requested.

    predictors selects keys from predictors.dispatch.PREDICTOR_REGISTRY (e.g.
    "random_forest", "xgboost", "lstm", "lstm_torch", "transformer", "gnn"); RQ3 runs once per
    (log_group, predictor, scope) combination, each to its own
    rq3_results_{predictor}*.csv.

    tasks (RQ3 only) narrows the run to specific tasks.catalog keys instead of
    the whole scope catalog; logs narrows it to specific LogSpec names within
    the selected log_groups. Both are for re-running one cell of an
    already-computed stage -- e.g. re-measuring a (log, task) pair whose
    timings were disturbed -- without repeating the stage. A narrowed run
    writes to its own suffixed filenames (see _subset_suffix) so it never
    overwrites the stage's canonical CSVs; merge the rows in deliberately.
    """
    base_cfg = cfg or ExperimentConfig()
    results = {}
    if tasks is not None:
        unknown = [t for t in tasks if t not in TASKS]
        if unknown:
            raise KeyError(f"Unknown task(s) {unknown}; valid keys: {', '.join(TASKS)}")
    subset_suffix = _subset_suffix(tasks, logs)

    for group in log_groups:
        group_logs = _LOG_GROUPS[group]()
        if logs is not None:
            wanted = set(logs)
            known = {spec.name for spec in group_logs}
            unknown = wanted - known
            if unknown:
                raise KeyError(f"Unknown log(s) {sorted(unknown)} for log group "
                               f"{group!r}; valid names: {', '.join(sorted(known))}")
            group_logs = [spec for spec in group_logs if spec.name in wanted]
        group_cfg = replace(base_cfg, logs=group_logs)
        suffix = f"_{group}"

        if "RQ2" in rqs:
            print(f"\n########## RQ2 — label fidelity ({group}) ##########")
            results[f"rq2{suffix}"] = run_rq2(group_cfg, out_name=f"rq2_fidelity{suffix}.csv")

        if "RQ3" in rqs:
            for predictor in predictors:
                predictor_cfg = replace(group_cfg, predictor=predictor)
                for scope in rq3_scopes:
                    scope_suffix = "" if scope == "partial" else "_full"
                    # One subdirectory per (log_group, scope) stage -- e.g.
                    # data/results/predictcollab_full/ -- mirroring
                    # plot_rq3_metrics.py's per-stage plot directories, so a
                    # stage's results/profile/progress files live alongside
                    # its plots instead of all stages mixing in a flat
                    # data/results/. folds_dir is NOT nested here: fold
                    # assignments are per-log only, deliberately reused
                    # across every predictor/scope for that log. Directory
                    # names always spell out the scope (_partial/_full), even
                    # though partial's own FILENAMES stay suffix-less per the
                    # module docstring's existing convention -- only the
                    # directory needs both scopes to read as symmetric.
                    stage_out_dir = os.path.join(base_cfg.out_dir, f"{group}_{scope}")
                    # `tasks`, when given, replaces the scope catalog; the
                    # scope itself still selects the stage directory and the
                    # _full naming, so a narrowed run lands beside the stage
                    # it patches rather than in a directory of its own.
                    scope_tasks = list(tasks) if tasks is not None else _RQ3_SCOPES[scope]()
                    scope_cfg = replace(predictor_cfg, rq3_tasks=scope_tasks,
                                        out_dir=stage_out_dir)
                    label = f"{scope} catalog" if tasks is None else f"tasks {', '.join(scope_tasks)}"
                    print(f"\n########## RQ3 — {label} ({group}, {predictor}) ##########")
                    results[f"rq3_{predictor}{suffix}{scope_suffix}{subset_suffix}"] = run_rq3(
                        scope_cfg,
                        out_name=f"rq3_results_{predictor}{suffix}{scope_suffix}{subset_suffix}.csv")

    if "RQ2" in rqs or "RQ3" in rqs:
        print("\n[note] RQ1 (transformation + P1 + schema) is the converter's tool.")

    return results


def _parse_args(argv):
    p = argparse.ArgumentParser(
        prog="python -m evaluation.run_evaluation",
        description="Run RQ2/RQ3 evaluation stages. Combine flags to run a "
                    "subset -- e.g. one predictor on one log group -- since some "
                    "(predictor, log_group) combinations are slow (see BPI2013 note "
                    "in the module docstring).")
    p.add_argument("--rqs", nargs="+", choices=["RQ2", "RQ3"],
                   default=["RQ2", "RQ3"], help="which RQs to run (default: both)")
    p.add_argument("--log-groups", nargs="+", choices=list(_LOG_GROUPS),
                   default=["predictcollab"],
                   help="which log groups to run (default: predictcollab; bpi2013 is opt-in, slow)")
    p.add_argument("--rq3-scopes", nargs="+", choices=list(_RQ3_SCOPES),
                   default=["partial"],
                   help="RQ3 task catalog scope; ignored for RQ2 (default: partial)")
    p.add_argument("--predictors", nargs="+", choices=sorted(PREDICTOR_REGISTRY),
                   default=["random_forest"],
                   help="RQ3 predictors to evaluate, one run per predictor (default: random_forest)")
    p.add_argument("--tasks", nargs="+", choices=sorted(TASKS), default=None, metavar="TASK",
                   help="RQ3 only: run just these tasks.catalog keys instead of "
                        "the whole --rq3-scopes catalog. For re-running one cell "
                        "of an already-computed stage; output goes to suffixed "
                        "filenames so it cannot overwrite the stage's own CSVs "
                        f"(choices: {', '.join(sorted(TASKS))})")
    p.add_argument("--logs", nargs="+", default=None, metavar="LOG",
                   help="RQ3 only: run just these LogSpec names within the "
                        "selected --log-groups (e.g. Healthcare, BPI2013). Same "
                        "suffixed-output rule as --tasks")
    p.add_argument("--out-dir", default=None,
                   help="override ExperimentConfig.out_dir (default: data/results)")
    p.add_argument("--folds-dir", default=None,
                   help="override ExperimentConfig.folds_dir (default: data/folds)")
    p.add_argument("--regenerate-folds", action="store_true",
                   help="regenerate each log's persisted CollaborationCase->fold "
                        "assignment even if data/folds/{log}_folds.csv already "
                        "exists (default: reuse it, or generate it once if "
                        "missing -- see rq3_pipeline.load_or_generate_folds)")
    p.add_argument("--profile", action="store_true",
                   help="RQ3 only: also write a per-stage wall-clock + RSS "
                        "memory profile to rq3_profile_{predictor}*.csv "
                        "(default: off, see evaluation.profiling)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None,
                   help="override ExperimentConfig.device, used by every "
                        "GPU-capable predictor (gnn, lstm_torch, transformer; "
                        "random_forest/xgboost are CPU-only regardless) "
                        "(default: cfg default, 'cpu' -- see config.py)")
    return p.parse_args(argv)


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, "-m", "evaluation.run_evaluation"] + sys.argv[1:])
    args = _parse_args(sys.argv[1:])
    cfg_overrides = {}
    if args.out_dir:
        cfg_overrides["out_dir"] = args.out_dir
    if args.folds_dir:
        cfg_overrides["folds_dir"] = args.folds_dir
    if args.regenerate_folds:
        cfg_overrides["regenerate_folds"] = True
    if args.profile:
        cfg_overrides["profile"] = True
    if args.device:
        cfg_overrides["device"] = args.device
    cfg = ExperimentConfig(**cfg_overrides) if cfg_overrides else None
    main(cfg=cfg,
        rqs=tuple(args.rqs),
        log_groups=tuple(args.log_groups),
        rq3_scopes=tuple(args.rq3_scopes),
        predictors=tuple(args.predictors),
        tasks=tuple(args.tasks) if args.tasks else None,
        logs=tuple(args.logs) if args.logs else None)
