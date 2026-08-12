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
xgboost, lstm, lstm_torch, gnn -- RQ3 only; RQ2 has no predictor axis). Output
filenames are derived from the combination, e.g. `rq3_results_random_forest_predictcollab.csv`
(predictcollab, partial, random_forest), `rq3_results_gnn_predictcollab_full.csv`
(predictcollab, full, gnn), `rq2_fidelity_bpi2013.csv` (RQ2 has no predictor
axis), `rq3_results_xgboost_bpi2013_full.csv` (bpi2013, full, xgboost). The
log group always appears in the name, so files stay self-describing as more
predictors are added. See `main()` below.

Some (predictor, log_group) combinations are slow (BPI2013 in particular, see
below), so each axis can be restricted independently to run the evaluation in
parts -- e.g. one predictor on one log group at a time -- either by calling
`main()` with a subset of each iterable, or via the CLI:

    python -m ocpm_eval.run_evaluation                                    # default: RQ2+RQ3 partial, Predict-Collab, random_forest
    python -m ocpm_eval.run_evaluation --rqs RQ3 --predictors gnn xgboost  # RQ3 only, two predictors, Predict-Collab
    python -m ocpm_eval.run_evaluation --log-groups bpi2013               # BPI2013 only
    python -m ocpm_eval.run_evaluation --rq3-scopes full --out-dir /tmp/x  # RQ3 full catalog, custom output dir
    python -m ocpm_eval.run_evaluation --help                              # full flag list

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
import sys
from dataclasses import replace
from typing import Iterable, Literal, Optional
from ocpm_tasks.catalog import EQUIVALENCE_TASKS, RQ3_SUBSET
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


def main(cfg: Optional[ExperimentConfig] = None,
        rqs: Iterable[RQ] = ("RQ2", "RQ3"),
        log_groups: Iterable[LogGroup] = ("predictcollab",),
        rq3_scopes: Iterable[Scope] = ("partial",),
        predictors: Iterable[str] = ("random_forest",)):
    """Run RQ2/RQ3 over the requested (rq, log_group, rq3_scope, predictor) combinations.

    rq3_scopes and predictors are only meaningful when "RQ3" is in rqs; both
    are ignored for RQ2 (RQ2 always evaluates all 14 tasks via
    EQUIVALENCE_TASKS, unconditionally, and has no predictor). RQ2 therefore
    runs once per log_group regardless of how many predictors are requested.

    predictors selects keys from predictors.dispatch.PREDICTOR_REGISTRY (e.g.
    "random_forest", "xgboost", "lstm", "lstm_torch", "gnn"); RQ3 runs once per
    (log_group, predictor, scope) combination, each to its own
    rq3_results_{predictor}*.csv.
    """
    base_cfg = cfg or ExperimentConfig()
    results = {}

    for group in log_groups:
        group_cfg = replace(base_cfg, logs=_LOG_GROUPS[group]())
        suffix = f"_{group}"

        if "RQ2" in rqs:
            print(f"\n########## RQ2 — label fidelity ({group}) ##########")
            results[f"rq2{suffix}"] = run_rq2(group_cfg, out_name=f"rq2_fidelity{suffix}.csv")

        if "RQ3" in rqs:
            for predictor in predictors:
                predictor_cfg = replace(group_cfg, predictor=predictor)
                for scope in rq3_scopes:
                    scope_cfg = replace(predictor_cfg, rq3_tasks=_RQ3_SCOPES[scope]())
                    scope_suffix = "" if scope == "partial" else "_full"
                    print(f"\n########## RQ3 — {scope} catalog ({group}, {predictor}) ##########")
                    results[f"rq3_{predictor}{suffix}{scope_suffix}"] = run_rq3(
                        scope_cfg,
                        out_name=f"rq3_results_{predictor}{suffix}{scope_suffix}.csv")

    if "RQ2" in rqs or "RQ3" in rqs:
        print("\n[note] RQ1 (transformation + P1 + schema) is the converter's tool.")

    return results


def _parse_args(argv):
    p = argparse.ArgumentParser(
        prog="python -m ocpm_eval.run_evaluation",
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
    p.add_argument("--out-dir", default=None,
                   help="override ExperimentConfig.out_dir (default: data/results)")
    return p.parse_args(argv)


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, "-m", "ocpm_eval.run_evaluation"] + sys.argv[1:])
    args = _parse_args(sys.argv[1:])
    cfg = ExperimentConfig(out_dir=args.out_dir) if args.out_dir else None
    main(cfg=cfg,
        rqs=tuple(args.rqs),
        log_groups=tuple(args.log_groups),
        rq3_scopes=tuple(args.rq3_scopes),
        predictors=tuple(args.predictors))
