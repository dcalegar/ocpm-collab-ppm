"""
Evaluation orchestrator. Runs the stages owned by the experimentation tool:
  * RQ2 — label fidelity (equivalence: R1 reads the source XES directly via the
    source accessors, R2 reads the OCEL 2.0 SQLite; see rq2_fidelity.py),
  * RQ3 — end-to-end feasibility + descriptive metrics on the representative subset,

RQ1 (XES->OCEL transformation, properties P1.1-P1.6 incl. the P1.2b order check,
OCEL 2.0 schema validation) is produced by the CONVERTER (a separate tool) and is
therefore out of scope here.

Three independent axes select what runs: `rqs` (RQ2, RQ3), `log_groups`
(predictcollab, bpi2013) and `rq3_scopes` (partial = the 6-task representative
subset, one task per anchor x problem-type combination; full = all 14 tasks,
supplementary coverage -- RQ3 only, ignored for RQ2). Output filenames are derived from the
combination plus the selected predictor, e.g. `rq3_results_random_forest.csv`
(predictcollab, partial), `rq3_results_random_forest_full.csv` (predictcollab,
full), `rq2_fidelity_bpi2013.csv` (RQ2 has no predictor axis),
`rq3_results_random_forest_bpi2013_full.csv` (bpi2013, full). See `main()` below.

RQ-EXT (object-enabled EXTENSION tasks X-Inf, X-MSt; ocpm_tasks/extensions.py) is run
separately, on a small hand-built toy log (data/logs/ToyCollab/toy_collab.*, see
mapping/support/build_toy_collab_log.py) with explicit send/receive pairs and msgId
correlation -- NOT on the four study logs, and its results (`rq_ext_results_toy.csv`)
are never combined with rq2/rq3. This is a feasibility demo of the object-enabled
extension tasks (ocpm_tasks/extensions.py), kept separate from the evaluated
RQ2/RQ3 results. The toy log is designed to exercise both X-Inf (variable
in-flight backlog) and X-MSt (message synchronization time with correlation).
Remaining extensions (X-PaL, X-Cmp, X-Lag) are not implemented here.

Usage: python -m ocpm_eval.run_evaluation   (adjust paths in config.py)

Reproducibility note: a fixed PYTHONHASHSEED is required across process runs.
OCPA's leading-type process-execution extraction and some of our own set(...)
usages (e.g. the activity vocabulary in features_ocpa.build_feature_set) order
their output by Python's per-string hash, which is randomized per process
unless PYTHONHASHSEED is pinned; without it, CollaborationCase/row order can
differ between runs, which in turn changes which rows RandomForest's bootstrap
sampling draws at each fixed random_state, changing the reported metrics by a
few percentage points despite the fixed seed. The __main__ guard below
re-execs the process once with PYTHONHASHSEED=0 if it is not already set, so
`python -m ocpm_eval.run_evaluation` is reproducible without the caller having
to remember the environment variable.
"""
import os
import sys
from dataclasses import replace
from typing import Iterable, Literal, Optional
from ocpm_tasks.catalog import EQUIVALENCE_TASKS, RQ3_SUBSET
from .config import ExperimentConfig, ExtExperimentConfig, predictcollab_ocel_logs, real_world_ocel_logs
from .rq2_fidelity import run_rq2
from .rq3_pipeline import run_rq3
from .rq_ext_pipeline import run_rq_ext as run_rq_ext_pipeline

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
        ext_cfg: Optional[ExtExperimentConfig] = None,
        rqs: Iterable[RQ] = ("RQ2", "RQ3"),
        log_groups: Iterable[LogGroup] = ("predictcollab",),
        rq3_scopes: Iterable[Scope] = ("partial",),
        predictor: Optional[str] = None,
        run_rq_ext: bool = True):
    """Run RQ2/RQ3 over the requested (rq, log_group, rq3_scope) combinations.

    rq3_scopes is only meaningful when "RQ3" is in rqs; it is ignored for RQ2
    (RQ2 always evaluates all 14 tasks via EQUIVALENCE_TASKS, unconditionally).
    RQ-EXT (run_rq_ext) is independent of these three axes -- see rq_ext_pipeline.py
    and ExtExperimentConfig: it always runs on the ToyCollab log, never on
    predictcollab/bpi2013, and has no partial/full scope.

    predictor selects a key from predictors.dispatch.PREDICTOR_REGISTRY (e.g.
    "random_forest"). Left as None by default so a predictor already set on a
    caller-supplied cfg is not silently overridden by this parameter's default;
    only overrides cfg.predictor when explicitly passed.
    """
    base_cfg = cfg or ExperimentConfig()
    if predictor is not None:
        base_cfg = replace(base_cfg, predictor=predictor)
    results = {}

    for group in log_groups:
        group_cfg = replace(base_cfg, logs=_LOG_GROUPS[group]())
        suffix = "" if group == "predictcollab" else f"_{group}"

        if "RQ2" in rqs:
            print(f"\n########## RQ2 — label fidelity ({group}) ##########")
            results[f"rq2{suffix}"] = run_rq2(group_cfg, out_name=f"rq2_fidelity{suffix}.csv")

        if "RQ3" in rqs:
            for scope in rq3_scopes:
                scope_cfg = replace(group_cfg, rq3_tasks=_RQ3_SCOPES[scope]())
                scope_suffix = "" if scope == "partial" else "_full"
                print(f"\n########## RQ3 — {scope} catalog ({group}) ##########")
                results[f"rq3{suffix}{scope_suffix}"] = run_rq3(
                    scope_cfg,
                    out_name=f"rq3_results_{scope_cfg.predictor}{suffix}{scope_suffix}.csv")

    if "RQ2" in rqs or "RQ3" in rqs:
        print("\n[note] RQ1 (transformation + P1 + schema) is the converter's tool.")

    if run_rq_ext:
        print("\n########## RQ-EXT — object-enabled extensions X-Inf/X-MSt (TOY LOG demo) ##########")
        results["rq_ext_toy"] = run_rq_ext_pipeline(ext_cfg or ExtExperimentConfig())

    return results


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, "-m", "ocpm_eval.run_evaluation"] + sys.argv[1:])
    main()
