"""
Evaluation orchestrator. Runs the stages owned by the experimentation tool:
  * RQ2 — label fidelity (equivalence: R1 and R2 both read from the OCEL logs),
  * RQ3 — end-to-end feasibility + descriptive metrics on the representative subset,

RQ1 (XES->OCEL transformation, properties P1.1-P1.5, OCEL 2.0 schema validation) is
produced by the CONVERTER (a separate tool) and is therefore out of scope here.

RQ3 is run twice: once on the representative subset (paper Table tab:rq3subset,
`rq3_results.csv`) and once on the full catalog of 14 tasks (`rq3_results_full.csv`),
intended as supplementary material rather than an in-paper table.

RQ-EXT (object-enabled EXTENSION tasks X-Inf, X-MSt; ocpm_tasks/extensions.py) is run
separately, on a small hand-built toy log (data/logs/toy_collab.*, see
mapping/aux/build_toy_collab_log.py) with explicit send/receive pairs and msgId
correlation -- NOT on the four study logs, and its results (`rq_ext_results_toy.csv`)
are never combined with rq2/rq3. This is a feasibility demo of the object-enabled
targets outlined in tasks.tex/discussion.tex, not part of the paper's evaluated
RQ2/RQ3 results. The toy log is designed to exercise both X-Inf (variable
in-flight backlog) and X-MSt (message synchronization time with correlation).
Remaining extensions (X-PaL, X-Cmp, X-Lag) are not implemented here.

Usage: python -m ocpm_eval.run_evaluation   (adjust paths in config.py)
"""
from dataclasses import replace
from typing import Optional
from ocpm_tasks.catalog import EQUIVALENCE_TASKS
from .config import ExperimentConfig, ExtExperimentConfig
from .rq2_fidelity import run_rq2
from .rq3_pipeline import run_rq3
from .rq_ext_pipeline import run_rq_ext


def main(cfg: Optional[ExperimentConfig] = None,
        ext_cfg: Optional[ExtExperimentConfig] = None):
    cfg = cfg or ExperimentConfig()
    results = {}
    print("\n########## RQ2 — label fidelity ##########")
    results["rq2"] = run_rq2(cfg)
    print("\n########## RQ3 — end-to-end feasibility (representative subset) ##########")
    results["rq3"] = run_rq3(cfg)
    print("\n########## RQ3 — full catalog (supplementary coverage) ##########")
    full_cfg = replace(cfg, rq3_tasks=list(EQUIVALENCE_TASKS))
    results["rq3_full"] = run_rq3(full_cfg, out_name="rq3_results_full.csv")
    print("\n[note] RQ1 (transformation + P1 + schema) is the converter's tool.")
    print("\n########## RQ-EXT — object-enabled extensions X-Inf/X-MSt (TOY LOG demo) ##########")
    results["rq_ext_toy"] = run_rq_ext(ext_cfg or ExtExperimentConfig())
    return results


if __name__ == "__main__":
    main()
