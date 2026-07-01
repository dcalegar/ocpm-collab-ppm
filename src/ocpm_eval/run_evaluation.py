"""
Evaluation orchestrator. Runs the stages owned by the experimentation tool:
  * RQ2 — label fidelity (equivalence: R1 and R2 both read from the OCEL logs;
          consistency for X-PaL/X-MSt),
  * RQ3 — end-to-end feasibility + descriptive metrics on the representative subset,
  * RQ4 — structural measures + expressiveness matrix.

RQ1 (XES->OCEL transformation, properties P1.1-P1.5, OCEL 2.0 schema validation) is
produced by the CONVERTER (a separate tool) and is therefore out of scope here.

RQ3 is run twice: once on the representative subset (paper Table tab:rq3subset,
`rq3_results.csv`) and once on the full catalog of single-case-counterpart tasks
(`rq3_results_full.csv`), intended as supplementary material rather than an
in-paper table. The two object-enabled extensions (X-PaL, X-MSt) are demonstrated
separately in `rq3_extensions_example.py`.

Usage: python -m ocpm_eval.run_evaluation   (adjust paths in config.py)
"""
from dataclasses import replace
from typing import Optional
from ocpm_tasks.catalog import EQUIVALENCE_TASKS
from .config import ExperimentConfig
from .rq2_fidelity import run_rq2
from .rq3_pipeline import run_rq3
from .rq4_structure import run_rq4_structure


def main(cfg: Optional[ExperimentConfig] = None):
    cfg = cfg or ExperimentConfig()
    results = {}
    print("\n########## RQ2 — label fidelity ##########")
    results["rq2"] = run_rq2(cfg)
    print("\n########## RQ3 — end-to-end feasibility (representative subset) ##########")
    results["rq3"] = run_rq3(cfg)
    print("\n########## RQ3 — full catalog (supplementary coverage) ##########")
    full_cfg = replace(cfg, rq3_tasks=list(EQUIVALENCE_TASKS))
    results["rq3_full"] = run_rq3(full_cfg, out_name="rq3_results_full.csv")
    print("\n########## RQ4 — structural measures ##########")
    results["rq4"] = run_rq4_structure(cfg)
    print("\n[note] RQ1 (transformation + P1 + schema) is the converter's tool.")
    print("[note] X-PaL/X-MSt (object-enabled extensions) are demonstrated in "
          "rq3_extensions_example.py, not included above.")
    return results


if __name__ == "__main__":
    main()
