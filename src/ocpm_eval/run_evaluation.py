"""
Evaluation orchestrator. Runs the stages owned by the experimentation tool:
  * RQ2 — label fidelity (equivalence needs the converter's R1 reader; consistency for
          X-PaL/X-MSt runs from R2 alone),
  * RQ3 — end-to-end feasibility + descriptive metrics on the representative subset,
  * RQ4 — structural measures + expressiveness matrix.

RQ1 (XES->OCEL transformation, properties P1.1-P1.5, OCEL 2.0 schema validation) is
produced by the CONVERTER (a separate tool) and is therefore out of scope here.

Usage: python -m ocpm_eval.run_evaluation   (adjust paths in config.py)
"""
from typing import Optional, Dict
from .config import ExperimentConfig
from .rq2_fidelity import run_rq2
from .rq3_pipeline import run_rq3
from .rq4_structure import run_rq4_structure


def main(cfg: Optional[ExperimentConfig] = None, r1_logs: Optional[Dict] = None):
    cfg = cfg or ExperimentConfig()
    results = {}
    print("\n########## RQ2 — label fidelity ##########")
    results["rq2"] = run_rq2(cfg, r1_logs)
    print("\n########## RQ3 — end-to-end feasibility ##########")
    results["rq3"] = run_rq3(cfg)
    print("\n########## RQ4 — structural measures ##########")
    results["rq4"] = run_rq4_structure(cfg)
    print("\n[note] RQ1 (transformation + P1 + schema) is the converter's tool.")
    return results


if __name__ == "__main__":
    main()
