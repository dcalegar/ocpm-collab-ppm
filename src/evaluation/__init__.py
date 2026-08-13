"""
evaluation — RQ2/RQ3 evaluation stages for the object-centric collaborative PPM study.

Orchestrates the decoupled ``tasks`` (task definitions and labels),
``features`` (OCEL reading and OCPA feature extraction) and ``predictors``
(per-learner fit/score) libraries into the RQ2/RQ3 pipelines. One module per
evaluation stage:
  config          log registry, CV/learner configuration
  rq2_fidelity    RQ2 — label equivalence (vs converter R1)
  rq3_pipeline    RQ3 — end-to-end feasibility, 5-fold CV grouped by CI
  run_evaluation  orchestrator (RQ2/RQ3; RQ1 is the converter's)
"""
__all__ = ["config", "rq2_fidelity", "rq3_pipeline", "run_evaluation"]
