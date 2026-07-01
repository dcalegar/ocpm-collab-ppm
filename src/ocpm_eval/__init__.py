"""
ocpm_eval — evaluation stages for the object-centric collaborative PPM study.

Uses the decoupled ``ocpm_tasks`` library for task definitions and labels. One module
per evaluation stage:
  config          log registry, CV/learner configuration
  io_ocel         read OCEL 2.0 SQLite into the neutral model (stdlib sqlite3)
  features_ocpa   native OCPA features + self-verified alignment (RQ3)
  models          per-fold training + descriptive metrics
  rq2_fidelity    RQ2 — label equivalence (vs converter R1) + X-* consistency
  rq3_pipeline    RQ3 — end-to-end feasibility, 5-fold CV grouped by CI
  run_evaluation  orchestrator (RQ2/RQ3; RQ1 is the converter's)
"""
__all__ = ["config", "io_ocel", "features_ocpa", "models",
           "rq2_fidelity", "rq3_pipeline", "run_evaluation"]
