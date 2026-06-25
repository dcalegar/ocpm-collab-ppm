"""
ocpm_tasks — object-centric collaborative prediction-task library.

A standalone, dependency-light library (pandas only at import; pm4py/ocpa are lazy)
that defines the 16 reformulated prediction tasks of the paper and computes their
ground-truth labels over a neutral object-centric model. It is decoupled from the
experimentation: it can be used inside OCPA- or pm4py-based pipelines by building the
model with the adapters and calling the label functions.

  schema      mapping vocabulary (object types/qualifiers/attributes)
  model       neutral object-centric structures (Event/Execution/ObjectCentricLog)
  catalog     the 16 tasks + the RQ2/RQ3 subsets
  labels      label/target functions ℓ + LabelContext + compute_label_rows
  fidelity    RQ2 comparators (equivalence; internal consistency)
  adapters    from_pm4py / from_ocpa / build_from_relations -> model
"""
from . import schema, model, catalog, labels, fidelity, adapters  # noqa: F401

__all__ = ["schema", "model", "catalog", "labels", "fidelity", "adapters"]
