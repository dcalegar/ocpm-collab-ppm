"""
tasks — object-centric collaborative prediction-task library.

A standalone, dependency-light library that defines the 14 reformulated prediction
tasks and computes their ground-truth labels over a neutral
object-centric model. It is decoupled from the experimentation: it can be used inside
OCPA- or pm4py-based pipelines by building the model with the adapters and calling the
label functions.

The pure task logic (schema/model/catalog/labels/fidelity) has NO third-party
dependency and is imported eagerly. Only `adapters` needs pandas (and, for the
corresponding adapter, pm4py/ocpa), so it is imported lazily on first attribute
access -- the task definitions, the label functions and the RQ2 comparator all
remain usable where none of those are installed.

  schema      mapping vocabulary (object types/qualifiers/attributes)
  model       neutral object-centric structures (Event/Execution/ObjectCentricLog)
  catalog     the 14 tasks + the RQ2/RQ3 subsets
  labels      label/target functions ℓ + LabelContext + compute_label_rows
  fidelity    RQ2 comparator (label equivalence)
  adapters    from_pm4py / from_ocpa / build_from_relations -> model  [lazy: needs pandas]
"""
from . import schema, model, catalog, labels, fidelity  # noqa: F401  (pure, no pandas)

__all__ = ["schema", "model", "catalog", "labels", "fidelity", "adapters"]

_LAZY = {"adapters"}


def __getattr__(name):
    # PEP 562: defer the pandas-dependent submodules so importing tasks (and using
    # labels/extensions) does not require pandas/pm4py/ocpa to be importable.
    if name in _LAZY:
        import importlib
        mod = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
