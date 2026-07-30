"""
Label-fidelity comparison for RQ2, independent of any predictor or encoding.

Equivalence (the 14 reformulated prediction tasks): per-prefix agreement between a reference
label source (R1) and the object-centric source (R2). Categorical, boolean and
count targets are compared by exact equality; temporal targets by equality up to
a tolerance. The tolerance is deliberately NOT applied to count targets (problem
type COUNT, e.g. NV-NMPr/NV-NMPa): a tolerant comparison there would treat e.g. 1
and 2 remaining messages as equivalent, which is not a legitimate encoding/rounding
slack the way a sub-second timestamp difference is.

Inputs are label-row lists (case_id, event_id, k, y) as produced by
``labels.compute_label_rows`` with ``drop_bottom=False``; both sides are aligned by
(case_id, k).
"""
from typing import List, Tuple, Dict
from .catalog import Task, COUNT

Row = Tuple[str, str, int, object]


def _index(rows: List[Row]) -> Dict[Tuple[str, int], object]:
    return {(c, k): y for (c, _e, k, y) in rows}


def compare_equivalence(ref_rows: List[Row], obj_rows: List[Row], task: Task,
                        bottom: str = "__BOTTOM__",
                        numeric_tol: float = 1.0,
                        max_examples: int = 5) -> Dict[str, object]:
    a, b = _index(ref_rows), _index(obj_rows)
    keys = set(a) & set(b)
    matches = 0
    exact_matches = 0
    examples = []
    for key in sorted(keys):
        av, bv = a[key], b[key]
        ok = exact_ok = False
        if av == bottom or bv == bottom:
            ok = exact_ok = (av == bv)
        elif task.kind == "numeric" and task.problem_type != COUNT:
            exact_ok = (av == bv)
            try:
                ok = abs(float(av) - float(bv)) <= numeric_tol
            except (TypeError, ValueError):
                ok = exact_ok
        else:
            ok = exact_ok = (av == bv)
        if ok:
            matches += 1
        elif len(examples) < max_examples:
            examples.append({"case_id": key[0], "k": key[1],
                             "reference": av, "object_centric": bv})
        if exact_ok:
            exact_matches += 1
    n = len(keys)
    n_only_ref = len(set(a) - set(b))
    n_only_obj = len(set(b) - set(a))
    return {
        "task": task.key,
        "check": "equivalence",
        "eval_labels": n,
        "matches": matches,
        "mismatches": n - matches,
        # Agreement over the INTERSECTION only: a row present on one side and
        # absent on the other never counts against it. agreement == 1.0 is
        # therefore compatible with a non-empty only_in_reference/
        # only_in_object_centric -- it demonstrates equivalence on the labels
        # both sides produced, not that both sides produced the same set of
        # labels. Use `full_equivalence` (below) for the stronger claim.
        "agreement": (matches / n) if n else float("nan"),
        # Equality up to `numeric_tol` for time-valued tasks (serialization
        # slack), never for COUNT tasks (see module docstring). This proves
        # APPROXIMATE equivalence; `exact_agreement` is the same comparison
        # with tol=0, i.e. genuine label equality, kept separate rather than
        # conflated into one number.
        "exact_agreement": (exact_matches / n) if n else float("nan"),
        "only_in_reference": n_only_ref,
        "only_in_object_centric": n_only_obj,
        # The claim "R1 and R2 are equivalent" requires BOTH: every evaluated
        # label matches AND neither side has a label the other lacks. Reading
        # only `agreement == 1.0` cannot establish this by itself.
        "full_equivalence": (n > 0 and matches == n
                            and n_only_ref == 0 and n_only_obj == 0),
        "examples": examples,
    }
