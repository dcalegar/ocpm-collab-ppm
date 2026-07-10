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
    examples = []
    for key in sorted(keys):
        av, bv = a[key], b[key]
        ok = False
        if av == bottom or bv == bottom:
            ok = (av == bv)
        elif task.kind == "numeric" and task.problem_type != COUNT:
            try:
                ok = abs(float(av) - float(bv)) <= numeric_tol
            except (TypeError, ValueError):
                ok = (av == bv)
        else:
            ok = (av == bv)
        if ok:
            matches += 1
        elif len(examples) < max_examples:
            examples.append({"case_id": key[0], "k": key[1],
                             "reference": av, "object_centric": bv})
    n = len(keys)
    return {
        "task": task.key,
        "check": "equivalence",
        "eval_labels": n,
        "matches": matches,
        "mismatches": n - matches,
        "agreement": (matches / n) if n else float("nan"),
        "only_in_reference": len(set(a) - set(b)),
        "only_in_object_centric": len(set(b) - set(a)),
        "examples": examples,
    }
