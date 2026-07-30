"""
Regression tests for ``ocpm_tasks.fidelity.compare_equivalence``.

Before this change, ``agreement`` was computed only over the KEY
INTERSECTION of the reference and object-centric label rows, with
``only_in_reference``/``only_in_object_centric`` reported but never folded
in: an ``agreement == 1.0`` result was therefore compatible with one side
missing rows the other has, which does not establish full equivalence. The
numeric tolerance (default 1.0s) also collapsed "equal up to slack" and
"exactly equal" into one number. These tests lock in the two additional
fields (`full_equivalence`, `exact_agreement`) that make both distinctions
explicit.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ocpm_tasks.catalog import TASKS               # noqa: E402
from ocpm_tasks.fidelity import compare_equivalence  # noqa: E402

BOTTOM = "__BOTTOM__"


def test_agreement_1_with_missing_rows_is_not_full_equivalence():
    """Every row the two sides share agrees, but the object-centric side is
    missing one row the reference has: agreement == 1.0, yet this is not full
    equivalence (there is a label one side never produced)."""
    task = TASKS["NV-PrT"]   # numeric, non-COUNT
    ref_rows = [("c1", "e1", 1, 10.0), ("c1", "e2", 2, 20.0)]
    obj_rows = [("c1", "e1", 1, 10.0)]
    res = compare_equivalence(ref_rows, obj_rows, task)
    assert res["agreement"] == 1.0
    assert res["only_in_reference"] == 1
    assert res["only_in_object_centric"] == 0
    assert not res["full_equivalence"]


def test_full_equivalence_true_when_rows_and_values_match_exactly():
    task = TASKS["NE-NEPr"]   # categorical
    rows = [("c1", "e1", 1, "A"), ("c1", "e2", 2, "B")]
    res = compare_equivalence(rows, rows, task)
    assert res["agreement"] == 1.0
    assert res["only_in_reference"] == 0
    assert res["only_in_object_centric"] == 0
    assert res["full_equivalence"]
    assert res["exact_agreement"] == 1.0


def test_exact_agreement_separates_tolerant_from_exact_match():
    """A time-valued task with a sub-tolerance difference (0.5s <= default
    1.0s tol) counts toward `agreement` but not `exact_agreement`."""
    task = TASKS["NV-TNE"]   # numeric, non-COUNT (time)
    ref_rows = [("c1", "e1", 1, 10.0)]
    obj_rows = [("c1", "e1", 1, 10.5)]
    res = compare_equivalence(ref_rows, obj_rows, task)
    assert res["agreement"] == 1.0
    assert res["exact_agreement"] == 0.0
    assert res["full_equivalence"]   # tolerant equivalence still holds


def test_count_task_never_gets_numeric_tolerance():
    """COUNT tasks (e.g. NV-NMPr) are never tolerant: a difference of 1 is a
    mismatch for both `agreement` and `exact_agreement`."""
    task = TASKS["NV-NMPr"]
    ref_rows = [("c1", "e1", 1, 1)]
    obj_rows = [("c1", "e1", 1, 2)]
    res = compare_equivalence(ref_rows, obj_rows, task)
    assert res["agreement"] == 0.0
    assert res["exact_agreement"] == 0.0
    assert not res["full_equivalence"]


def test_bottom_only_matches_bottom():
    task = TASKS["NE-NEPr"]
    ref_rows = [("c1", "e1", 1, BOTTOM)]
    obj_rows = [("c1", "e1", 1, "A")]
    res = compare_equivalence(ref_rows, obj_rows, task, bottom=BOTTOM)
    assert res["agreement"] == 0.0
    assert res["exact_agreement"] == 0.0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
