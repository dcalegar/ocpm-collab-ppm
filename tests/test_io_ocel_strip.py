"""
Regression test for ``features.io_ocel._strip_participant_e2o``.

Context: OCPA's leading-type execution extraction connects ALL E2O-related
objects of an event pairwise, regardless of qualifier or type. A log-wide
object (Participant, M2 scope) reached from events of two DIFFERENT
CollaborationCases therefore merges those cases into one OCPA process
execution. The direct `in_participant` E2O edge (M6) was already stripped down
to one witness row per object for this reason; this test locks in that the
same treatment now also covers the D25 export-reachability witness edges
(qualifier `from`/`to`, reused at the E2O level for a Participant that is
only ever a message counterparty and never itself `collab:participant` of an
event -- mapping.collab_xes_to_ocel._add_export_reachability_witnesses).
Before this fix, such a Participant could carry one witness edge per
referencing event, including events of different cases, silently
reintroducing the exact cross-case merge that D25's own docstring set out to
avoid on the export side.

Pure stdlib sqlite3, no pm4py/OCPA dependency: builds a minimal
``event_object`` table directly, since that is the only table
``_strip_participant_e2o`` reads or writes.
"""
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features.io_ocel import _strip_participant_e2o  # noqa: E402
from ocpm_tasks.schema import Schema  # noqa: E402


def _build_event_object_db(rows):
    fd, path = __import__("tempfile").mkstemp(suffix=".sqlite")
    os.close(fd)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE event_object (ocel_event_id TEXT, ocel_object_id TEXT, "
                "ocel_qualifier TEXT)")
    con.executemany("INSERT INTO event_object VALUES (?,?,?)", rows)
    con.commit()
    con.close()
    return path


def test_strip_removes_cross_case_from_to_witness_edges():
    # Participant "B" is an endpoint-only counterparty (D25 witness, qualifier
    # `to`) referenced from one send event in C1 and another in C2 -- exactly
    # the pattern that would let OCPA's leading-type extraction merge C1/C2.
    rows = [
        ("e::C1::01", "part::B", "to"),
        ("e::C2::00", "part::B", "to"),
        ("e::C1::00", "part::A", "in_participant"),
    ]
    path = _build_event_object_db(rows)
    out = _strip_participant_e2o(path, Schema())
    assert out != path  # a copy was made because there was something to strip

    con = sqlite3.connect(out)
    b_rows = con.execute(
        "SELECT ocel_event_id FROM event_object WHERE ocel_object_id = 'part::B'"
    ).fetchall()
    con.close()
    assert len(b_rows) == 1, f"expected exactly one surviving witness row for B, got {b_rows}"


def test_strip_is_noop_without_participant_or_witness_edges():
    rows = [("e::C1::00", "msg::e::C1::00", "send")]
    path = _build_event_object_db(rows)
    out = _strip_participant_e2o(path, Schema())
    assert out == path  # nothing to strip -> original path returned unchanged


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n[ok] {len(fns)} io_ocel strip tests passed.")


if __name__ == "__main__":
    _run_all()
