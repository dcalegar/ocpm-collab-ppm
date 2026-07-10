"""
Regression tests for the P1.1-P1.3 consistency checks in
``mapping.collab_xes_to_ocel`` (rules M1-M8, checks P1.1-P1.6).

These tests exist because, before this change, P1.1/P1.2/P1.2b compared only
AGGREGATE counts/timestamps, not per-event identity -- so a bug that permuted
activities, swapped events between same-size cases, or reordered two
same-timestamp events would keep every check green. Each test below builds a
tiny well-formed log, transforms it, then deliberately corrupts a COPY of the
transform's output to simulate exactly one such bug and asserts the
corresponding check (and only that one) now fails. A companion test locks in
the counterparty-completeness behaviour (Normalization paragraph, appendix):
a send/receive event whose OWN side is implicit gets backfilled from
collab:participant, but a missing COUNTERPARTY side is left undefined rather
than guessed, and is reported explicitly (not silently) by P1.3.

Run directly: python tests/test_mapping_checks.py   (also pytest-compatible)
"""
import os
import sys
from dataclasses import replace

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mapping.collab_xes_to_ocel import (   # noqa: E402
    transform, run_consistency_checks, MappingConfig, TransformResult,
    COL_EID, COL_ACTIVITY, COL_TIMESTAMP, COL_OID, COL_OTYPE, COL_QUALIFIER,
    Q_WITHIN, OT_CC,
)

CASE_KEY, ACT_KEY, TS_KEY = "case:concept:name", "concept:name", "time:timestamp"
ELEM_KEY, PART_KEY, FROM_KEY, TO_KEY = (
    "collab:elemType", "collab:participant", "collab:fromParticipant", "collab:toParticipant")


def _row(case, act, ts, part, elem="task", frm=None, to=None):
    return {CASE_KEY: case, ACT_KEY: act, TS_KEY: pd.Timestamp(ts, tz="UTC"),
            ELEM_KEY: elem, PART_KEY: part, FROM_KEY: frm, TO_KEY: to}


def _well_formed_log() -> pd.DataFrame:
    """Two cases, four events each, all send/receive endpoints complete."""
    rows = [
        _row("C1", "Start", "2024-01-01T00:00:00Z", "A"),
        _row("C1", "Send1", "2024-01-01T00:00:01Z", "A", "SendTask", frm="A", to="B"),
        _row("C1", "Recv1", "2024-01-01T00:00:02Z", "B", "ReceiveTask", frm="A", to="B"),
        _row("C1", "End", "2024-01-01T00:00:03Z", "B"),
        _row("C2", "Start", "2024-01-02T00:00:00Z", "A"),
        _row("C2", "Send2", "2024-01-02T00:00:01Z", "A", "SendTask", frm="A", to="B"),
        _row("C2", "Recv2", "2024-01-02T00:00:02Z", "B", "ReceiveTask", frm="A", to="B"),
        _row("C2", "End", "2024-01-02T00:00:03Z", "B"),
    ]
    return pd.DataFrame(rows)


def _checks_by_name(checks):
    return {c.name: c for c in checks}


def test_well_formed_log_all_pass():
    src = _well_formed_log()
    res = transform(src, MappingConfig())
    checks = run_consistency_checks(src, res, MappingConfig())
    assert all(c.passed for c in checks), [c for c in checks if not c.passed]


def test_p11_catches_activity_permutation():
    """Swapping the activity of two events (same counts, same timestamps
    overall) must fail P1.1, since it now checks per-event identity."""
    src = _well_formed_log()
    res = transform(src, MappingConfig())
    ev = res.events_df.copy()
    i, j = ev.index[0], ev.index[1]
    ev.loc[i, COL_ACTIVITY], ev.loc[j, COL_ACTIVITY] = (
        ev.loc[j, COL_ACTIVITY], ev.loc[i, COL_ACTIVITY])
    corrupted = replace(res, events_df=ev)
    checks = _checks_by_name(run_consistency_checks(src, corrupted, MappingConfig()))
    assert not checks["P1.1 Totality"].passed
    assert "activity mismatches=2" in checks["P1.1 Totality"].detail


def test_p12_catches_cross_case_swap():
    """Two cases with the same event count: moving one event's 'within'
    edge to the other case's CC object must fail P1.2 (set, not count)."""
    src = _well_formed_log()
    res = transform(src, MappingConfig())
    rel = res.relations_df.copy()
    within_mask = (rel[COL_QUALIFIER] == Q_WITHIN)
    c1_event = rel[within_mask & (rel[COL_OID] == "cc::C1")].iloc[0]
    idx = rel[(rel[COL_EID] == c1_event[COL_EID]) & within_mask].index[0]
    rel.loc[idx, COL_OID] = "cc::C2"   # reassign one C1 event's case to C2
    corrupted = replace(res, relations_df=rel)
    checks = _checks_by_name(run_consistency_checks(src, corrupted, MappingConfig()))
    assert not checks["P1.2 Per-case partition"].passed
    assert "set mismatches=2" in checks["P1.2 Per-case partition"].detail  # both cases now wrong


def test_p12b_catches_unpadded_id_regression():
    """Direct regression test for the exact scenario the fixed-width padding
    in `_event_id` exists to prevent: without zero-padding, the id string
    'e::C1::9' sorts (lexicographically) *before* 'e::C1::10', inverting the
    chronological order those two ids are supposed to encode. Build a
    minimal two-event case with such unpadded ids and confirm P1.2b -- which
    compares id ORDER against the true prec_L sequence, not just whether
    consecutive timestamps happen to be non-decreasing -- flags the
    resulting deviation."""
    src = pd.DataFrame([
        _row("C1", "Ev9", "2024-01-01T00:00:09Z", "A"),
        _row("C1", "Ev10", "2024-01-01T00:00:10Z", "A"),
    ])
    ts9 = pd.Timestamp("2024-01-01T00:00:09Z")
    ts10 = pd.Timestamp("2024-01-01T00:00:10Z")
    # Unpadded ids: "e::C1::10" < "e::C1::9" lexicographically, so ORDER BY
    # eid yields [Ev10, Ev9] -- the inverse of the true chronological order.
    ev = pd.DataFrame([
        {COL_EID: "e::C1::9", COL_ACTIVITY: "Ev9", COL_TIMESTAMP: ts9},
        {COL_EID: "e::C1::10", COL_ACTIVITY: "Ev10", COL_TIMESTAMP: ts10},
    ])
    rel = pd.DataFrame([
        {COL_EID: "e::C1::9", COL_OID: "cc::C1", COL_OTYPE: OT_CC, COL_QUALIFIER: Q_WITHIN},
        {COL_EID: "e::C1::10", COL_OID: "cc::C1", COL_OTYPE: OT_CC, COL_QUALIFIER: Q_WITHIN},
    ])
    obj = pd.DataFrame([{COL_OID: "cc::C1", COL_OTYPE: OT_CC, "caseId": "C1"}])
    o2o = pd.DataFrame(columns=["ocel:oid", "ocel:oid_2", COL_QUALIFIER])
    corrupted = TransformResult(ev, obj, rel, o2o, stats={})
    checks = _checks_by_name(run_consistency_checks(src, corrupted, MappingConfig()))
    assert not checks["P1.2b Per-case order (identifier order equals prec_L, incl. tie-break)"].passed
    detail = checks["P1.2b Per-case order (identifier order equals prec_L, incl. tie-break)"].detail
    assert "deviation from prec_L=1" in detail


def test_p13_reports_missing_counterparty_without_failing():
    """A SendTask whose toParticipant the source never recorded: the OWN
    side (fromParticipant) is backfilled from collab:participant, but the
    COUNTERPARTY side stays undefined -- not guessed -- and P1.3 reports it
    explicitly (not silently) while still passing, since this is a source
    data-completeness fact, not a construction defect."""
    src = _well_formed_log().copy()
    src.loc[src[ACT_KEY] == "Send1", TO_KEY] = None
    res = transform(src, MappingConfig())
    checks = _checks_by_name(run_consistency_checks(src, res, MappingConfig()))
    assert res.stats["n_messages_missing_receiver"] == 1
    assert res.stats["n_messages_missing_sender"] == 0
    p13 = checks["P1.3 Message well-formedness"]
    assert p13.passed
    assert "messages missing receiver (source never recorded 'to'): 1" in p13.detail
    # No 'to' O2O relation should have been fabricated for that message.
    send1_msg = res.objects_df[(res.objects_df["ocel:type"] == "Message")
                               & (res.objects_df["receiver"].isna())]
    assert len(send1_msg) == 1


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n[ok] {len(fns)} mapping consistency-check tests passed.")


if __name__ == "__main__":
    _run_all()
