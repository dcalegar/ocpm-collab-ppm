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
    COL_EID, COL_ACTIVITY, COL_TIMESTAMP, COL_OID, COL_OID2, COL_OTYPE, COL_QUALIFIER,
    Q_WITHIN, Q_FROM, Q_SEND, Q_PARTICIPANT, Q_IN_PROJECTION, Q_EXCHANGED_IN,
    OT_CC, OT_MESSAGE, OT_PP,
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


def test_p13_catches_deleted_message_object():
    """Deleting a Message object and ALL its relations (not merely leaving
    one attribute partial) must fail P1.3: the send/receive EVENT is still
    there, but nothing relates it to any Message at all. The checks above
    only ever iterate the Message objects that still exist, so this needs
    a separate coverage check from the event's side."""
    src = _well_formed_log()
    res = transform(src, MappingConfig())
    obj = res.objects_df
    msg_oid = obj[obj[COL_OTYPE] == OT_MESSAGE][COL_OID].iloc[0]
    obj2 = obj[obj[COL_OID] != msg_oid].reset_index(drop=True)
    rel2 = res.relations_df[res.relations_df[COL_OID] != msg_oid].reset_index(drop=True)
    o2o2 = res.o2o_df[(res.o2o_df[COL_OID] != msg_oid)
                      & (res.o2o_df[COL_OID2] != msg_oid)].reset_index(drop=True)
    corrupted = replace(res, objects_df=obj2, relations_df=rel2, o2o_df=o2o2)
    checks = _checks_by_name(run_consistency_checks(src, corrupted, MappingConfig()))
    p13 = checks["P1.3 Message well-formedness"]
    assert not p13.passed
    assert "send/receive events with no Message relation: 1" in p13.detail


def test_p13_catches_from_multiplicity_violation():
    """A Message with TWO 'from' O2O edges, one of them contradictory, must
    fail P1.3 -- not silently pass by only ever inspecting the first edge
    found (froms[0])."""
    src = _well_formed_log()
    res = transform(src, MappingConfig())
    obj = res.objects_df
    msg_oid = obj[(obj[COL_OTYPE] == OT_MESSAGE) & (obj["sender"] == "A")][COL_OID].iloc[0]
    extra = pd.DataFrame([{COL_OID: msg_oid, COL_OID2: "part::B", COL_QUALIFIER: Q_FROM}])
    o2o2 = pd.concat([res.o2o_df, extra], ignore_index=True)
    corrupted = replace(res, o2o_df=o2o2)
    checks = _checks_by_name(run_consistency_checks(src, corrupted, MappingConfig()))
    p13 = checks["P1.3 Message well-formedness"]
    assert not p13.passed
    assert "multiplicity violations (>1 edge on one side): 1" in p13.detail


def test_p13_catches_stripped_event_attribute():
    """The Message's 'sender' object attribute is defined, but the event's
    preserved fromParticipant attribute is corrupted to missing: M8
    guarantees these track each other's definedness, so this must fail
    P1.3, not be silently skipped by only comparing when both happen to be
    present."""
    src = _well_formed_log()
    res = transform(src, MappingConfig())
    ev = res.events_df.copy()
    idx = ev[ev[COL_ACTIVITY] == "Send1"].index[0]
    ev.loc[idx, "fromParticipant"] = None
    corrupted = replace(res, events_df=ev)
    checks = _checks_by_name(run_consistency_checks(src, corrupted, MappingConfig()))
    p13 = checks["P1.3 Message well-formedness"]
    assert not p13.passed
    assert "inconsistent partial state (attribute/relation disagree on definedness): 1" in p13.detail


def test_p13_catches_duplicate_message_on_same_event():
    """TWO Message objects, each with its own single 'send' edge, hanging off
    the SAME send event: the XOR check (edges per MESSAGE) passes for both,
    and set-based coverage sees the event as covered, so only the per-event
    edge count catches it. P1.3 must fail."""
    src = _well_formed_log()
    res = transform(src, MappingConfig())
    obj = res.objects_df.copy()
    rel = res.relations_df.copy()
    orig = obj[obj[COL_OTYPE] == OT_MESSAGE].iloc[0]
    dup = orig.copy()
    dup[COL_OID] = str(orig[COL_OID]) + "_dup"
    obj.loc[len(obj)] = dup
    edge = rel[(rel[COL_OID] == orig[COL_OID]) & (rel[COL_QUALIFIER] == Q_SEND)].iloc[0].copy()
    edge[COL_OID] = dup[COL_OID]
    rel.loc[len(rel)] = edge
    extra_o2o = res.o2o_df[res.o2o_df[COL_OID] == orig[COL_OID]].copy()
    extra_o2o[COL_OID] = dup[COL_OID]
    corrupted = replace(res, objects_df=obj, relations_df=rel,
                        o2o_df=pd.concat([res.o2o_df, extra_o2o], ignore_index=True))
    checks = _checks_by_name(run_consistency_checks(src, corrupted, MappingConfig()))
    p13 = checks["P1.3 Message well-formedness"]
    assert not p13.passed
    assert "events related to more than one Message: 1" in p13.detail


def test_p13_and_p11_catch_elemtype_flip_with_message_dropped():
    """Flip a send event's elemType to 'task' in the OUTPUT and drop its
    Message object with all relations. If the set of communication events
    were read off the output's own elemType column, the corrupted output
    would vouch for itself and every check would pass; deriving it from the
    SOURCE makes P1.3 fail (uncovered comm event), and the per-event
    attribute comparison makes P1.1 fail (elemType mismatch)."""
    src = _well_formed_log()
    res = transform(src, MappingConfig())
    rel = res.relations_df
    victim = rel[(rel[COL_OTYPE] == OT_MESSAGE) & (rel[COL_QUALIFIER] == Q_SEND)].iloc[0]
    ev = res.events_df.copy()
    ev.loc[ev[COL_EID] == victim[COL_EID], "elemType"] = "task"
    corrupted = replace(
        res, events_df=ev,
        objects_df=res.objects_df[res.objects_df[COL_OID] != victim[COL_OID]],
        relations_df=rel[rel[COL_OID] != victim[COL_OID]],
        o2o_df=res.o2o_df[res.o2o_df[COL_OID] != victim[COL_OID]])
    checks = _checks_by_name(run_consistency_checks(src, corrupted, MappingConfig()))
    assert not checks["P1.1 Totality"].passed
    assert "elemType mismatches=1" in checks["P1.1 Totality"].detail
    p13 = checks["P1.3 Message well-formedness"]
    assert not p13.passed
    assert "send/receive events with no Message relation: 1" in p13.detail


def test_p11_catches_stripped_participant_attribute():
    """Stripping the preserved collab:participant attribute from the output
    events must fail P1.1 (part is total in the source, Definition 1, and
    M8 preserves it), not be silently skipped by comparisons that only run
    when the attribute happens to be present."""
    src = _well_formed_log()
    res = transform(src, MappingConfig())
    ev = res.events_df.copy()
    ev["participant"] = None
    corrupted = replace(res, events_df=ev)
    checks = _checks_by_name(run_consistency_checks(src, corrupted, MappingConfig()))
    p11 = checks["P1.1 Totality"]
    assert not p11.passed
    assert "participant mismatches=8" in p11.detail
    # P1.3 also reports (though P1.1 is the failing check) the comm events
    # that lost their participant attribute.
    assert "communication events missing their participant attribute" \
        in checks["P1.3 Message well-formedness"].detail


def test_p11_catches_stripped_elemtype_on_task_event():
    """Stripping the preserved collab:elemType attribute from a plain 'task'
    event must fail P1.1: the transform always materializes an explicit
    elemType (normalized to 'task', M5/M8), so its absence is a preservation
    defect -- it must not be silently equated with 'task' by a fallback."""
    src = _well_formed_log()
    res = transform(src, MappingConfig())
    ev = res.events_df.copy()
    idx = ev[ev[COL_ACTIVITY] == "Start"].index[0]  # a task event
    ev.loc[idx, "elemType"] = None
    corrupted = replace(res, events_df=ev)
    checks = _checks_by_name(run_consistency_checks(src, corrupted, MappingConfig()))
    p11 = checks["P1.1 Totality"]
    assert not p11.passed
    assert "elemType mismatches=1" in p11.detail


def test_p13_catches_send_edge_flipped_to_receive_when_receiver_undefined():
    """Flip a send event's E2O qualifier from 'send' to 'receive' on a
    Message whose receiver the source never recorded. The XOR, coverage,
    and per-event counts all still hold, and the participant/sender
    comparison is inconclusive (the receiver side is undefined), so only
    the direction check -- edge qualifier vs source elemType -- catches
    it. P1.3 must fail."""
    src = _well_formed_log().copy()
    src.loc[src[ACT_KEY] == "Send1", TO_KEY] = None  # receiver unrecorded
    res = transform(src, MappingConfig())
    rel = res.relations_df.copy()
    obj = res.objects_df
    victim_msg = obj[(obj[COL_OTYPE] == OT_MESSAGE) & (obj["receiver"].isna())][COL_OID].iloc[0]
    mask = (rel[COL_OID] == victim_msg) & (rel[COL_QUALIFIER] == Q_SEND)
    assert mask.sum() == 1
    rel.loc[mask, COL_QUALIFIER] = "receive"
    corrupted = replace(res, relations_df=rel)
    checks = _checks_by_name(run_consistency_checks(src, corrupted, MappingConfig()))
    p13 = checks["P1.3 Message well-formedness"]
    assert not p13.passed
    assert "send/receive qualifier contradicts the source elemType: 1" in p13.detail


# ---------------------------------------------------------------------
# D24: the five confirmed false-positive scenarios (P1 all-PASS on
# structurally corrupted output). Each test reproduces exactly one of
# them and asserts the hardened check now fails, with the specific
# counter that must have caught it.
# ---------------------------------------------------------------------

def test_p16_catches_simultaneous_participant_and_inprojection_removal():
    """Deleting BOTH the direct 'participant' edge and the 'in_projection'
    edge for one event. Before D24, P1.6's universe of checked events was
    the union of eids that still had at least one of the two edges, so
    removing both together removed that event from the universe entirely
    instead of being caught by 'missing one side'."""
    src = _well_formed_log()
    res = transform(src, MappingConfig())
    rel = res.relations_df.copy()
    eid = rel[rel[COL_QUALIFIER] == Q_PARTICIPANT][COL_EID].iloc[0]
    rel2 = rel[~((rel[COL_EID] == eid)
                & (rel[COL_QUALIFIER].isin([Q_PARTICIPANT, Q_IN_PROJECTION])))]
    corrupted = replace(res, relations_df=rel2)
    checks = _checks_by_name(run_consistency_checks(src, corrupted, MappingConfig()))
    p16 = checks["P1.6 Participant coherence"]
    assert not p16.passed
    assert "missing one side=1" in p16.detail


def test_p15_catches_dangling_relation_target():
    """A relation edge whose target object id was never materialized (e.g.
    left over after a partial deletion): distinct from an orphan object
    (P1.5's original statement). Before D24, no check verified that E2O/O2O
    TARGETS actually resolve to an existing object."""
    src = _well_formed_log()
    res = transform(src, MappingConfig())
    rel = res.relations_df.copy()
    send_mask = (rel[COL_OTYPE] == OT_MESSAGE) & (rel[COL_QUALIFIER] == Q_SEND)
    idx = rel[send_mask].index[0]
    rel.loc[idx, COL_OID] = "msg::DOES_NOT_EXIST"
    corrupted = replace(res, relations_df=rel)
    checks = _checks_by_name(run_consistency_checks(src, corrupted, MappingConfig()))
    p15 = checks["P1.5 No orphan objects"]
    assert not p15.passed
    assert "dangling E2O object targets=1" in p15.detail


def test_p13_catches_corrupted_exchanged_in():
    """A Message's 'exchanged_in' O2O edge retargeted to a different but
    EXISTING CollaborationCase. The Message keeps its send/receive edge and
    its from/to endpoints, so it is not an orphan, and before D24 nothing
    checked the exchanged_in target at all."""
    src = _well_formed_log()
    res = transform(src, MappingConfig())
    o2o = res.o2o_df.copy()
    msg_oid = res.objects_df[res.objects_df[COL_OTYPE] == OT_MESSAGE][COL_OID].iloc[0]
    mask = (o2o[COL_OID] == msg_oid) & (o2o[COL_QUALIFIER] == Q_EXCHANGED_IN)
    assert mask.sum() == 1
    o2o.loc[mask, COL_OID2] = "cc::C2"  # wrong, but an existing CC
    corrupted = replace(res, o2o_df=o2o)
    checks = _checks_by_name(run_consistency_checks(src, corrupted, MappingConfig()))
    p13 = checks["P1.3 Message well-formedness"]
    assert not p13.passed
    assert "exchanged_in disagreements with the related event's case: 1" in p13.detail


def test_p11_catches_dropped_residual_attribute():
    """A residual (M8) source attribute -- preserved verbatim, not consumed
    by M1-M8 -- silently dropped from the OUTPUT event. Before D24, no
    check compared residual attributes at all, only the fixed structural
    ones (activity/timestamp/participant/elemType)."""
    src = _well_formed_log().copy()
    src.loc[src[ACT_KEY] == "Send1", "msgInstanceId"] = "prescription_1"
    res = transform(src, MappingConfig())
    ev = res.events_df.copy()
    idx = ev[ev[COL_ACTIVITY] == "Send1"].index[0]
    assert ev.loc[idx, "msgInstanceId"] == "prescription_1"
    ev.loc[idx, "msgInstanceId"] = None
    corrupted = replace(res, events_df=ev)
    checks = _checks_by_name(run_consistency_checks(src, corrupted, MappingConfig()))
    p11 = checks["P1.1 Totality"]
    assert not p11.passed
    assert "residual (M8) attribute mismatches=1" in p11.detail


def test_p12_and_p14_catch_altered_caseid():
    """Corrupting the caseId attribute of a CollaborationCase object (P1.2)
    or a ParticipantProjection object (P1.4). caseId affects no relation,
    so every edge/set-based check stays green; before D24 this was
    invisible to all six checks."""
    src = _well_formed_log()
    res = transform(src, MappingConfig())
    obj = res.objects_df.copy()
    cc_idx = obj[(obj[COL_OTYPE] == OT_CC) & (obj[COL_OID] == "cc::C1")].index[0]
    obj.loc[cc_idx, "caseId"] = "C2"
    pp_idx = obj[obj[COL_OTYPE] == OT_PP].index[0]
    obj.loc[pp_idx, "caseId"] = "WRONG"
    corrupted = replace(res, objects_df=obj)
    checks = _checks_by_name(run_consistency_checks(src, corrupted, MappingConfig()))
    p12 = checks["P1.2 Per-case partition"]
    p14 = checks["P1.4 Participant-projection coherence"]
    assert not p12.passed
    assert "caseId disagreements=1" in p12.detail
    assert not p14.passed
    # The corrupted PP object is shared by both C1 events with participant
    # "A" (Start and Send1), so the source-driven identity check flags it
    # once per event that references it.
    assert "caseId/participant identity disagreements=2" in p14.detail


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n[ok] {len(fns)} mapping consistency-check tests passed.")


if __name__ == "__main__":
    _run_all()
