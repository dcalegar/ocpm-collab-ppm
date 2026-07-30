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
import tempfile
from dataclasses import replace

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mapping.collab_xes_to_ocel import (   # noqa: E402
    transform, run_consistency_checks, MappingConfig, TransformResult,
    _add_export_reachability_witnesses, _raw_timestamps_in_file_order,
    COL_EID, COL_ACTIVITY, COL_TIMESTAMP, COL_OID, COL_OID2, COL_OTYPE, COL_QUALIFIER,
    Q_WITHIN, Q_FROM, Q_TO, Q_SEND, Q_PARTICIPANT, Q_IN_ORCHESTRATION, Q_EXCHANGED_IN,
    OT_CC, OT_MESSAGE, OT_OC,
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
    """Deleting BOTH the direct 'participant' edge and the 'in_orchestration'
    edge for one event. Before D24, P1.6's universe of checked events was
    the union of eids that still had at least one of the two edges, so
    removing both together removed that event from the universe entirely
    instead of being caught by 'missing one side'."""
    src = _well_formed_log()
    res = transform(src, MappingConfig())
    rel = res.relations_df.copy()
    eid = rel[rel[COL_QUALIFIER] == Q_PARTICIPANT][COL_EID].iloc[0]
    rel2 = rel[~((rel[COL_EID] == eid)
                & (rel[COL_QUALIFIER].isin([Q_PARTICIPANT, Q_IN_ORCHESTRATION])))]
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
    or an OrchestrationCase object (P1.4). caseId affects no relation,
    so every edge/set-based check stays green; before D24 this was
    invisible to all six checks."""
    src = _well_formed_log()
    res = transform(src, MappingConfig())
    obj = res.objects_df.copy()
    cc_idx = obj[(obj[COL_OTYPE] == OT_CC) & (obj[COL_OID] == "cc::C1")].index[0]
    obj.loc[cc_idx, "caseId"] = "C2"
    oc_idx = obj[obj[COL_OTYPE] == OT_OC].index[0]
    obj.loc[oc_idx, "caseId"] = "WRONG"
    corrupted = replace(res, objects_df=obj)
    checks = _checks_by_name(run_consistency_checks(src, corrupted, MappingConfig()))
    p12 = checks["P1.2 Per-case partition"]
    p14 = checks["P1.4 OrchestrationCase coherence"]
    assert not p12.passed
    assert "caseId disagreements=1" in p12.detail
    assert not p14.passed
    # The corrupted OC object is shared by both C1 events with participant
    # "A" (Start and Send1), so the source-driven identity check flags it
    # once per event that references it.
    assert "caseId/participant identity disagreements=2" in p14.detail


def test_export_witness_recovers_endpoint_only_participant():
    """D25: a participant that is only ever a message counterparty (never
    collab:participant of its own event) is correctly materialized by
    transform() -- Participant B, plus the O2O `to` edge from the Message to
    it -- but M6's direct `participant` E2O edge only covers participants
    that own at least one event, so B is reachable via relations_df (E2O)
    not at all. Before D25, exporting such a log with pm4py's OCEL 2.0
    writers silently dropped B, because they discard every object absent
    from the E2O table (filtering_utils.propagate_relations_filtering),
    regardless of O2O reachability -- confirmed by reading the installed
    pm4py source, not just the module docstring's claim. This locks in the
    export-time fix without needing pm4py installed:
    _add_export_reachability_witnesses augments a COPY of relations_df with
    a witness edge anchored on the Send event, while leaving
    res.relations_df (P1's input) untouched.
    """
    rows = [
        _row("C1", "Start", "2024-01-01T00:00:00Z", "A"),
        _row("C1", "Send1", "2024-01-01T00:00:01Z", "A", "SendTask", frm="A", to="B"),
        _row("C1", "End", "2024-01-01T00:00:02Z", "A"),
    ]
    src = pd.DataFrame(rows)
    res = transform(src, MappingConfig())

    b_oid = "part::B"
    assert b_oid in set(res.objects_df[COL_OID])         # M2/M7: B is materialized
    assert b_oid not in set(res.relations_df[COL_OID])   # but unreachable via E2O

    checks = _checks_by_name(run_consistency_checks(src, res, MappingConfig()))
    assert all(c.passed for c in checks.values())  # P1 is unaffected -- this is an export-only gap

    augmented = _add_export_reachability_witnesses(res.objects_df, res.relations_df, res.o2o_df)
    assert b_oid not in set(res.relations_df[COL_OID])   # original left untouched by the patch
    witness = augmented[augmented[COL_OID] == b_oid]
    assert len(witness) == 1
    assert witness.iloc[0][COL_QUALIFIER] == Q_TO
    assert witness.iloc[0][COL_OTYPE] == res.participant_types.type_of("B")
    send_eid = res.events_df.loc[res.events_df[COL_ACTIVITY] == "Send1", COL_EID].iloc[0]
    assert witness.iloc[0][COL_EID] == send_eid


def test_export_witness_is_one_row_per_object_across_cases():
    """D25: an endpoint-only Participant shared across two different
    CollaborationCases (B is a message counterparty in both C1 and C2, never
    collab:participant of its own event) must receive EXACTLY ONE witness E2O
    row, not one per referencing event. A single E2O edge already suffices for
    pm4py's exporter to keep the object, so emitting one per (event, object)
    would only add more non-M6 `to`/`from` rows to the serialized OCEL -- rows
    an external consumer could misread as semantic -- pushing the exported
    artefact further from the E2O set of mu(L) for no benefit. Deduplicating on
    the target object keeps that surplus at its minimum (1 row) and matches the
    per-object reduction the read-side strip already performs.
    """
    rows = [
        _row("C1", "Send1", "2024-01-01T00:00:00Z", "A", "SendTask", frm="A", to="B"),
        _row("C2", "Send2", "2024-01-02T00:00:00Z", "C", "SendTask", frm="C", to="B"),
    ]
    src = pd.DataFrame(rows)
    res = transform(src, MappingConfig())

    b_oid = "part::B"
    assert b_oid in set(res.objects_df[COL_OID])         # materialized in both cases
    assert b_oid not in set(res.relations_df[COL_OID])   # but endpoint-only, no E2O

    augmented = _add_export_reachability_witnesses(res.objects_df, res.relations_df, res.o2o_df)
    witness = augmented[augmented[COL_OID] == b_oid]
    assert len(witness) == 1                              # one row per object, not per event
    assert witness.iloc[0][COL_QUALIFIER] == Q_TO


_XES_TEMPLATE = """<?xml version="1.0" encoding="utf-8" ?>
<log{ns}>
  <trace>
    <string key="concept:name" value="C1" />
    <event>
      <date key="time:timestamp" value="2024-01-01T00:00:00.000000+00:00" />
    </event>
    <event>
      <date key="time:timestamp" value="2024-01-01T00:00:01.000000+00:00" />
    </event>
  </trace>
  <trace>
    <string key="concept:name" value="C2" />
    <event>
      <date key="time:timestamp" value="2024-01-02T00:00:00.000000+00:00" />
    </event>
  </trace>
</log>
"""


def _write_temp_xes(namespaced: bool) -> str:
    ns = ' xmlns="http://www.xes-standard.org/"' if namespaced else ""
    content = _XES_TEMPLATE.format(ns=ns)
    fd, path = tempfile.mkstemp(suffix=".xes")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path


def test_raw_timestamps_namespace_aware():
    """D26: a XES file may declare a default namespace on the root <log>
    element (xmlns=...), as pm4py's own XES writer does -- toy_collab.xes
    is generated this way. Before D26, _raw_timestamps_in_file_order()
    searched for bare tag names ("trace"/"event"/"date"), which silently
    match nothing once every element's real tag is "{namespace-uri}trace"
    etc., so a namespaced file produced 0 timestamps against N parsed
    events and _correct_utc_timestamps aborted. Both variants must yield
    the same 3 timestamps in file order.
    """
    expected = [
        "2024-01-01T00:00:00.000000+00:00",
        "2024-01-01T00:00:01.000000+00:00",
        "2024-01-02T00:00:00.000000+00:00",
    ]
    for namespaced in (False, True):
        path = _write_temp_xes(namespaced)
        try:
            assert _raw_timestamps_in_file_order(path) == expected, (
                f"namespaced={namespaced}")
        finally:
            os.remove(path)


def test_p11_catches_undefined_source_participant():
    """B14: Definition app-r1 declares `part` a TOTAL function on E_L, unlike
    `from`/`to` (which the Normalization paragraph explicitly allows to be
    partial on the counterparty side). A source event with no
    collab:participant value at all violates this precondition. Before B14,
    transform() silently propagated the absence (no participant/in_orchestration
    edge created, M6) and P1.1 never flagged it -- the output's participant
    attribute is equally absent, so the existing mismatch counter saw
    None == None and stayed green.
    """
    src = _well_formed_log().copy()
    idx = src[src[ACT_KEY] == "Start"].index[0]
    assert src.loc[idx, PART_KEY] == "A"
    src.loc[idx, PART_KEY] = None
    res = transform(src, MappingConfig())
    checks = _checks_by_name(run_consistency_checks(src, res, MappingConfig()))
    p11 = checks["P1.1 Totality"]
    assert not p11.passed
    assert "collab:participant undefined" in p11.detail
    assert "undefined (violates the total-function precondition of Definition app-r1)=1" in p11.detail


def test_p11_catches_elemtype_out_of_domain():
    """E30: _require_columns only checks that collab:elemType is present, not
    that its values lie in {task, SendTask, ReceiveTask}. An out-of-domain
    value is silently treated as a plain task by M5/M6 while the garbage
    string is preserved verbatim as the output elemType attribute, so the
    existing elemtype_mismatches counter stays 0 (both sides agree on the
    same string) -- surfaced here as its own count.
    """
    src = _well_formed_log().copy()
    idx = src[src[ACT_KEY] == "Start"].index[0]
    src.loc[idx, ELEM_KEY] = "BogusTask"
    res = transform(src, MappingConfig())
    checks = _checks_by_name(run_consistency_checks(src, res, MappingConfig()))
    p11 = checks["P1.1 Totality"]
    assert not p11.passed
    assert "elemType absent or outside" in p11.detail
    assert p11.detail.rstrip(".").endswith("=1")


def test_p11_catches_absent_and_blank_elemtype():
    """E30 (reviewer round 2): an ABSENT/empty/whitespace collab:elemType is
    silently defaulted to 'task' by _sorted_case_events before the domain
    check runs, so before this fix only a non-empty bogus string like
    'BogusTask' was ever flagged; None/''/'   ' passed vacuously (the
    default made both source and output read 'task'). The domain check now
    reads the RAW pre-default value, so an absent/blank elemType is counted
    too."""
    for bad in (None, "", "   "):
        src = _well_formed_log().copy()
        # Use a plain 'task' event so no Message/send/receive machinery is
        # involved -- isolates the elemType-absence signal itself.
        idx = src[src[ACT_KEY] == "Start"].index[0]
        src.loc[idx, ELEM_KEY] = bad
        res = transform(src, MappingConfig())
        checks = _checks_by_name(run_consistency_checks(src, res, MappingConfig()))
        p11 = checks["P1.1 Totality"]
        assert not p11.passed, f"expected P1.1 to fail for elemType={bad!r}"
        assert "elemType absent or outside" in p11.detail
        assert p11.detail.rstrip(".").endswith("=1"), \
            f"expected exactly one out-of-domain elemType for {bad!r}: {p11.detail}"


def test_transform_rejects_residual_attribute_name_collision():
    """E30: a residual (M8) source column with the bare name "participant"
    (distinct from the prefixed collab:participant, which is excluded from
    residual_keys) is not filtered out by consumed_keys, and the residual
    loop in transform() runs AFTER the reserved output attribute of the same
    name is set -- before this fix it silently overwrote the mapped
    collab:participant value in ev_row, and P1.1's residual check (reading
    that same clobbered cell) would not catch it since both comparisons hit
    the identical overwritten value. transform() now raises instead of
    materializing this ambiguity.
    """
    src = _well_formed_log().copy()
    src["participant"] = "SPURIOUS"
    try:
        transform(src, MappingConfig())
        assert False, "expected ValueError for reserved-name collision"
    except ValueError as ex:
        assert "participant" in str(ex)


def test_oc_id_no_separator_collision():
    """E30: _oc_id joined case/participant with an unescaped "::", so
    _oc_id("x", "y::z") and _oc_id("x::y", "z") both created "oc::x::y::z" --
    two distinct (case, participant) pairs colliding on one object id.
    Vacuous on the 6 evaluated logs (no case id or participant name contains
    ":"), but a real injectivity gap in the id-creation scheme. After the
    fix, escaping "\\" and ":" per component before joining makes distinct
    pairs create distinct ids even when a component itself contains "::".
    """
    from mapping.collab_xes_to_ocel import _oc_id
    assert _oc_id("x", "y::z") != _oc_id("x::y", "z")
    # Still stable/unaffected for the common case (no separator inside components).
    assert _oc_id("case_1", "PartyA") == "oc::case_1::PartyA"


def test_p12_catches_deleted_caseid():
    """Deleting (not merely altering) the caseId attribute of a
    CollaborationCase object. Before the D24 reviewer round-2 fix, P1.2's
    comparison only ran when the attribute happened to be present
    (`pd.notna(got)`), so a deleted caseId passed vacuously -- distinct from
    test_p12_and_p14_catch_altered_caseid, which corrupts the value to a
    different but PRESENT string."""
    src = _well_formed_log()
    res = transform(src, MappingConfig())
    obj = res.objects_df.copy()
    cc_idx = obj[(obj[COL_OTYPE] == OT_CC) & (obj[COL_OID] == "cc::C1")].index[0]
    obj.loc[cc_idx, "caseId"] = None
    corrupted = replace(res, objects_df=obj)
    checks = _checks_by_name(run_consistency_checks(src, corrupted, MappingConfig()))
    p12 = checks["P1.2 Per-case partition"]
    assert not p12.passed
    assert "caseId disagreements=1" in p12.detail


def test_p14_catches_deleted_oc_participant_attribute():
    """Deleting an OrchestrationCase's own `participant` attribute.
    Before the D24 reviewer round-2 fix, both the for_participant name
    check and the OC identity check only ran when the attribute was
    present, so a deleted value passed P1.4 vacuously."""
    src = _well_formed_log()
    res = transform(src, MappingConfig())
    obj = res.objects_df.copy()
    oc_idx = obj[obj[COL_OTYPE] == OT_OC].index[0]
    obj.loc[oc_idx, "participant"] = None
    corrupted = replace(res, objects_df=obj)
    checks = _checks_by_name(run_consistency_checks(src, corrupted, MappingConfig()))
    p14 = checks["P1.4 OrchestrationCase coherence"]
    assert not p14.passed
    assert "for_participant name disagreements=1" in p14.detail
    assert "caseId/participant identity disagreements=2" in p14.detail


def test_p14_catches_deleted_participant_name():
    """Deleting a participant object's own `name` attribute. Reached by
    BOTH C1's and C2's OrchestrationCase for that participant, so
    every for_participant edge into it is affected."""
    src = _well_formed_log()
    res = transform(src, MappingConfig())
    obj = res.objects_df.copy()
    part_idx = obj[obj[COL_OTYPE].map(res.participant_types.is_participant_type)].index[0]
    obj.loc[part_idx, "name"] = None
    corrupted = replace(res, objects_df=obj)
    checks = _checks_by_name(run_consistency_checks(src, corrupted, MappingConfig()))
    p14 = checks["P1.4 OrchestrationCase coherence"]
    assert not p14.passed
    assert "for_participant name disagreements=2" in p14.detail


def test_p16_catches_contradictory_duplicate_participant_edge():
    """Adding a SECOND, contradictory 'participant' E2O edge for the same
    event (pointing at a different Participant than the correct one).
    Before the D24 reviewer round-2 fix, P1.6 built its per-event lookup
    with `dict(zip(eids, oids))`, which silently collapsed the duplicate to
    whichever edge pandas iterated last -- the ambiguity itself was never
    reported, only whatever value happened to survive the collapse."""
    src = _well_formed_log()
    res = transform(src, MappingConfig())
    rel = res.relations_df.copy()
    row = rel[rel[COL_QUALIFIER] == Q_PARTICIPANT].iloc[0].copy()
    row[COL_OID] = "part::B" if row[COL_OID] != "part::B" else "part::A"
    rel2 = pd.concat([rel, pd.DataFrame([row])], ignore_index=True)
    corrupted = replace(res, relations_df=rel2)
    checks = _checks_by_name(run_consistency_checks(src, corrupted, MappingConfig()))
    p16 = checks["P1.6 Participant coherence"]
    assert not p16.passed
    assert "events with >1 distinct 'participant' edge target=1" in p16.detail


def test_p13_catches_coherent_endpoint_corruption_vs_source():
    """Coherently rewriting one endpoint's value across ALL THREE
    representations (event fromParticipant attribute, Message.sender
    attribute, and the O2O 'from' edge target) to the SAME wrong
    participant "Z". Before the D24 reviewer round-2 fix, P1.3 only
    compared these three representations against EACH OTHER, so a mutation
    that changes them together in lockstep agreed with itself on every
    comparison and passed; P1.3 must also compare each representation
    against the value re-derived independently from the SOURCE log."""
    src = _well_formed_log()
    res = transform(src, MappingConfig())
    send_eid = res.events_df[res.events_df[COL_ACTIVITY] == "Send1"][COL_EID].iloc[0]
    msg_oid = res.relations_df[
        (res.relations_df[COL_EID] == send_eid) & (res.relations_df[COL_QUALIFIER] == Q_SEND)
    ][COL_OID].iloc[0]

    ev = res.events_df.copy()
    ev.loc[ev[COL_EID] == send_eid, "fromParticipant"] = "Z"

    obj = res.objects_df.copy()
    obj.loc[obj[COL_OID] == msg_oid, "sender"] = "Z"
    obj = pd.concat([obj, pd.DataFrame([
        {COL_OID: "part::Z", COL_OTYPE: res.participant_types.register("Z"), "name": "Z"}])], ignore_index=True)

    o2o = res.o2o_df.copy()
    mask = (o2o[COL_OID] == msg_oid) & (o2o[COL_QUALIFIER] == Q_FROM)
    o2o.loc[mask, COL_OID2] = "part::Z"

    corrupted = replace(res, events_df=ev, objects_df=obj, o2o_df=o2o)
    checks = _checks_by_name(run_consistency_checks(src, corrupted, MappingConfig()))
    p13 = checks["P1.3 Message well-formedness"]
    assert not p13.passed
    assert "endpoint representations (event/Message/O2O) disagreeing with the source value: 2" in p13.detail


# =====================================================================
# Rule M2: one object type per participant identifier (tau)
# =====================================================================

def test_tau_types_each_participant_and_keeps_name_attribute():
    """M2: the object type IS the participant identifier (encoded), and the
    identifier is additionally kept as the `name` object attribute. Before
    this change every participant shared a single "Participant" type."""
    src = _well_formed_log()
    res = transform(src, MappingConfig())
    assert res.participant_types.types == {"A", "B"}
    obj = res.objects_df
    parts = obj[obj[COL_OTYPE].map(res.participant_types.is_participant_type)]
    assert dict(zip(parts["name"], parts[COL_OTYPE])) == {"A": "A", "B": "B"}
    # The structural types are disjoint from T_Pa.
    assert not res.participant_types.is_participant_type(OT_CC)
    assert not res.participant_types.is_participant_type(OT_OC)
    assert not res.participant_types.is_participant_type(OT_MESSAGE)


def test_tau_is_identifier_safe_and_injective_under_collision():
    """tau must (a) yield a valid Python identifier -- OCPA resolves object
    types with getattr over itertuples and silently drops any type it cannot
    address -- and (b) stay injective, including against pm4py's own
    object-type name stripping, which derives the physical SQLite table name
    and would otherwise make two participants share one table."""
    from mapping.collab_xes_to_ocel import (
        ParticipantTypes, _encode_object_type, _pm4py_table_name)
    assert _encode_object_type("Org line A2") == "OrgLineA2"
    assert _encode_object_type("Hospital") == "Hospital"   # fixed point
    assert _encode_object_type("2Fast").isidentifier()     # leading digit
    tau = ParticipantTypes()
    # Two distinct identifiers whose base encodings collide, plus one that
    # collides with a structural type of the mapping.
    got = [tau.register(p) for p in ("Org line A2", "Org  line  A2", "Message")]
    assert len(set(got)) == 3
    assert all(t.isidentifier() for t in got)
    assert "Message" not in got                            # reserved
    assert len({_pm4py_table_name(t) for t in got}) == 3    # distinct tables
    assert tau.participant_of(got[0]) == "Org line A2"      # inverse


def test_p14_catches_for_participant_retargeted_to_wrong_type():
    """M2 encodes the participant identifier twice, as the object type and as
    the `name` value, and the revised P1.4 requires BOTH to agree. Retarget a
    for_participant edge to an object that carries the right name but the
    wrong type: the name clause alone would still pass."""
    src = _well_formed_log()
    res = transform(src, MappingConfig())
    obj = res.objects_df.copy()
    a_name = obj.loc[obj[COL_OID] == "part::A", "name"].iloc[0]
    obj = pd.concat([obj, pd.DataFrame([
        {COL_OID: "part::A_impostor", COL_OTYPE: OT_MESSAGE, "name": a_name}])],
        ignore_index=True)
    o2o = res.o2o_df.copy()
    mask = ((o2o[COL_QUALIFIER] == "for_participant") & (o2o[COL_OID2] == "part::A"))
    o2o.loc[o2o[mask].index[:1], COL_OID2] = "part::A_impostor"
    corrupted = replace(res, objects_df=obj, o2o_df=o2o)
    checks = _checks_by_name(run_consistency_checks(src, corrupted, MappingConfig()))
    p14 = checks["P1.4 OrchestrationCase coherence"]
    assert not p14.passed
    assert "for_participant name disagreements=0" in p14.detail   # name still agrees
    assert "for_participant type disagreements=1" in p14.detail


# =====================================================================
# Refinement layers: resource (R1-R3, PR.1/PR.2), correlation (C1, PC.1)
# =====================================================================

def _layer_log() -> pd.DataFrame:
    """The well-formed log with an actor attribute and a correlation
    identifier that pairs each send with its receive."""
    src = _well_formed_log()
    src["org:resource"] = ["r1", "r1", "r2", "r2", "r1", "r1", "r2", "r2"]
    src["msgId"] = [None, "m1", "m1", None, None, "m2", "m2", None]
    return src


def test_layers_are_off_by_default():
    """The core mapping creates no Resource object and no correlated_with
    relation, and reports exactly the P1 checks."""
    res = transform(_layer_log(), MappingConfig())
    assert "Resource" not in set(res.objects_df[COL_OTYPE])
    assert "correlated_with" not in set(res.o2o_df[COL_QUALIFIER])
    names = [c.name for c in run_consistency_checks(_layer_log(), res, MappingConfig())]
    assert not [n for n in names if n.startswith(("PR.", "PC."))]


def test_resource_layer_adds_objects_relations_and_passes_its_criteria():
    src = _layer_log()
    cfg = MappingConfig(resource_attr="org:resource")
    res = transform(src, cfg)
    obj, rel, o2o = res.objects_df, res.relations_df, res.o2o_df
    assert set(obj[obj[COL_OTYPE] == "Resource"]["name"]) == {"r1", "r2"}       # R1
    assert int((rel[COL_QUALIFIER] == "resource").sum()) == len(src)            # R2
    acts_for = o2o[o2o[COL_QUALIFIER] == "acts_for"]                            # R3
    assert set(zip(acts_for[COL_OID], acts_for[COL_OID2])) == {
        ("res::r1", "part::A"), ("res::r2", "part::B")}
    checks = _checks_by_name(run_consistency_checks(src, res, cfg))
    assert checks["PR.1 Actor-participant coherence"].passed
    assert checks["PR.2 No orphan resources"].passed
    assert all(c.passed for c in checks.values())


def test_pr1_catches_resource_acting_for_an_unrecorded_participant():
    """Deleting the acts_for edge that backs one event's resource relation:
    the event is then attributed to a resource the log does not record as
    acting for that participant."""
    src = _layer_log()
    cfg = MappingConfig(resource_attr="org:resource")
    res = transform(src, cfg)
    o2o = res.o2o_df
    o2o = o2o.drop(o2o[(o2o[COL_QUALIFIER] == "acts_for")
                       & (o2o[COL_OID] == "res::r1")].index)
    checks = _checks_by_name(run_consistency_checks(
        src, replace(res, o2o_df=o2o), cfg))
    assert not checks["PR.1 Actor-participant coherence"].passed
    assert all(c.passed for n, c in checks.items() if not n.startswith("PR.1"))


def test_correlation_layer_pairs_send_and_receive_and_passes_pc1():
    src = _layer_log()
    cfg = MappingConfig(correlation_attr="msgId")
    res = transform(src, cfg)
    corr = res.o2o_df[res.o2o_df[COL_QUALIFIER] == "correlated_with"]
    assert len(corr) == 2                       # one per (send, receive) pair
    # Directed from the send observation to the receive observation.
    send_eids = set(res.relations_df[res.relations_df[COL_QUALIFIER] == Q_SEND][COL_EID])
    assert all(src_oid.replace("msg::", "") in send_eids for src_oid in corr[COL_OID])
    checks = _checks_by_name(run_consistency_checks(src, res, cfg))
    assert checks["PC.1 Correlation well-formedness"].passed
    assert all(c.passed for c in checks.values())


def test_pc1_catches_correlation_id_shared_by_more_than_two_observations():
    """Rule C1 is unconditional: an identifier shared by one send and TWO
    receives yields two relations out of the same Message. Filtering such a
    group out at construction time would let PC.1 pass vacuously on a source
    attribute that is not a usable correlation identifier."""
    src = _layer_log()
    src.loc[src["msgId"].isna() & (src[ACT_KEY] == "End"), "msgId"] = "m1"
    src.loc[src[ACT_KEY] == "End", ELEM_KEY] = "ReceiveTask"
    src.loc[src[ACT_KEY] == "End", TO_KEY] = "B"
    cfg = MappingConfig(correlation_attr="msgId")
    res = transform(src, cfg)
    checks = _checks_by_name(run_consistency_checks(src, res, cfg))
    pc1 = checks["PC.1 Correlation well-formedness"]
    assert not pc1.passed
    assert "with >1 outgoing=1" in pc1.detail


def test_layers_are_additive_independent_and_degenerate():
    """The three properties the layered construction rests on: each layer only
    appends (additivity), they commute (independence), and each is the
    identity on a log that records no such identifier (degeneracy)."""
    src = _layer_log()
    def sig(r):
        f = lambda d: sorted(map(tuple, d.astype(str).values.tolist()))
        return f(r.objects_df), f(r.relations_df), f(r.o2o_df)
    core = sig(transform(src, MappingConfig()))
    rc = sig(transform(src, MappingConfig(resource_attr="org:resource",
                                          correlation_attr="msgId")))
    cr = sig(transform(src, MappingConfig(correlation_attr="msgId",
                                          resource_attr="org:resource")))
    assert rc == cr                                              # independence
    assert all(set(core[i]) <= set(rc[i]) for i in range(3))     # additivity
    only_r = sig(transform(src, MappingConfig(resource_attr="org:resource")))
    only_c = sig(transform(src, MappingConfig(correlation_attr="msgId")))
    for i in range(3):   # the layers' contributions are disjoint and exhaustive
        er, ec = set(only_r[i]) - set(core[i]), set(only_c[i]) - set(core[i])
        assert not (er & ec) and er | ec == set(rc[i]) - set(core[i])
    # Degeneracy: naming an attribute no event carries changes nothing.
    assert sig(transform(src, MappingConfig(resource_attr="absent:attr",
                                            correlation_attr="absent:attr"))) == core


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n[ok] {len(fns)} mapping consistency-check tests passed.")


if __name__ == "__main__":
    _run_all()
