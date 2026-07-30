"""
Regression tests for ``ocpm_tasks.adapters.build_from_relations``'s resolution
of ``Event.corr_id`` (used by the X-MSt extension task).

Before this change, X-MSt's correlation was populated ONLY from a raw residual
``corr_attr`` event attribute, entirely independent of the mapping's own
correlation refinement layer (C1's ``correlated_with`` O2O relation, already
checked by PC.1 in ``mapping.collab_xes_to_ocel``). A log converted with
``--correlation-attr`` but read back without re-specifying ``corr_attr`` to
the adapter silently produced BOTTOM for every X-MSt row, and an ambiguous
correlation (which PC.1 treats as a defect) was never surfaced to X-MSt at
all. These tests lock in the fix: `correlated_with` is preferred when
unambiguous, ambiguous pairs resolve to no correlation (not an arbitrary
pick), and the raw-attribute path remains a fallback for logs that never
applied C1.
"""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ocpm_tasks.adapters import build_from_relations, _Ev, _Ob   # noqa: E402
from ocpm_tasks.schema import Schema                              # noqa: E402


def _t(s):
    return datetime(2024, 1, 1, 0, 0, s, tzinfo=timezone.utc)


def _minimal_log(corr_edges, attrs=None):
    """One CollaborationCase, one send event (-> msgA) and one receive event
    (-> msgB). ``corr_edges`` maps a message oid to the list of
    `correlated_with` targets it carries; ``attrs`` maps an event id to its
    residual event-attribute dict (for the corr_attr fallback path)."""
    attrs = attrs or {}
    objects = {
        "cc1": _Ob("cc1", "CollaborationCase"),
        "msgA": _Ob("msgA", "Message",
                   o2o=[(t, "correlated_with") for t in corr_edges.get("msgA", [])]),
        "msgB": _Ob("msgB", "Message",
                   o2o=[(t, "correlated_with") for t in corr_edges.get("msgB", [])]),
    }
    if "msgC" in corr_edges or any("msgC" in v for v in corr_edges.values()):
        objects["msgC"] = _Ob("msgC", "Message",
                              o2o=[(t, "correlated_with") for t in corr_edges.get("msgC", [])])
    events = [
        _Ev("e1", "Send", _t(1), attrs=attrs.get("e1", {}),
           e2o=[("cc1", "within"), ("msgA", "send")]),
        _Ev("e2", "Recv", _t(2), attrs=attrs.get("e2", {}),
           e2o=[("cc1", "within"), ("msgB", "receive")]),
    ]
    return events, objects


def _send_recv(log):
    ex = log.get("cc1")
    send_ev = next(e for e in ex.events if e.is_send)
    recv_ev = next(e for e in ex.events if e.is_receive)
    return send_ev, recv_ev


def test_corr_id_resolved_from_unambiguous_correlated_with_o2o():
    events, objects = _minimal_log({"msgA": ["msgB"]})
    log = build_from_relations(events, objects, Schema())
    send_ev, recv_ev = _send_recv(log)
    assert send_ev.corr_id is not None
    assert send_ev.corr_id == recv_ev.corr_id


def test_ambiguous_correlated_with_resolves_to_no_correlation():
    """A Message with more than one outgoing `correlated_with` edge is
    exactly the defect PC.1 reports; the adapter must not pick one
    arbitrarily -- corr_id stays None, same as an unenriched log."""
    events, objects = _minimal_log({"msgA": ["msgB", "msgC"]})
    log = build_from_relations(events, objects, Schema())
    send_ev, _ = _send_recv(log)
    assert send_ev.corr_id is None


def test_correlated_with_takes_precedence_over_corr_attr_fallback():
    events, objects = _minimal_log(
        {"msgA": ["msgB"]},
        attrs={"e1": {"msgId": "raw1"}, "e2": {"msgId": "raw1"}})
    log = build_from_relations(events, objects, Schema(), corr_attr="msgId")
    send_ev, recv_ev = _send_recv(log)
    # resolved via the validated O2O relation (msgA's own oid), not the raw
    # attribute value "raw1"
    assert send_ev.corr_id == recv_ev.corr_id == "msgA"


def test_corr_attr_fallback_used_when_no_correlated_with_edge():
    events, objects = _minimal_log(
        {}, attrs={"e1": {"msgId": "raw1"}, "e2": {"msgId": "raw1"}})
    log = build_from_relations(events, objects, Schema(), corr_attr="msgId")
    send_ev, recv_ev = _send_recv(log)
    assert send_ev.corr_id == recv_ev.corr_id == "raw1"


def test_no_correlation_source_leaves_corr_id_none():
    events, objects = _minimal_log({})
    log = build_from_relations(events, objects, Schema())
    send_ev, recv_ev = _send_recv(log)
    assert send_ev.corr_id is None
    assert recv_ev.corr_id is None


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
