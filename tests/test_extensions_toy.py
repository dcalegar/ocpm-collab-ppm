"""
Dedicated toy-log test for the object-enabled EXTENSION tasks (X-Inf, X-MSt).

Kept SEPARATE from the reformulation evaluation (RQ2/RQ3) and from the four study
logs on purpose: it exercises the extensions on a small synthetic collaboration case
crafted so both targets have non-trivial, hand-checkable values (a rising/falling
in-flight backlog and some unmatched — still in-flight — sends).

Toy case ``C1`` (participants A and B), one event per second:

    e1  Start   task     A
    e2  SendX   send  A->B  corr=k1
    e3  SendY   send  A->B  corr=k2
    e4  SendZ   send  A->B  corr=k3   (never received)
    e5  RecvX   recv  A->B  corr=k1
    e6  RecvY   recv  A->B  corr=k2
    e7  SendW   send  A->B  corr=k4   (never received)
    e8  End     task     B

Running in-flight backlog after each event: [0,1,2,3,2,1,2,2]  (terminal = 2: k3,k4).

Run directly:  python tests/test_extensions_toy.py    (also pytest-compatible)
"""
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ocpm_tasks.model import Event, Execution, ObjectCentricLog          # noqa: E402
from ocpm_tasks.extensions import (                                       # noqa: E402
    X_INF, X_MST, in_flight_trajectory, compute_ext_label_rows)
from ocpm_tasks.labels import build_context, BOTTOM                       # noqa: E402

BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _t(sec):
    return BASE + timedelta(seconds=sec)


def _toy_case():
    A, B, C = "A", "B", "C1"
    ev = [
        Event("e1", "Start", _t(1), A),
        Event("e2", "SendX", _t(2), A, is_send=True,    msg_from=A, msg_to=B, corr_id="k1"),
        Event("e3", "SendY", _t(3), A, is_send=True,    msg_from=A, msg_to=B, corr_id="k2"),
        Event("e4", "SendZ", _t(4), A, is_send=True,    msg_from=A, msg_to=B, corr_id="k3"),
        Event("e5", "RecvX", _t(5), B, is_receive=True, msg_from=A, msg_to=B, corr_id="k1"),
        Event("e6", "RecvY", _t(6), B, is_receive=True, msg_from=A, msg_to=B, corr_id="k2"),
        Event("e7", "SendW", _t(7), A, is_send=True,    msg_from=A, msg_to=B, corr_id="k4"),
        Event("e8", "End",   _t(8), B),
    ]
    return ObjectCentricLog([Execution(C, ev)])


def _by_k(rows):
    """{k: y} from (case, event_id, k, y) rows."""
    return {k: y for (_c, _e, k, y) in rows}


def test_in_flight_trajectory():
    ex = next(iter(_toy_case()))
    assert in_flight_trajectory(ex) == [0, 1, 2, 3, 2, 1, 2, 2]


def test_x_inf_peak_future_backlog():
    log = _toy_case()
    rows = compute_ext_label_rows(log, X_INF)
    # rows for k=1..7 (cut i=0..6); peak of backlog over events after the cut
    assert _by_k(rows) == {1: 3, 2: 3, 3: 3, 4: 2, 5: 2, 6: 2, 7: 2}


def test_x_inf_needs_no_correlation():
    """X-Inf is a pure aggregate count: dropping the correlation ids must not
    change its labels (unlike X-MSt)."""
    log = _toy_case()
    with_corr = _by_k(compute_ext_label_rows(log, X_INF))
    for ex in log:
        for e in ex.events:
            e.corr_id = None
    without_corr = _by_k(compute_ext_label_rows(log, X_INF))
    assert with_corr == without_corr


def test_x_mst_next_send_latency():
    log = _toy_case()
    rows = compute_ext_label_rows(log, X_MST)          # BOTTOM rows dropped
    # Only k1 (cut i=0 -> next send e2, matched by e5) and k2 (cut i=1 -> e3/e6)
    # yield a latency; e4(k3) and e7(k4) are unmatched -> BOTTOM.
    assert _by_k(rows) == {1: 3.0, 2: 3.0}


def test_x_mst_undefined_without_enrichment():
    """Without native correlation ids the send<->receive pairing cannot be
    recovered, so X-MSt is undefined everywhere (all BOTTOM -> no rows)."""
    log = _toy_case()
    for ex in log:
        for e in ex.events:
            e.corr_id = None
    rows = compute_ext_label_rows(log, X_MST)
    assert rows == []
    # ... and keeping BOTTOM shows every cut is undefined:
    kept = compute_ext_label_rows(log, X_MST, drop_bottom=False)
    assert all(y == BOTTOM for (_c, _e, _k, y) in kept)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n[ok] {len(fns)} toy-log extension tests passed (X-Inf, X-MSt).")


if __name__ == "__main__":
    _run_all()
