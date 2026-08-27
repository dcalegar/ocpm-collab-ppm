"""
Expected-value tests for the fourteen label functions of ``tasks.labels``.

These are the only tests that pin the targets to concrete values. Every other
check of label correctness in this repository is an *agreement* check: RQ2
compares two derivation paths (R1 over the source XES, R2 over the generated
OCEL 2.0) and reports zero mismatches, but both share the converter's
timestamp-correction utility and the same timestamp+insertion ordering
convention, so an error in either convention would produce agreement rather
than expose it. The values below are computed by hand from the toy log and
committed as literals; they are never derived by running the code under test.

The toy log is built directly as the neutral model of ``tasks.model`` -- the
label functions are pure and read nothing else -- so these tests need no OCEL
file, no fixture and no special environment.

Run:  python tests/test_labels_golden.py    (also pytest-compatible)
"""
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tasks.catalog import TASKS                              # noqa: E402
from tasks.labels import (BOTTOM, build_context,             # noqa: E402
                          compute_label_rows, label_value)
from tasks.model import Event, Execution, ObjectCentricLog   # noqa: E402

T0 = datetime(2026, 1, 1, 12, 0, 0)


def _at(seconds):
    return T0 + timedelta(seconds=seconds)


# --- Toy log -----------------------------------------------------------------
# Case c1, six events, participants A and B, deliberately non-uniform timestamps
# so that NV-TNE cannot be satisfied by a constant:
#
#   i  k  id  activity  actor  kind      from  to   t(s)
#   0  1  e1  Start     A      task                   0
#   1  2  e2  Ask       A      send      A     B     10
#   2  3  e3  Ask       B      receive   A     B     20
#   3  4  e4  Work      B      task                  45
#   4  5  e5  Reply     B      send      B     A     55
#   5  -  e6  End       A      task                 100
#
# Cut points are i in [0, n-2], i.e. k = i+1 in [1, 5]. Case c2 is a two-event
# case contributing exactly one cut point, so row counts distinguish a per-case
# from a per-log iteration.
C1 = Execution("c1", [
    Event("e1", "Start", _at(0),   "A"),
    Event("e2", "Ask",   _at(10),  "A", is_send=True,    msg_id="m1",
          msg_type="Ask", msg_from="A", msg_to="B"),
    Event("e3", "Ask",   _at(20),  "B", is_receive=True, msg_id="m2",
          msg_type="Ask", msg_from="A", msg_to="B"),
    Event("e4", "Work",  _at(45),  "B"),
    Event("e5", "Reply", _at(55),  "B", is_send=True,    msg_id="m3",
          msg_type="Reply", msg_from="B", msg_to="A"),
    Event("e6", "End",   _at(100), "A"),
])
C2 = Execution("c2", [
    Event("e7", "Start", _at(0), "A"),
    Event("e8", "End",   _at(5), "A"),
])
LOG = ObjectCentricLog([C1, C2])
CTX = build_context(LOG)

B = BOTTOM

# Expected value at each cut point i = 0..4 of case c1, computed by hand.
GOLDEN = {
    ("NE-NEPr", None): ["Ask", "Ask", "Work", "Reply", "End"],
    ("NE-NPaA", None): ["A", "B", "B", "B", "A"],
    ("NE-NEPa", None): [("Ask", "A"), ("Ask", "B"), ("Work", "B"),
                        ("Reply", "B"), ("End", "A")],
    # next participant to SEND: sends are e2 (i=1) and e5 (i=4); `from` endpoint
    ("NE-NPaM", "send"):    ["A", "B", "B", "B", B],
    # next participant to RECEIVE: the only receive is e3 (i=2); `to` endpoint
    ("NE-NPaM", "receive"): ["B", "B", B, B, B],
    # next message kind of participant A: A's only message event is e2
    ("NE-NMPa", "A"): ["Ask", B, B, B, B],
    ("NE-NMPa", "B"): ["Ask", "Ask", "Reply", "Reply", B],
    # next message kind in the process: message events are e2, e3, e5
    ("NE-NMPr", None): ["Ask", "Ask", "Reply", "Reply", B],
    # remaining process time: t(e6) - t(e_i)
    ("NV-PrT", None): [100.0, 90.0, 80.0, 55.0, 45.0],
    # A acts last (e6), so A's remaining time coincides with the process's
    ("NV-PaT", "A"): [100.0, 90.0, 80.0, 55.0, 45.0],
    # B's last event is e5 (t=55); at i=4 no later B event remains -> 0.0, the
    # one numeric task whose fallback is 0 rather than BOTTOM
    ("NV-PaT", "B"): [55.0, 45.0, 35.0, 10.0, 0.0],
    # time to next event: t(e_{i+1}) - t(e_i)
    ("NV-TNE", None): [10.0, 10.0, 25.0, 10.0, 45.0],
    # time to next send (e2 at 10, e5 at 55)
    ("NV-TNM", "send"):    [10.0, 45.0, 35.0, 10.0, B],
    # time to next receive (e3 at 20 only)
    ("NV-TNM", "receive"): [20.0, 10.0, B, B, B],
    # remaining messages: BOTH directions count (e2 send, e3 receive, e5 send)
    ("NV-NMPr", None): [3, 2, 1, 1, 0],
    ("NV-NMPa", "A"): [1, 0, 0, 0, 0],
    ("NV-NMPa", "B"): [2, 2, 1, 1, 0],
    # A acts again at e6 from every cut point; B does not, after e5
    ("OB-P", "A"): [True, True, True, True, True],
    ("OB-P", "B"): [True, True, True, True, False],
    # "Ask" is carried by e2 (send) and e3 (receive): both directions count
    ("OB-M", "Ask"):   [True, True, False, False, False],
    ("OB-M", "Reply"): [True, True, True, True, False],
    ("OB-M", "Work"):  [False] * 5,      # a non-message activity is never matched
}


def test_every_task_has_golden_values():
    """Guard against a task silently losing coverage here."""
    covered = {key for key, _ in GOLDEN}
    assert covered == set(TASKS), (sorted(set(TASKS) - covered),
                                   sorted(covered - set(TASKS)))


def test_label_values_match_golden():
    for (key, param), expected in sorted(GOLDEN.items(), key=lambda kv: kv[0]):
        assert len(expected) == C1.n - 1, (key, param, len(expected))
        for i, want in enumerate(expected):
            got = label_value(TASKS[key], CTX, C1, i, param)
            assert got == want, (key, param, i, got, want)


def test_pair_target_is_not_concatenated():
    """NE-NEPa must return the (activity, participant) pair: concatenating them
    is not injective, so a string encoding would conflate distinct labels."""
    got = label_value(TASKS["NE-NEPa"], CTX, C1, 0, None)
    assert isinstance(got, tuple) and len(got) == 2, got


def test_boolean_tasks_never_return_bottom():
    for key in ("OB-P", "OB-M"):
        param = "A" if key == "OB-P" else "Ask"
        for i in range(C1.n - 1):
            assert label_value(TASKS[key], CTX, C1, i, param) is not BOTTOM


def test_parameterized_tasks_require_their_parameter():
    for key in ("NE-NMPa", "NV-PaT", "NV-NMPa", "OB-P", "OB-M"):
        try:
            label_value(TASKS[key], CTX, C1, 0, None)
        except ValueError:
            continue
        raise AssertionError(f"{key} accepted a missing parameter")


def test_compute_label_rows_drops_bottom_and_spans_cases():
    """Rows are (case_id, event_id, k, y) for k in [1, n-1], with BOTTOM rows
    dropped for categorical and numeric tasks but never for Boolean ones."""
    rows = compute_label_rows(LOG, TASKS["NE-NEPr"], None, CTX)
    assert [(c, k) for c, _e, k, _y in rows] == [
        ("c1", 1), ("c1", 2), ("c1", 3), ("c1", 4), ("c1", 5), ("c2", 1)]
    assert [y for *_x, y in rows] == ["Ask", "Ask", "Work", "Reply", "End", "End"]

    # NE-NMPa/A: one non-BOTTOM cut point in c1, none in c2 (no message events)
    rows = compute_label_rows(LOG, TASKS["NE-NMPa"], "A", CTX)
    assert rows == [("c1", "e1", 1, "Ask")], rows

    # Boolean targets keep every cut point, BOTTOM never arising. Only A acts
    # in c2, so its single cut point is False for p=B.
    rows = compute_label_rows(LOG, TASKS["OB-P"], "B", CTX)
    assert len(rows) == 6, rows
    assert [y for *_x, y in rows] == [True, True, True, True, False, False]

    # Keeping BOTTOM restores the full grid
    rows = compute_label_rows(LOG, TASKS["NE-NMPa"], "A", CTX, drop_bottom=False)
    assert len(rows) == 6, rows


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n[ok] {len(fns)} golden label tests passed.")


if __name__ == "__main__":
    _run_all()
