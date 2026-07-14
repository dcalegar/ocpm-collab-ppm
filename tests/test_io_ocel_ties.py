"""
Regression test for ``ocpm_eval.io_ocel._break_timestamp_ties``.

Context (D23/B9): OCPA's positional features cut the prefix with
``event_timestamp <= cut_time``, not with the total order prec_L, so two
events of the same CollaborationCase sharing an ``ocel_time`` leak into each
other's "past" count. ``_break_timestamp_ties`` nudges tied within-case
timestamps by whole microseconds in prec_L (``ocel_id`` idx) order so every
event of a case gets a strictly increasing timestamp.

This test locks in the reviewer round-2 fix: the previous implementation
shifted a tied run by offsets from the run's OWN base, without checking the
next distinct timestamp, so an input like [0us, 0us, 1us] became
[0us, 1us, 1us] -- a brand-new tie. The forward-pass version compares each
event against the running previous value, yielding [0us, 1us, 2us] (or, more
precisely, three strictly increasing instants), leaving zero residual ties.

Pure stdlib sqlite3, no OCPA/pm4py dependency: builds the minimal
``event_map_type`` + ``event_<suffix>`` tables the function reads.
"""
import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ocpm_eval.io_ocel import _break_timestamp_ties  # noqa: E402


def _build_db(events):
    """events: list of (suffix, ocel_id, ocel_time_str)."""
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE event_map_type (ocel_type_map TEXT)")
    suffixes = sorted({s for s, _, _ in events})
    con.executemany("INSERT INTO event_map_type VALUES (?)", [(s,) for s in suffixes])
    for s in suffixes:
        con.execute(f'CREATE TABLE "event_{s}" (ocel_id TEXT, ocel_time TEXT)')
    for s, oid, t in events:
        con.execute(f'INSERT INTO "event_{s}" VALUES (?,?)', (oid, t))
    con.commit()
    con.close()
    return path


def _all_times(path, suffixes):
    con = sqlite3.connect(path)
    out = {}
    for s in suffixes:
        for oid, t in con.execute(f'SELECT ocel_id, ocel_time FROM "event_{s}"'):
            out[oid] = t
    con.close()
    return out


def test_break_ties_no_residual_tie_on_cascade():
    # One case C1, three events in prec_L (idx) order: two share 0us, the
    # third is at 1us -- the exact [0us, 0us, 1us] cascade pattern.
    T0 = "2024-01-01 00:00:00.000000"
    T0b = "2024-01-01 00:00:00.000000"
    T1 = "2024-01-01 00:00:00.000001"
    events = [
        ("task", "e::C1::00", T0),
        ("task", "e::C1::01", T0b),
        ("task", "e::C1::02", T1),
    ]
    path = _build_db(events)
    out = _break_timestamp_ties(path)
    assert out != path  # a shift was needed -> a copy was made
    times = _all_times(out, ["task"])
    assert len(set(times.values())) == 3, f"expected 3 distinct instants, got {times}"
    # Order must still follow prec_L (idx) order.
    ordered = [times["e::C1::00"], times["e::C1::01"], times["e::C1::02"]]
    assert ordered == sorted(ordered), f"timestamps not monotonic in prec_L order: {ordered}"


def test_break_ties_noop_when_all_distinct():
    events = [
        ("task", "e::C1::00", "2024-01-01 00:00:00.000000"),
        ("task", "e::C1::01", "2024-01-01 00:00:01.000000"),
        ("task", "e::C1::02", "2024-01-01 00:00:02.000000"),
    ]
    path = _build_db(events)
    out = _break_timestamp_ties(path)
    assert out == path  # nothing to shift -> original path returned unchanged


def test_break_ties_independent_across_cases():
    # Same instant in two different cases is NOT a tie to break (OCPA
    # executions are per CollaborationCase); only within-case ties matter.
    T0 = "2024-01-01 00:00:00.000000"
    events = [
        ("task", "e::C1::00", T0),
        ("task", "e::C2::00", T0),
    ]
    path = _build_db(events)
    out = _break_timestamp_ties(path)
    assert out == path  # no within-case tie -> no-op


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n[ok] {len(fns)} io_ocel tie-break tests passed.")


if __name__ == "__main__":
    _run_all()
