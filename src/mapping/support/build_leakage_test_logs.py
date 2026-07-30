#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_leakage_test_logs.py
=====================================================================
Builds tiny synthetic OCEL 2.0 SQLite fixtures used ONLY by
tests/test_features_leakage.py (A7, point 3: leakage tests per feature
family, beyond the remaining-time oracle). Not a paper artifact, not
one of the four study logs, not the toy_collab log used for X-Inf/X-MSt.

Two pairs of fixtures, written to tests/fixtures/:

  * leak_temporal_full.sqlite / leak_temporal_truncated.sqlite
    Same single collaboration case C1, but the truncated fixture drops
    its last event. Used to check that features at cut points 1..5 are
    unchanged whether or not event 6 exists (no future/temporal leakage).

  * leak_crosscase_single.sqlite / leak_crosscase_combined.sqlite
    C1 alone vs. C1 + C2, where C2 shares the "Hospital" participant
    identifier with C1 (rule M2: log-wide scope). Used to check that
    C1's per-event features are unchanged whether or not C2 exists in
    the same log (no cross-case leakage through the shared participant
    object).

The builder calls the real converter's core (mapping.collab_xes_to_ocel:
transform / build_ocel_object / write_ocel2_sqlite) directly on an
in-memory DataFrame -- no XES round-trip needed, since `transform` is
pm4py-independent and only object construction/export touches pm4py.

Run (mapping venv, pm4py >= 2.7, matching the converter's requirement):
    arch -x86_64 .venv-mapping/bin/python3.10 \
        src/mapping/support/build_leakage_test_logs.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from mapping.collab_xes_to_ocel import (  # noqa: E402
    transform, build_ocel_object, write_ocel2_sqlite, MappingConfig)

BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                       "tests", "fixtures")


def _row(case, offset_s, activity, elem, participant, frm=None, to=None):
    return {
        "case:concept:name": case,
        "concept:name": activity,
        "time:timestamp": BASE + timedelta(seconds=offset_s),
        "collab:elemType": elem,
        "collab:participant": participant,
        "collab:fromParticipant": frm,
        "collab:toParticipant": to,
    }


def _case_c1():
    """6 events, exercising task/send/receive so previous_type_count(PP),
    previous_type_count(Message), preceding_activities, and elapsed_time
    all vary across the trace."""
    return [
        _row("C1", 0, "Start", "task", "Hospital"),
        _row("C1", 1, "SendMsg", "SendTask", "Hospital", frm="Hospital", to="Lab"),
        _row("C1", 2, "RecvAck", "ReceiveTask", "Lab", frm="Hospital", to="Lab"),
        _row("C1", 3, "Review", "task", "Hospital"),
        _row("C1", 4, "SendResult", "SendTask", "Hospital", frm="Hospital", to="Lab"),
        _row("C1", 5, "End", "task", "Hospital"),
    ]


def _case_c2():
    """Shares the 'Hospital' participant identifier with C1 (rule M2: same
    identifier, log-wide scope -> same Pa object). Uses an activity
    ('Dispense') not present in C1, and a distinct partner ('Pharmacy'),
    so any effect of C2 on C1's features can only come through the shared
    participant object, not through a shared activity vocabulary."""
    return [
        _row("C2", 100, "Start", "task", "Hospital"),
        _row("C2", 101, "Dispense", "task", "Hospital"),
        _row("C2", 102, "End", "task", "Hospital"),
    ]


def _build(rows, path):
    df = pd.DataFrame(rows)
    res = transform(df, MappingConfig())
    ocel = build_ocel_object(res.events_df, res.objects_df, res.relations_df, res.o2o_df)
    write_ocel2_sqlite(ocel, path)
    print(f"[ok] wrote {path} ({len(rows)} events)")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    c1 = _case_c1()
    c2 = _case_c2()

    _build(c1, os.path.join(OUT_DIR, "leak_temporal_full.sqlite"))
    _build(c1[:-1], os.path.join(OUT_DIR, "leak_temporal_truncated.sqlite"))

    _build(c1, os.path.join(OUT_DIR, "leak_crosscase_single.sqlite"))
    _build(c1 + c2, os.path.join(OUT_DIR, "leak_crosscase_combined.sqlite"))


if __name__ == "__main__":
    raise SystemExit(main())
