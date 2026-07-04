#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_toy_collab_log.py
=====================================================================
Builds a small, hand-crafted extended collaborative XES log used ONLY to
demonstrate and test the object-enabled EXTENSION tasks (X-Inf, X-MSt;
see ocpm_tasks/extensions.py). Its files live alongside the four study
logs in data/logs/ (named "toy_collab.*" so it is never mistaken for one
of them); it stays out of RQ2/RQ3 by task/config separation, not by
directory -- its own results go to a dedicated "rq_ext_results_toy.csv"
in data/results/.

Three collaboration cases, participants PartyA/PartyB, one send/receive
event per communication. Each Send/Receive event carries a residual
"msgId" event attribute (NOT part of the core M1-M8 vocabulary) that a
downstream ENRICHMENT step (ocpm_tasks.adapters, ``corr_attr="msgId"``)
uses to populate Event.corr_id for X-MSt. The core mapping (M4) still
mints one independent Message object per send/receive observation and
infers no correspondence on its own -- msgId is an extra, explicit,
opt-in correlation source, exactly the kind of "native message
identifier" the paper's X-MSt discussion requires.

Case shapes (by design):
  T1 (9 events) -- Order/Ack exchanged in full; a final SendUpdate is
                   NEVER received -> backlog stays at 1 to the end;
                   X-MSt undefined (BOTTOM) for that send.
  T2 (8 events) -- two overlapping in-flight sends (peak backlog = 2),
                   both later received; a final SendZ is never received.
  T3 (6 events) -- fully synchronized (send immediately followed by its
                   receive both ways) -- backlog never exceeds 1.

Run (mapping venv, pm4py >= 2.7):
    arch -x86_64 .venv-mapping/bin/python3.10 \
        src/mapping/aux/build_toy_collab_log.py data/logs/toy_collab.xes
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone

from pm4py.objects.log.obj import EventLog, Trace, Event as Pm4pyEvent

BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _ev(offset_s, activity, elem, participant, frm=None, to=None, msg_id=None):
    e = Pm4pyEvent()
    e["concept:name"] = activity
    e["time:timestamp"] = BASE + timedelta(seconds=offset_s)
    e["collab:elemType"] = elem
    if participant is not None:
        e["collab:participant"] = participant
    if frm is not None:
        e["collab:fromParticipant"] = frm
    if to is not None:
        e["collab:toParticipant"] = to
    if msg_id is not None:
        e["msgId"] = msg_id
    return e


def _trace(case_id, events):
    t = Trace()
    t.attributes["concept:name"] = case_id
    for e in events:
        t.append(e)
    return t


def build_toy_log() -> EventLog:
    A, B = "PartyA", "PartyB"

    t1 = _trace("T1", [
        _ev(1, "Start",       "task",        A),
        _ev(2, "SendOrder",   "SendTask",    A, frm=A, to=B, msg_id="T1-m1"),
        _ev(3, "RecvOrder",   "ReceiveTask", B, frm=A, to=B, msg_id="T1-m1"),
        _ev(4, "Prepare",     "task",        B),
        _ev(5, "SendAck",     "SendTask",    B, frm=B, to=A, msg_id="T1-m2"),
        _ev(6, "RecvAck",     "ReceiveTask", A, frm=B, to=A, msg_id="T1-m2"),
        _ev(7, "SendUpdate",  "SendTask",    A, frm=A, to=B, msg_id="T1-m3"),  # unmatched
        _ev(8, "Review",      "task",        A),
        _ev(9, "Close",       "task",        B),
    ])

    t2 = _trace("T2", [
        _ev(1, "Start",   "task",        A),
        _ev(2, "SendX",   "SendTask",    A, frm=A, to=B, msg_id="T2-m1"),
        _ev(3, "SendY",   "SendTask",    A, frm=A, to=B, msg_id="T2-m2"),   # backlog -> 2
        _ev(4, "RecvX",   "ReceiveTask", B, frm=A, to=B, msg_id="T2-m1"),
        _ev(5, "RecvY",   "ReceiveTask", B, frm=A, to=B, msg_id="T2-m2"),
        _ev(6, "SendZ",   "SendTask",    B, frm=B, to=A, msg_id="T2-m3"),   # unmatched
        _ev(7, "Wrapup",  "task",        B),
        _ev(8, "End",     "task",        A),
    ])

    t3 = _trace("T3", [
        _ev(1, "Start",  "task",        A),
        _ev(2, "SendP",  "SendTask",    A, frm=A, to=B, msg_id="T3-m1"),
        _ev(3, "RecvP",  "ReceiveTask", B, frm=A, to=B, msg_id="T3-m1"),
        _ev(4, "SendQ",  "SendTask",    B, frm=B, to=A, msg_id="T3-m2"),
        _ev(5, "RecvQ",  "ReceiveTask", A, frm=B, to=A, msg_id="T3-m2"),
        _ev(6, "End",    "task",        B),
    ])

    log = EventLog()
    for t in (t1, t2, t3):
        log.append(t)
    return log


def main(argv=None):
    import pm4py
    argv = argv if argv is not None else sys.argv[1:]
    out_path = argv[0] if argv else "data/logs/toy_collab.xes"
    log = build_toy_log()
    pm4py.write_xes(log, out_path)
    n_events = sum(len(t) for t in log)
    print(f"[ok] wrote {out_path} ({len(log)} cases, {n_events} events)")


if __name__ == "__main__":
    raise SystemExit(main())
