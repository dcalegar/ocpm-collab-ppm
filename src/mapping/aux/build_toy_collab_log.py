#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_toy_collab_log.py
=====================================================================
Builds a synthetic extended collaborative XES log (100 cases, 3 participants)
used ONLY to demonstrate and test the object-enabled EXTENSION tasks (X-Inf,
X-MSt; see ocpm_tasks/extensions.py). Its files live alongside the four study
logs in data/logs/ (named "toy_collab.*" so it is never mistaken for one of
them); it stays out of RQ2/RQ3 by task/config separation, not by directory
-- its own results go to a dedicated "rq_ext_results_toy.csv" in data/results/.

100 collaboration cases with 3 participants (PartyA, PartyB, PartyC), each with
a variable number of events. Cases are generated to exercise both X-Inf and
X-MSt, with backlog/latency tied to a prefix-observable structural property
(2 vs. 3 participants in the case) rather than drawn independently of the
observed prefix, so the targets carry a genuine, learnable signal:
  - Variable in-flight backlogs, higher in 3-party ("congested") cases
  - Variable send-to-receive latencies, higher in 3-party cases
  - Some sends that are never received (driving X-Inf backlog), more likely
    in 3-party cases
  - Explicit msgId correlation for X-MSt enrichment

Each Send/Receive event carries a residual "msgId" attribute (NOT part of the
core M1-M8 vocabulary) that a downstream ENRICHMENT step (ocpm_tasks.adapters,
``corr_attr="msgId"``) uses to populate Event.corr_id for X-MSt. The core
mapping (M4) still mints one independent Message object per send/receive
observation and infers no correspondence on its own -- msgId is an extra,
explicit, opt-in correlation source.

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
    """Generate 100 collaboration cases with 3 participants and variable patterns."""
    import random

    A, B, C = "PartyA", "PartyB", "PartyC"
    participants = [A, B, C]
    activities = ["Start", "Process", "Review", "Prepare", "Finalize", "End", "Wrapup"]

    random.seed(42)  # reproducible

    log = EventLog()
    global_time = 0
    msg_counter = 0

    for case_num in range(1, 101):
        case_id = f"C{case_num:03d}"
        events = []
        time_offset = 0

        # Vary case structure: 2 or 3 participants per case. Coordination
        # overhead (backlog, latency) is tied to this structural, prefix-
        # observable property -- more parties means more concurrent
        # in-flight messages and slower synchronization, which the
        # ParticipantProjection object count already reveals to the
        # feature extractor from the case's first few events onward. This
        # gives X-Inf/X-MSt a genuine, non-degenerate signal to learn,
        # instead of drawing backlog/latency i.i.d. of the observed prefix.
        n_participants = random.choice([2, 3])
        case_participants = random.sample(participants, n_participants)
        primary = case_participants[0]
        congested = n_participants == 3

        # Start event
        events.append(_ev(time_offset, "Start", "task", primary))
        time_offset += random.randint(1, 3)

        # Join event per remaining participant, right after Start: establishes
        # every case participant's ParticipantProjection early, so the case-size
        # signal is available to the feature extractor (previous-PP-count) from
        # near the beginning of the prefix, not only once that participant
        # happens to act in a later interaction.
        for p in case_participants[1:]:
            events.append(_ev(time_offset, "Join", "task", p))
            time_offset += random.randint(1, 2)

        # More interactions, and more unmatched sends, in congested cases.
        num_interactions = random.randint(7, 10) if congested else random.randint(2, 3)

        for inter in range(num_interactions):
            # Decide: message interaction (any pair of case participants --
            # 3-party cases have more distinct channels, hence more concurrent
            # messages) or a regular task
            if random.random() < 0.7:
                sender, receiver = random.sample(case_participants, 2)
                msg_id = f"{case_id}-m{msg_counter}"
                msg_counter += 1

                # Send event
                send_activity = random.choice(["Send", "Request", "Query"])
                events.append(_ev(time_offset, send_activity, "SendTask", sender,
                                 frm=sender, to=receiver, msg_id=msg_id))
                time_offset += random.randint(1, 2)

                # Receive event (sometimes delayed, sometimes missing entirely);
                # congested (3-party) cases run slower and drop more sends.
                latency = random.randint(5, 12) if congested else random.randint(1, 3)
                unmatched_prob = 0.35 if congested else 0.03
                if random.random() >= unmatched_prob:  # received
                    recv_activity = random.choice(["Receive", "Process", "Handle"])
                    events.append(_ev(time_offset + latency, recv_activity,
                                     "ReceiveTask", receiver, frm=sender, to=receiver,
                                     msg_id=msg_id))
                    time_offset += latency + random.randint(1, 3)
                else:  # unmatched send (drives X-Inf backlog)
                    time_offset += random.randint(2, 4)
            else:
                # Just a regular task
                task_activity = random.choice(activities)
                actor = random.choice(case_participants)
                events.append(_ev(time_offset, task_activity, "task", actor))
                time_offset += random.randint(1, 3)

        # End event
        end_actor = random.choice(case_participants)
        events.append(_ev(time_offset, "End", "task", end_actor))

        trace = _trace(case_id, events)
        log.append(trace)

    return log


def main(argv=None):
    import pm4py
    argv = argv if argv is not None else sys.argv[1:]
    out_path = argv[0] if argv else "data/logs/toy_collab.xes"
    log = build_toy_log()
    pm4py.write_xes(log, out_path)
    n_events = sum(len(t) for t in log)
    n_messages = sum(1 for t in log for e in t
                     if e.get("collab:elemType") in ("SendTask", "ReceiveTask"))
    print(f"[ok] wrote {out_path} ({len(log)} cases, {n_events} events, "
          f"~{n_messages//2} messages)")
    print(f"    Participants: 3 (PartyA, PartyB, PartyC)")
    print(f"    Events/case: {n_events // len(log):.1f} avg")


if __name__ == "__main__":
    raise SystemExit(main())
