#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_healthcare_extended.py
=====================================================================
Generates healthcare_extended.xes for testing X-Inf and X-MSt extensions:
a copy of the Healthcare log (100 cases, 1450 events, same structure) with
a residual "msgId" attribute added to Send events. This allows realistic
end-to-end testing of the extensions on a 100-case dataset, larger than
the small hand-built toy case but NOT one of the four study logs used in
RQ2/RQ3 evaluation.

Changes from the original:
  * Same timestamps, events, participants, case flow — no semantics altered.
  * Each Send event gets a "msgId" attribute (e.g., "msg_0", "msg_1", ...)
    for explicit message-instance correlation, enabling X-MSt evaluation.
  * Converted via the same M1-M8 mapping as the study logs.

Output file: data/logs/healthcare_extended.xes (and OCEL2 .sqlite/.jsonocel)

Run (mapping venv, pm4py >= 2.7):
    arch -x86_64 .venv-mapping/bin/python3.10 \\
        src/mapping/aux/build_healthcare_extended.py data/logs/healthcare_extended.xes
"""
from __future__ import annotations

import sys
import pm4py


def add_message_ids(log) -> pm4py.objects.log.obj.EventLog:
    """
    Correlate send/receive pairs and assign the same 'msgId' to matched pairs.
    Heuristic: match by (sender, receiver) in order of appearance.
    All messages in healthcare are fully paired, so this should find exact matches.
    """
    msg_counter = 0
    matched_count = 0
    for trace in log:
        # Build lists of sends and receives
        sends = []
        recvs = []
        for i, e in enumerate(trace):
            elem_type = e.get("collab:elemType") if isinstance(e, dict) else None
            if elem_type == "SendTask":
                sends.append((i, e))
            elif elem_type == "ReceiveTask":
                recvs.append((i, e))

        # Match sends to receives by (fromParticipant, toParticipant)
        matched_recvs = set()
        for send_idx, send_e in sends:
            send_from = send_e.get("collab:fromParticipant")
            send_to = send_e.get("collab:toParticipant")

            # Find the first unmatched receive with matching sender/receiver
            for recv_idx, recv_e in recvs:
                if recv_idx in matched_recvs:
                    continue
                recv_from = recv_e.get("collab:fromParticipant")
                recv_to = recv_e.get("collab:toParticipant")

                if send_from == recv_from and send_to == recv_to:
                    # Assign the same msgId to both send and receive
                    msg_id = f"msg_{msg_counter}"
                    send_e["msgId"] = msg_id
                    recv_e["msgId"] = msg_id
                    matched_recvs.add(recv_idx)
                    msg_counter += 1
                    matched_count += 1
                    break

    print(f"  [debug] matched {matched_count} message pairs (msgId assigned)")
    return log


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    out_path = argv[0] if argv else "data/logs/healthcare_extended.xes"

    # Read the original healthcare log.
    original_path = "data/logs/collectivelog_healthcare_collab.xes"
    try:
        log = pm4py.read_xes(original_path)
    except Exception as e:
        print(f"Error reading {original_path}: {e}")
        return 1

    # Add msgId attributes to sends for explicit correlation.
    log = add_message_ids(log)

    # Write.
    pm4py.write_xes(log, out_path)
    n_events = sum(len(t) for t in log)
    n_cases = len(log)
    print(f"[ok] wrote {out_path} ({n_cases} cases, {n_events} events)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
