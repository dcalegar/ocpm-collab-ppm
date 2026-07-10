#!/usr/bin/env python3
"""
collab_convert.py

Transform the BPI Challenge 2013 *incidents* event log (Volvo IT VINST) into a
collaborative event log, using the IT organizational line ("organization
involved") as the collaboration participant (BPMN pool) and detecting inter-
participant messages from ticket hand-overs surfaced as ``Queued`` events.

Design decisions (consistent with the project bitacora):
  * Participant = value of the source attribute ``organization involved``.
    Each participant is treated as an autonomous organisation (pool) that runs
    its own internal handling process on the ticket while it holds it.
  * Collaboration instance (CI) = one incident case (one source trace).
  * Local case = the projection of a CI onto a single participant (the M3
    (case, participant) pair); collected even when the participant is left and
    re-entered (ping-pong).
  * Message detection: an event whose activity status is ``Queued`` and whose
    participant differs from the immediately preceding event's participant is
    read as a *receive* observation of a ticket hand-over. One Message is
    minted per such observation, correlating exactly two observations:
      - a ``ReceiveTask`` (the real, recorded ``Queued`` event), with
        toParticipant = the receiving line (the event's own participant) and
        fromParticipant = the previous event's participant (DERIVED; declared
        as a pre-processing inference, not a recorded fact).
      - a ``SendTask`` (SYNTHESIZED: the source system does not record a
        separate send event for a hand-over), attributed to the previous
        line and dated at the same timestamp as the triggering ``Queued``
        event. Its organizational attributes (``org:resource``, ``org:role``,
        ``org:group``, countries) are copied from that same ``Queued`` event
        -- only "organization involved" is re-attributed to the previous
        line -- justified by a prior empirical check (not reproduced by this
        script) that 98.2% of such hand-overs are executed by the resource of
        the previous handler. This is a deterministic re-attribution of data already
        present on the event, not a fabrication of unobserved data, and it is
        declared explicitly (never silently blended with real observations)
        via the residual attribute ``collab:synthesized="true"``, present
        only on the synthesized ``SendTask``.
      - Both observations of a pair share the same message identifier
        (serialized as the residual attribute ``msgId``), enabling explicit
        message correlation (the same ``corr_attr`` mechanism already used by
        ToyCollab).
      - A ``Queued`` line-change event in first position of a CI has no observed
        predecessor -> flagged as an UNMATCHED receive (external / unobserved
        origin); kept, not corrected, and no send is synthesized for it.

The converter FLAGS but does NOT silently correct data-quality issues
(unmatched receives, group->line inconsistencies, catch-all "Other" line,
missing roles, "Unmatched" activity, etc.).

NOTE ON SCHEMA KEYS: the ``collab:*`` attribute keys used on output
(``elemType``, ``participant``, ``fromParticipant``, ``toParticipant``) have
been reconciled with, and match exactly, the authoritative collaborative-XES
extension used by the project repository (``src/mapping/support/collab.xesext``).

Author-facing, reproducible. Python 3, standard library only.
"""

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
import xml.etree.ElementTree as ET

XES_NS = "http://www.xes-standard.org/"
NS = "{" + XES_NS + "}"

# Source attribute carrying the participant (BPMN pool) for the incidents log.
PARTICIPANT_KEY = "organization involved"
# Status value that, on a participant change, is read as a receive observation.
TRANSFER_STATUS = "Queued"
# Original attributes preserved verbatim on every collaborative event (M8).
# NOTE: "impact" and "product" are deliberately excluded here -- they are
# constant within every case and are hoisted to CI (trace) level instead
# (see write_collab_xes()).
PRESERVE_KEYS = [
    "concept:name", "lifecycle:transition", "org:group", "org:role",
    "org:resource", "organization involved", "organization country",
    "resource country",
]
# Case-level attributes, constant within every CI, hoisted to <trace> instead
# of being repeated on every <event>: these attributes are constant within
# every case, so they are preserved as attributes of the collaboration
# instance instead.
CASE_LEVEL_KEYS = ["product", "impact"]


# --------------------------------------------------------------------------- #
# Reading                                                                      #
# --------------------------------------------------------------------------- #
def read_incidents(path):
    """Stream-parse the XES file into a list of (case_id, [event_dict]) traces.

    Transparently decompresses gzip-compressed sources (``.xes.gz``), since
    the BPI Challenge 2013 originals are distributed that way.
    """
    traces = []
    op = gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")
    with op as fh:
        for _, el in ET.iterparse(fh, events=("end",)):
            if el.tag != NS + "trace":
                continue
            case_id = None
            events = []
            for ch in el:
                if ch.tag == NS + "string" and ch.get("key") == "concept:name":
                    case_id = ch.get("value")
                elif ch.tag == NS + "event":
                    d = {}
                    for a in ch:
                        d[a.get("key")] = a.get("value")
                    events.append(d)
            traces.append((case_id, events))
            el.clear()
    return traces


# --------------------------------------------------------------------------- #
# Transformation                                                               #
# --------------------------------------------------------------------------- #
def transform(traces):
    """
    Return (collab_cases, stats). Each collab case is a dict with the CI id and
    an ordered list of enriched events carrying collaborative annotations.
    """
    collab_cases = []
    stats = {
        "n_ci": 0,
        "n_events": 0,
        "n_task": 0,
        "n_receive_matched": 0,
        "n_receive_unmatched": 0,
        "n_send_synthesized": 0,
        "n_messages": 0,
        "elem_by_status": Counter(),
        "participants": Counter(),          # events per participant
        "participants_per_ci": [],
        "events_per_ci": [],
        "messages_per_ci": [],
        "localcases_total": 0,
        "localcase_lengths": [],
        # flags / quality
        "flag_queued_first_of_ci": 0,       # Queued in first position (entry)
        "flag_line_change_non_queued": 0,   # hand-over not surfaced as Queued
        "flag_participant_absent": 0,
        "flag_activity_unmatched": 0,
        "flag_other_line_events": 0,
        "from_is_external": 0,
        "flag_missing_org_role": 0,          # (v) events lacking org:role
        "flag_product_impact_not_constant": 0,  # CIs where product/impact vary
        "group_to_lines": defaultdict(set),  # (iv) org:group -> {organization involved}
    }

    eid = 0  # global event id, minted in order of appearance (D15)
    mid = 0  # global message id

    for case_id, events in traces:
        # order of appearance == source file order (already time-sorted, but we
        # do NOT rely on timestamps: appearance order defines the source order).
        participants_here = set()
        local_events = defaultdict(list)   # participant -> [enriched events]
        enriched = []
        n_msg_ci = 0

        prev_part = None
        for i, e in enumerate(events):
            part = e.get(PARTICIPANT_KEY)
            status = e.get("concept:name")

            if part is None or part == "UNKNOWN":
                stats["flag_participant_absent"] += 1
            if status == "Unmatched":
                stats["flag_activity_unmatched"] += 1
            if part == "Other":
                stats["flag_other_line_events"] += 1
            if not e.get("org:role"):
                stats["flag_missing_org_role"] += 1
            group = e.get("org:group")
            if group is not None and part is not None:
                stats["group_to_lines"][group].add(part)

            elem_type = "task"
            from_part = None
            to_part = None
            msg_id = None

            is_line_change = (prev_part is not None and part != prev_part)
            if status == TRANSFER_STATUS:
                if i == 0:
                    # Queued at CI entry: no observed predecessor.
                    stats["flag_queued_first_of_ci"] += 1
                elif is_line_change:
                    # Receive observation -> mint one Message, correlating a
                    # real ReceiveTask with a synthesized SendTask.
                    mid += 1
                    n_msg_ci += 1
                    elem_type = "ReceiveTask"
                    to_part = part
                    from_part = prev_part
                    msg_id = f"msg{mid:07d}"
                    stats["n_receive_matched"] += 1
                    stats["n_messages"] += 1
                    if from_part in (None, "UNKNOWN"):
                        stats["from_is_external"] += 1

                    # Synthesize the paired send observation, attributed to
                    # the previous line: the source log records only the
                    # receive side of a hand-over. Shallow-copy the
                    # triggering Queued event so org:resource/org:role/
                    # org:group/countries (which already describe the real
                    # sender empirically, 98.2% agreement per a prior check
                    # not reproduced here -- see module docstring) and
                    # time:timestamp are preserved verbatim; only
                    # "organization involved" is re-attributed to prev_part.
                    # Inserted immediately before the ReceiveTask so
                    # insertion order (the project's stable tie-break
                    # convention) places it first.
                    eid += 1
                    send_src = dict(e)
                    send_src[PARTICIPANT_KEY] = prev_part
                    send_ev = {
                        "collab_event_id": f"e{eid:07d}",
                        "participant": prev_part,
                        "elem_type": "SendTask",
                        "from_participant": prev_part,
                        "to_participant": part,
                        "message_id": msg_id,
                        "synthesized": True,
                        "src": send_src,
                    }
                    stats["elem_by_status"][(status, "SendTask")] += 1
                    stats["n_send_synthesized"] += 1
                    enriched.append(send_ev)
                    local_events[prev_part].append(send_ev)
                    participants_here.add(prev_part)
                # else: same-line Queued -> internal re-queue, stays a task
            elif is_line_change:
                # Hand-over NOT surfaced as a Queued event: flagged residual,
                # NOT minted as a message under the stated algorithm.
                stats["flag_line_change_non_queued"] += 1

            eid += 1
            stats["elem_by_status"][(status, elem_type)] += 1
            enriched_ev = {
                "collab_event_id": f"e{eid:07d}",
                "participant": part,
                "elem_type": elem_type,
                "from_participant": from_part,
                "to_participant": to_part,
                "message_id": msg_id,
                "synthesized": False,
                "src": e,
            }
            enriched.append(enriched_ev)
            if part is not None:
                participants_here.add(part)
                local_events[part].append(enriched_ev)
            prev_part = part

        # local cases for this CI
        for p, evs in local_events.items():
            stats["localcases_total"] += 1
            stats["localcase_lengths"].append(len(evs))

        stats["n_ci"] += 1
        stats["n_events"] += len(enriched)
        stats["participants_per_ci"].append(len(participants_here))
        stats["events_per_ci"].append(len(enriched))
        stats["messages_per_ci"].append(n_msg_ci)
        for p in participants_here:
            stats["participants"][p] += sum(1 for x in enriched if x["participant"] == p)

        # CI-level attributes: product/impact are constant within a case
        # (verified, not assumed) and hoisted to the <trace> rather than
        # repeated on every <event>; flag (do not silently fix) exceptions.
        case_attrs = {}
        for k in CASE_LEVEL_KEYS:
            values = {e.get(k) for e in events if e.get(k) is not None}
            if len(values) > 1:
                stats["flag_product_impact_not_constant"] += 1
            if values:
                case_attrs[k] = sorted(values)[0]

        collab_cases.append({"ci": case_id, "events": enriched,
                             "participants": sorted(participants_here),
                             "case_attrs": case_attrs})

    stats["n_task"] = sum(v for (s, t), v in stats["elem_by_status"].items() if t == "task")
    return collab_cases, stats


# --------------------------------------------------------------------------- #
# Writing collaborative XES                                                    #
# --------------------------------------------------------------------------- #
def _esc(v):
    return (str(v).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))

EXTENSIONS = [
    ("Organizational", "org", "http://www.xes-standard.org/org.xesext"),
    ("Time", "time", "http://www.xes-standard.org/time.xesext"),
    ("Lifecycle", "lifecycle", "http://www.xes-standard.org/lifecycle.xesext"),
    ("Collaborative Processes", "collab", "http://www.xes-standard.org/collab.xesext"),
    ("Concept", "concept", "http://www.xes-standard.org/concept.xesext"),
]
LOG_NAME = "BPI Challenge 2013 incidents - With Collaboration"


def write_collab_xes(collab_cases, path):
    """
    Emit a collaborative XES log aligned with the official collaborative
    extension (collab.xesext): the collab keys written are ``elemType``,
    ``participant`` and, on receive observations, ``fromParticipant`` and
    ``toParticipant``. The collaboration instance is the ``<trace>``
    (identified by ``concept:name``), which also carries the CI-constant
    ``product``/``impact`` attributes.

    Each receive observation is paired with a synthesized send observation
    (element type ``SendTask``), correlated via the residual ``msgId``
    attribute (serialized on both sides of the pair -- the same ``corr_attr``
    mechanism already used by ToyCollab). The synthesized side additionally
    carries the residual attribute ``collab:synthesized="true"``, never
    present on the real (recorded) side.
    """
    op = gzip.open(path, "wt", encoding="utf-8") if path.endswith(".gz") \
        else open(path, "w", encoding="utf-8")
    with op as f:
        f.write('<?xml version="1.0" encoding="UTF-8" ?>\n')
        f.write('<!-- Collaborative log derived from BPI Challenge 2013, incidents (Volvo IT VINST). -->\n')
        f.write('<!-- Participant = source attribute "organization involved" (IT organizational line). -->\n')
        f.write('<!-- Messages: Queued events on a participant change are read as receive -->\n')
        f.write('<!-- observations, paired with a synthesized SendTask (collab:synthesized="true") -->\n')
        f.write('<!-- attributed to the previous line and correlated via msgId. -->\n')
        f.write('<log xes.version="1.0" xes.features="nested-attributes">\n')
        for nm, pref, uri in EXTENSIONS:
            f.write(f'\t<extension name="{nm}" prefix="{pref}" uri="{uri}"/>\n')
        f.write('\t<classifier name="Activity" keys="concept:name lifecycle:transition"/>\n')
        f.write('\t<classifier name="Collaborative" keys="collab:participant collab:elemType"/>\n')
        f.write(f'\t<string key="concept:name" value="{_esc(LOG_NAME)}"/>\n')
        for case in collab_cases:
            f.write('\t<trace>\n')
            f.write(f'\t\t<string key="concept:name" value="{_esc(case["ci"])}"/>\n')
            for k in CASE_LEVEL_KEYS:
                if k in case["case_attrs"]:
                    f.write(f'\t\t<string key="{_esc(k)}" value="{_esc(case["case_attrs"][k])}"/>\n')
            for ev in case["events"]:
                src = ev["src"]
                f.write('\t\t<event>\n')
                f.write(f'\t\t\t<string key="collab:elemType" value="{ev["elem_type"]}"/>\n')
                f.write(f'\t\t\t<string key="collab:participant" value="{_esc(ev["participant"])}"/>\n')
                if ev["elem_type"] in ("ReceiveTask", "SendTask"):
                    f.write(f'\t\t\t<string key="collab:fromParticipant" value="{_esc(ev["from_participant"])}"/>\n')
                    f.write(f'\t\t\t<string key="collab:toParticipant" value="{_esc(ev["to_participant"])}"/>\n')
                    f.write(f'\t\t\t<string key="msgId" value="{_esc(ev["message_id"])}"/>\n')
                if ev.get("synthesized"):
                    f.write('\t\t\t<string key="collab:synthesized" value="true"/>\n')
                for k in PRESERVE_KEYS:
                    if k in src and k != "time:timestamp":
                        f.write(f'\t\t\t<string key="{_esc(k)}" value="{_esc(src[k])}"/>\n')
                if "time:timestamp" in src:
                    f.write(f'\t\t\t<date key="time:timestamp" value="{_esc(src["time:timestamp"])}"/>\n')
                f.write('\t\t</event>\n')
            f.write('\t</trace>\n')
        f.write('</log>\n')


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
def summarize(stats):
    def q(lst, p):
        if not lst: return 0
        s = sorted(lst); return s[min(len(s)-1, int(p*len(s)))]
    ppc = stats["participants_per_ci"]
    collaborative = sum(1 for x in ppc if x >= 2)
    group_to_lines = stats["group_to_lines"]
    groups_multi_line = sum(1 for lines in group_to_lines.values() if len(lines) > 1)
    out = {
        "collaboration_instances": stats["n_ci"],
        "genuinely_collaborative_ci_ge2_participants": collaborative,
        "single_participant_ci": stats["n_ci"] - collaborative,
        "events_total": stats["n_events"],
        "participants_distinct": len(stats["participants"]),
        "local_cases_total": stats["localcases_total"],
        "messages_total": stats["n_messages"],
        "receive_matched": stats["n_receive_matched"],
        "send_synthesized": stats["n_send_synthesized"],
        "elem_type_events": {
            "task": stats["n_task"],
            "ReceiveTask": stats["n_receive_matched"],
            "SendTask": stats["n_send_synthesized"],
        },
        "participants_per_ci": {
            "mean": round(sum(ppc)/len(ppc), 3), "max": max(ppc),
        },
        "events_per_ci": {
            "mean": round(sum(stats["events_per_ci"])/len(ppc), 3),
            "max": max(stats["events_per_ci"]),
        },
        "messages_per_ci": {
            "mean": round(sum(stats["messages_per_ci"])/len(ppc), 3),
            "max": max(stats["messages_per_ci"]),
            "ci_with_ge1_message": sum(1 for x in stats["messages_per_ci"] if x >= 1),
        },
        "local_case_length": {
            "mean": round(sum(stats["localcase_lengths"])/max(1, len(stats["localcase_lengths"])), 3),
            "median": q(stats["localcase_lengths"], 0.5),
            "max": max(stats["localcase_lengths"]) if stats["localcase_lengths"] else 0,
        },
        "flags": {
            "queued_at_ci_entry_external_or_entry": stats["flag_queued_first_of_ci"],
            "line_change_not_surfaced_as_queued": stats["flag_line_change_non_queued"],
            "receive_from_external_unobserved": stats["from_is_external"],
            "participant_absent_events": stats["flag_participant_absent"],
            "activity_unmatched_events": stats["flag_activity_unmatched"],
            "catchall_other_line_events": stats["flag_other_line_events"],
            "groups_spanning_multiple_lines": groups_multi_line,
            "groups_distinct": len(group_to_lines),
            "events_missing_org_role": stats["flag_missing_org_role"],
            "ci_with_product_or_impact_not_constant": stats["flag_product_impact_not_constant"],
        },
        "events_by_participant_top": stats["participants"].most_common(10),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="bpi2013/BPI_Challenge_2013_incidents.xes")
    ap.add_argument("--out", dest="out", default="out/BPI2013_incidents_collaborative.xes.gz")
    ap.add_argument("--metrics", dest="metrics", default="out/metrics.json")
    args = ap.parse_args()

    traces = read_incidents(args.inp)
    collab, stats = transform(traces)
    import os
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    write_collab_xes(collab, args.out)
    summary = summarize(stats)
    with open(args.metrics, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
