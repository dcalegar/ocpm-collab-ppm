#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collab_xes_to_ocel.py
=====================================================================
Model-to-model transformation mu: extended collaborative XES log
  -->  OCEL 2.0 log (conforming to Berti et al. 2023, Definition 2),
exported to the .jsonocel and .sqlite formats.

This implements rules M1-M8 and the consistency criteria P1.1-P1.5 of
the mapping section. It is a transformation that produces an event log
CONFORMING to the OCEL 2.0 metamodel; it does NOT define a new
metamodel.

Source side (extended collaborative XES), per the collab.xesext
extension, carries these event-level string attributes:
    collab:elemType        in {task, SendTask, ReceiveTask}
    collab:participant     participant executing the event
    collab:fromParticipant sender (defined on Send/Receive events)
    collab:toParticipant   receiver (defined on Send/Receive events)
plus the XES keys for activity, timestamp, and global case id.

The Message object is characterized by its sender, its receiver, and
the send/receive events that reference it; collab:elemType already
distinguishes task / SendTask / ReceiveTask, so no separate message-type
attribute is read or stored.

Target side (OCEL 2.0):
    Object types : CollaborationInstance, Participant, LocalCase,
                   Message.
                   NOTE on Participant: in the collaborative source
                   process (BPMN 2.0 terminology) a participant is an
                   orchestration / pool (e.g. "Laboratory",
                   "Gynecologist"), not a role or a person. The object
                   type keeps the name "Participant" for traceability
                   with the collaborative process, but it represents the
                   orchestration: a global object, reused across cases.
                   LocalCase is the execution of that orchestration
                   within one global case; it is a distinct object,
                   linked to its Participant by the O2O qualifier
                   executed_by.
    E2O qualifiers: within, local, send, receive
                   (there is no direct event->Participant relation: the
                   participant of an event is reached via
                   local -> executed_by, so the orchestration is never
                   confused with its own per-case projection).
    O2O qualifiers: part_of, executed_by, from, to, exchanged_in

IMPORTANT VERIFICATION NOTES (read before running elsewhere):
  * Requires pm4py >= 2.7.16 (the SQLite timestamp fix in 2.7.16 also
    hardened the OCEL2 exporters; both the JSON and SQLite exporters are
    available from 2.7.x). pandas is pulled in as a pm4py dependency.
  * This script does not call pm4py at import time for anything other
    than the I/O endpoints, so the pure-Python transformation/checks
    can be unit-tested without a full pm4py install if desired.

OCEL 2.0 JSON SCHEMA CONFORMANCE:
  The exported .jsonocel must conform to the official OCEL 2.0 JSON
  schema (draft-07). The schema types every attribute `value` as a
  string and requires a `time` field on every object attribute. To
  guarantee conformance regardless of the exporter's type inference,
  build_ocel_object() casts all attribute values to string and adds an
  epoch timestamp column to the objects table (static-attribute
  encoding). After export, validate_jsonocel() checks the file against
  the embedded schema: it uses the `jsonschema` package when present
  (full draft-07 validation) and otherwise applies a dependency-free
  structural fallback. Validation runs by default in convert(); pass
  --no-validate to skip it, or --strict to make a violation fatal.
=====================================================================
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# pm4py is imported lazily inside read/write helpers so that the core
# transformation and the consistency checks remain importable even in
# an environment where only pandas is present.


# =====================================================================
# Configuration
# =====================================================================

# --- Source (extended collaborative XES) attribute keys -------------
CASE_KEY = "concept:name"
ACTIVITY_KEY = "concept:name"
TIMESTAMP_KEY = "time:timestamp"
ELEMTYPE_KEY = "collab:elemType"
PARTICIPANT_KEY = "collab:participant"
FROM_KEY = "collab:fromParticipant"
TO_KEY = "collab:toParticipant"

# Element-type literal values, per the collaborative extension.
ELEM_TASK = "task"
ELEM_SEND = "SendTask"
ELEM_RECEIVE = "ReceiveTask"

# --- OCEL 2.0 object types ------------------------------------------
OT_CI = "CollaborationInstance"
OT_PARTICIPANT = "Participant"
OT_LOCALCASE = "LocalCase"
OT_MESSAGE = "Message"

# --- E2O qualifiers (rule M6) ---------------------------------------
# No 'actor' qualifier: an event does not relate directly to a
# Participant (orchestration). The participant is reached through
# local -> executed_by, keeping the orchestration distinct from its
# per-case execution (LocalCase).
Q_WITHIN = "within"
Q_LOCAL = "local"
Q_SEND = "send"
Q_RECEIVE = "receive"

# --- O2O qualifiers (rule M7) ---------------------------------------
Q_PART_OF = "part_of"
Q_EXECUTED_BY = "executed_by"
Q_FROM = "from"
Q_TO = "to"
Q_EXCHANGED_IN = "exchanged_in"

# --- OCEL 2.0 canonical column names used by pm4py ------------------
# These are the standard pm4py OCEL column identifiers. The transform
# builds the DataFrames with exactly these names so that the OCEL2
# JSON exporter serializes them correctly. They are also re-read from
# the constructed OCEL object after instantiation, as a safety check.
COL_EID = "ocel:eid"
COL_OID = "ocel:oid"
COL_OID2 = "ocel:oid_2"
COL_ACTIVITY = "ocel:activity"
COL_TIMESTAMP = "ocel:timestamp"
COL_OTYPE = "ocel:type"
COL_QUALIFIER = "ocel:qualifier"

# Reference timestamp 0 (epoch) for static object attribute encoding,
# following the OCEL 2.0 convention (Berti et al., Def. 2).
EPOCH = pd.Timestamp("1970-01-01 00:00:00", tz="UTC")


logger = logging.getLogger("collab_xes_to_ocel")


@dataclass
class MappingConfig:
    """All source-side knobs in one place, so the transformation can be
    adapted to a concrete log without editing the rules."""
    case_key: str = CASE_KEY
    activity_key: str = ACTIVITY_KEY
    timestamp_key: str = TIMESTAMP_KEY
    elemtype_key: str = ELEMTYPE_KEY
    participant_key: str = PARTICIPANT_KEY
    from_key: str = FROM_KEY
    to_key: str = TO_KEY
    # Attribute keys that the mapping consumes (M1-M8) and must NOT be
    # re-emitted as residual event attributes by M8.
    consumed_keys: Tuple[str, ...] = field(default_factory=lambda: (
        CASE_KEY, ACTIVITY_KEY, TIMESTAMP_KEY,
        PARTICIPANT_KEY, FROM_KEY, TO_KEY,
    ))


# =====================================================================
# Identifier minting (disjoint object-id ranges; Appendix A)
# =====================================================================
# Source identifiers are stored as attribute VALUES, never reused as
# object ids (U_obj and U_val are disjoint in OCEL 2.0). We therefore
# mint type-prefixed object ids.

def _ci_id(case: str) -> str:
    return f"ci::{case}"

def _participant_id(p: str) -> str:
    return f"part::{p}"

def _localcase_id(case: str, p: str) -> str:
    return f"lc::{case}::{p}"

def _message_id(send_eid: str) -> str:
    # One Message per send event (M4); the send event id makes it unique.
    return f"msg::{send_eid}"

def _event_id(case: str, idx: int) -> str:
    # Stable per-case event id. idx is the within-case source order.
    return f"e::{case}::{idx}"


# =====================================================================
# I/O endpoints (the only pm4py-dependent functions)
# =====================================================================

def read_collaborative_xes(path: str, encoding: str = "utf-8") -> pd.DataFrame:
    """M0 - read the extended collaborative XES file into a DataFrame.

    Uses the DataFrame importer (return_legacy_log_object=False) which
    does not attempt to resolve the extension URI over the network and
    preserves all collab:* attributes as columns.
    """
    import pm4py
    logger.info("Reading collaborative XES: %s", path)
    df = pm4py.read_xes(path, return_legacy_log_object=False, encoding=encoding)
    # Normalize to a plain pandas DataFrame (pm4py may hand back a
    # pandas or polars-backed frame depending on options/version).
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    logger.info("Read %d events.", len(df))
    return df


def write_ocel2_json(ocel: Any, path: str) -> None:
    """Export the constructed OCEL 2.0 log to .jsonocel."""
    import pm4py
    logger.info("Writing OCEL 2.0 (JSON) to: %s", path)
    pm4py.write.write_ocel2_json(ocel, path)
    logger.info("Done.")

def write_ocel2_sqlite(ocel: Any, path: str) -> None:
    """Export the constructed OCEL 2.0 log to .sqlite."""
    import pm4py
    logger.info("Writing OCEL 2.0 (SQLite) to: %s", path)
    pm4py.write.write_ocel2_sqlite(ocel, path)
    logger.info("Done.")


# OCEL 2.0 JSON schema (draft-07), as published with the standard.
# Embedded so validation needs no external schema file.
OCEL2_JSON_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "eventTypes": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "attributes": {"type": "array", "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "type": {"type": "string"}},
                    "required": ["name", "type"]}}},
            "required": ["name", "attributes"]}},
        "objectTypes": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "attributes": {"type": "array", "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "type": {"type": "string"}},
                    "required": ["name", "type"]}}},
            "required": ["name", "attributes"]}},
        "events": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "type": {"type": "string"},
                "time": {"type": "string", "format": "date-time"},
                "attributes": {"type": "array", "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "value": {"type": "string"}},
                    "required": ["name", "value"]}},
                "relationships": {"type": "array", "items": {
                    "type": "object",
                    "properties": {"objectId": {"type": "string"}, "qualifier": {"type": "string"}},
                    "required": ["objectId", "qualifier"]}}},
            "required": ["id", "type", "time"]}},
        "objects": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "type": {"type": "string"},
                "relationships": {"type": "array", "items": {
                    "type": "object",
                    "properties": {"objectId": {"type": "string"}, "qualifier": {"type": "string"}},
                    "required": ["objectId", "qualifier"]}},
                "attributes": {"type": "array", "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "value": {"type": "string"},
                                   "time": {"type": "string", "format": "date-time"}},
                    "required": ["name", "value", "time"]}}},
            "required": ["id", "type"]}},
    },
    "required": ["eventTypes", "objectTypes", "events", "objects"],
}


def validate_jsonocel(path: str) -> List[str]:
    """Validate an exported .jsonocel file against the OCEL 2.0 JSON
    schema. Returns a list of human-readable problems (empty == valid).

    Uses the `jsonschema` package when available (full draft-07 check);
    otherwise falls back to a structural check of the schema's required
    keys and the string-typing of attribute values. The fallback needs
    no extra dependency.
    """
    import json
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    # Preferred path: full schema validation if jsonschema is installed.
    try:
        import jsonschema  # type: ignore
        validator = jsonschema.Draft7Validator(OCEL2_JSON_SCHEMA)
        errors = sorted(validator.iter_errors(data), key=lambda e: list(e.path))
        return [f"{list(e.path)}: {e.message}" for e in errors]
    except ImportError:
        logger.info("jsonschema not installed; using structural fallback check.")

    # Fallback: structural validation of the parts most likely to break.
    problems: List[str] = []
    for top in ("eventTypes", "objectTypes", "events", "objects"):
        if top not in data:
            problems.append(f"missing required top-level key '{top}'")
        elif not isinstance(data[top], list):
            problems.append(f"top-level '{top}' must be an array")

    def _check_attr_types(entries: List[dict], where: str) -> None:
        for i, et in enumerate(entries or []):
            if "name" not in et:
                problems.append(f"{where}[{i}] missing 'name'")
            for j, a in enumerate(et.get("attributes", []) or []):
                if "name" not in a or "type" not in a:
                    problems.append(f"{where}[{i}].attributes[{j}] needs name+type")

    _check_attr_types(data.get("eventTypes", []), "eventTypes")
    _check_attr_types(data.get("objectTypes", []), "objectTypes")

    for i, e in enumerate(data.get("events", []) or []):
        for k in ("id", "type", "time"):
            if k not in e:
                problems.append(f"events[{i}] missing required '{k}'")
        for j, a in enumerate(e.get("attributes", []) or []):
            if not isinstance(a.get("value"), str):
                problems.append(f"events[{i}].attributes[{j}].value must be a string")
        for j, r in enumerate(e.get("relationships", []) or []):
            if "objectId" not in r or "qualifier" not in r:
                problems.append(f"events[{i}].relationships[{j}] needs objectId+qualifier")

    for i, o in enumerate(data.get("objects", []) or []):
        for k in ("id", "type"):
            if k not in o:
                problems.append(f"objects[{i}] missing required '{k}'")
        for j, a in enumerate(o.get("attributes", []) or []):
            for k in ("name", "value", "time"):
                if k not in a:
                    problems.append(f"objects[{i}].attributes[{j}] missing '{k}'")
            if "value" in a and not isinstance(a["value"], str):
                problems.append(f"objects[{i}].attributes[{j}].value must be a string")
        for j, r in enumerate(o.get("relationships", []) or []):
            if "objectId" not in r or "qualifier" not in r:
                problems.append(f"objects[{i}].relationships[{j}] needs objectId+qualifier")

    return problems


def _stringify_attribute_columns(df: pd.DataFrame, reserved: Tuple[str, ...]) -> pd.DataFrame:
    """Cast every attribute column (i.e., every non-reserved column) to
    string, leaving NaN as NaN. The OCEL 2.0 JSON schema requires
    attribute `value` fields to be strings, so we coerce here rather than
    relying on the exporter's type inference.
    """
    if df.empty:
        return df
    df = df.copy()
    for c in df.columns:
        if c in reserved:
            continue
        df[c] = df[c].apply(lambda v: v if pd.isna(v) else str(v))
    return df


def build_ocel_object(events_df: pd.DataFrame,
                      objects_df: pd.DataFrame,
                      relations_df: pd.DataFrame,
                      o2o_df: pd.DataFrame) -> Any:
    """Instantiate a pm4py OCEL object from the four DataFrames and
    attach the O2O table. Kept isolated so the rest of the module is
    pm4py-independent.

    Two schema-driven hardenings are applied here:
      * object-attribute and event-attribute values are cast to string
        (the OCEL 2.0 JSON schema types every attribute `value` as a
        string);
      * the objects table is given an `ocel:timestamp` column set to the
        epoch (0), so that the exporter can emit the `time` field that
        the schema requires for every object attribute (static-attribute
        encoding of OCEL 2.0).
    """
    from pm4py.objects.ocel.obj import OCEL

    ev_reserved = (COL_EID, COL_ACTIVITY, COL_TIMESTAMP)
    ob_reserved = (COL_OID, COL_OTYPE, COL_TIMESTAMP)

    events_df = _stringify_attribute_columns(events_df, ev_reserved)
    objects_df = _stringify_attribute_columns(objects_df, ob_reserved)

    # Ensure the objects table carries a timestamp column at the epoch,
    # so static object attributes serialize with a `time` field (t=0).
    if not objects_df.empty and COL_TIMESTAMP not in objects_df.columns:
        objects_df = objects_df.copy()
        objects_df[COL_TIMESTAMP] = EPOCH

    ocel = OCEL(events=events_df, objects=objects_df, relations=relations_df)
    # Attach O2O. pm4py stores O2O in ocel.o2o; set it explicitly so the
    # exporter serializes the qualified object-to-object relations.
    ocel.o2o = o2o_df
    return ocel


# =====================================================================
# Source-log preparation
# =====================================================================

def _require_columns(df: pd.DataFrame, cfg: MappingConfig) -> None:
    """Fail early and clearly if mandatory source columns are missing."""
    mandatory = [cfg.case_key, cfg.activity_key, cfg.timestamp_key,
                 cfg.elemtype_key, cfg.participant_key]
    missing = [c for c in mandatory if c not in df.columns]
    if missing:
        raise KeyError(
            "Missing mandatory source columns: %s. Present columns: %s"
            % (missing, list(df.columns))
        )
    # from/to are mandatory only for non-task events; checked per-row later.
    for opt in (cfg.from_key, cfg.to_key):
        if opt not in df.columns:
            logger.warning("Optional column '%s' absent; will treat as empty.", opt)


def _sorted_case_events(df: pd.DataFrame, cfg: MappingConfig
                        ) -> Dict[str, List[Dict[str, Any]]]:
    """Group events by global case and order each trace by timestamp,
    breaking ties by the original source order (Definition 1).
    Returns, per case, a list of event dicts enriched with a minted eid
    and the within-case index.
    """
    work = df.copy()
    # Preserve the original file order as the tie-breaker.
    work["__src_order__"] = range(len(work))
    work[cfg.timestamp_key] = pd.to_datetime(work[cfg.timestamp_key], utc=True,
                                             errors="coerce")
    if work[cfg.timestamp_key].isna().any():
        n_bad = int(work[cfg.timestamp_key].isna().sum())
        logger.warning("%d events have unparseable timestamps (set to NaT).", n_bad)

    cases: Dict[str, List[Dict[str, Any]]] = {}
    for case_val, grp in work.groupby(cfg.case_key, sort=False):
        grp = grp.sort_values(by=[cfg.timestamp_key, "__src_order__"],
                              kind="mergesort")  # stable
        evlist: List[Dict[str, Any]] = []
        for idx, (_, row) in enumerate(grp.iterrows()):
            evlist.append({
                "eid": _event_id(str(case_val), idx),
                "case": str(case_val),
                "idx": idx,
                "activity": row[cfg.activity_key],
                "timestamp": row[cfg.timestamp_key],
                "elem": (row[cfg.elemtype_key]
                         if pd.notna(row.get(cfg.elemtype_key)) else ELEM_TASK),
                "participant": (str(row[cfg.participant_key])
                                if pd.notna(row.get(cfg.participant_key)) else None),
                "from": (str(row[cfg.from_key])
                         if cfg.from_key in work.columns and pd.notna(row.get(cfg.from_key))
                         else None),
                "to": (str(row[cfg.to_key])
                       if cfg.to_key in work.columns and pd.notna(row.get(cfg.to_key))
                       else None),
                "row": row,
            })
        cases[str(case_val)] = evlist
    return cases


# =====================================================================
# Rule M4 - send/receive correlation (declared ETL heuristic)
# =====================================================================

def _correlate_messages(case_events: List[Dict[str, Any]]
                        ) -> Tuple[Dict[str, str], List[str], List[str]]:
    """For one case, pair each SendTask with the earliest subsequent
    unpaired ReceiveTask whose (from,to) mirror the send's (participant,to).

    Returns:
      rho        : map send_eid -> receive_eid (only for matched sends)
      send_eids  : all send event ids (each mints a Message)
      unmatched_recv : receive event ids never matched (reported)
    """
    rho: Dict[str, str] = {}
    send_eids: List[str] = []
    receive_events = [e for e in case_events if e["elem"] == ELEM_RECEIVE]
    matched_recv: set = set()

    for ev in case_events:
        if ev["elem"] != ELEM_SEND:
            continue
        send_eids.append(ev["eid"])
        # candidate receives: later-or-equal timestamp, mirrored parties
        sender = ev["participant"]
        receiver = ev["to"]
        best = None
        for rv in receive_events:
            if rv["eid"] in matched_recv:
                continue
            # time order: receive not earlier than send
            if pd.notna(ev["timestamp"]) and pd.notna(rv["timestamp"]):
                if rv["timestamp"] < ev["timestamp"]:
                    continue
            elif rv["idx"] < ev["idx"]:
                # if timestamps missing, fall back to within-case order
                continue
            # mirrored parties: receive.from == send.participant
            #                   receive.to   == send.to
            if rv["from"] is not None and sender is not None and rv["from"] != sender:
                continue
            if rv["to"] is not None and receiver is not None and rv["to"] != receiver:
                continue
            # earliest by (timestamp, idx)
            key = (rv["timestamp"], rv["idx"])
            if best is None or key < (best["timestamp"], best["idx"]):
                best = rv
        if best is not None:
            rho[ev["eid"]] = best["eid"]
            matched_recv.add(best["eid"])

    unmatched_recv = [rv["eid"] for rv in receive_events
                      if rv["eid"] not in matched_recv]
    return rho, send_eids, unmatched_recv


# =====================================================================
# The transformation mu (rules M1-M8)
# =====================================================================

@dataclass
class TransformResult:
    events_df: pd.DataFrame
    objects_df: pd.DataFrame
    relations_df: pd.DataFrame
    o2o_df: pd.DataFrame
    stats: Dict[str, Any]


def transform(df: pd.DataFrame, cfg: Optional[MappingConfig] = None) -> TransformResult:
    """Apply rules M1-M8 and return the four OCEL DataFrames plus stats.

    The function is pure pandas/Python; it does not touch pm4py, so it
    can be tested in isolation.
    """
    cfg = cfg or MappingConfig()
    _require_columns(df, cfg)
    cases = _sorted_case_events(df, cfg)

    # Accumulators -----------------------------------------------------
    event_rows: List[Dict[str, Any]] = []
    object_rows: Dict[str, Dict[str, Any]] = {}   # oid -> object row
    e2o_rows: List[Dict[str, Any]] = []
    o2o_rows: List[Dict[str, Any]] = []

    # Residual (M8) event-attribute keys: every column not consumed.
    residual_keys = [c for c in df.columns
                     if c not in cfg.consumed_keys
                     and c not in (cfg.elemtype_key,)  # elemType handled by M5
                     and not c.startswith("__")]

    participant_seen: set = set()
    n_unmatched_recv = 0
    n_messages = 0

    def _ensure_object(oid: str, otype: str, attrs: Dict[str, Any]) -> None:
        if oid not in object_rows:
            row = {COL_OID: oid, COL_OTYPE: otype}
            row.update(attrs)
            object_rows[oid] = row

    for case, evlist in cases.items():
        # ---- M1: CollaborationInstance object -----------------------
        ci_oid = _ci_id(case)
        _ensure_object(ci_oid, OT_CI, {"caseId": case})

        # ---- M4 (correlation) before emitting message E2O/O2O -------
        rho, send_eids, unmatched_recv = _correlate_messages(evlist)
        n_unmatched_recv += len(unmatched_recv)

        # index events by eid for receive lookups
        by_eid = {e["eid"]: e for e in evlist}

        # ---- M4: Message objects (one per send) ---------------------
        # built here so attributes are available; E2O/O2O below.
        send_to_msg: Dict[str, str] = {}
        for ev in evlist:
            if ev["elem"] != ELEM_SEND:
                continue
            msg_oid = _message_id(ev["eid"])
            send_to_msg[ev["eid"]] = msg_oid
            _ensure_object(msg_oid, OT_MESSAGE, {
                "sender": ev["participant"],
                "receiver": ev["to"],
            })
            n_messages += 1

        for ev in evlist:
            p = ev["participant"]

            # ---- M2: Participant object (orchestration/pool;
            #          log-level scope) -------------------------------
            # Participant represents the orchestration (BPMN pool), not a
            # role or person. Also create participant objects for the
            # from/to (sender/receiver) identifiers of messages.
            for pid in (p, ev["from"], ev["to"]):
                if pid is not None and pid not in participant_seen:
                    _ensure_object(_participant_id(pid), OT_PARTICIPANT, {"name": pid})
                    participant_seen.add(pid)

            # ---- M3: LocalCase object (execution of the orchestration
            #          within this global case) -----------------------
            lc_oid = None
            if p is not None:
                lc_oid = _localcase_id(case, p)
                _ensure_object(lc_oid, OT_LOCALCASE,
                               {"caseId": case, "participant": p})

            # ---- M5: Event (evtype = activity; elemType attribute) --
            ev_row = {
                COL_EID: ev["eid"],
                COL_ACTIVITY: ev["activity"],
                COL_TIMESTAMP: ev["timestamp"],
                "elemType": ev["elem"],
            }
            # ---- M8: residual source attributes (unchanged) ---------
            for k in residual_keys:
                val = ev["row"].get(k)
                if pd.notna(val):
                    ev_row[k] = val
            event_rows.append(ev_row)

            # ---- M6: structural E2O relations -----------------------
            # within (-> CollaborationInstance) and local (-> LocalCase).
            # There is NO direct actor edge to the Participant: the
            # orchestration is reached via local -> executed_by (O2O).
            e2o_rows.append({COL_EID: ev["eid"], COL_OID: ci_oid,
                             COL_OTYPE: OT_CI, COL_QUALIFIER: Q_WITHIN})
            if lc_oid is not None:
                e2o_rows.append({COL_EID: ev["eid"], COL_OID: lc_oid,
                                 COL_OTYPE: OT_LOCALCASE, COL_QUALIFIER: Q_LOCAL})

            # ---- M6: send E2O ---------------------------------------
            if ev["elem"] == ELEM_SEND:
                msg_oid = send_to_msg[ev["eid"]]
                e2o_rows.append({COL_EID: ev["eid"], COL_OID: msg_oid,
                                 COL_OTYPE: OT_MESSAGE, COL_QUALIFIER: Q_SEND})

                # ---- M7: O2O relations for this Message -------------
                o2o_rows.append({COL_OID: msg_oid, COL_OID2: _participant_id(ev["participant"]),
                                 COL_QUALIFIER: Q_FROM})
                if ev["to"] is not None:
                    o2o_rows.append({COL_OID: msg_oid, COL_OID2: _participant_id(ev["to"]),
                                     COL_QUALIFIER: Q_TO})
                o2o_rows.append({COL_OID: msg_oid, COL_OID2: ci_oid,
                                 COL_QUALIFIER: Q_EXCHANGED_IN})

        # ---- M6: receive E2O (only matched receives) ----------------
        for send_eid, recv_eid in rho.items():
            msg_oid = send_to_msg[send_eid]
            e2o_rows.append({COL_EID: recv_eid, COL_OID: msg_oid,
                             COL_OTYPE: OT_MESSAGE, COL_QUALIFIER: Q_RECEIVE})

        # ---- M7: LocalCase O2O relations ----------------------------
        # one per (case, participant) seen in this case
        seen_lc: set = set()
        for ev in evlist:
            p = ev["participant"]
            if p is None:
                continue
            lc_oid = _localcase_id(case, p)
            if lc_oid in seen_lc:
                continue
            seen_lc.add(lc_oid)
            o2o_rows.append({COL_OID: lc_oid, COL_OID2: ci_oid,
                             COL_QUALIFIER: Q_PART_OF})
            o2o_rows.append({COL_OID: lc_oid, COL_OID2: _participant_id(p),
                             COL_QUALIFIER: Q_EXECUTED_BY})

    # ---- assemble DataFrames ----------------------------------------
    events_df = pd.DataFrame(event_rows)
    objects_df = pd.DataFrame(list(object_rows.values()))
    relations_df = pd.DataFrame(e2o_rows)
    o2o_df = pd.DataFrame(o2o_rows)

    # Ensure canonical dtypes / ordering of key columns where present.
    if not events_df.empty:
        events_df[COL_TIMESTAMP] = pd.to_datetime(events_df[COL_TIMESTAMP], utc=True,
                                                  errors="coerce")
        front = [COL_EID, COL_ACTIVITY, COL_TIMESTAMP]
        events_df = events_df[[c for c in front if c in events_df.columns]
                              + [c for c in events_df.columns if c not in front]]

    stats = {
        "n_events": len(events_df),
        "n_objects": len(objects_df),
        "n_objects_by_type": (objects_df[COL_OTYPE].value_counts().to_dict()
                              if not objects_df.empty else {}),
        "n_e2o": len(relations_df),
        "n_o2o": len(o2o_df),
        "n_messages": n_messages,
        "n_unmatched_receives": n_unmatched_recv,
        "n_cases": len(cases),
    }
    return TransformResult(events_df, objects_df, relations_df, o2o_df, stats)


# =====================================================================
# Consistency checks P1.1 - P1.5
# =====================================================================

@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def run_consistency_checks(src_df: pd.DataFrame,
                           res: TransformResult,
                           cfg: Optional[MappingConfig] = None) -> List[CheckResult]:
    """Machine-check P1.1-P1.5 against the constructed DataFrames.

    These guard against implementation defects; they are independent of
    the by-construction argument in the appendix.
    """
    cfg = cfg or MappingConfig()
    out: List[CheckResult] = []
    ev = res.events_df
    rel = res.relations_df
    o2o = res.o2o_df
    obj = res.objects_df

    # ---- P1.1 Totality: one OCEL event per source event, ts preserved
    n_src = len(src_df)
    n_ev = len(ev)
    p11 = (n_src == n_ev) and (COL_TIMESTAMP in ev.columns) \
        and (ev[COL_TIMESTAMP].notna().sum() == src_df[cfg.timestamp_key].notna().sum()
             if cfg.timestamp_key in src_df.columns else True)
    out.append(CheckResult(
        "P1.1 Totality",
        bool(p11),
        f"source events={n_src}, ocel events={n_ev}; "
        f"timestamps preserved (non-null parity checked)."))

    # ---- P1.2 Per-case partition: within-image of each ci == case set
    # Build, per CI object, the set of event ids related by 'within';
    # compare its size to the number of source events of that case.
    if not rel.empty:
        within = rel[(rel[COL_QUALIFIER] == Q_WITHIN) & (rel[COL_OTYPE] == OT_CI)]
        within_counts = within.groupby(COL_OID)[COL_EID].nunique().to_dict()
    else:
        within_counts = {}
    # expected per-case counts from source
    if cfg.case_key in src_df.columns:
        src_case_counts = src_df.groupby(cfg.case_key).size().to_dict()
        expected = {_ci_id(str(c)): n for c, n in src_case_counts.items()}
    else:
        expected = {}
    mismatches = {k: (within_counts.get(k, 0), v)
                  for k, v in expected.items() if within_counts.get(k, 0) != v}
    # also: every within edge points at an existing CI object
    ci_ids = set(obj[obj[COL_OTYPE] == OT_CI][COL_OID]) if not obj.empty else set()
    dangling = set(within_counts) - ci_ids
    p12 = (len(mismatches) == 0) and (len(dangling) == 0)
    out.append(CheckResult(
        "P1.2 Per-case partition",
        bool(p12),
        f"CI objects={len(ci_ids)}; count mismatches={len(mismatches)}; "
        f"dangling within-targets={len(dangling)}."
        + ("" if p12 else f" first mismatches={dict(list(mismatches.items())[:5])}")))

    # ---- P1.3 Message well-formedness ------------------------------
    # exactly one send, at most one receive per Message; from/to agree
    # with sender/receiver attributes; receive ts >= send ts.
    p13_ok = True
    p13_detail_bits: List[str] = []
    if not rel.empty:
        msg_rel = rel[rel[COL_OTYPE] == OT_MESSAGE]
        send_counts = msg_rel[msg_rel[COL_QUALIFIER] == Q_SEND].groupby(COL_OID).size()
        recv_counts = msg_rel[msg_rel[COL_QUALIFIER] == Q_RECEIVE].groupby(COL_OID).size()
        msg_ids = set(obj[obj[COL_OTYPE] == OT_MESSAGE][COL_OID]) if not obj.empty else set()

        bad_send = [m for m in msg_ids if int(send_counts.get(m, 0)) != 1]
        bad_recv = [m for m in msg_ids if int(recv_counts.get(m, 0)) > 1]
        p13_detail_bits.append(f"messages={len(msg_ids)}")
        p13_detail_bits.append(f"!=1 send: {len(bad_send)}")
        p13_detail_bits.append(f">1 receive: {len(bad_recv)}")
        if bad_send or bad_recv:
            p13_ok = False

        # from/to O2O agreement with sender/receiver object attributes
        if not o2o.empty and not obj.empty:
            obj_idx = obj.set_index(COL_OID)
            o2o_msg = o2o[o2o[COL_QUALIFIER].isin([Q_FROM, Q_TO])]
            disagreements = 0
            ts_violations = 0
            for m in msg_ids:
                if m not in obj_idx.index:
                    continue
                sender = obj_idx.at[m, "sender"] if "sender" in obj_idx.columns else None
                receiver = obj_idx.at[m, "receiver"] if "receiver" in obj_idx.columns else None
                froms = o2o_msg[(o2o_msg[COL_OID] == m) & (o2o_msg[COL_QUALIFIER] == Q_FROM)][COL_OID2].tolist()
                tos = o2o_msg[(o2o_msg[COL_OID] == m) & (o2o_msg[COL_QUALIFIER] == Q_TO)][COL_OID2].tolist()
                if sender is not None and froms and froms[0] != _participant_id(str(sender)):
                    disagreements += 1
                if receiver is not None and tos and tos[0] != _participant_id(str(receiver)):
                    disagreements += 1
            p13_detail_bits.append(f"from/to disagreements: {disagreements}")
            if disagreements:
                p13_ok = False

            # receive ts >= send ts
            if not ev.empty:
                ev_ts = ev.set_index(COL_EID)[COL_TIMESTAMP].to_dict()
                send_edges = msg_rel[msg_rel[COL_QUALIFIER] == Q_SEND][[COL_OID, COL_EID]]
                recv_edges = msg_rel[msg_rel[COL_QUALIFIER] == Q_RECEIVE][[COL_OID, COL_EID]]
                send_eid_by_msg = dict(zip(send_edges[COL_OID], send_edges[COL_EID]))
                for _, r in recv_edges.iterrows():
                    s_eid = send_eid_by_msg.get(r[COL_OID])
                    if s_eid is None:
                        continue
                    t_send = ev_ts.get(s_eid)
                    t_recv = ev_ts.get(r[COL_EID])
                    if pd.notna(t_send) and pd.notna(t_recv) and t_recv < t_send:
                        ts_violations += 1
                p13_detail_bits.append(f"receive<send ts: {ts_violations}")
                if ts_violations:
                    p13_ok = False
    out.append(CheckResult("P1.3 Message well-formedness", bool(p13_ok),
                           "; ".join(p13_detail_bits) or "no messages"))

    # ---- P1.4 Local-case coherence ---------------------------------
    # Without a direct actor edge, coherence is checked in two parts:
    #  (a) for every event with a 'local' object lc, lc is 'part_of' the
    #      event's 'within' object (the global instance);
    #  (b) every LocalCase is 'executed_by' exactly one Participant, and
    #      that Participant's name equals the LocalCase's 'participant'
    #      attribute.
    p14_ok = True
    p14_detail = ""
    if not rel.empty and not o2o.empty:
        ev_local = rel[(rel[COL_QUALIFIER] == Q_LOCAL)][[COL_EID, COL_OID]]
        ev_within = dict(zip(
            rel[rel[COL_QUALIFIER] == Q_WITHIN][COL_EID],
            rel[rel[COL_QUALIFIER] == Q_WITHIN][COL_OID]))
        part_of = {(r[COL_OID]): r[COL_OID2] for _, r in
                   o2o[o2o[COL_QUALIFIER] == Q_PART_OF].iterrows()}

        # (a) local/within coherence
        bad_partof = 0
        for _, r in ev_local.iterrows():
            eid, lc = r[COL_EID], r[COL_OID]
            if part_of.get(lc) != ev_within.get(eid):
                bad_partof += 1

        # (b) executed_by well-formedness, per LocalCase
        lc_ids = set(obj[obj[COL_OTYPE] == OT_LOCALCASE][COL_OID]) if not obj.empty else set()
        exec_edges = o2o[o2o[COL_QUALIFIER] == Q_EXECUTED_BY]
        exec_counts = exec_edges.groupby(COL_OID).size().to_dict()
        exec_target = dict(zip(exec_edges[COL_OID], exec_edges[COL_OID2]))
        obj_idx = obj.set_index(COL_OID) if not obj.empty else None
        bad_exec = 0
        bad_name = 0
        for lc in lc_ids:
            if int(exec_counts.get(lc, 0)) != 1:
                bad_exec += 1
                continue
            # name agreement: executed_by target's 'name' == lc.participant
            tgt = exec_target.get(lc)
            if obj_idx is not None and tgt in obj_idx.index:
                tgt_name = obj_idx.at[tgt, "name"] if "name" in obj_idx.columns else None
                lc_part = obj_idx.at[lc, "participant"] if "participant" in obj_idx.columns else None
                if tgt_name is not None and lc_part is not None and str(tgt_name) != str(lc_part):
                    bad_name += 1

        p14_ok = (bad_partof == 0) and (bad_exec == 0) and (bad_name == 0)
        p14_detail = (f"local/within mismatches={bad_partof}; "
                      f"local cases !=1 executed_by={bad_exec}; "
                      f"executed_by name disagreements={bad_name}")
    out.append(CheckResult("P1.4 Local-case coherence", bool(p14_ok), p14_detail))

    # ---- P1.5 No orphan objects ------------------------------------
    related_oids = set()
    if not rel.empty:
        related_oids |= set(rel[COL_OID])
    if not o2o.empty:
        related_oids |= set(o2o[COL_OID]) | set(o2o[COL_OID2])
    all_oids = set(obj[COL_OID]) if not obj.empty else set()
    orphans = all_oids - related_oids
    out.append(CheckResult("P1.5 No orphan objects", len(orphans) == 0,
                           f"objects={len(all_oids)}; orphans={len(orphans)}"
                           + ("" if not orphans else f"; e.g. {list(orphans)[:5]}")))
    return out


def print_check_report(checks: List[CheckResult], stats: Dict[str, Any]) -> bool:
    logger.info("---- transformation stats ----")
    for k, v in stats.items():
        logger.info("  %-22s %s", k, v)
    logger.info("---- consistency checks (P1.1-P1.5) ----")
    all_ok = True
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        logger.info("  [%s] %-26s %s", status, c.name, c.detail)
        all_ok = all_ok and c.passed
    logger.info("---- overall: %s ----", "PASS" if all_ok else "FAIL")
    return all_ok


# =====================================================================
# Orchestration
# =====================================================================

def convert(input_xes: str,
            output: str,
            cfg: Optional[MappingConfig] = None,
            strict: bool = False,
            validate: bool = True,
            encoding: str = "utf-8") -> TransformResult:
    """Full pipeline: read XES -> transform (M1-M8) -> check (P1) ->
    build OCEL -> export -> validate against the OCEL 2.0 JSON
    schema.

    If strict=True, a failing consistency check aborts before export.
    If validate=True (default), the .jsonocel file is checked against the
    OCEL 2.0 JSON schema; problems are logged, and under strict=True a
    schema violation raises.
    """
    cfg = cfg or MappingConfig()
    # Strip any extension the caller may have supplied so the base is clean.
    base = output_base
    for ext in (".jsonocel", ".sqlite", ".json"):
        if base.lower().endswith(ext):
            base = base[: -len(ext)]
            break

    json_path = base + ".jsonocel"
    sqlite_path = base + ".sqlite"

    src_df = read_collaborative_xes(input_xes, encoding=encoding)
    res = transform(src_df, cfg)
    checks = run_consistency_checks(src_df, res, cfg)
    all_ok = print_check_report(checks, res.stats)
    if strict and not all_ok:
        raise RuntimeError("Consistency checks failed under strict mode; not exporting.")
    ocel = build_ocel_object(res.events_df, res.objects_df,
                             res.relations_df, res.o2o_df)
    write_ocel2_json(ocel, output+".jsonocel")
    write_ocel2_sqlite(ocel, output+".sqlite")

    if validate:
        problems = validate_jsonocel(output+".jsonocel")
        if problems:
            logger.warning("OCEL 2.0 JSON schema validation found %d issue(s):",
                           len(problems))
            for p in problems[:25]:
                logger.warning("  - %s", p)
            if len(problems) > 25:
                logger.warning("  ... and %d more.", len(problems) - 25)
            if strict:
                raise RuntimeError("Exported .jsonocel does not conform to the "
                                   "OCEL 2.0 JSON schema (strict mode).")
        else:
            logger.info("OCEL 2.0 JSON schema validation: PASS.")
    return res


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Transform an extended collaborative XES log into an "
                    "OCEL 2.0 log (.jsonocel + .sqlite) per mapping rules M1-M8.")
    p.add_argument("input_xes", help="Path to the extended collaborative XES file.")
    p.add_argument("output", help="Path to the output files. Provide <output_base> name.")
    p.add_argument("--strict", action="store_true",
                   help="Abort if any P1 check or schema validation fails.")
    p.add_argument("--no-validate", action="store_true",
                   help="Skip OCEL 2.0 JSON schema validation of the output.")
    p.add_argument("--encoding", default="utf-8")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s")
    cfg = MappingConfig()
    convert(args.input_xes, args.output, cfg=cfg,
            strict=args.strict, validate=not args.no_validate,
            encoding=args.encoding)
    return 0


if __name__ == "__main__":
    sys.exit(main())
