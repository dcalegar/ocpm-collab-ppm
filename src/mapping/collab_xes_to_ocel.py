#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collab_xes_to_ocel.py
=====================================================================
Model-to-model transformation mu: extended collaborative XES log
  -->  OCEL 2.0 log (conforming to Berti et al. 2024, Definition 2),
exported to the .jsonocel and .sqlite formats.

This implements rules M1-M8 and the consistency criteria P1.1-P1.6 of
the mapping section. It is a transformation that produces an event log
CONFORMING to the OCEL 2.0 metamodel; it does NOT define a new
metamodel.

Source side (extended collaborative XES), per the collab.xesext
extension, carries these event-level string attributes:
    collab:elemType        in {task, SendTask, ReceiveTask}
    collab:participant     participant executing the event
    collab:fromParticipant sender (own side always backfilled from
                            collab:participant on Send events; counterparty
                            side on Receive events, may be absent)
    collab:toParticipant   receiver (own side always backfilled from
                            collab:participant on Receive events; counterparty
                            side on Send events, may be absent)
plus the XES keys for activity, timestamp, and global case id.

The Message object represents a single communication OBSERVATION, not a
correlated message instance: rule M4 mints one Message per send event and
one Message per receive event, each related to exactly its own event (by
`send` or `receive`) and carrying that event's recorded sender/receiver --
when the source records the counterparty side; when it does not, the
sender/receiver attribute and the corresponding from/to O2O relation are
left undefined rather than guessed (see _sorted_case_events, P1.3 below).
Because the source logs do not guarantee message identifiers or other
reliable correlation information, the core mapping does NOT infer any
correspondence between a send observation and a receive observation.
collab:elemType already distinguishes task / SendTask / ReceiveTask, so no
separate message-type attribute is read or stored.

Target side (OCEL 2.0):
    Object types : CollaborationCase, Participant, ParticipantProjection,
                   Message.
                   NOTE on Participant: in the collaborative source
                   process (BPMN 2.0 terminology) a participant is an
                   orchestration / pool (e.g. "Laboratory",
                   "Gynecologist"), not a role or a person. The object
                   type keeps the name "Participant" for traceability
                   with the collaborative process, but it represents the
                   orchestration: a global object, reused across cases.
                   ParticipantProjection is the execution of that
                   orchestration within one collaboration case; it is a
                   distinct object, linked to its Participant by the O2O
                   qualifier for_participant.
    E2O qualifiers: within, in_projection, send, receive, participant
                   (M6). The participant of an event is reachable two
                   ways: directly, via the `participant` edge below, and
                   indirectly via in_projection -> for_participant,
                   keeping the orchestration distinct from its own
                   per-case projection. Both are part of the conceptual
                   model -- the direct edge keeps Participant objects
                   reachable at the event-to-object level for
                   serialization and downstream feature extraction (see
                   NOTE below); their agreement is machine-checked by
                   P1.6. `send`/`receive` each relate a communication
                   event to its OWN Message observation object; no
                   relation between distinct send and receive events is
                   inferred (M4).
    O2O qualifiers: projection_of, for_participant, from, to, exchanged_in

    NOTE on the `participant` E2O edge: pm4py's OCEL 2.0 exporters (JSON
    and SQLite) call filtering_utils.propagate_relations_filtering(),
    which keeps only the objects that appear in the E2O relations table
    -- silently dropping any object reachable only via O2O. Since
    Participant would otherwise be reached only via O2O (for_participant,
    from, to), it and those edges would be lost on export without a
    direct event edge. Rule M6 keeps the direct `participant` edge for
    exactly this reason, so the exported .jsonocel/.sqlite stays
    lossless.

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
CASE_KEY = "case:concept:name"
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
OT_CC = "CollaborationCase"
OT_PARTICIPANT = "Participant"
OT_PP = "ParticipantProjection"
OT_MESSAGE = "Message"

# --- E2O qualifiers (rule M6) ---------------------------------------
Q_WITHIN = "within"
Q_IN_PROJECTION = "in_projection"
Q_SEND = "send"
Q_RECEIVE = "receive"
# Direct event->Participant edge (M6). Also reachable indirectly via
# in_projection -> for_participant; P1.6 checks the two agree. See the
# module docstring NOTE above for why the direct edge is kept.
Q_PARTICIPANT = "participant"

# --- O2O qualifiers (rule M7) ---------------------------------------
Q_PROJECTION_OF = "projection_of"
Q_FOR_PARTICIPANT = "for_participant"
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

def _cc_id(case: str) -> str:
    return f"cc::{case}"

def _participant_id(p: str) -> str:
    return f"part::{p}"

def _pp_id(case: str, p: str) -> str:
    return f"pp::{case}::{p}"

def _message_id(eid: str) -> str:
    # One Message per send OR receive event (M4, m_e for e in S_L u R_L);
    # the event id makes it unique. No correlation between a send Message
    # and a receive Message is inferred.
    return f"msg::{eid}"

def _event_id(case: str, idx: int, width: int = 1) -> str:
    # Stable per-case event id. idx is the within-case source order (the
    # rank of e in prec_L, Definition 1). Zero-padded to `width` digits so
    # that lexicographic string order on the id agrees with numeric idx
    # order -- required for mu_E to be an order-embedding of prec_L:
    # without padding, "e::459::10" sorts before "e::459::9".
    return f"e::{case}::{str(idx).zfill(width)}"


# =====================================================================
# I/O endpoints (the only pm4py-dependent functions)
# =====================================================================

def _raw_timestamps_in_file_order(path: str) -> List[str]:
    """Extract every event's raw time:timestamp attribute string directly
    from the XML, in exact file order (trace-by-trace, event-by-event),
    bypassing pm4py's datetime parser entirely -- see _correct_utc_timestamps
    for why."""
    from lxml import etree
    root = etree.parse(path).getroot()
    out: List[str] = []
    for trace in root.findall("trace"):
        for ev in trace.findall("event"):
            for d in ev.findall("date"):
                if d.get("key") == TIMESTAMP_KEY:
                    out.append(d.get("value"))
                    break
    return out


def _correct_utc_timestamps(path: str, n_events: int) -> pd.Series:
    """Re-derive the timestamp column directly from the raw XES, in UTC.

    pm4py's ISO8601 datetime parsers -- both the default
    ``strpfromiso`` variant (pm4py.util.dt_parsing.variants.strpfromiso.
    fix_naivety) and its ``dummy`` fallback -- normalize to UTC via
    ``datetime.replace(tzinfo=timezone.utc)`` instead of
    ``astimezone(timezone.utc)``. ``replace`` only overwrites the tzinfo
    label; it does not shift the clock reading, so any timestamp with a
    non-zero UTC offset comes back with its original LOCAL wall-clock
    digits silently mislabeled as UTC. pm4py itself warns about this
    ("ISO8601 strings are not fully supported with strpfromiso for
    Python versions below 3.11") but still returns the wrong value
    instead of raising.

    This is silent for logs whose events all share one fixed UTC offset
    (it cancels out in any time delta), but is wrong for any source log
    whose offset varies across events -- e.g. a timezone that observes
    DST, where a duration whose two endpoints straddle the DST change
    comes out off by exactly one hour. Example: source
    ``2012-05-23T01:22:25+02:00`` (true UTC 2012-05-22 23:22:25) comes
    back from pm4py as ``2012-05-23 01:22:25+00:00`` -- the local digits,
    not the UTC instant.

    Bypasses the bug by re-parsing the raw attribute strings ourselves
    with pandas (which converts non-UTC offsets to UTC correctly)
    instead of trusting pm4py's parsed column.
    """
    raw = _raw_timestamps_in_file_order(path)
    if len(raw) != n_events:
        raise ValueError(
            f"Raw timestamp count from XML ({len(raw)}) does not match "
            f"pm4py's parsed event count ({n_events}); cannot safely "
            f"realign timestamps to fix the UTC-offset bug (see "
            f"_correct_utc_timestamps docstring)."
        )
    return pd.to_datetime(pd.Series(raw), utc=True, errors="coerce")


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
    # Work around pm4py's UTC-offset parsing bug (see
    # _correct_utc_timestamps): re-derive TIMESTAMP_KEY from the raw XML
    # rather than trusting pm4py's own parsed column.
    df = df.reset_index(drop=True)
    df[TIMESTAMP_KEY] = _correct_utc_timestamps(path, len(df))
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
    breaking ties by the original source appearance order (the per-case
    order prec_L of Definition 1). Ordering by timestamp with a stable
    tie-break guarantees the trace order is preserved before and after
    the mapping; event ids are minted in this order, so reconstruction
    (P1.2) reads the event-identifier order without re-sorting by
    timestamp.
    Returns, per case, a list of event dicts enriched with a minted eid
    and the within-case index.
    """
    def _clean(v: Any) -> Optional[str]:
        # Treat NaN, None, and empty/whitespace strings as absent, so a
        # blank fromParticipant/toParticipant/participant never mints a
        # spurious object (e.g. an empty-named Participant).
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        if pd.isna(v):
            return None
        s = str(v).strip()
        return s if s != "" else None

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
        # prec_L: order by timestamp, ties broken by original source
        # appearance (__src_order__); a stable sort keeps the trace order
        # identical before and after the mapping (M5).
        grp = grp.sort_values(by=[cfg.timestamp_key, "__src_order__"],
                              kind="mergesort")  # stable
        # Width to zero-pad idx so id order matches prec_L order (see
        # _event_id): must cover the largest index in this case, n-1.
        idx_width = max(1, len(str(max(len(grp) - 1, 0))))
        evlist: List[Dict[str, Any]] = []
        for idx, (_, row) in enumerate(grp.iterrows()):
            elem = _clean(row.get(cfg.elemtype_key)) or ELEM_TASK
            participant = _clean(row.get(cfg.participant_key))
            from_p = (_clean(row.get(cfg.from_key))
                      if cfg.from_key in work.columns else None)
            to_p = (_clean(row.get(cfg.to_key))
                    if cfg.to_key in work.columns else None)
            # Def. app-r1 well-formedness (i)/(ii): part(e)=from(e) for a
            # SendTask and part(e)=to(e) for a ReceiveTask; from/to are
            # total on S_L u R_L. The source log may leave the "own side"
            # of a send/receive event implicit (recording only the
            # counterparty), relying on collab:participant to supply it;
            # backfill it here so from/to are total, as M7/M8 assume.
            if elem == ELEM_SEND and from_p is None:
                from_p = participant
            if elem == ELEM_RECEIVE and to_p is None:
                to_p = participant
            evlist.append({
                "eid": _event_id(str(case_val), idx, idx_width),
                "case": str(case_val),
                "idx": idx,
                "activity": row[cfg.activity_key],
                "timestamp": row[cfg.timestamp_key],
                "elem": elem,
                "participant": participant,
                "from": from_p,
                "to": to_p,
                "row": row,
            })
        cases[str(case_val)] = evlist
    return cases


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

    # Residual (M8) event-attribute keys: every column not consumed. The
    # consumed keys (participant/from/to/elemType) are still preserved as
    # event attributes below, but under fixed, explicit names (M8), so
    # they are excluded here to avoid emitting them twice.
    residual_keys = [c for c in df.columns
                     if c not in cfg.consumed_keys
                     and c not in (cfg.elemtype_key,)  # elemType handled by M5
                     and not c.startswith("__")]

    participant_seen: set = set()
    n_messages = 0
    # Counterparty completeness (Normalization paragraph, appendix): the source
    # format may leave a send/receive event's OWN side implicit (backfilled
    # above from collab:participant), but it may also leave the COUNTERPARTY
    # side (toParticipant for a Send, fromParticipant for a Receive) genuinely
    # absent. That side is NOT guessed: from(e)/to(e) stay undefined (None),
    # no 'from'/'to' O2O relation is created for that side, and the Message's
    # sender/receiver object attribute stays unset. Tracked here so it is
    # visible in stats/P1.3 rather than silently dropped.
    n_messages_missing_sender = 0
    n_messages_missing_receiver = 0

    def _ensure_object(oid: str, otype: str, attrs: Dict[str, Any]) -> None:
        if oid not in object_rows:
            row = {COL_OID: oid, COL_OTYPE: otype}
            row.update(attrs)
            object_rows[oid] = row

    for case, evlist in cases.items():
        # ---- M1: CollaborationCase object -----------------------
        cc_oid = _cc_id(case)
        _ensure_object(cc_oid, OT_CC, {"caseId": case})

        # ---- M4: Message objects (one per send OR receive event) ----
        # Each communication event mints its OWN Message observation
        # object (m_e for e in S_L u R_L); the core mapping does not
        # infer any correspondence between a send and a receive
        # observation, since the source logs do not guarantee message
        # identifiers or other reliable correlation information.
        # Built here so attributes are available; E2O/O2O below.
        msg_by_eid: Dict[str, str] = {}
        for ev in evlist:
            if ev["elem"] not in (ELEM_SEND, ELEM_RECEIVE):
                continue
            msg_oid = _message_id(ev["eid"])
            msg_by_eid[ev["eid"]] = msg_oid
            _ensure_object(msg_oid, OT_MESSAGE, {
                "sender": ev["from"],
                "receiver": ev["to"],
            })
            n_messages += 1
            if ev["from"] is None:
                n_messages_missing_sender += 1
            if ev["to"] is None:
                n_messages_missing_receiver += 1

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

            # ---- M3: ParticipantProjection object (execution of the
            #          orchestration within this collaboration case) ---
            pp_oid = None
            if p is not None:
                pp_oid = _pp_id(case, p)
                _ensure_object(pp_oid, OT_PP,
                               {"caseId": case, "participant": p})

            # ---- M5: Event (evtype = activity; elemType attribute) --
            ev_row = {
                COL_EID: ev["eid"],
                COL_ACTIVITY: ev["activity"],
                COL_TIMESTAMP: ev["timestamp"],
                "elemType": ev["elem"],
            }
            # ---- M8: structural attribute preservation ---------------
            # collab:participant, collab:elemType, fromParticipant, and
            # toParticipant are retained as event attributes even though
            # the corresponding participant/endpoints are also
            # materialized by E2O/O2O relations (M6/M7).
            if p is not None:
                ev_row["participant"] = p
            if ev["elem"] in (ELEM_SEND, ELEM_RECEIVE):
                if ev["from"] is not None:
                    ev_row["fromParticipant"] = ev["from"]
                if ev["to"] is not None:
                    ev_row["toParticipant"] = ev["to"]
            # ---- M8: residual source attributes (unchanged) ---------
            for k in residual_keys:
                val = ev["row"].get(k)
                if pd.notna(val):
                    ev_row[k] = val
            event_rows.append(ev_row)

            # ---- M6: structural E2O relations -----------------------
            # within (-> CollaborationCase), in_projection (-> Participant-
            # Projection), and the direct participant edge (-> Participant),
            # whose agreement with in_projection -> for_participant is
            # checked by P1.6 (see module docstring).
            e2o_rows.append({COL_EID: ev["eid"], COL_OID: cc_oid,
                             COL_OTYPE: OT_CC, COL_QUALIFIER: Q_WITHIN})
            if pp_oid is not None:
                e2o_rows.append({COL_EID: ev["eid"], COL_OID: pp_oid,
                                 COL_OTYPE: OT_PP, COL_QUALIFIER: Q_IN_PROJECTION})
            if p is not None:
                e2o_rows.append({COL_EID: ev["eid"], COL_OID: _participant_id(p),
                                 COL_OTYPE: OT_PARTICIPANT, COL_QUALIFIER: Q_PARTICIPANT})

            # ---- M6: send/receive E2O --------------------------------
            # Each communication event is related only to its OWN Message
            # observation object; no send/receive correlation is made.
            if ev["elem"] in (ELEM_SEND, ELEM_RECEIVE):
                msg_oid = msg_by_eid[ev["eid"]]
                msg_qualifier = Q_SEND if ev["elem"] == ELEM_SEND else Q_RECEIVE
                e2o_rows.append({COL_EID: ev["eid"], COL_OID: msg_oid,
                                 COL_OTYPE: OT_MESSAGE, COL_QUALIFIER: msg_qualifier})

                # ---- M7: O2O relations for this Message -------------
                if ev["from"] is not None:
                    o2o_rows.append({COL_OID: msg_oid, COL_OID2: _participant_id(ev["from"]),
                                     COL_QUALIFIER: Q_FROM})
                if ev["to"] is not None:
                    o2o_rows.append({COL_OID: msg_oid, COL_OID2: _participant_id(ev["to"]),
                                     COL_QUALIFIER: Q_TO})
                o2o_rows.append({COL_OID: msg_oid, COL_OID2: cc_oid,
                                 COL_QUALIFIER: Q_EXCHANGED_IN})

        # ---- M7: ParticipantProjection O2O relations ----------------
        # one per (case, participant) seen in this case
        seen_pp: set = set()
        for ev in evlist:
            p = ev["participant"]
            if p is None:
                continue
            pp_oid = _pp_id(case, p)
            if pp_oid in seen_pp:
                continue
            seen_pp.add(pp_oid)
            o2o_rows.append({COL_OID: pp_oid, COL_OID2: cc_oid,
                             COL_QUALIFIER: Q_PROJECTION_OF})
            o2o_rows.append({COL_OID: pp_oid, COL_OID2: _participant_id(p),
                             COL_QUALIFIER: Q_FOR_PARTICIPANT})

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
        "n_messages_missing_sender": n_messages_missing_sender,
        "n_messages_missing_receiver": n_messages_missing_receiver,
        "n_cases": len(cases),
    }
    if n_messages_missing_sender or n_messages_missing_receiver:
        logger.warning(
            "%d Message object(s) have no recorded sender and %d have no "
            "recorded receiver (source log leaves the counterparty side "
            "of some send/receive events implicit; see Normalization, "
            "appendix). Their 'from'/'to' O2O relation and sender/receiver "
            "object attribute are left undefined rather than guessed.",
            n_messages_missing_sender, n_messages_missing_receiver)
    return TransformResult(events_df, objects_df, relations_df, o2o_df, stats)


# =====================================================================
# Consistency checks P1.1 - P1.6
# =====================================================================

@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def run_consistency_checks(src_df: pd.DataFrame,
                           res: TransformResult,
                           cfg: Optional[MappingConfig] = None) -> List[CheckResult]:
    """Machine-check P1.1-P1.6 against the constructed DataFrames.

    These guard against implementation defects; they are independent of
    the by-construction argument in the appendix.
    """
    cfg = cfg or MappingConfig()
    out: List[CheckResult] = []
    ev = res.events_df
    rel = res.relations_df
    o2o = res.o2o_df
    obj = res.objects_df

    # Independently recompute the expected per-event identity (case, eid,
    # activity, timestamp, prec_L order) straight from the source log, using
    # the same normalization/ordering logic the transform itself uses. This
    # is deliberately NOT read off `res`: P1.1/P1.2/P1.2b below compare the
    # transform's output against this freshly-derived expectation, rather
    # than against internal state the transform already trusted, so they
    # catch defects that only checking cardinalities/timestamps would miss
    # (e.g. a permuted activity or a swapped case membership that keeps
    # every count the same).
    expected_cases = _sorted_case_events(src_df, cfg)
    expected_by_eid: Dict[str, Tuple[str, Any]] = {
        e["eid"]: (str(e["activity"]), e["timestamp"])
        for evlist in expected_cases.values() for e in evlist}
    expected_eids_by_case: Dict[str, List[str]] = {
        case: [e["eid"] for e in evlist] for case, evlist in expected_cases.items()}

    # ---- P1.1 Totality: one OCEL event per source event, same activity
    # and timestamp as its source counterpart (checked by event identity,
    # not merely by matching aggregate counts).
    n_src = len(src_df)
    n_ev = len(ev)
    ev_by_eid = ev.set_index(COL_EID) if not ev.empty else None
    missing_eids = 0
    activity_mismatches = 0
    timestamp_mismatches = 0
    for eid, (exp_act, exp_ts) in expected_by_eid.items():
        if ev_by_eid is None or eid not in ev_by_eid.index:
            missing_eids += 1
            continue
        if str(ev_by_eid.at[eid, COL_ACTIVITY]) != exp_act:
            activity_mismatches += 1
        got_ts = ev_by_eid.at[eid, COL_TIMESTAMP] if COL_TIMESTAMP in ev_by_eid.columns else None
        if pd.notna(exp_ts) and pd.notna(got_ts) and pd.Timestamp(got_ts) != pd.Timestamp(exp_ts):
            timestamp_mismatches += 1
        elif pd.isna(exp_ts) != pd.isna(got_ts):
            timestamp_mismatches += 1
    p11 = (n_src == n_ev) and missing_eids == 0 \
        and activity_mismatches == 0 and timestamp_mismatches == 0
    out.append(CheckResult(
        "P1.1 Totality",
        bool(p11),
        f"source events={n_src}, ocel events={n_ev}; missing event ids={missing_eids}; "
        f"activity mismatches={activity_mismatches}; timestamp mismatches={timestamp_mismatches}."))

    # ---- P1.2 Per-case partition: within-image of each cc equals the
    # exact SET of source event ids of that case (not merely its size, so a
    # swap of same-count events between two cases is caught).
    if not rel.empty:
        within = rel[(rel[COL_QUALIFIER] == Q_WITHIN) & (rel[COL_OTYPE] == OT_CC)]
        within_sets: Dict[str, set] = within.groupby(COL_OID)[COL_EID].apply(set).to_dict()
    else:
        within = pd.DataFrame(columns=[COL_EID, COL_OID, COL_OTYPE, COL_QUALIFIER])
        within_sets = {}
    expected_sets = {_cc_id(case): set(eids) for case, eids in expected_eids_by_case.items()}
    mismatches = {k: (len(within_sets.get(k, set())), len(v))
                  for k, v in expected_sets.items() if within_sets.get(k, set()) != v}
    # also: every within edge points at an existing CC object
    cc_ids = set(obj[obj[COL_OTYPE] == OT_CC][COL_OID]) if not obj.empty else set()
    dangling = set(within_sets) - cc_ids
    p12 = (len(mismatches) == 0) and (len(dangling) == 0)
    out.append(CheckResult(
        "P1.2 Per-case partition",
        bool(p12),
        f"CC objects={len(cc_ids)}; set mismatches={len(mismatches)}; "
        f"dangling within-targets={len(dangling)}."
        + ("" if p12 else f" first mismatches={dict(list(mismatches.items())[:5])}")))

    # ---- P1.2b Per-case order: identifier order equals prec_L exactly
    # (timestamp order with ties broken by source appearance order, M5).
    # A timestamp-monotonicity-only check (b < a) would accept two same-
    # timestamp events swapped in identifier order, since b == a is not a
    # decrease -- silently violating the tie-break half of prec_L that
    # criterion P1.2/M5 also promises. Instead, compare the identifier
    # order directly against `expected_eids_by_case`, which is prec_L
    # itself (timestamp, then __src_order__): identifiers are minted in
    # that exact sequence (M5), so this also re-catches the id-padding
    # regression that motivated fixed-width `_event_id` (e.g. "e::9"
    # sorting after "e::10" under an unpadded scheme) as a special case.
    order_violations: Dict[str, int] = {}
    if not within.empty:
        for case, expected_order in expected_eids_by_case.items():
            cc_oid = _cc_id(case)
            grp = within[within[COL_OID] == cc_oid]
            if grp.empty:
                continue
            actual_order = sorted(grp[COL_EID].unique())  # lexicographic == identifier order
            if actual_order != expected_order:
                n_bad = sum(1 for a, b in zip(actual_order, expected_order) if a != b)
                order_violations[cc_oid] = max(n_bad, 1)
    p12_order = len(order_violations) == 0
    out.append(CheckResult(
        "P1.2b Per-case order (identifier order equals prec_L, incl. tie-break)",
        bool(p12_order),
        f"cases checked={len(expected_eids_by_case)}; "
        f"cases with an identifier-order deviation from prec_L={len(order_violations)}."
        + ("" if p12_order else f" first offenders={dict(list(order_violations.items())[:5])}")))

    # ---- P1.3 Message well-formedness ------------------------------
    # Every Message is related to exactly one communication event, either
    # by 'send' or by 'receive', but not both (M4: no send/receive
    # correlation is inferred). Its from/to O2O relations agree with its
    # sender/receiver object attributes and with the preserved
    # fromParticipant/toParticipant attributes of that single related
    # event. For a send observation the event participant is the sender;
    # for a receive observation it is the receiver.
    #
    # Completeness of from/to: the source format may leave a send/receive
    # event's COUNTERPARTY side (toParticipant of a Send, fromParticipant
    # of a Receive) unrecorded (see Normalization, appendix). That side is
    # not guessed: from(e)/to(e) stay undefined for that Message, no O2O
    # 'from'/'to' relation is created for it, and its sender/receiver
    # object attribute stays unset. This is a data-completeness property,
    # not a construction defect, so it is reported explicitly below
    # (n_messages_missing_sender/receiver) rather than silently skipped;
    # P1.3 only fails on an actual DISAGREEMENT between two sources that
    # both claim to have a value (or a value present on one side of the
    # attribute/relation pair without its counterpart), never merely on
    # an endpoint the source log never recorded.
    p13_ok = True
    p13_detail_bits: List[str] = []
    if not rel.empty:
        msg_rel = rel[rel[COL_OTYPE] == OT_MESSAGE]
        send_edges = msg_rel[msg_rel[COL_QUALIFIER] == Q_SEND][[COL_OID, COL_EID]]
        recv_edges = msg_rel[msg_rel[COL_QUALIFIER] == Q_RECEIVE][[COL_OID, COL_EID]]
        send_counts = send_edges.groupby(COL_OID).size()
        recv_counts = recv_edges.groupby(COL_OID).size()
        msg_ids = set(obj[obj[COL_OTYPE] == OT_MESSAGE][COL_OID]) if not obj.empty else set()

        bad_xor = [m for m in msg_ids
                  if int(send_counts.get(m, 0)) + int(recv_counts.get(m, 0)) != 1]
        p13_detail_bits.append(f"messages={len(msg_ids)}")
        p13_detail_bits.append(f"not exactly-one send-xor-receive: {len(bad_xor)}")
        if bad_xor:
            p13_ok = False

        # the single (event, qualifier) related to each Message
        msg_event: Dict[str, Tuple[str, str]] = {
            oid: (eid, Q_SEND) for oid, eid in zip(send_edges[COL_OID], send_edges[COL_EID])}
        msg_event.update({oid: (eid, Q_RECEIVE)
                          for oid, eid in zip(recv_edges[COL_OID], recv_edges[COL_EID])})

        if not obj.empty:
            obj_idx = obj.set_index(COL_OID)
            ev_idx = ev.set_index(COL_EID) if not ev.empty else None
            o2o_msg = (o2o[o2o[COL_QUALIFIER].isin([Q_FROM, Q_TO])] if not o2o.empty
                      else pd.DataFrame(columns=[COL_OID, COL_OID2, COL_QUALIFIER]))
            oa_disagreements = 0
            ea_disagreements = 0
            participant_disagreements = 0
            missing_sender = 0
            missing_receiver = 0
            inconsistent_partial = 0  # attribute defined but relation missing, or vice versa
            for m in msg_ids:
                if m not in obj_idx.index:
                    continue
                sender = obj_idx.at[m, "sender"] if "sender" in obj_idx.columns else None
                receiver = obj_idx.at[m, "receiver"] if "receiver" in obj_idx.columns else None
                froms = o2o_msg[(o2o_msg[COL_OID] == m) & (o2o_msg[COL_QUALIFIER] == Q_FROM)][COL_OID2].tolist()
                tos = o2o_msg[(o2o_msg[COL_OID] == m) & (o2o_msg[COL_QUALIFIER] == Q_TO)][COL_OID2].tolist()

                sender_defined, from_defined = pd.notna(sender), bool(froms)
                receiver_defined, to_defined = pd.notna(receiver), bool(tos)
                if not sender_defined:
                    missing_sender += 1
                if not receiver_defined:
                    missing_receiver += 1
                # A genuine construction bug: the object attribute and the O2O
                # relation for the SAME side disagree on whether it is defined.
                if sender_defined != from_defined:
                    inconsistent_partial += 1
                if receiver_defined != to_defined:
                    inconsistent_partial += 1

                if sender_defined and froms and froms[0] != _participant_id(str(sender)):
                    oa_disagreements += 1
                if receiver_defined and tos and tos[0] != _participant_id(str(receiver)):
                    oa_disagreements += 1

                eid_qual = msg_event.get(m)
                if eid_qual is None or ev_idx is None or eid_qual[0] not in ev_idx.index:
                    continue
                eid, qual = eid_qual
                ev_from = ev_idx.at[eid, "fromParticipant"] if "fromParticipant" in ev_idx.columns else None
                ev_to = ev_idx.at[eid, "toParticipant"] if "toParticipant" in ev_idx.columns else None
                ev_participant = ev_idx.at[eid, "participant"] if "participant" in ev_idx.columns else None
                if pd.notna(sender) and pd.notna(ev_from) and str(ev_from) != str(sender):
                    ea_disagreements += 1
                if pd.notna(receiver) and pd.notna(ev_to) and str(ev_to) != str(receiver):
                    ea_disagreements += 1
                if pd.notna(ev_participant):
                    expected = sender if qual == Q_SEND else receiver
                    if pd.notna(expected) and str(ev_participant) != str(expected):
                        participant_disagreements += 1

            p13_detail_bits.append(f"from/to O2O vs sender/receiver: {oa_disagreements}")
            p13_detail_bits.append(
                f"sender/receiver vs event fromParticipant/toParticipant: {ea_disagreements}")
            p13_detail_bits.append(f"event participant disagreements: {participant_disagreements}")
            p13_detail_bits.append(
                f"messages missing sender (source never recorded 'from'): {missing_sender}")
            p13_detail_bits.append(
                f"messages missing receiver (source never recorded 'to'): {missing_receiver}")
            p13_detail_bits.append(
                f"inconsistent partial state (attribute/relation disagree on definedness): "
                f"{inconsistent_partial}")
            if oa_disagreements or ea_disagreements or participant_disagreements or inconsistent_partial:
                p13_ok = False
    out.append(CheckResult("P1.3 Message well-formedness", bool(p13_ok),
                           "; ".join(p13_detail_bits) or "no messages"))

    # ---- P1.4 Participant-projection coherence ----------------------
    # Checked in two parts:
    #  (a) for every event with an 'in_projection' object pp, pp is
    #      'projection_of' the event's 'within' object (the collaboration
    #      case);
    #  (b) every ParticipantProjection is 'for_participant' exactly one
    #      Participant, and that Participant's name equals the
    #      ParticipantProjection's 'participant' attribute.
    p14_ok = True
    p14_detail = ""
    if not rel.empty and not o2o.empty:
        ev_inproj = rel[(rel[COL_QUALIFIER] == Q_IN_PROJECTION)][[COL_EID, COL_OID]]
        ev_within = dict(zip(
            rel[rel[COL_QUALIFIER] == Q_WITHIN][COL_EID],
            rel[rel[COL_QUALIFIER] == Q_WITHIN][COL_OID]))
        projection_of = {(r[COL_OID]): r[COL_OID2] for _, r in
                         o2o[o2o[COL_QUALIFIER] == Q_PROJECTION_OF].iterrows()}

        # (a) in_projection/within coherence
        bad_projof = 0
        for _, r in ev_inproj.iterrows():
            eid, pp = r[COL_EID], r[COL_OID]
            if projection_of.get(pp) != ev_within.get(eid):
                bad_projof += 1

        # (b) for_participant well-formedness, per ParticipantProjection
        pp_ids = set(obj[obj[COL_OTYPE] == OT_PP][COL_OID]) if not obj.empty else set()
        forpart_edges = o2o[o2o[COL_QUALIFIER] == Q_FOR_PARTICIPANT]
        forpart_counts = forpart_edges.groupby(COL_OID).size().to_dict()
        forpart_target = dict(zip(forpart_edges[COL_OID], forpart_edges[COL_OID2]))
        obj_idx = obj.set_index(COL_OID) if not obj.empty else None
        bad_forpart = 0
        bad_name = 0
        for pp in pp_ids:
            if int(forpart_counts.get(pp, 0)) != 1:
                bad_forpart += 1
                continue
            # name agreement: for_participant target's 'name' == pp.participant
            tgt = forpart_target.get(pp)
            if obj_idx is not None and tgt in obj_idx.index:
                tgt_name = obj_idx.at[tgt, "name"] if "name" in obj_idx.columns else None
                pp_part = obj_idx.at[pp, "participant"] if "participant" in obj_idx.columns else None
                if tgt_name is not None and pp_part is not None and str(tgt_name) != str(pp_part):
                    bad_name += 1

        p14_ok = (bad_projof == 0) and (bad_forpart == 0) and (bad_name == 0)
        p14_detail = (f"in_projection/within mismatches={bad_projof}; "
                      f"projections !=1 for_participant={bad_forpart}; "
                      f"for_participant name disagreements={bad_name}")
    out.append(CheckResult("P1.4 Participant-projection coherence", bool(p14_ok), p14_detail))

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

    # ---- P1.6 Participant coherence ---------------------------------
    # For every event, the Participant reached by the direct 'participant'
    # E2O edge must equal the Participant reached via the two-step
    # 'in_projection' -> 'for_participant' path.
    p16_ok = True
    p16_detail = ""
    if not rel.empty:
        ev_participant = dict(zip(
            rel[(rel[COL_QUALIFIER] == Q_PARTICIPANT) & (rel[COL_OTYPE] == OT_PARTICIPANT)][COL_EID],
            rel[(rel[COL_QUALIFIER] == Q_PARTICIPANT) & (rel[COL_OTYPE] == OT_PARTICIPANT)][COL_OID]))
        ev_inproj = dict(zip(
            rel[rel[COL_QUALIFIER] == Q_IN_PROJECTION][COL_EID],
            rel[rel[COL_QUALIFIER] == Q_IN_PROJECTION][COL_OID]))
        forpart_target = (dict(zip(o2o[o2o[COL_QUALIFIER] == Q_FOR_PARTICIPANT][COL_OID],
                                   o2o[o2o[COL_QUALIFIER] == Q_FOR_PARTICIPANT][COL_OID2]))
                          if not o2o.empty else {})
        all_eids = set(ev_participant) | set(ev_inproj)
        mismatches6 = 0
        missing_one_side = 0
        for eid in all_eids:
            direct = ev_participant.get(eid)
            pp_oid = ev_inproj.get(eid)
            indirect = forpart_target.get(pp_oid) if pp_oid is not None else None
            if direct is None or indirect is None:
                missing_one_side += 1
                continue
            if direct != indirect:
                mismatches6 += 1
        p16_ok = (mismatches6 == 0) and (missing_one_side == 0)
        p16_detail = (f"events checked={len(all_eids)}; "
                      f"direct/indirect mismatches={mismatches6}; "
                      f"missing one side={missing_one_side}")
    out.append(CheckResult("P1.6 Participant coherence", bool(p16_ok), p16_detail))

    return out


def print_check_report(checks: List[CheckResult], stats: Dict[str, Any]) -> bool:
    logger.info("---- transformation stats ----")
    for k, v in stats.items():
        logger.info("  %-22s %s", k, v)
    logger.info("---- consistency checks (P1.1-P1.6) ----")
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
    base = output
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
    write_ocel2_json(ocel, json_path)
    write_ocel2_sqlite(ocel, sqlite_path)

    if validate:
        problems = validate_jsonocel(json_path)
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
