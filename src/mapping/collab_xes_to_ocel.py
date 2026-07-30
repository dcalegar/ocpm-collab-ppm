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
correlated message instance: rule M4 creates one Message per send event and
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
    Object types : CollaborationCase, OrchestrationCase, Message, plus ONE
                   OBJECT TYPE PER PARTICIPANT IDENTIFIER (rule M2): the
                   participants occurring in the log are themselves declared
                   as object types, so a participant object of "Laboratory"
                   has object type "Laboratory". There is no single
                   "Participant" type. The set of participant types is
                   written T_Pa; participant_types() below implements the
                   injective encoding tau that maps an identifier to its
                   object type (see NOTE on tau).
                   In BPMN 2.0 terminology a participant is an
                   orchestration / pool (e.g. "Laboratory",
                   "Gynecologist"), not a role or a person: it is a global
                   object, reused across cases. OrchestrationCase is the
                   execution of that participant's local process within one
                   collaboration case (M3, one per (case, participant) pair
                   under the single-instantiation assumption); it is a
                   distinct object, linked to its participant object by the
                   O2O qualifier for_participant.
    E2O qualifiers: within, in_orchestration, send, receive, participant
                   (M6). The participant of an event is reachable two
                   ways: directly, via the `participant` edge below, and
                   indirectly via in_orchestration -> for_participant,
                   keeping the participant distinct from its own per-case
                   execution. Both are part of the conceptual
                   model -- the direct edge keeps participant objects
                   reachable at the event-to-object level for
                   serialization and downstream feature extraction (see
                   NOTE below); their agreement is machine-checked by
                   P1.6. `send`/`receive` each relate a communication
                   event to its OWN Message observation object; no
                   relation between distinct send and receive events is
                   inferred (M4).
    O2O qualifiers: part_of, for_participant, from, to, exchanged_in

    NOTE on the `participant` E2O edge: pm4py's OCEL 2.0 exporters (JSON
    and SQLite) call filtering_utils.propagate_relations_filtering(),
    which keeps only the objects that appear in the E2O relations table
    -- silently dropping any object reachable only via O2O. Since a
    participant object would otherwise be reached only via O2O
    (for_participant, from, to), it and those edges would be lost on
    export without a direct event edge. Rule M6 keeps the direct
    `participant` edge for exactly this reason, so the exported
    .jsonocel/.sqlite stays lossless.

    NOTE on tau (the participant -> object type encoding): the object type
    is required to be an INJECTIVE encoding of the participant identifier,
    not the raw identifier. The encoding here also keeps the type a valid
    Python identifier ("Org line A2" -> "OrgLineA2"; "Hospital", "PartyA"
    and every other alphanumeric identifier are fixed points, so the type
    reads as the participant name itself in the common case). This matters
    downstream: OCPA's OCEL 2.0 importer materializes one DataFrame column
    per object type and resolves it with getattr(row, object_type) over
    pandas itertuples, which silently drops any type whose name is not a
    valid identifier. Injectivity is enforced both on tau itself and on
    pm4py's own table-name stripping of tau (the SQLite exporter writes one
    object_<stripped name> table per object type), so two participants can
    never collapse into one type or one table.

    Refinement layers (OPT-IN, off by default; see MappingConfig):
    resource layer  -- R1: one Resource object per actor identifier the
                       source records (e.g. org:resource); R2: E2O
                       `resource`; R3: O2O `acts_for` from a Resource to
                       every participant it acts for. Criteria PR.1/PR.2.
    correlation layer -- C1: O2O `correlated_with` from a send observation's
                       Message to the receive observation's Message when
                       both carry the same correlation identifier (e.g. a
                       native message id). No object is created. Criterion
                       PC.1.
    Both layers only APPEND objects and relations, so they are additive
    (P1.1-P1.6 are unaffected) and mutually independent, and each
    degenerates to the identity on a log that records no such identifier.

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
import re
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

# --- Fixed event-attribute names M8 writes for the consumed source keys
# (elemType/participant/from/toParticipant, unprefixed). A residual source
# column that happens to share one of these bare names (distinct from its
# prefixed cfg.*_key counterpart, which IS excluded from residual_keys)
# would silently overwrite the consumed value instead of being preserved
# alongside it (E30); transform() raises on this instead of masking it.
RESERVED_EVENT_OUTPUT_KEYS = ("elemType", "participant", "fromParticipant", "toParticipant")

# --- OCEL 2.0 object types ------------------------------------------
# The participant types are NOT listed here: rule M2 declares one object
# type per participant identifier occurring in the log, so T_Pa is
# log-dependent and is built at transform time (see ParticipantTypes).
OT_CC = "CollaborationCase"
OT_OC = "OrchestrationCase"
OT_MESSAGE = "Message"
# Resource layer only (R1); never created by the core mapping.
OT_RESOURCE = "Resource"

# Object types tau must never produce, so that a participant identifier can
# never be confused with a structural type of the mapping.
RESERVED_OBJECT_TYPES = (OT_CC, OT_OC, OT_MESSAGE, OT_RESOURCE)

# --- E2O qualifiers (rule M6) ---------------------------------------
Q_WITHIN = "within"
Q_IN_ORCHESTRATION = "in_orchestration"
Q_SEND = "send"
Q_RECEIVE = "receive"
# Direct event->participant edge (M6). Also reachable indirectly via
# in_orchestration -> for_participant; P1.6 checks the two agree. See the
# module docstring NOTE above for why the direct edge is kept.
Q_PARTICIPANT = "participant"

# --- O2O qualifiers (rule M7) ---------------------------------------
Q_PART_OF = "part_of"
Q_FOR_PARTICIPANT = "for_participant"
Q_FROM = "from"
Q_TO = "to"
Q_EXCHANGED_IN = "exchanged_in"

# --- Refinement-layer qualifiers (opt-in; R2/R3 and C1) -------------
Q_RESOURCE = "resource"            # E2O, R2
Q_ACTS_FOR = "acts_for"            # O2O, R3
Q_CORRELATED_WITH = "correlated_with"  # O2O, C1

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
    # --- Refinement layers (opt-in) ---------------------------------
    # Both name a RESIDUAL source attribute (one M8 preserves verbatim and
    # the core mapping does not consume). Left None, the corresponding
    # layer is not applied at all. Naming an attribute no event carries is
    # not an error either: the layer then degenerates to the identity,
    # adding no object and no relation.
    resource_attr: Optional[str] = None      # R1-R3, e.g. "org:resource"
    correlation_attr: Optional[str] = None   # C1, e.g. "msgInstanceId"


# =====================================================================
# Identifier creation (disjoint object-id ranges)
# =====================================================================
# Source identifiers are stored as attribute VALUES, never reused as
# object ids (U_obj and U_val are disjoint in OCEL 2.0). We therefore
# create type-prefixed object ids.

def _cc_id(case: str) -> str:
    return f"cc::{case}"

def _participant_id(p: str) -> str:
    return f"part::{p}"

def _resource_id(a: str) -> str:
    # Resource layer (R1). Its own id range, disjoint from every core one,
    # which is what makes the layer additive.
    return f"res::{a}"

def _oc_id(case: str, p: str) -> str:
    # Escape "\" and ":" in each component before joining with the
    # unescaped delimiter "::": without this, _oc_id("x", "y::z") and
    # _oc_id("x::y", "z") both create "oc::x::y::z" (E30) -- a real
    # injectivity failure of the id-creation scheme, though vacuous on the
    # 6 evaluated logs (no case id or participant name contains ":").
    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace(":", "\\:")
    return f"oc::{esc(case)}::{esc(p)}"

def _message_id(eid: str) -> str:
    # One Message per send OR receive event (M4, m_e for e in S_L u R_L);
    # the event id makes it unique. No correlation between a send Message
    # and a receive Message is inferred.
    return f"msg::{eid}"

def _event_id(case: str, idx: int, width: int = 1) -> str:
    # Stable per-case event id. idx is the within-case source order (the
    # rank of e in the source total order prec_L). Zero-padded to `width` digits so
    # that lexicographic string order on the id agrees with numeric idx
    # order -- required for mu_E to be order-preserving on prec_L
    # (deliberately not "order-embedding", since the
    # converse fails across cases): without padding, "e::459::10" sorts
    # before "e::459::9".
    return f"e::{case}::{str(idx).zfill(width)}"


# =====================================================================
# tau: participant identifier -> OCEL object type (rule M2)
# =====================================================================

_NON_ALNUM = re.compile(r"[^0-9a-zA-Z]+")


def _pm4py_table_name(otype: str) -> str:
    """Reproduce pm4py's object-type name stripping
    (pm4py.objects.ocel.util.names_stripping.apply), which the OCEL 2.0
    SQLite exporter uses to derive the physical ``object_<name>`` table of
    each object type. Two object types whose stripped names coincide would
    make the exporter write the same table twice; the registry below
    therefore keeps this image injective as well, not only tau itself."""
    words = [w.capitalize() for w in otype.split(" ")]
    return _NON_ALNUM.sub("", "".join(words)).strip()[:100]


def _encode_object_type(p: str) -> str:
    """Base encoding of a participant identifier as an object type: split on
    runs of non-alphanumerics, upper-case the first character of each part
    (keeping the rest), and join. Alphanumeric identifiers are fixed points,
    so "Hospital" and "PartyA" stay themselves and only identifiers that are
    not already valid Python identifiers change ("Org line A2" ->
    "OrgLineA2"). See the module docstring NOTE on tau for why the result
    must be a valid identifier."""
    parts = [w for w in _NON_ALNUM.split(p) if w]
    s = "".join(w[:1].upper() + w[1:] for w in parts)
    if not s or s[0].isdigit():
        s = "P" + s
    return s


class ParticipantTypes:
    """The injective encoding tau of rule M2, built incrementally as
    participants are first seen, plus its inverse.

    Injectivity is maintained by construction rather than checked
    afterwards: a candidate type is accepted only when neither it nor its
    pm4py table name is already taken (by another participant or by one of
    the structural object types of the mapping), and otherwise a numeric
    suffix is appended until a free one is found. Since the identifiers are
    processed in first-occurrence order, the assignment is deterministic
    for a given source log."""

    def __init__(self) -> None:
        self._by_participant: Dict[str, str] = {}
        self._by_type: Dict[str, str] = {}
        self._taken_tables: Dict[str, str] = {}
        for t in RESERVED_OBJECT_TYPES:
            self._by_type[t] = ""            # reserved, owned by no participant
            self._taken_tables[_pm4py_table_name(t)] = ""

    def register(self, participant: str) -> str:
        """Return tau(participant), assigning it on first sight."""
        known = self._by_participant.get(participant)
        if known is not None:
            return known
        base = _encode_object_type(participant)
        candidate, n = base, 1
        while (candidate in self._by_type
               or _pm4py_table_name(candidate) in self._taken_tables):
            n += 1
            candidate = f"{base}{n}"
        self._by_participant[participant] = candidate
        self._by_type[candidate] = participant
        self._taken_tables[_pm4py_table_name(candidate)] = participant
        return candidate

    def type_of(self, participant: str) -> Optional[str]:
        """tau(participant), or None if the participant was never seen."""
        return self._by_participant.get(participant)

    def participant_of(self, otype: str) -> Optional[str]:
        """tau^-1(otype), or None if the type is not a participant type."""
        return self._by_type.get(otype) or None

    def is_participant_type(self, otype: Any) -> bool:
        return bool(self._by_type.get(otype))

    @property
    def types(self) -> frozenset:
        """T_Pa: the set of participant object types of this log."""
        return frozenset(t for t, p in self._by_type.items() if p)

    @property
    def participants(self) -> frozenset:
        return frozenset(self._by_participant)

    def __len__(self) -> int:
        return len(self._by_participant)


# =====================================================================
# I/O endpoints (the only pm4py-dependent functions)
# =====================================================================

def _raw_timestamps_in_file_order(path: str) -> List[str]:
    """Extract every event's raw time:timestamp attribute string directly
    from the XML, in exact file order (trace-by-trace, event-by-event),
    bypassing pm4py's datetime parser entirely -- see _correct_utc_timestamps
    for why.

    Namespace-aware (D26): a XES file may declare a default namespace on
    the root <log> element (e.g. xmlns="http://www.xes-standard.org/",
    as pm4py's own XES writer does -- toy_collab.xes is generated this
    way). When it does, every element's actual tag is
    "{namespace-uri}trace"/"{namespace-uri}event"/etc., not the bare
    name; an unqualified findall("trace") then silently matches zero
    elements. The namespace (if any) is read once from the root tag and
    prefixed onto every findall so both namespaced and un-namespaced XES
    (the source format does not mandate one) parse identically.
    """
    from lxml import etree
    root = etree.parse(path).getroot()
    ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
    out: List[str] = []
    for trace in root.findall(f"{ns}trace"):
        for ev in trace.findall(f"{ns}event"):
            for d in ev.findall(f"{ns}date"):
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


def _add_export_reachability_witnesses(objects_df: pd.DataFrame,
                                       relations_df: pd.DataFrame,
                                       o2o_df: pd.DataFrame) -> pd.DataFrame:
    """Return relations_df augmented with witness E2O edges for objects
    that transform() correctly materializes but that are reachable ONLY
    via O2O relations -- e.g. a participant that is always a message
    counterparty (collab:toParticipant/fromParticipant) and never itself
    collab:participant of any event, so M6 never gives it the direct
    `participant` edge (nor in_orchestration -> for_participant, which
    requires the participant to own an event too). pm4py's OCEL 2.0
    exporters call filtering_utils.propagate_relations_filtering(),
    which drops every object absent from the E2O relations table
    regardless of O2O reachability (see module docstring NOTE); without
    this, such participant objects -- and the O2O edges pointing at them --
    are silently lost on export even though transform()/P1 correctly
    account for them (D25).

    This is an export-completeness patch, not a mapping rule change: it
    does not modify relations_df as returned by transform(), so P1 and
    all downstream evaluation code keep seeing exactly the E2O set M6
    defines; it only feeds a superset into the pm4py OCEL object built
    right before write_ocel2_json/write_ocel2_sqlite. The witness edge
    reuses the O2O qualifier itself (`from`/`to`) applied at the E2O
    level, anchored on the very send/receive event whose Message O2O
    relation references the otherwise-unreachable object -- the same
    "keep one edge so the exporter doesn't drop the object" rationale
    already used for the direct `participant` edge (M6, see module
    docstring) and for BPIC's Participant objects in
    ocpm_eval.io_ocel._strip_participant_e2o.
    """
    if objects_df.empty or relations_df.empty or o2o_df.empty:
        return relations_df

    reachable = set(relations_df[COL_OID])
    orphan_oids = set(objects_df[COL_OID]) - reachable
    if not orphan_oids:
        return relations_df

    oid_to_otype = dict(zip(objects_df[COL_OID], objects_df[COL_OTYPE]))

    # Message object id -> the event it was created from (its send/receive
    # E2O edge), so a witness edge can be anchored on that same event.
    msg_to_event = {
        row[COL_OID]: row[COL_EID]
        for row in relations_df.loc[
            relations_df[COL_QUALIFIER].isin((Q_SEND, Q_RECEIVE))
        ].to_dict("records")
    }

    witness_rows = []
    seen = set()
    for row in o2o_df.to_dict("records"):
        target = row[COL_OID2]
        qualifier = row[COL_QUALIFIER]
        if target not in orphan_oids or qualifier not in (Q_FROM, Q_TO):
            continue
        if target in seen:
            continue
        eid = msg_to_event.get(row[COL_OID])
        if eid is None:
            continue
        # One witness row per ORPHAN OBJECT, not per (event, object): a single
        # E2O edge already suffices for pm4py's exporter to keep the object
        # (and for OCPA's object table, cf. ocpm_eval.io_ocel._strip_participant_e2o,
        # which reduces to exactly one row per object on read). Deduplicating on
        # the target here keeps the count of non-M6 rows in the serialized OCEL
        # minimal -- an endpoint-only Participant shared across N CollaborationCases
        # contributes 1 witness row, not N -- so the exported artefact stays as
        # close as possible to the E2O set of mu(L) (D25).
        #
        # The witness row must carry the object's OWN type: under rule M2
        # each participant identifier is its own object type, so there is no
        # single fallback type to default to, and a wrong one here would
        # contradict the `object` table on export.
        otype = oid_to_otype.get(target)
        if otype is None:
            continue
        seen.add(target)
        witness_rows.append({COL_EID: eid, COL_OID: target,
                             COL_OTYPE: otype,
                             COL_QUALIFIER: qualifier})

    if not witness_rows:
        return relations_df
    return pd.concat([relations_df, pd.DataFrame(witness_rows)], ignore_index=True)


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
    total order prec_L). Ordering by timestamp with a stable
    tie-break guarantees the trace order is preserved before and after
    the mapping; event ids are created in this order, so reconstruction
    (P1.2) reads the event-identifier order without re-sorting by
    timestamp.
    Returns, per case, a list of event dicts enriched with a created eid
    and the within-case index.
    """
    def _clean(v: Any) -> Optional[str]:
        # Treat NaN, None, and empty/whitespace strings as absent, so a
        # blank fromParticipant/toParticipant/participant never creates a
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
            # elem_raw is the cleaned but NOT-yet-defaulted value (None when
            # the source leaves elemType absent/empty/whitespace); elem is the
            # value transform() builds on, defaulted to 'task'. Both are kept
            # so the E30 domain check in run_consistency_checks can flag an
            # ABSENT elemType too, instead of the default silently masking it.
            elem_raw = _clean(row.get(cfg.elemtype_key))
            elem = elem_raw or ELEM_TASK
            participant = _clean(row.get(cfg.participant_key))
            from_p = (_clean(row.get(cfg.from_key))
                      if cfg.from_key in work.columns else None)
            to_p = (_clean(row.get(cfg.to_key))
                    if cfg.to_key in work.columns else None)
            # Normalization nu: the source
            # log may leave the "own side" of a send/receive event implicit
            # (recording only the counterparty), relying on
            # collab:participant to supply it. Backfill ONLY a missing own
            # side; a recorded value is never overwritten, so a source
            # value contradicting collab:participant surfaces in P1.3
            # rather than being silently corrected. The counterparty side
            # is never guessed: from/to remain genuinely partial when the
            # source omits it (well-formedness (i)/(ii) constrain only the
            # own side).
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
                "elem_raw": elem_raw,
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
    # tau and its inverse for this log (rule M2). The consistency checks and
    # the export path need it to tell a participant object from a structural
    # one, since there is no longer a single participant object type.
    participant_types: ParticipantTypes = field(default_factory=ParticipantTypes)


def _layer_attribute(ev: Dict[str, Any], key: str) -> Optional[str]:
    """Read a residual source attribute off an event, treating NaN, None and
    blank strings as unrecorded -- the same normalization the core mapping
    applies to participants, so a blank actor never creates an empty-named
    Resource and a blank correlation id never pairs two observations."""
    val = ev["row"].get(key)
    if val is None or pd.isna(val):
        return None
    s = str(val).strip()
    return s or None


def _apply_resource_layer(cases: Dict[str, List[Dict[str, Any]]],
                          attr: str,
                          ensure_object,
                          e2o_rows: List[Dict[str, Any]],
                          o2o_rows: List[Dict[str, Any]]) -> Dict[str, int]:
    """Resource layer (R1-R3): promote the actor identifier the source
    records per event to an object of its own.

    R1 creates one Resource object per distinct actor identifier, R2 relates
    every event whose actor the source records to it by `resource`, and R3
    relates that Resource to every participant it acts for by `acts_for`.
    An event whose actor is unrecorded carries no relation, rather than
    being attributed to a synthetic resource.

    The pass only appends: it creates objects in an id range of its own
    (res::) and uses qualifiers the core mapping never emits, so it leaves
    P1.1-P1.6 and the core object and relation sets untouched. On a log
    where no event carries `attr` it adds nothing at all, which is the
    degeneracy property.
    """
    n_events, n_resources, acts_for_pairs = 0, set(), set()
    for evlist in cases.values():
        for ev in evlist:
            actor = _layer_attribute(ev, attr)
            if actor is None:
                continue
            res_oid = _resource_id(actor)
            ensure_object(res_oid, OT_RESOURCE, {"name": actor})   # R1
            n_resources.add(actor)
            e2o_rows.append({COL_EID: ev["eid"], COL_OID: res_oid,     # R2
                             COL_OTYPE: OT_RESOURCE, COL_QUALIFIER: Q_RESOURCE})
            n_events += 1
            p = ev["participant"]
            if p is not None and (actor, p) not in acts_for_pairs:     # R3
                acts_for_pairs.add((actor, p))
                o2o_rows.append({COL_OID: res_oid, COL_OID2: _participant_id(p),
                                 COL_QUALIFIER: Q_ACTS_FOR})
    return {"n_resources": len(n_resources),
            "n_resource_e2o": n_events,
            "n_acts_for": len(acts_for_pairs)}


def _apply_correlation_layer(cases: Dict[str, List[Dict[str, Any]]],
                             attr: str,
                             msg_by_eid: Dict[str, str],
                             o2o_rows: List[Dict[str, Any]]) -> Dict[str, int]:
    """Correlation layer (C1): relate the send observation and the receive
    observation of one logical message to each other.

    Rule C1 is unconditional -- every send observation and every receive
    observation carrying the same correlation identifier, WITHIN THE SAME
    CollaborationCase, are related by `correlated_with`, directed from the
    send to the receive. Grouping is scoped to (case, correlation identifier),
    not to the identifier alone: a correlation attribute is a per-conversation
    token (e.g. a message id local to one collaboration instance), and
    treating it as globally unique across cases would relate unrelated
    exchanges whenever two different cases happen to reuse the same
    identifier -- a real possibility the source log's own identifier scheme
    does not rule out. Criterion PC.1 independently checks that every
    `correlated_with` relation stays within one case (see `bad_case` in
    `_run_layer_checks`); scoping the grouping here is what makes that check
    hold by construction rather than by coincidence of the sample data.

    Multiplicity within one case is deliberately NOT filtered here: a
    correlation attribute that groups three or more observations, or two
    sends, in the same case then produces a Message with more than one
    outgoing or incoming relation, which is exactly what criterion PC.1
    reports. Filtering such groups out at construction time would make PC.1
    pass vacuously on a source attribute that is not a usable correlation
    identifier.

    No object is created, so P1.5 is unaffected, and `correlated_with` is
    distinct from `exchanged_in`, so the message objects of a collaboration
    case are unchanged. An observation whose identifier the source does not
    record, or whose counterpart it never observes (in the SAME case), simply
    carries no relation -- a send with no outgoing `correlated_with` is
    precisely an exchange the log observes on one side only.
    """
    sends: Dict[Tuple[str, str], List[str]] = {}
    receives: Dict[Tuple[str, str], List[str]] = {}
    for case, evlist in cases.items():
        for ev in evlist:
            if ev["elem"] not in (ELEM_SEND, ELEM_RECEIVE):
                continue
            corr = _layer_attribute(ev, attr)
            if corr is None:
                continue
            side = sends if ev["elem"] == ELEM_SEND else receives
            side.setdefault((case, corr), []).append(ev["eid"])

    n_relations = 0
    for key, send_eids in sends.items():
        for s_eid in send_eids:
            for r_eid in receives.get(key, ()):
                o2o_rows.append({COL_OID: msg_by_eid[s_eid],
                                 COL_OID2: msg_by_eid[r_eid],
                                 COL_QUALIFIER: Q_CORRELATED_WITH})
                n_relations += 1
    corr_ids_seen = {corr for _case, corr in (set(sends) | set(receives))}
    return {"n_correlated_with": n_relations,
            "n_correlation_ids": len(corr_ids_seen),
            "n_uncorrelated_sends": sum(1 for key, eids in sends.items()
                                        for _ in eids if key not in receives)}


def transform(df: pd.DataFrame, cfg: Optional[MappingConfig] = None) -> TransformResult:
    """Apply rules M1-M8 and return the four OCEL DataFrames plus stats.

    When ``cfg`` enables a refinement layer, its rules (R1-R3 and/or C1) are
    applied as a post-pass that only appends objects and relations.

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
    _colliding = [c for c in residual_keys if c in RESERVED_EVENT_OUTPUT_KEYS]
    if _colliding:
        raise ValueError(
            f"Residual source column(s) {_colliding} collide with the fixed "
            f"M8 output attribute name(s) of the same name -- rename the "
            f"source column(s) before mapping (E30).")

    participant_types = ParticipantTypes()
    participant_seen: set = set()
    msg_by_eid_all: Dict[str, str] = {}
    n_messages = 0
    # Counterparty completeness (Normalization nu): the source
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
        # (msg_by_eid is per case; msg_by_eid_all accumulates the same
        # mapping over the whole log for the correlation layer, which pairs
        # observations by a shared identifier rather than by case.)
        # Each communication event creates its OWN Message observation
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
            msg_by_eid_all[ev["eid"]] = msg_oid
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

            # ---- M2: participant object, typed by the participant
            #          identifier itself (log-level scope) -------------
            # A participant is the orchestration (BPMN pool), not a role or
            # person, and its OBJECT TYPE is tau(identifier) -- there is no
            # single "Participant" type. Also create participant objects for
            # the from/to (sender/receiver) identifiers of messages.
            for pid in (p, ev["from"], ev["to"]):
                if pid is not None and pid not in participant_seen:
                    _ensure_object(_participant_id(pid),
                                   participant_types.register(pid), {"name": pid})
                    participant_seen.add(pid)

            # ---- M3: OrchestrationCase object (the participant's local
            #          execution within this collaboration case) -------
            oc_oid = None
            if p is not None:
                oc_oid = _oc_id(case, p)
                _ensure_object(oc_oid, OT_OC,
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
            # ---- M8: residual source attributes, unchanged in
            # TransformResult (res.events_df/objects_df keep native types).
            # The exported .jsonocel/.sqlite is a SEPARATE serialization step
            # (build_ocel_object/_stringify_attribute_columns) that casts
            # every attribute value to string, because the OCEL 2.0 JSON
            # schema itself types `value` as a string (see the module
            # docstring's OCEL 2.0 JSON SCHEMA CONFORMANCE note) -- not
            # because M8 stops preserving the value. A residual numeric,
            # boolean, or datetime attribute therefore keeps its native type
            # here and in P1.1, but reaches disk as its string
            # representation, same as every other OCEL 2.0 attribute value.
            for k in residual_keys:
                val = ev["row"].get(k)
                if pd.notna(val):
                    ev_row[k] = val
            event_rows.append(ev_row)

            # ---- M6: structural E2O relations -----------------------
            # within (-> CollaborationCase), in_orchestration (->
            # OrchestrationCase), and the direct participant edge (-> the
            # participant object), whose agreement with in_orchestration ->
            # for_participant is checked by P1.6 (see module docstring).
            e2o_rows.append({COL_EID: ev["eid"], COL_OID: cc_oid,
                             COL_OTYPE: OT_CC, COL_QUALIFIER: Q_WITHIN})
            if oc_oid is not None:
                e2o_rows.append({COL_EID: ev["eid"], COL_OID: oc_oid,
                                 COL_OTYPE: OT_OC, COL_QUALIFIER: Q_IN_ORCHESTRATION})
            if p is not None:
                e2o_rows.append({COL_EID: ev["eid"], COL_OID: _participant_id(p),
                                 COL_OTYPE: participant_types.type_of(p),
                                 COL_QUALIFIER: Q_PARTICIPANT})

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

        # ---- M7: OrchestrationCase O2O relations --------------------
        # one per (case, participant) seen in this case
        seen_oc: set = set()
        for ev in evlist:
            p = ev["participant"]
            if p is None:
                continue
            oc_oid = _oc_id(case, p)
            if oc_oid in seen_oc:
                continue
            seen_oc.add(oc_oid)
            o2o_rows.append({COL_OID: oc_oid, COL_OID2: cc_oid,
                             COL_QUALIFIER: Q_PART_OF})
            o2o_rows.append({COL_OID: oc_oid, COL_OID2: _participant_id(p),
                             COL_QUALIFIER: Q_FOR_PARTICIPANT})

    # ---- refinement layers (opt-in, additive) -----------------------
    # Applied after every core rule has run, appending only. Neither layer
    # redefines a component of the core log, so P1.1-P1.6 hold of the result
    # exactly as they hold of the core mapping; the two use disjoint object
    # id ranges and disjoint qualifier vocabularies, so either may be applied
    # alone, or both in any order.
    layer_stats: Dict[str, Any] = {}
    degenerate_layers: List[Tuple[str, str]] = []
    if cfg.resource_attr:
        s = _apply_resource_layer(
            cases, cfg.resource_attr, _ensure_object, e2o_rows, o2o_rows)
        layer_stats.update(s)
        if not s["n_resources"]:
            degenerate_layers.append(("resource", cfg.resource_attr))
    if cfg.correlation_attr:
        s = _apply_correlation_layer(
            cases, cfg.correlation_attr, msg_by_eid_all, o2o_rows)
        layer_stats.update(s)
        if not s["n_correlation_ids"]:
            degenerate_layers.append(("correlation", cfg.correlation_attr))

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
        "n_participant_types": len(participant_types),
    }
    stats.update(layer_stats)
    if n_messages_missing_sender or n_messages_missing_receiver:
        logger.warning(
            "%d Message object(s) have no recorded sender and %d have no "
            "recorded receiver (source log leaves the counterparty side "
            "of some send/receive events implicit; see Normalization nu "
            "above). Their 'from'/'to' O2O relation and sender/receiver "
            "object attribute are left undefined rather than guessed.",
            n_messages_missing_sender, n_messages_missing_receiver)
    for name, attr in degenerate_layers:
        logger.warning(
            "The %s layer was requested on attribute '%s', which no event "
            "records; the layer degenerates to the identity and the output "
            "is exactly the core mapping.", name, attr)
    return TransformResult(events_df, objects_df, relations_df, o2o_df, stats,
                           participant_types)


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
    """Machine-check P1.1-P1.6 against the constructed DataFrames, plus the
    criteria of whichever refinement layer ``cfg`` enables (PR.1/PR.2 for the
    resource layer, PC.1 for the correlation layer).

    These guard against implementation defects; they are independent of
    the by-construction correctness argument for the mapping.
    """
    cfg = cfg or MappingConfig()
    out: List[CheckResult] = []
    ev = res.events_df
    rel = res.relations_df
    o2o = res.o2o_df
    obj = res.objects_df
    tau = res.participant_types

    def is_participant_type(otype: Any) -> bool:
        """Rule M2 leaves no single participant object type, so every check
        that used to filter on one filters on membership in T_Pa instead."""
        return tau.is_participant_type(otype)

    # Indexes reused by several checks below (D24 hardening): the object
    # table by oid, and the event->CollaborationCase ('within') map. Built
    # once here instead of locally inside P1.4, so P1.2/P1.3/P1.5/P1.6 can
    # also use them without re-deriving from `rel`/`obj` each time.
    obj_idx_all = obj.set_index(COL_OID) if not obj.empty else None
    ev_within: Dict[str, str] = (
        dict(zip(rel[rel[COL_QUALIFIER] == Q_WITHIN][COL_EID],
                rel[rel[COL_QUALIFIER] == Q_WITHIN][COL_OID]))
        if not rel.empty else {})

    # Recompute the expected per-event identity (case, eid, activity,
    # timestamp, participant, elemType, prec_L order) straight from the
    # source log, calling `_sorted_case_events` a second time rather than
    # reading it off `res`. This is independent of transform()'s in-memory
    # state (its accumulators, DataFrames, and object-id bookkeeping), so it
    # catches defects that only checking cardinalities/timestamps would miss
    # (e.g. a permuted activity, a swapped case membership that keeps every
    # count the same, a stripped collab:participant attribute, or an
    # elemType flipped to 'task' with its Message dropped -- the output
    # cannot vouch for itself). It is NOT independent of `_sorted_case_events`
    # itself: both transform() and this call route through the same
    # normalization/ordering function, so a systematic defect in that
    # function (its sort key, its nu backfill rule, its elemType
    # normalization) would reproduce identically on both sides and pass
    # undetected here. Closing that residual gap needs a second, differently
    # implemented oracle (e.g. a property-based spec derived directly from
    # the mapping's mathematical definition), which this module does not
    # provide.
    expected_cases = _sorted_case_events(src_df, cfg)
    # D24 (reviewer round 2): from/to are included here too (positions 4/5)
    # so P1.3 can compare the transform's event/Message/O2O endpoint
    # representations against the SOURCE-derived value directly, rather
    # than only against each other -- a mutation that coherently rewrites
    # all three representations to the same wrong value would otherwise
    # agree with itself and pass.
    expected_by_eid: Dict[str, Tuple[str, Any, Optional[str], str, Optional[str], Optional[str]]] = {
        e["eid"]: (str(e["activity"]), e["timestamp"], e["participant"], e["elem"],
                   e["from"], e["to"])
        for evlist in expected_cases.values() for e in evlist}
    expected_endpoints_by_eid: Dict[str, Tuple[Optional[str], Optional[str]]] = {
        e["eid"]: (e["from"], e["to"]) for evlist in expected_cases.values() for e in evlist}
    # E30: the RAW (pre-default) elemType per event -- None when the source
    # left it absent/empty/whitespace. Used by the domain check in P1.1 so
    # an absent elemType is flagged, not masked by the 'task' default.
    expected_elem_raw_by_eid: Dict[str, Optional[str]] = {
        e["eid"]: e["elem_raw"] for evlist in expected_cases.values() for e in evlist}
    expected_eids_by_case: Dict[str, List[str]] = {
        case: [e["eid"] for e in evlist] for case, evlist in expected_cases.items()}
    case_by_eid: Dict[str, str] = {
        e["eid"]: case for case, evlist in expected_cases.items() for e in evlist}
    # Raw source row per event id (D24: residual-attribute preservation
    # check in P1.1) and the residual (M8) attribute names -- the same
    # exclusion logic `transform()` uses, so this checks exactly the
    # columns the mapping is supposed to re-emit verbatim.
    expected_row_by_eid: Dict[str, Any] = {
        e["eid"]: e["row"] for evlist in expected_cases.values() for e in evlist}
    residual_keys = [c for c in src_df.columns
                     if c not in cfg.consumed_keys
                     and c != cfg.elemtype_key
                     and not str(c).startswith("__")]
    # Source-side sets of communication events (elem != task), used by P1.3,
    # split by direction so the send/receive QUALIFIER of each edge can be
    # compared against the source elemType, not merely event coverage.
    expected_send_eids = {
        e["eid"] for evlist in expected_cases.values() for e in evlist
        if e["elem"] == ELEM_SEND}
    expected_recv_eids = {
        e["eid"] for evlist in expected_cases.values() for e in evlist
        if e["elem"] == ELEM_RECEIVE}
    expected_comm_eids = expected_send_eids | expected_recv_eids

    # ---- P1.1 Totality: one OCEL event per source event, same activity
    # and timestamp as its source counterpart, and the preserved structural
    # attributes collab:participant and collab:elemType intact (checked by
    # event identity, not merely by matching aggregate counts; part/elem are
    # total functions on the source event set, so a stripped or altered value
    # is a construction defect, M8).
    n_src = len(src_df)
    n_ev = len(ev)
    ev_by_eid = ev.set_index(COL_EID) if not ev.empty else None

    def _got_attr(eid_: str, col: str) -> Optional[str]:
        if ev_by_eid is None or col not in ev_by_eid.columns:
            return None
        v = ev_by_eid.at[eid_, col]
        if v is None or pd.isna(v):
            return None
        s = str(v).strip()
        return s if s != "" else None

    missing_eids = 0
    activity_mismatches = 0
    timestamp_mismatches = 0
    participant_mismatches = 0
    elemtype_mismatches = 0
    residual_mismatches = 0
    # B14: `part` is a TOTAL function on E_L (unlike
    # from/to, which Normalization nu explicitly allows to be
    # partial on the counterparty side). A source event whose collab:participant
    # is itself absent violates that precondition; the transform still
    # propagates the absence faithfully (no participant/in_orchestration edge is
    # created, M6), so participant_mismatches above stays 0 (None == None) and
    # this would otherwise pass unnoticed -- surfaced here as its own count,
    # separate from preservation mismatches, since it flags a source
    # precondition violation, not a construction defect.
    part_undefined_in_source = 0
    # E30: _require_columns only checks that collab:elemType is PRESENT, not
    # that its values lie in the codomain {task, SendTask, ReceiveTask} the
    # mapping assumes; an out-of-domain value (e.g. a typo) is
    # silently treated as a plain task by the send/receive comparisons
    # (M5/M6) while the raw garbage string is still preserved verbatim as
    # the output elemType attribute -- so elemtype_mismatches above would
    # stay 0 (both sides preserve the same string) and this would pass
    # unnoticed. Counted separately since it flags a source precondition
    # violation, same category as B14's part_undefined_in_source.
    # E30 (reviewer round 2): the check reads the RAW pre-default value
    # (expected_elem_raw_by_eid, None when the source left elemType
    # absent/empty/whitespace) rather than exp_elem -- otherwise the
    # 'task' default in _sorted_case_events coerces an ABSENT value to
    # 'task' before this check runs, masking it (only a non-empty bogus
    # string like 'BogusTask' survived to be caught).
    elemtype_out_of_domain = 0
    for eid, (exp_act, exp_ts, exp_part, exp_elem, _, _) in expected_by_eid.items():
        if exp_part is None:
            part_undefined_in_source += 1
        if expected_elem_raw_by_eid.get(eid) not in (ELEM_TASK, ELEM_SEND, ELEM_RECEIVE):
            elemtype_out_of_domain += 1
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
        if _got_attr(eid, "participant") != exp_part:
            participant_mismatches += 1
        # Strict comparison, no fallback to 'task' on absence: the transform
        # always materializes an explicit elemType (normalized to 'task',
        # M5), so a missing value in the output is a preservation defect
        # even for a plain task event, not an equivalent spelling of 'task'.
        if _got_attr(eid, "elemType") != exp_elem:
            elemtype_mismatches += 1
        # D24: residual (M8) source attributes must survive unchanged. Only
        # checked when the source actually carries a value for that column
        # on this event -- an attribute the source never set is not a
        # preservation defect if it is also absent downstream (P1.1 must
        # not fail on the ordinary case of per-activity-type attributes).
        src_row = expected_row_by_eid.get(eid)
        if src_row is not None:
            for k in residual_keys:
                src_val = src_row.get(k)
                if pd.isna(src_val):
                    continue
                if _got_attr(eid, k) != str(src_val).strip():
                    residual_mismatches += 1
    p11 = (n_src == n_ev) and missing_eids == 0 \
        and activity_mismatches == 0 and timestamp_mismatches == 0 \
        and participant_mismatches == 0 and elemtype_mismatches == 0 \
        and residual_mismatches == 0 and part_undefined_in_source == 0 \
        and elemtype_out_of_domain == 0
    out.append(CheckResult(
        "P1.1 Totality",
        bool(p11),
        f"source events={n_src}, ocel events={n_ev}; missing event ids={missing_eids}; "
        f"activity mismatches={activity_mismatches}; timestamp mismatches={timestamp_mismatches}; "
        f"participant mismatches={participant_mismatches}; "
        f"elemType mismatches={elemtype_mismatches}; "
        f"residual (M8) attribute mismatches={residual_mismatches}; "
        f"source events with collab:participant undefined (violates the "
        f"total-function precondition of Definition app-r1)={part_undefined_in_source}; "
        f"source events with collab:elemType absent or outside "
        f"{{task, SendTask, ReceiveTask}}={elemtype_out_of_domain}."))

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
    # D24: the CollaborationCase object's own `caseId` attribute must agree
    # with the case it was built from -- a corrupted caseId keeps every
    # within edge and set-membership check above intact, so it is otherwise
    # invisible to P1.2. `caseId` is always set at object-creation time
    # (M1), so a MISSING value is itself the defect (D24 reviewer round 2:
    # comparing only when present let a deleted caseId pass vacuously).
    bad_cc_caseid = 0
    for case in expected_eids_by_case:
        cc_oid = _cc_id(case)
        if obj_idx_all is not None and cc_oid in obj_idx_all.index:
            got = obj_idx_all.at[cc_oid, "caseId"] if "caseId" in obj_idx_all.columns else None
            if pd.isna(got) or str(got) != case:
                bad_cc_caseid += 1
        else:
            bad_cc_caseid += 1
    p12 = (len(mismatches) == 0) and (len(dangling) == 0) and (bad_cc_caseid == 0)
    out.append(CheckResult(
        "P1.2 Per-case partition",
        bool(p12),
        f"CC objects={len(cc_ids)}; set mismatches={len(mismatches)}; "
        f"dangling within-targets={len(dangling)}; caseId disagreements={bad_cc_caseid}."
        + ("" if p12 else f" first mismatches={dict(list(mismatches.items())[:5])}")))

    # ---- P1.2b Per-case order: identifier order equals prec_L exactly
    # (timestamp order with ties broken by source appearance order, M5).
    # A timestamp-monotonicity-only check (b < a) would accept two same-
    # timestamp events swapped in identifier order, since b == a is not a
    # decrease -- silently violating the tie-break half of prec_L that
    # criterion P1.2/M5 also promises. Instead, compare the identifier
    # order directly against `expected_eids_by_case`, which is prec_L
    # itself (timestamp, then __src_order__): identifiers are created in
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
    # of a Receive) unrecorded (see Normalization nu above). That side is
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

        # Coverage and per-event multiplicity (the converse of the above):
        # every source SendTask/ReceiveTask event must be related to EXACTLY
        # ONE Message by 'send'/'receive' (M4/M6, unconditionally -- unlike
        # the counterparty endpoint, this does not depend on source
        # completeness), and no source 'task' event may carry such a
        # relation. Which events are communication events is derived from
        # the SOURCE log (expected_comm_eids), never from the output's own
        # elemType column: an output corrupted by flipping elemType to
        # 'task' and dropping the Message would otherwise vouch for itself
        # and pass. The XOR check above only bounds edges per MESSAGE, so a
        # second Message hanging off the same event (each with one edge)
        # would also go undetected without the per-event count below.
        send_covered = set(send_edges[COL_EID])
        recv_covered = set(recv_edges[COL_EID])
        covered_eids = send_covered | recv_covered
        uncovered_events = expected_comm_eids - covered_eids
        spurious_covered = covered_eids - expected_comm_eids
        # Direction: the edge's QUALIFIER must match the source elemType --
        # a SendTask event related by 'receive' (or vice versa) is a defect
        # even when every count above is right, and it cannot be left to the
        # participant/sender comparison further down, which is silently
        # inconclusive when the counterparty endpoint is unrecorded or when
        # sender == receiver (a self-message).
        wrong_direction = ((send_covered & expected_recv_eids)
                           | (recv_covered & expected_send_eids))
        edge_counts = pd.concat([send_edges[COL_EID], recv_edges[COL_EID]]).value_counts()
        multi_message_events = int((edge_counts > 1).sum())
        p13_detail_bits.append(
            f"send/receive events with no Message relation: {len(uncovered_events)}")
        p13_detail_bits.append(
            f"non-communication events with a Message relation: {len(spurious_covered)}")
        p13_detail_bits.append(
            f"events whose send/receive qualifier contradicts the source "
            f"elemType: {len(wrong_direction)}")
        p13_detail_bits.append(
            f"events related to more than one Message: {multi_message_events}")
        if (uncovered_events or spurious_covered or wrong_direction
                or multi_message_events):
            p13_ok = False

        # the single (event, qualifier) related to each Message
        msg_event: Dict[str, Tuple[str, str]] = {
            oid: (eid, Q_SEND) for oid, eid in zip(send_edges[COL_OID], send_edges[COL_EID])}
        msg_event.update({oid: (eid, Q_RECEIVE)
                          for oid, eid in zip(recv_edges[COL_OID], recv_edges[COL_EID])})

        # D24: every Message's `exchanged_in` O2O relation (M7) must target
        # the SAME CollaborationCase as the `within` edge of its related
        # event -- a deleted or retargeted exchanged_in edge is otherwise
        # invisible (the Message still has its send/receive edge and its
        # from/to endpoints, so it is not an orphan under P1.5).
        exch_target: Dict[str, str] = (
            dict(zip(o2o[o2o[COL_QUALIFIER] == Q_EXCHANGED_IN][COL_OID],
                    o2o[o2o[COL_QUALIFIER] == Q_EXCHANGED_IN][COL_OID2]))
            if not o2o.empty else {})
        bad_exchanged_in = sum(
            1 for m, (eid, _q) in msg_event.items()
            if ev_within.get(eid) != exch_target.get(m))
        p13_detail_bits.append(
            f"exchanged_in disagreements with the related event's case: {bad_exchanged_in}")
        if bad_exchanged_in:
            p13_ok = False

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
            multiplicity_violations = 0  # more than one from/to O2O edge for a single Message
            event_participant_missing = 0  # comm event lacking its participant attribute
            source_endpoint_mismatches = 0  # any endpoint representation vs the SOURCE value
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

                # Multiplicity: at most one 'from' and one 'to' edge per
                # Message (M7). A second, contradictory edge on the same
                # side must not be silently ignored by only ever comparing
                # froms[0]/tos[0].
                if len(froms) > 1:
                    multiplicity_violations += 1
                elif sender_defined and froms and froms[0] != _participant_id(str(sender)):
                    oa_disagreements += 1
                if len(tos) > 1:
                    multiplicity_violations += 1
                elif receiver_defined and tos and tos[0] != _participant_id(str(receiver)):
                    oa_disagreements += 1

                eid_qual = msg_event.get(m)
                if eid_qual is None or ev_idx is None or eid_qual[0] not in ev_idx.index:
                    continue
                eid, qual = eid_qual
                ev_from = ev_idx.at[eid, "fromParticipant"] if "fromParticipant" in ev_idx.columns else None
                ev_to = ev_idx.at[eid, "toParticipant"] if "toParticipant" in ev_idx.columns else None
                ev_participant = ev_idx.at[eid, "participant"] if "participant" in ev_idx.columns else None
                # M8 preserves fromParticipant/toParticipant on the event
                # exactly when from(e)/to(e) is defined (M5/M8): the object attribute and the preserved
                # event attribute must agree on DEFINEDNESS, not only on
                # value when both happen to be present.
                if sender_defined != pd.notna(ev_from):
                    inconsistent_partial += 1
                if receiver_defined != pd.notna(ev_to):
                    inconsistent_partial += 1
                if pd.notna(sender) and pd.notna(ev_from) and str(ev_from) != str(sender):
                    ea_disagreements += 1
                if pd.notna(receiver) and pd.notna(ev_to) and str(ev_to) != str(receiver):
                    ea_disagreements += 1
                if pd.notna(ev_participant):
                    expected = sender if qual == Q_SEND else receiver
                    if pd.notna(expected) and str(ev_participant) != str(expected):
                        participant_disagreements += 1
                else:
                    # collab:participant is total in the source,
                    # so a communication event without it is a preservation
                    # defect (M8). Reported here explicitly (not silently
                    # skipped); the per-event identity comparison in P1.1 is
                    # what fails on it, against the source expectation.
                    event_participant_missing += 1

                # D24 (reviewer round 2): every comparison above is between
                # the transform's OWN representations (Message attribute,
                # event attribute, O2O target) -- a mutation that coherently
                # rewrites all three to the SAME wrong value for one
                # endpoint agrees with itself on every check above and
                # would otherwise pass. Re-derive the expected endpoint
                # straight from the source log (expected_endpoints_by_eid,
                # the same normalization _sorted_case_events/nu already
                # applies) and compare each representation against it
                # independently.
                exp_from, exp_to = expected_endpoints_by_eid.get(eid, (None, None))
                sender_val = str(sender) if pd.notna(sender) else None
                receiver_val = str(receiver) if pd.notna(receiver) else None
                ev_from_val = str(ev_from) if pd.notna(ev_from) else None
                ev_to_val = str(ev_to) if pd.notna(ev_to) else None
                if sender_val != exp_from:
                    source_endpoint_mismatches += 1
                if receiver_val != exp_to:
                    source_endpoint_mismatches += 1
                if ev_from_val != exp_from:
                    source_endpoint_mismatches += 1
                if ev_to_val != exp_to:
                    source_endpoint_mismatches += 1

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
            p13_detail_bits.append(
                f"from/to O2O multiplicity violations (>1 edge on one side): "
                f"{multiplicity_violations}")
            p13_detail_bits.append(
                f"communication events missing their participant attribute "
                f"(fails P1.1): {event_participant_missing}")
            p13_detail_bits.append(
                f"endpoint representations (event/Message/O2O) disagreeing with "
                f"the source value: {source_endpoint_mismatches}")
            if (oa_disagreements or ea_disagreements or participant_disagreements
                    or inconsistent_partial or multiplicity_violations
                    or source_endpoint_mismatches):
                p13_ok = False
    out.append(CheckResult("P1.3 Message well-formedness", bool(p13_ok),
                           "; ".join(p13_detail_bits) or "no messages"))

    # ---- P1.4 OrchestrationCase coherence ---------------------------
    # Checked in three parts:
    #  (a) totality: every event with a defined participant has EXACTLY ONE
    #      'in_orchestration' edge, targeting the OrchestrationCase its
    #      source (case, participant) pair implies (M3/M6, checked against
    #      the source-derived expectation, not the output's own edges --
    #      D24: a mutation that deletes an event's in_orchestration edge,
    #      e.g. together with its 'participant' edge, used to be invisible
    #      because this check only ever iterated over edges that survived);
    #  (b) for every existing 'in_orchestration' edge, its target oc is
    #      'part_of' the event's 'within' object (the collaboration case);
    #  (c) every OrchestrationCase is 'for_participant' exactly one
    #      participant object, whose OBJECT TYPE and 'name' attribute both
    #      equal the OrchestrationCase's 'participant' attribute (M2 encodes
    #      the identifier twice, as tau(p) and as the name value, and the
    #      revised criterion requires both to agree), and the
    #      OrchestrationCase's own caseId/participant attributes agree with
    #      the (case, participant) pair it was created for (D24: a corrupted
    #      caseId/participant attribute affects no edge, so it was otherwise
    #      invisible to every check).
    p14_ok = True
    p14_detail = ""
    if not rel.empty:
        inorch_rows = rel[rel[COL_QUALIFIER] == Q_IN_ORCHESTRATION]
        ev_inorch = dict(zip(inorch_rows[COL_EID], inorch_rows[COL_OID]))
        inorch_counts = inorch_rows.groupby(COL_EID).size().to_dict()
        part_of = ({r[COL_OID]: r[COL_OID2] for _, r in
                    o2o[o2o[COL_QUALIFIER] == Q_PART_OF].iterrows()}
                   if not o2o.empty else {})

        # (a) totality + target identity, against the source expectation
        missing_inorch = 0
        wrong_oc_target = 0
        multi_inorch = 0
        for eid, (_, _, exp_part, _, _, _) in expected_by_eid.items():
            if exp_part is None:
                continue
            if int(inorch_counts.get(eid, 0)) > 1:
                multi_inorch += 1
                continue
            got_oc = ev_inorch.get(eid)
            if got_oc is None:
                missing_inorch += 1
            elif got_oc != _oc_id(case_by_eid[eid], exp_part):
                wrong_oc_target += 1

        # (b) in_orchestration/within coherence, for existing edges
        bad_partof = sum(1 for eid, oc in ev_inorch.items()
                         if part_of.get(oc) != ev_within.get(eid))

        # (c) for_participant well-formedness + OC identity, per OC object
        oc_ids = set(obj[obj[COL_OTYPE] == OT_OC][COL_OID]) if not obj.empty else set()
        forpart_edges = (o2o[o2o[COL_QUALIFIER] == Q_FOR_PARTICIPANT] if not o2o.empty
                         else pd.DataFrame(columns=[COL_OID, COL_OID2]))
        forpart_counts = forpart_edges.groupby(COL_OID).size().to_dict()
        forpart_target = dict(zip(forpart_edges[COL_OID], forpart_edges[COL_OID2]))
        bad_forpart = 0
        bad_name = 0
        bad_type = 0
        for oc in oc_ids:
            if int(forpart_counts.get(oc, 0)) != 1:
                bad_forpart += 1
                continue
            # name agreement: for_participant target's 'name' == oc.participant.
            # D24 (reviewer round 2): 'name'/'participant' are always set at
            # object-creation time (M2/M3), so a MISSING value is itself a
            # construction defect, not merely inconclusive -- comparing only
            # when both happen to be present let a deleted attribute pass
            # vacuously (reproduced: deleting a participant's name or the
            # OC's caseId/participant kept every check above PASS).
            tgt = forpart_target.get(oc)
            if obj_idx_all is not None and tgt in obj_idx_all.index:
                tgt_name = obj_idx_all.at[tgt, "name"] if "name" in obj_idx_all.columns else None
                oc_part = obj_idx_all.at[oc, "participant"] if "participant" in obj_idx_all.columns else None
                if pd.isna(tgt_name) or pd.isna(oc_part) or str(tgt_name) != str(oc_part):
                    bad_name += 1
                # The type side of the same identity: tau is injective, so
                # the target's object type must be exactly tau(oc.participant).
                # Without this, a for_participant edge retargeted to a
                # correctly-named object of the wrong type would still pass.
                tgt_type = obj_idx_all.at[tgt, COL_OTYPE]
                if pd.isna(oc_part) or tgt_type != tau.type_of(str(oc_part)):
                    bad_type += 1
            else:
                bad_name += 1
                bad_type += 1

        bad_oc_identity = 0
        for eid, (_, _, exp_part, _, _, _) in expected_by_eid.items():
            if exp_part is None:
                continue
            oc_oid = _oc_id(case_by_eid[eid], exp_part)
            if obj_idx_all is not None and oc_oid in obj_idx_all.index:
                got_case = obj_idx_all.at[oc_oid, "caseId"] if "caseId" in obj_idx_all.columns else None
                got_part = obj_idx_all.at[oc_oid, "participant"] if "participant" in obj_idx_all.columns else None
                if pd.isna(got_case) or str(got_case) != case_by_eid[eid]:
                    bad_oc_identity += 1
                if pd.isna(got_part) or str(got_part) != exp_part:
                    bad_oc_identity += 1
            else:
                bad_oc_identity += 1

        p14_ok = (missing_inorch == 0 and wrong_oc_target == 0 and multi_inorch == 0
                 and bad_partof == 0 and bad_forpart == 0 and bad_name == 0
                 and bad_type == 0 and bad_oc_identity == 0)
        p14_detail = (f"events missing in_orchestration={missing_inorch}; "
                      f"in_orchestration wrong target={wrong_oc_target}; "
                      f"events with >1 in_orchestration edge={multi_inorch}; "
                      f"in_orchestration/within mismatches={bad_partof}; "
                      f"orchestration cases !=1 for_participant={bad_forpart}; "
                      f"for_participant name disagreements={bad_name}; "
                      f"for_participant type disagreements={bad_type}; "
                      f"OrchestrationCase caseId/participant identity disagreements={bad_oc_identity}")
    out.append(CheckResult("P1.4 OrchestrationCase coherence", bool(p14_ok), p14_detail))

    # ---- P1.5 No orphan objects (+ D24: no dangling relations) ------
    # Orphans (an object with no relation at all) is the formal P1.5
    # statement. D24 adds the converse, which P1.5 never checked: a
    # relation whose target does not exist as an object (E2O's oid, or
    # O2O's oid/oid2) is a construction defect just as much as an orphan
    # object is, and neither the per-relation checks above (which only
    # ever compare edges that ARE present) nor "no orphan objects" catches
    # a relation pointing at an id that was never materialized.
    related_oids = set()
    if not rel.empty:
        related_oids |= set(rel[COL_OID])
    if not o2o.empty:
        related_oids |= set(o2o[COL_OID]) | set(o2o[COL_OID2])
    all_oids = set(obj[COL_OID]) if not obj.empty else set()
    orphans = all_oids - related_oids
    all_eids_actual = set(ev[COL_EID]) if not ev.empty else set()
    dangling_e2o_eid = (set(rel[COL_EID]) - all_eids_actual) if not rel.empty else set()
    dangling_e2o_oid = (set(rel[COL_OID]) - all_oids) if not rel.empty else set()
    dangling_o2o = ((set(o2o[COL_OID]) | set(o2o[COL_OID2])) - all_oids) if not o2o.empty else set()
    p15_ok = (len(orphans) == 0 and len(dangling_e2o_eid) == 0
             and len(dangling_e2o_oid) == 0 and len(dangling_o2o) == 0)
    out.append(CheckResult("P1.5 No orphan objects", bool(p15_ok),
                           f"objects={len(all_oids)}; orphans={len(orphans)}"
                           + ("" if not orphans else f"; e.g. {list(orphans)[:5]}")
                           + f"; dangling E2O event ids={len(dangling_e2o_eid)}"
                           + f"; dangling E2O object targets={len(dangling_e2o_oid)}"
                           + f"; dangling O2O targets={len(dangling_o2o)}"))

    # ---- P1.6 Participant coherence ---------------------------------
    # For every event, the participant object reached by the direct
    # 'participant' E2O edge must equal the one reached via the two-step
    # 'in_orchestration' -> 'for_participant' path. D24: the universe of
    # events checked is every source event with a defined participant
    # (from `expected_by_eid`), not the union of eids that happen to still
    # have a 'participant' or 'in_orchestration' edge -- the previous universe
    # meant that deleting BOTH edges for one event removed it from the
    # universe entirely, so the mutation passed vacuously instead of being
    # caught by "missing one side".
    p16_ok = True
    p16_detail = ""
    if not rel.empty:
        participant_rows = rel[(rel[COL_QUALIFIER] == Q_PARTICIPANT)
                               & (rel[COL_OTYPE].map(is_participant_type))]
        ev_participant = dict(zip(participant_rows[COL_EID], participant_rows[COL_OID]))
        # D24 (reviewer round 2): `dict(zip(...))` above silently collapses
        # a SECOND, contradictory 'participant' edge for the same event to
        # whichever row pandas iterates last -- a duplicate-edge mutation
        # could therefore agree with the (arbitrarily chosen) surviving
        # value and pass. Count distinct targets per event id separately so
        # ambiguity itself is caught, independent of which value the dict
        # happened to keep.
        participant_targets_per_eid = participant_rows.groupby(COL_EID)[COL_OID].nunique().to_dict()
        ev_inorch = dict(zip(
            rel[rel[COL_QUALIFIER] == Q_IN_ORCHESTRATION][COL_EID],
            rel[rel[COL_QUALIFIER] == Q_IN_ORCHESTRATION][COL_OID]))
        forpart_target = (dict(zip(o2o[o2o[COL_QUALIFIER] == Q_FOR_PARTICIPANT][COL_OID],
                                   o2o[o2o[COL_QUALIFIER] == Q_FOR_PARTICIPANT][COL_OID2]))
                          if not o2o.empty else {})
        all_eids = {eid for eid, (_, _, exp_part, _, _, _) in expected_by_eid.items()
                   if exp_part is not None}
        mismatches6 = 0
        missing_one_side = 0
        ambiguous_participant_edge = 0
        for eid in all_eids:
            if participant_targets_per_eid.get(eid, 0) > 1:
                ambiguous_participant_edge += 1
                continue
            direct = ev_participant.get(eid)
            oc_oid = ev_inorch.get(eid)
            indirect = forpart_target.get(oc_oid) if oc_oid is not None else None
            if direct is None or indirect is None:
                missing_one_side += 1
                continue
            if direct != indirect:
                mismatches6 += 1
        p16_ok = (mismatches6 == 0) and (missing_one_side == 0) and (ambiguous_participant_edge == 0)
        p16_detail = (f"events checked={len(all_eids)}; "
                      f"direct/indirect mismatches={mismatches6}; "
                      f"missing one side={missing_one_side}; "
                      f"events with >1 distinct 'participant' edge target={ambiguous_participant_edge}")
    out.append(CheckResult("P1.6 Participant coherence", bool(p16_ok), p16_detail))

    out.extend(_run_layer_checks(res, cfg, is_participant_type, ev_within))

    return out


def _run_layer_checks(res: TransformResult,
                      cfg: MappingConfig,
                      is_participant_type,
                      ev_within: Dict[str, str]) -> List[CheckResult]:
    """Criteria of the refinement layers: PR.1/PR.2 for the resource layer,
    PC.1 for the correlation layer. Each runs only when its layer is enabled,
    so a core-mapping run reports exactly P1.1-P1.6 as before."""
    out: List[CheckResult] = []
    rel, o2o, obj = res.relations_df, res.o2o_df, res.objects_df

    # ---- PR.1 / PR.2 (resource layer) -------------------------------
    if cfg.resource_attr:
        res_e2o = (rel[rel[COL_QUALIFIER] == Q_RESOURCE] if not rel.empty
                   else pd.DataFrame(columns=[COL_EID, COL_OID]))
        acts_for = (o2o[o2o[COL_QUALIFIER] == Q_ACTS_FOR] if not o2o.empty
                    else pd.DataFrame(columns=[COL_OID, COL_OID2]))
        acts_for_pairs = set(zip(acts_for[COL_OID], acts_for[COL_OID2]))
        # PR.1: for every event whose 'resource' relation is defined, its
        # resource object and its participant object are related by
        # 'acts_for'. The two dimensions therefore never disagree, and no
        # event is attributed to a resource the log never records as acting
        # for that participant. Events without a participant are vacuous:
        # there is no participant object for the pair to be formed with.
        part_e2o = (rel[(rel[COL_QUALIFIER] == Q_PARTICIPANT)
                        & (rel[COL_OTYPE].map(is_participant_type))]
                    if not rel.empty else pd.DataFrame(columns=[COL_EID, COL_OID]))
        ev_participant = dict(zip(part_e2o[COL_EID], part_e2o[COL_OID]))
        pr1_bad = 0
        for eid, res_oid in zip(res_e2o[COL_EID], res_e2o[COL_OID]):
            pa_oid = ev_participant.get(eid)
            if pa_oid is None:
                continue
            if (res_oid, pa_oid) not in acts_for_pairs:
                pr1_bad += 1
        out.append(CheckResult(
            "PR.1 Actor-participant coherence", pr1_bad == 0,
            f"resource E2O relations={len(res_e2o)}; acts_for pairs="
            f"{len(acts_for_pairs)}; events whose (resource, participant) pair "
            f"is not related by acts_for={pr1_bad}"))

        # PR.2: every Resource object is the target of at least one
        # 'resource' relation, so P1.5 keeps holding of the enlarged object
        # set. Also flag the converse -- a 'resource' or 'acts_for' relation
        # whose Resource endpoint was never materialized -- since the layer
        # must not introduce a dangling relation either.
        res_oids = (set(obj[obj[COL_OTYPE] == OT_RESOURCE][COL_OID])
                    if not obj.empty else set())
        unreached = res_oids - set(res_e2o[COL_OID])
        dangling = ((set(res_e2o[COL_OID]) | set(acts_for[COL_OID])) - res_oids)
        out.append(CheckResult(
            "PR.2 No orphan resources", not unreached and not dangling,
            f"Resource objects={len(res_oids)}; with no resource relation="
            f"{len(unreached)}; relations with an unmaterialized Resource "
            f"endpoint={len(dangling)}"))

    # ---- PC.1 (correlation layer) -----------------------------------
    if cfg.correlation_attr:
        corr = (o2o[o2o[COL_QUALIFIER] == Q_CORRELATED_WITH] if not o2o.empty
                else pd.DataFrame(columns=[COL_OID, COL_OID2]))
        msg_oids = (set(obj[obj[COL_OTYPE] == OT_MESSAGE][COL_OID])
                    if not obj.empty else set())
        send_of = (dict(zip(rel[rel[COL_QUALIFIER] == Q_SEND][COL_OID],
                            rel[rel[COL_QUALIFIER] == Q_SEND][COL_EID]))
                   if not rel.empty else {})
        recv_of = (dict(zip(rel[rel[COL_QUALIFIER] == Q_RECEIVE][COL_OID],
                            rel[rel[COL_QUALIFIER] == Q_RECEIVE][COL_EID]))
                   if not rel.empty else {})
        obj_idx = obj.set_index(COL_OID) if not obj.empty else None

        def endpoint(oid: str, key: str) -> Optional[str]:
            if obj_idx is None or oid not in obj_idx.index or key not in obj_idx.columns:
                return None
            v = obj_idx.at[oid, key]
            return None if pd.isna(v) else str(v)

        # Multiplicity: at most one outgoing and one incoming relation per
        # Message. This is what fails on a correlation attribute that does
        # not induce a one-to-one pairing -- e.g. an identifier shared by
        # three or more observations, which rule C1 (deliberately
        # unconditional) turns into several relations rather than silently
        # dropping.
        multi_out = int((corr.groupby(COL_OID).size() > 1).sum()) if len(corr) else 0
        multi_in = int((corr.groupby(COL_OID2).size() > 1).sum()) if len(corr) else 0
        bad_direction = bad_case = bad_endpoints = bad_order = dangling_corr = 0
        for src, tgt in zip(corr[COL_OID], corr[COL_OID2]):
            if src not in msg_oids or tgt not in msg_oids:
                dangling_corr += 1
                continue
            s_eid, r_eid = send_of.get(src), recv_of.get(tgt)
            # Direction: from the observation of a send event to that of a
            # receive event, so the direction of the exchange is recoverable
            # from the relation itself.
            if s_eid is None or r_eid is None:
                bad_direction += 1
                continue
            if ev_within.get(s_eid) != ev_within.get(r_eid):
                bad_case += 1
            # Endpoints agree wherever both are defined.
            for key in ("sender", "receiver"):
                a, b = endpoint(src, key), endpoint(tgt, key)
                if a is not None and b is not None and a != b:
                    bad_endpoints += 1
            # The send event precedes the receive event. Within a case the
            # event identifier embeds prec_L (M5), so comparing identifiers
            # is the same order P1.2 reconstructs; across cases the
            # comparison is meaningless, but such a relation is already
            # counted by bad_case.
            if ev_within.get(s_eid) == ev_within.get(r_eid) and not (s_eid < r_eid):
                bad_order += 1
        pc1_ok = not (multi_out or multi_in or bad_direction or bad_case
                      or bad_endpoints or bad_order or dangling_corr)
        out.append(CheckResult(
            "PC.1 Correlation well-formedness", pc1_ok,
            f"correlated_with relations={len(corr)}; messages with >1 outgoing="
            f"{multi_out}; with >1 incoming={multi_in}; not send->receive="
            f"{bad_direction}; across different cases={bad_case}; endpoint "
            f"disagreements={bad_endpoints}; receive not after send={bad_order}; "
            f"relations with a non-Message endpoint={dangling_corr}"))

    return out


def check_export_reachability(objects_df: pd.DataFrame,
                              export_relations_df: pd.DataFrame) -> CheckResult:
    """Export-level completeness check, distinct from P1.1-P1.6: those checks
    verify TransformResult (the pre-export DataFrames), not the E2O table that
    is actually handed to the OCEL exporters. pm4py's exporters call
    filtering_utils.propagate_relations_filtering(), which drops any object
    absent from the E2O relations table regardless of O2O reachability (see
    the module docstring NOTE on the `participant` edge and
    _add_export_reachability_witnesses). This check runs AFTER that witness
    pass, over export_relations_df, so a regression that reintroduces an
    export-only orphan (e.g. a new O2O-only object type the witness pass does
    not yet cover) is caught here rather than silently shipping an
    incomplete .jsonocel/.sqlite.
    """
    if objects_df.empty:
        return CheckResult("P1.7 Export reachability", True, "no objects")
    all_oids = set(objects_df[COL_OID])
    reachable = set(export_relations_df[COL_OID]) if not export_relations_df.empty else set()
    unreachable = all_oids - reachable
    return CheckResult(
        "P1.7 Export reachability", len(unreachable) == 0,
        f"objects={len(all_oids)}; unreachable in the exported E2O table="
        f"{len(unreachable)}"
        + ("" if not unreachable else f"; e.g. {sorted(unreachable)[:5]}"))


def print_check_report(checks: List[CheckResult], stats: Dict[str, Any]) -> bool:
    logger.info("---- transformation stats ----")
    for k, v in stats.items():
        logger.info("  %-22s %s", k, v)
    logger.info("---- consistency checks ----")
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
    # D25: witness E2O edges only for the pm4py write path, so objects
    # reachable exclusively via O2O (e.g. endpoint-only Participants)
    # survive export; res.relations_df itself (returned to the caller,
    # and already used above by run_consistency_checks) is untouched.
    export_relations_df = _add_export_reachability_witnesses(
        res.objects_df, res.relations_df, res.o2o_df)
    # P1.7 runs over export_relations_df -- the E2O table actually handed to
    # the exporters -- so the printed report and --strict cover the
    # delivered artifact's completeness, not only the pre-export model that
    # P1.1-P1.6 verify.
    checks = checks + [check_export_reachability(res.objects_df, export_relations_df)]
    all_ok = print_check_report(checks, res.stats)
    if strict and not all_ok:
        raise RuntimeError("Consistency checks failed under strict mode; not exporting.")
    ocel = build_ocel_object(res.events_df, res.objects_df,
                             export_relations_df, res.o2o_df)
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
                   help="Abort if any consistency check or schema validation fails.")
    p.add_argument("--no-validate", action="store_true",
                   help="Skip OCEL 2.0 JSON schema validation of the output.")
    p.add_argument("--resource-attr", metavar="ATTR", default=None,
                   help="Apply the resource layer (R1-R3) over this residual "
                        "source attribute, e.g. 'org:resource'. Adds Resource "
                        "objects, `resource` E2O and `acts_for` O2O relations, "
                        "and checks PR.1/PR.2. Omitted, no Resource is created.")
    p.add_argument("--correlation-attr", metavar="ATTR", default=None,
                   help="Apply the correlation layer (C1) over this residual "
                        "source attribute, e.g. 'msgInstanceId'. Relates the "
                        "send and receive observations sharing an identifier by "
                        "`correlated_with` and checks PC.1. Omitted, no "
                        "send/receive correspondence is inferred.")
    p.add_argument("--encoding", default="utf-8")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s")
    cfg = MappingConfig(resource_attr=args.resource_attr,
                        correlation_attr=args.correlation_attr)
    convert(args.input_xes, args.output, cfg=cfg,
            strict=args.strict, validate=not args.no_validate,
            encoding=args.encoding)
    return 0


if __name__ == "__main__":
    sys.exit(main())
