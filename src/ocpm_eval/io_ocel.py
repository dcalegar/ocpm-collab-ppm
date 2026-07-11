"""I/O helper for reading OCEL 2.0 for the LABEL side, into the neutral model.

Reading uses OCPA's native OCEL 2.0 importer (ocpa.objects.log.importer.ocel2.sqlite),
the same library used for feature extraction, so both sides share the same reading path.
``load_ocpa_ocel`` is also imported by features_ocpa and rq3_pipeline so the OCEL object
can be loaded once and passed to both label and feature extraction.
"""
import os
from typing import Dict, List, Tuple
from ocpm_tasks.adapters import from_ocpa, from_ocel2_sqlite, _normalize_ocel_sqlite_timestamps
from ocpm_tasks.model import ObjectCentricLog

# Direct event->Participant E2O qualifier added by the converter
# (collab_xes_to_ocel.py). It is a genuine relation of the conceptual
# model (rule M6): a participant is reachable both directly, via this
# edge, and indirectly via in_projection -> for_participant (see
# ocpm_tasks/schema.py); their agreement is checked by P1.6. OCPA's
# leading-type extraction connects ALL E2O-related objects of an event
# pairwise regardless of type, so this edge merges every
# CollaborationCase's process execution with every other one that shares
# a Participant -- and since a handful of Participants are shared across
# the whole log, every execution collapses into (almost) the entire log,
# which is what makes feature extraction stall. Strip it (mostly) before
# handing the file to OCPA; this is an OCPA-specific import workaround,
# not a change to the conceptual model.
#
# OCPA's own SQLite importer (ocpa/objects/log/importer/ocel2/sqlite/...)
# builds its object table (``OCEL.obj.raw.objects``) exclusively from rows
# surviving in ``event_object``: an object type with no surviving E2O row
# at all is dropped from that table entirely, even though it may still be
# present, correctly typed, in ``object`` and referenced by O2O edges in
# ``object_object`` (``o2o_graph`` is read from ``object_object`` directly
# and is unaffected). Deleting EVERY 'participant' row therefore drops
# every Participant object from OCPA's object table, so any O2O lookup
# that must confirm a target's type is Participant (e.g. Message
# from/to, used by the X-MSt extension) silently fails even though the
# O2O edge itself is intact. Keeping exactly one witness row per distinct
# Participant object avoids this: that participant is still E2O-linked to
# a single event (so OCPA's object table keeps it, correctly typed), while
# every other occurrence -- the ones that would otherwise transitively
# merge unrelated CollaborationCase executions -- is still removed.
_E2O_PARTICIPANT_QUALIFIER = "participant"


def _strip_participant_e2o(path: str) -> str:
    """Return a temp path to a copy of the SQLite file with the direct
    'participant' E2O rows removed, except for one witness row per
    distinct Participant object (see _E2O_PARTICIPANT_QUALIFIER above).
    Returns ``path`` unchanged if there is nothing to strip."""
    import sqlite3, shutil, tempfile

    con = sqlite3.connect(path)
    try:
        has_participant = con.execute(
            "SELECT 1 FROM event_object WHERE ocel_qualifier = ? LIMIT 1",
            (_E2O_PARTICIPANT_QUALIFIER,)).fetchone()
    finally:
        con.close()
    if not has_participant:
        return path

    fd, tmp = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    shutil.copyfile(path, tmp)
    con = sqlite3.connect(tmp)
    try:
        keep_rowids = [r[0] for r in con.execute(
            "SELECT MIN(rowid) FROM event_object WHERE ocel_qualifier = ? "
            "GROUP BY ocel_object_id", (_E2O_PARTICIPANT_QUALIFIER,))]
        placeholders = ",".join("?" * len(keep_rowids))
        con.execute(
            f"DELETE FROM event_object WHERE ocel_qualifier = ? "
            f"AND rowid NOT IN ({placeholders})",
            (_E2O_PARTICIPANT_QUALIFIER, *keep_rowids))
        con.commit()
    finally:
        con.close()
    return tmp


def _break_timestamp_ties(path: str) -> str:
    """Return a temp path to a copy of the SQLite file with per-event
    timestamps nudged by whole microseconds so that no two events of the
    SAME CollaborationCase share an identical ``ocel_time`` (D23).

    Rationale: OCPA's positional features (``previous_type_count``, used
    for X-Inf/OB-M/NV-* etc. via ``features_ocpa.build_feature_set``) cut
    the prefix with ``event_timestamp <= cut_time``
    (ocpa/algo/predictive_monitoring/event_based_features/
    extraction_functions.py::_get_recent_events), NOT with the total order
    prec_L (Definition 1) the paper's feature definitions rely on
    (tasks.tex). When two events of the same execution share a timestamp,
    ``<=`` includes BOTH at each other's cut point, leaking one event's
    existence into the other's "past" count. This is a real, non-uniform
    effect in BPIC2013 (4,051 same-instant Send/Receive pairs); the four
    study logs and ToyCollab have no ties and are unaffected (this
    function is then a no-op and returns ``path`` unchanged).

    The nudge uses ``ocel_id``, which already encodes the correct
    within-case order prec_L (``_event_id`` in collab_xes_to_ocel.py:
    ``e::{case}::{idx}``, idx zero-padded so lexicographic order agrees
    with prec_L -- see the "Verificado y correcto" note on order-embedding
    in the mapping). Events tied within a case are reassigned strictly
    increasing offsets of whole microseconds in idx order (first event of
    a tied run keeps its original timestamp), which resolves the ``<=``
    ambiguity while shifting EVENT_ELAPSED_TIME/EVENT_REMAINING_TIME by at
    most a few microseconds per run -- far below the already-documented
    1s tolerance (C18) and negligible next to this data's second-level
    timestamp granularity. Must run AFTER ``_normalize_ocel_sqlite_timestamps``
    so every ``ocel_time`` carries an explicit ``.NNNNNN`` microsecond
    field the increment can be parsed against.

    Only affects the copy handed to OCPA (feature extraction + the
    ground-truth labels derived from the same ``ocpa_ocel`` object via
    ``from_ocpa``, so oracle/labels/features stay mutually consistent);
    RQ2's ``from_ocel2_sqlite`` reads the untouched original file directly
    and is unaffected."""
    import sqlite3, os, shutil, tempfile
    from datetime import datetime, timedelta

    con = sqlite3.connect(path)
    try:
        suffixes = [r[0] for r in con.execute(
            "SELECT ocel_type_map FROM event_map_type")]
        rows = []  # (suffix, ocel_id, case, idx, time_str)
        for suffix in suffixes:
            for ocel_id, time_str in con.execute(
                    f'SELECT ocel_id, ocel_time FROM "event_{suffix}" '
                    f'WHERE ocel_time IS NOT NULL'):
                case, _, idx = ocel_id.rpartition("::")
                if not idx.isdigit():
                    continue
                rows.append((suffix, ocel_id, case, int(idx), time_str))
    finally:
        con.close()

    by_case: Dict[str, List[Tuple[int, str, str, str]]] = {}
    for suffix, ocel_id, case, idx, time_str in rows:
        by_case.setdefault(case, []).append((idx, suffix, ocel_id, time_str))

    updates = []  # (suffix, ocel_id, new_time_str)
    for case_rows in by_case.values():
        case_rows.sort(key=lambda r: r[0])  # idx order == prec_L within the case
        run: List[Tuple[int, str, str, str]] = []

        def _flush(run):
            if len(run) < 2:
                return
            base = datetime.fromisoformat(run[0][3])
            for offset, (_, suffix, ocel_id, _) in enumerate(run):
                if offset == 0:
                    continue
                new_dt = base + timedelta(microseconds=offset)
                updates.append((suffix, ocel_id, new_dt.isoformat(sep=" ")))

        prev_time = None
        for entry in case_rows:
            if prev_time is not None and entry[3] != prev_time:
                _flush(run)
                run = []
            run.append(entry)
            prev_time = entry[3]
        _flush(run)

    if not updates:
        return path

    fd, tmp = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    shutil.copyfile(path, tmp)
    con = sqlite3.connect(tmp)
    try:
        for suffix, ocel_id, new_time_str in updates:
            con.execute(
                f'UPDATE "event_{suffix}" SET ocel_time = ? WHERE ocel_id = ?',
                (new_time_str, ocel_id))
        con.commit()
    finally:
        con.close()
    return tmp


def load_ocpa_ocel(schema, path: str):
    """Load an OCEL 2.0 SQLite log via OCPA's native importer with leading-type
    execution extraction (one process execution per CollaborationCase)."""
    from ocpa.objects.log.importer.ocel2.sqlite import factory as ocel2_import_factory
    stripped_path = _strip_participant_e2o(path)
    norm_path = _normalize_ocel_sqlite_timestamps(stripped_path)
    tie_broken_path = _break_timestamp_ties(norm_path)
    params = {"execution_extraction": "leading_type", "leading_type": schema.ot_cc}
    try:
        try:
            return ocel2_import_factory.apply(tie_broken_path, parameters=params)
        except TypeError:
            # Older OCPA signature without parameters: import with default extraction.
            # NOTE: default "connected components" may merge instances sharing a Participant;
            # verify the partitioning (the alignment oracle in features_ocpa will flag it).
            return ocel2_import_factory.apply(tie_broken_path)
    finally:
        if tie_broken_path != norm_path:
            os.unlink(tie_broken_path)
        if norm_path != stripped_path:
            os.unlink(norm_path)
        if stripped_path != path:
            os.unlink(stripped_path)


def read_ocel2_labels(path: str, schema,
                      ocpa_ocel=None, corr_attr=None) -> ObjectCentricLog:
    """Build the neutral ObjectCentricLog from an OCEL 2.0 SQLite file.
    If ``ocpa_ocel`` is provided it is used via from_ocpa; otherwise the
    stdlib sqlite3 reader is used to avoid OCPA's itertuples/getattr path
    which breaks on attribute names containing non-identifier characters
    (e.g. 'org:group' → 'event_org:group').
    ``corr_attr``: optional enrichment attribute name for Event.corr_id
    (X-MSt only; see ocpm_tasks.adapters.build_from_relations). None for
    the four study logs (unchanged core-mapping behaviour)."""
    if ocpa_ocel is not None:
        return from_ocpa(ocpa_ocel, schema, corr_attr=corr_attr)
    return from_ocel2_sqlite(path, schema, corr_attr=corr_attr)
