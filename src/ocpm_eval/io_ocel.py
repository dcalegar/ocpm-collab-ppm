"""I/O helper for reading OCEL 2.0 for the LABEL side, into the neutral model.

Reading uses OCPA's native OCEL 2.0 importer (ocpa.objects.log.importer.ocel2.sqlite),
the same library used for feature extraction, so both sides share the same reading path.
``load_ocpa_ocel`` is also imported by features_ocpa and rq3_pipeline so the OCEL object
can be loaded once and passed to both label and feature extraction.
"""
import gc
import os
import time
import warnings
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
# Concrete example (Healthcare log): Participant["Gynecologist"] is a
# single, log-wide object (M2 scope), and it performs events in many
# different CollaborationCases -- say case 459 (events e2, e3) and case
# 460 (events e5, e6). With the direct edge kept, the event-object graph
# has e2/e3 --participant--> Participant["Gynecologist"] <--participant--
# e5/e6: a path connects an event of case 459 to an event of case 460
# through that single shared object. OCPA's leading-type extraction
# treats such a path as evidence the two belong to the same process
# execution, so it merges case 459 and case 460 (and, transitively,
# every other case the same Gynecologist appears in) into one execution
# -- prefixes/timestamps from unrelated cases get mixed, remaining-time
# targets become meaningless, and the per-case grouped-CV assumption
# (that all prefixes of one case stay within a single fold) silently breaks. Removing the
# edge for every occurrence except one witness per Participant leaves no
# live edge shared between two DIFFERENT cases, so case 459 and case 460
# no longer connect through Participant["Gynecologist"] -- while the one
# surviving witness edge (on whichever event happens to keep it) still
# lets OCPA's importer register the object at all (see the paragraph
# below on OCPA's object-table construction).
#
# The same merging risk applies to the D25 witness E2O edges
# (mapping.collab_xes_to_ocel._add_export_reachability_witnesses): those
# reuse the O2O qualifier ('from'/'to') at the E2O level to keep an
# endpoint-only Participant (one that is always a message counterparty
# and never itself collab:participant of an event) from being dropped by
# pm4py's exporter. Such a Participant is log-wide (M2 scope) just like
# any other, so if it is the counterparty of messages in more than one
# CollaborationCase, each referencing event gets its own witness edge and
# the same cross-case path reappears -- this time through 'from'/'to'
# instead of 'participant'. The core mapping (M6) never emits 'from'/'to'
# as an E2O qualifier itself (only within/in_projection/participant/send/
# receive), so every E2O row under these two
# qualifiers is, by construction, a D25 witness row; and D25 only adds a
# witness for an object with zero prior E2O rows, so an object is never
# reached by both 'participant' and a 'from'/'to' witness at once. Both
# patterns are stripped identically below (keep one witness row per
# distinct object).
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
_E2O_CROSS_CASE_QUALIFIERS = (_E2O_PARTICIPANT_QUALIFIER, "from", "to")


def _strip_participant_e2o(path: str) -> str:
    """Return a temp path to a copy of the SQLite file with the E2O rows
    that can reconnect distinct CollaborationCase executions through a
    shared, log-wide object removed -- the direct 'participant' edge (M6)
    and the D25 'from'/'to' witness edges alike -- except for one witness
    row per distinct object (see _E2O_CROSS_CASE_QUALIFIERS above).
    Returns ``path`` unchanged if there is nothing to strip."""
    import sqlite3, shutil, tempfile

    qual_placeholders = ",".join("?" * len(_E2O_CROSS_CASE_QUALIFIERS))
    con = sqlite3.connect(path)
    try:
        has_any = con.execute(
            f"SELECT 1 FROM event_object WHERE ocel_qualifier IN ({qual_placeholders}) LIMIT 1",
            _E2O_CROSS_CASE_QUALIFIERS).fetchone()
    finally:
        con.close()
    if not has_any:
        return path

    fd, tmp = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    shutil.copyfile(path, tmp)
    con = sqlite3.connect(tmp)
    try:
        keep_rowids = [r[0] for r in con.execute(
            f"SELECT MIN(rowid) FROM event_object WHERE ocel_qualifier IN ({qual_placeholders}) "
            "GROUP BY ocel_object_id", _E2O_CROSS_CASE_QUALIFIERS)]
        keep_placeholders = ",".join("?" * len(keep_rowids))
        con.execute(
            f"DELETE FROM event_object WHERE ocel_qualifier IN ({qual_placeholders}) "
            f"AND rowid NOT IN ({keep_placeholders})",
            (*_E2O_CROSS_CASE_QUALIFIERS, *keep_rowids))
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
    extraction_functions.py::_get_recent_events), NOT with the source log's
    total event order prec_L the feature definitions rely on. When two
    events of the same execution share a timestamp,
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
        # Single forward pass in prec_L order enforcing a STRICTLY increasing
        # ocel_time: each event keeps its own timestamp unless that would be
        # <= the previous (already-placed) event's, in which case it is
        # nudged to prev + 1 microsecond. This resolves ties generally,
        # including the cascade the previous per-run implementation missed
        # (D23/B9 reviewer round 2): shifting a tied run by offsets from its
        # own base alone could push its last member ONTO the next distinct
        # timestamp, e.g. [0us, 0us, 1us] -> [0us, 1us, 1us], reintroducing a
        # tie. Comparing against the running previous value instead closes
        # that: [0us, 0us, 1us] -> [0us, 1us, 2us]. Untied cases never enter
        # the `dt <= prev_dt` branch, so `updates` stays empty and the file
        # is returned unchanged (no-op). The accumulated shift is bounded by
        # the length of a same-instant run in whole microseconds -- still far
        # below the 1s tolerance (C18) and this data's second-level
        # timestamp granularity.
        prev_dt = None
        for idx, suffix, ocel_id, time_str in case_rows:
            dt = datetime.fromisoformat(time_str)
            if prev_dt is not None and dt <= prev_dt:
                dt = prev_dt + timedelta(microseconds=1)
                updates.append((suffix, ocel_id, dt.isoformat(sep=" ")))
            prev_dt = dt

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


def _unlink_temp_sqlite(path: str, attempts: int = 10, delay: float = 0.2) -> None:
    """Delete a temp SQLite file created by this module, tolerating Windows'
    file-locking semantics.

    On POSIX (Mac/Linux) unlinking a file while another handle is still open
    on it is harmless -- the OS keeps the inode alive until the last handle
    closes, so ``os.unlink`` always succeeds immediately. On Windows the same
    call raises ``PermissionError`` ([WinError 32]) if ANY handle, including
    one held by this same process, is still open -- notably a ``sqlite3``
    connection that OCPA's importer (``ocel2_import_factory.apply`` in
    ``load_ocpa_ocel`` below) may keep alive on the returned OCEL object and
    only release when that connection is garbage-collected. ``gc.collect()``
    forces those lingering connections closed via their ``__del__``; retrying
    briefly covers the rest (e.g. antivirus/indexer scans transiently holding
    the file open on Windows). If the file is still locked after all
    attempts, warn and leave it for the OS temp-dir cleanup rather than
    failing the whole evaluation over a leftover temp file.
    """
    for attempt in range(attempts):
        try:
            os.unlink(path)
            return
        except PermissionError:
            gc.collect()
            if attempt == attempts - 1:
                warnings.warn(
                    f"Could not delete temporary file {path!r} "
                    "(still locked by another handle); leaving it in place.")
                return
            time.sleep(delay)
        except FileNotFoundError:
            return


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
            _unlink_temp_sqlite(tie_broken_path)
        if norm_path != stripped_path:
            _unlink_temp_sqlite(norm_path)
        if stripped_path != path:
            _unlink_temp_sqlite(stripped_path)


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
