"""I/O helper for reading OCEL 2.0 for the LABEL side, into the neutral model.

Reading uses OCPA's native OCEL 2.0 importer (ocpa.objects.log.importer.ocel2.sqlite),
the same library used for feature extraction, so both sides share the same reading path.
``load_ocpa_ocel`` is also imported by features_ocpa and rq3_pipeline so the OCEL object
can be loaded once and passed to both label and feature extraction.
"""
import os
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
# which is what makes feature extraction stall. Strip it before handing
# the file to OCPA; this is an OCPA-specific import workaround, not a
# change to the conceptual model.
_E2O_PARTICIPANT_QUALIFIER = "participant"


def _strip_participant_e2o(path: str) -> str:
    """Return a temp path to a copy of the SQLite file with the direct
    'participant' E2O rows removed (see _E2O_PARTICIPANT_QUALIFIER above).
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
        con.execute("DELETE FROM event_object WHERE ocel_qualifier = ?",
                    (_E2O_PARTICIPANT_QUALIFIER,))
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
    params = {"execution_extraction": "leading_type", "leading_type": schema.ot_cc}
    try:
        try:
            return ocel2_import_factory.apply(norm_path, parameters=params)
        except TypeError:
            # Older OCPA signature without parameters: import with default extraction.
            # NOTE: default "connected components" may merge instances sharing a Participant;
            # verify the partitioning (the alignment oracle in features_ocpa will flag it).
            return ocel2_import_factory.apply(norm_path)
    finally:
        if norm_path != stripped_path:
            os.unlink(norm_path)
        if stripped_path != path:
            os.unlink(stripped_path)


def read_ocel2_labels(path: str, schema,
                      ocpa_ocel=None) -> ObjectCentricLog:
    """Build the neutral ObjectCentricLog from an OCEL 2.0 SQLite file.
    If ``ocpa_ocel`` is provided it is used via from_ocpa; otherwise the
    stdlib sqlite3 reader is used to avoid OCPA's itertuples/getattr path
    which breaks on attribute names containing non-identifier characters
    (e.g. 'org:group' → 'event_org:group')."""
    if ocpa_ocel is not None:
        return from_ocpa(ocpa_ocel, schema)
    return from_ocel2_sqlite(path, schema)
