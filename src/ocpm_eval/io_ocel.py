"""I/O helper for reading OCEL 2.0 for the LABEL side, into the neutral model.

Reading uses OCPA's native OCEL 2.0 importer (ocpa.objects.log.importer.ocel2.sqlite),
the same library used for feature extraction, so both sides share the same reading path.
``load_ocpa_ocel`` is also imported by features_ocpa and rq3_pipeline so the OCEL object
can be loaded once and passed to both label and feature extraction.
"""
from ocpm_tasks.adapters import from_ocpa
from ocpm_tasks.model import ObjectCentricLog


def load_ocpa_ocel(schema, path: str):
    """Load an OCEL 2.0 SQLite log via OCPA's native importer with leading-type
    execution extraction (one process execution per CollaborationInstance)."""
    from ocpa.objects.log.importer.ocel2.sqlite import factory as ocel2_import_factory
    params = {"execution_extraction": "leading_type", "leading_type": schema.ot_ci}
    try:
        return ocel2_import_factory.apply(path, parameters=params)
    except TypeError:
        # Older OCPA signature without parameters: import with default extraction.
        # NOTE: default "connected components" may merge instances sharing a Participant;
        # verify the partitioning (the alignment oracle in features_ocpa will flag it).
        return ocel2_import_factory.apply(path)


def read_ocel2_labels(path: str, schema,
                      ocpa_ocel=None) -> ObjectCentricLog:
    """Build the neutral ObjectCentricLog from an OCEL 2.0 SQLite file.
    If ``ocpa_ocel`` is provided (already loaded by the caller) it is reused
    directly, avoiding a second read of the same file."""
    ocel = ocpa_ocel if ocpa_ocel is not None else load_ocpa_ocel(schema, path)
    return from_ocpa(ocel, schema)
