"""I/O helper for reading OCEL 2.0 for the LABEL side, into the neutral model.

Reading is done with the stdlib SQLite reader (``from_ocel2_sqlite``), not pm4py:
OCPA pins pm4py==2.2.32, which cannot read OCEL 2.0. The features side reads the same
OCEL 2.0 SQLite file through OCPA's native importer (see features_ocpa.py)."""
from ocpm_tasks.adapters import from_ocel2_sqlite
from ocpm_tasks.model import ObjectCentricLog


def read_ocel2_labels(path: str, schema) -> ObjectCentricLog:
    return from_ocel2_sqlite(path, schema)
