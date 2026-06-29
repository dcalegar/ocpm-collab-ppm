"""
Native object-centric feature extraction with OCPA over OCEL 2.0. Stage used by RQ3.

Follows the official OCPA usage (github.com/ocpm/ocpa, ocpa.readthedocs.io):
  * OCEL 2.0 is imported NATIVELY (no OCEL 1.0 anywhere):
        from ocpa.objects.log.importer.ocel2.sqlite import factory
        ocel = factory.apply(path)
  * process executions must be one per CollaborationInstance: with the default
    "connected components" the shared Participant objects would merge all instances,
    so leading-type extraction is required. The OCEL-1.0 jsonocel importer documents
    a ``parameters={"execution_extraction":"leading_type","leading_type":...}`` dict;
    we pass the same to the OCEL 2.0 importer. VERIFY this is honoured by the ocel2
    importer in your OCPA build; if not, configure the extraction technique accordingly
    (the alignment oracle below will catch a wrong partitioning).
  * predictive_monitoring.apply(ocel, event_based_features, []) -> Feature_Storage;
  * tabular.construct_table(feature_storage) -> one row per event; feature_set[0]
    (remaining time) is the NV-PrT target and the alignment oracle.

Row<->event alignment uses the documented Feature_Storage structure
(feature_graphs -> nodes -> node.event_id), and labels are joined by event_id; the
oracle (OCPA remaining time == NV-PrT) validates it and also detects an event-id
mismatch. VERIFY: that node.event_id matches the OCEL event ids.
"""
import os
from typing import List, Dict
import pandas as pd

from ocpm_tasks.model import ObjectCentricLog
from .io_ocel import load_ocpa_ocel  # noqa: F401  (re-exported for callers)


def build_feature_set(ocel, schema):
    """Object-centric, PAST-RELATIVE event features only (prefix-respecting), so a
    per-event row encodes the observable prefix at its cut point without leaking the
    future: elapsed time, counts of previously-seen objects, preceding activities. No
    execution-based features (they would leak). feature_set[0] = remaining time is the
    NV-PrT target and the alignment oracle (excluded from model features by the
    caller). This is the tabular encoding (one row per cut point); OCPA's sequential /
    graph encodings are alternatives the same features support."""
    from ocpa.algo.predictive_monitoring import factory as predictive_monitoring
    activities = list(set(ocel.log.log["event_activity"].tolist()))
    return [
        (predictive_monitoring.EVENT_REMAINING_TIME, ()),     # [0]: target + oracle
        (predictive_monitoring.EVENT_ELAPSED_TIME, ()),
        (predictive_monitoring.EVENT_PREVIOUS_TYPE_COUNT, (schema.ot_participant,)),
        (predictive_monitoring.EVENT_PREVIOUS_TYPE_COUNT, (schema.ot_message,)),
    ] + [(predictive_monitoring.EVENT_PRECEDING_ACTIVITIES, (a,)) for a in activities]


def _row_event_ids(feature_storage) -> List[str]:
    """Event id per table row, in construct_table order (feature_graphs -> nodes)."""
    return [str(node.event_id)
            for fg in feature_storage.feature_graphs for node in fg.nodes]


def _prt_by_event_id(log: ObjectCentricLog) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for ex in log:
        end = ex.events[-1].timestamp
        for e in ex.events:
            out[str(e.event_id)] = (end - e.timestamp).total_seconds()
    return out


def _case_by_event_id(log: ObjectCentricLog) -> Dict[str, str]:
    return {str(e.event_id): ex.case_id for ex in log for e in ex.events}


def extract_feature_table(name: str, schema, ocel2_sqlite_path: str,
                          ocel_log: ObjectCentricLog,
                          ocpa_ocel=None) -> Dict[str, object]:
    from ocpa.algo.predictive_monitoring import factory as predictive_monitoring
    from ocpa.algo.predictive_monitoring import tabular

    ocel = ocpa_ocel if ocpa_ocel is not None else load_ocpa_ocel(schema, ocel2_sqlite_path)
    feature_set = build_feature_set(ocel, schema)
    feature_storage = predictive_monitoring.apply(ocel, feature_set, [])
    table = tabular.construct_table(feature_storage).reset_index(drop=True)

    row_ids = _row_event_ids(feature_storage)
    if len(row_ids) != len(table):
        raise RuntimeError(
            f"[{name}] size mismatch: {len(row_ids)} nodes vs {len(table)} rows.")
    table["event_id"] = row_ids
    case_by_eid = _case_by_event_id(ocel_log)
    table["case_id"] = [case_by_eid.get(e) for e in row_ids]

    prt = _prt_by_event_id(ocel_log)
    rem_col = feature_set[0]
    bad = sum(1 for eid, r in zip(row_ids, table[rem_col].astype(float))
              if prt.get(eid) is None or abs(prt[eid] - r) > 1.0)
    if bad:
        raise RuntimeError(
            f"[{name}] alignment oracle FAILED on {bad}/{len(table)} rows "
            "(OCPA remaining time != PrT label, or event ids/partitioning differ). "
            "Verify leading-type extraction and that node.event_id matches the OCEL.")

    feature_cols = [c for c in table.columns
                    if c not in (rem_col, "event_id", "case_id")]
    return {"table": table, "feature_cols": feature_cols}
