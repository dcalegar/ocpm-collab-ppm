"""
RQ4 — structural measures and expressiveness (descriptive only; not a cost claim).

Structural measures per OCEL 2.0 SQLite log, read directly with the stdlib sqlite3
(no pm4py): number of objects per type, |E2O|, |O2O|, |E|, |O|, file size, and the
structural ratio (|O| + |E2O| + |O2O|) / |E|. The expressiveness/preprocessing matrix
is a fixed qualitative table (evaluation Table tab:expressiveness).
"""
import os
import sqlite3
from typing import List, Optional, Dict
import pandas as pd

from .config import ExperimentConfig, LogSpec


EXPRESSIVENESS: List[Dict[str, str]] = [
    {"operation": "Participant-local view", "collaborative_xes": "Attribute filtering",
     "rich_ocel": "Direct navigation", "ocel_relations": "local, executed_by"},
    {"operation": "Message lifecycle", "collaborative_xes": "Correlation required",
     "rich_ocel": "Persistent object", "ocel_relations": "send, receive, from, to"},
    {"operation": "Cross-case participant view", "collaborative_xes": "Additional indexing",
     "rich_ocel": "Shared object", "ocel_relations": "executed_by"},
]


def structural_measures(spec: LogSpec, cfg: ExperimentConfig) -> dict:
    con = sqlite3.connect(spec.ocel_path)
    c = con.cursor()

    def scalar(q):
        return c.execute(q).fetchone()[0]

    n_events = scalar("SELECT COUNT(*) FROM event")
    n_objects = scalar("SELECT COUNT(*) FROM object")
    n_e2o = scalar("SELECT COUNT(*) FROM event_object")
    n_o2o = scalar("SELECT COUNT(*) FROM object_object")
    by_type = dict(c.execute("SELECT ocel_type, COUNT(*) FROM object GROUP BY ocel_type"))
    con.close()

    rec = {
        "log": spec.name,
        "events": int(n_events),
        "objects": int(n_objects),
        "E2O": int(n_e2o),
        "O2O": int(n_o2o),
        "ocel_size_bytes": (os.path.getsize(spec.ocel_path)
                            if os.path.exists(spec.ocel_path) else None),
        "struct_ratio": ((n_objects + n_e2o + n_o2o) / n_events) if n_events else None,
    }
    for ot, cnt in by_type.items():
        rec[f"obj_{ot}"] = int(cnt)
    return rec


def run_rq4_structure(cfg: Optional[ExperimentConfig] = None) -> pd.DataFrame:
    cfg = cfg or ExperimentConfig()
    os.makedirs(cfg.out_dir, exist_ok=True)
    rows = []
    for spec in cfg.logs:
        print(f"\n===== RQ4 STRUCTURE LOG: {spec.name} =====")
        try:
            rows.append(structural_measures(spec, cfg))
        except Exception as ex:                       # noqa: BLE001
            print(f"[{spec.name}] ERROR: {ex}")
            rows.append({"log": spec.name, "error": str(ex)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg.out_dir, "rq4_structure.csv"), index=False)
    pd.DataFrame(EXPRESSIVENESS).to_csv(
        os.path.join(cfg.out_dir, "rq4_expressiveness.csv"), index=False)
    print("[ok] wrote rq4_structure.csv and rq4_expressiveness.csv")
    return df
