"""
RQ2 — label fidelity of the reformulated tasks (predictor-independent).

For the 14 single-case-counterpart tasks the check is label EQUIVALENCE between a
reference reader on the original collaborative log (R1) and the object-centric reader
(R2). For X-PaL and X-MSt the check is INTERNAL CONSISTENCY of the object-centric
target.

The R1 reader belongs to the converter (a separate tool); this stage therefore takes
an optional ``r1_logs`` mapping {log_name: ObjectCentricLog} built by the converter
from R1. When it is not provided, only the consistency checks (X-PaL, X-MSt) are
produced, and equivalence rows are marked as requiring the converter's R1 reader.
"""
import os
from typing import Optional, Dict, List
import pandas as pd

from ocpm_tasks.catalog import TASKS, EQUIVALENCE_TASKS, CONSISTENCY_TASKS
from ocpm_tasks import labels as TL
from ocpm_tasks import fidelity as FID
from ocpm_tasks.model import ObjectCentricLog
from .config import ExperimentConfig, LogSpec
from .io_ocel import read_ocel2_labels


def _params_for(task, ocel_log, cfg):
    """Relevant parameter values to evaluate for a parameterized task (per log)."""
    if task.param == "activity":
        acts = sorted({e.activity for ex in ocel_log for e in ex.events if e.is_send})
        return acts or [None]
    if task.param == "participant":
        ps = sorted({e.actor for ex in ocel_log for e in ex.events if e.actor})
        return ps or [None]
    return [None]


def run_one_log(spec: LogSpec, cfg: ExperimentConfig,
                r1_log: Optional[ObjectCentricLog]) -> List[dict]:
    r2_log = read_ocel2_labels(spec.ocel_path, cfg.schema)
    ctx2 = TL.build_context(r2_log, cfg.bottom)
    ctx1 = TL.build_context(r1_log, cfg.bottom) if r1_log is not None else None
    out: List[dict] = []

    for key in EQUIVALENCE_TASKS:
        task = TASKS[key]
        for param in _params_for(task, r2_log, cfg):
            obj_rows = TL.compute_label_rows(r2_log, task, param, ctx2,
                                             drop_bottom=False)
            if ctx1 is None:
                out.append({"log": spec.name, "task": key, "param": param,
                            "check": "equivalence", "eval_labels": len(obj_rows),
                            "note": "needs converter R1 reader"})
                continue
            ref_rows = TL.compute_label_rows(r1_log, task, param, ctx1,
                                             drop_bottom=False)
            res = FID.compare_equivalence(ref_rows, obj_rows, task,
                                          cfg.bottom, cfg.numeric_tol)
            res.update({"log": spec.name, "param": param})
            out.append(res)

    for key in CONSISTENCY_TASKS:
        task = TASKS[key]
        for param in _params_for(task, r2_log, cfg):
            rows = TL.compute_label_rows(r2_log, task, param, ctx2, drop_bottom=False)
            res = FID.check_consistency(rows, task, cfg.bottom)
            res.update({"log": spec.name, "param": param})
            out.append(res)
    return out


def run_rq2(cfg: Optional[ExperimentConfig] = None,
            r1_logs: Optional[Dict[str, ObjectCentricLog]] = None) -> pd.DataFrame:
    cfg = cfg or ExperimentConfig()
    os.makedirs(cfg.out_dir, exist_ok=True)
    out: List[dict] = []
    for spec in cfg.logs:
        print(f"\n===== RQ2 LOG: {spec.name} =====")
        r1 = (r1_logs or {}).get(spec.name)
        try:
            out.extend(run_one_log(spec, cfg, r1))
        except Exception as ex:                       # noqa: BLE001
            print(f"[{spec.name}] ERROR: {ex}")
            out.append({"log": spec.name, "error": str(ex)})
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(cfg.out_dir, "rq2_fidelity.csv"), index=False)
    print("[ok] wrote rq2_fidelity.csv")
    return df
