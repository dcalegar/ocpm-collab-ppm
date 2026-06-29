"""
RQ2 — label fidelity of the reformulated tasks (predictor-independent).

For the 14 single-case-counterpart tasks the check is label EQUIVALENCE between:
  R1 — Θ_τ^L: labels computed directly from the collaborative XES log using the
       source accessors act/time_L/part/elem (Definition 1 of the appendix). No
       intermediate ObjectCentricLog is built; each task definition is evaluated
       on the sorted per-case event sequence from the XES file.
  R2 — Θ_τ: labels computed from the OCEL 2.0 SQLite via the OCEL accessors
       evtype/time/pa/snd/rcv/msg/Msgs/pos, using labels.compute_label_rows.
Proposition P2 guarantees Θ_τ = Θ_τ^L; empirical agreement=1.0 is expected.
For X-PaL and X-MSt the check is INTERNAL CONSISTENCY of R2 only (no R1
counterpart exists per the remark following Proposition P2).
"""
import os
from datetime import datetime
from typing import Optional, List, Dict, Tuple
import pandas as pd

from ocpm_tasks.catalog import TASKS, EQUIVALENCE_TASKS, CONSISTENCY_TASKS
from ocpm_tasks import labels as TL
from ocpm_tasks import fidelity as FID
from ocpm_tasks.fidelity import Row
from .config import ExperimentConfig, LogSpec
from .io_ocel import read_ocel2_labels

_ELEM_SEND = "SendTask"
_ELEM_RECV = "ReceiveTask"


# ---------------------------------------------------------------------------
# Θ_τ^L — source-level label computation (R1)
# ---------------------------------------------------------------------------

def _clean(v) -> Optional[str]:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    return s if s else None


def _read_xes_cases(path: str) -> Dict[str, List[dict]]:
    """Read XES and return per-case sorted event dicts using source accessors."""
    import pm4py
    log = pm4py.read_xes(path)
    df = pm4py.convert_to_dataframe(log)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    cases: Dict[str, List[dict]] = {}
    for case_id, grp in df.groupby("case:concept:name", sort=False):
        grp = grp.sort_values(by=["time:timestamp"], kind="mergesort").reset_index(drop=True)
        evs = []
        for _, row in grp.iterrows():
            ts = pd.to_datetime(row["time:timestamp"])
            if hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()
            evs.append({
                "activity":    str(row["concept:name"]),
                "timestamp":   ts,
                "elem":        _clean(row.get("collab:elemType")) or "task",
                "participant": _clean(row.get("collab:participant")) or "",
                "from":        _clean(row.get("collab:fromParticipant")),
                "to":          _clean(row.get("collab:toParticipant")),
            })
        cases[str(case_id)] = evs
    return cases


def _correlate(evs: List[dict]) -> Dict[int, int]:
    """M4 heuristic ρ for one case: send_pos -> recv_pos."""
    rho: Dict[int, int] = {}
    matched: set = set()
    recvs = [(i, e) for i, e in enumerate(evs) if e["elem"] == _ELEM_RECV]
    for i, ev in enumerate(evs):
        if ev["elem"] != _ELEM_SEND:
            continue
        sender, receiver = ev["participant"], ev["to"]
        best_pos, best_ts = None, None
        for j, rv in recvs:
            if j in matched:
                continue
            if rv["timestamp"] < ev["timestamp"]:
                continue
            if rv["from"] is not None and sender and rv["from"] != sender:
                continue
            if rv["to"] is not None and receiver is not None and rv["to"] != receiver:
                continue
            if best_pos is None or (rv["timestamp"], j) < (best_ts, best_pos):
                best_pos, best_ts = j, rv["timestamp"]
        if best_pos is not None:
            rho[i] = best_pos
            matched.add(best_pos)
    return rho


def _src_label(key: str, evs: List[dict], i: int, param, bottom: str,
               rho: Dict[int, int]) -> object:
    """Θ_τ^L for one cut point i in one case (Definition 1 / Proposition P2)."""
    n = len(evs)

    if key == "NE-NEPr":
        return evs[i + 1]["activity"] if i + 1 < n else bottom

    if key == "NE-NPaA":
        return evs[i + 1]["participant"] if i + 1 < n else bottom

    if key == "NE-NEPa":
        if i + 1 < n:
            return f"{evs[i + 1]['activity']}||{evs[i + 1]['participant']}"
        return bottom

    if key == "NE-NPaM":
        # part(e_j) via 'from' O2O = collab:participant of the send event
        for j in range(i + 1, n):
            if evs[j]["elem"] == _ELEM_SEND:
                return evs[j]["participant"]
        return bottom

    if key == "NE-NMPr":
        for j in range(i + 1, n):
            if evs[j]["elem"] in (_ELEM_SEND, _ELEM_RECV):
                return evs[j]["activity"]
        return bottom

    if key == "NE-NMPa":
        for j in range(i + 1, n):
            if evs[j]["elem"] in (_ELEM_SEND, _ELEM_RECV) and evs[j]["participant"] == param:
                return evs[j]["activity"]
        return bottom

    if key == "NV-PrT":
        return (evs[-1]["timestamp"] - evs[i]["timestamp"]).total_seconds()

    if key == "NV-PaT":
        zs = [j for j in range(n) if evs[j]["participant"] == param]
        if zs:
            z = max(zs)
            return (evs[z]["timestamp"] - evs[i]["timestamp"]).total_seconds() if z > i else 0.0
        return 0.0

    if key == "NV-TNE":
        if i + 1 < n:
            return (evs[i + 1]["timestamp"] - evs[i]["timestamp"]).total_seconds()
        return bottom

    if key == "NV-TNM":
        for j in range(i + 1, n):
            if evs[j]["elem"] == _ELEM_SEND:
                return (evs[j]["timestamp"] - evs[i]["timestamp"]).total_seconds()
        return bottom

    if key == "NV-NMPr":
        return sum(1 for j in range(i + 1, n) if evs[j]["elem"] == _ELEM_SEND)

    if key == "NV-NMPa":
        # part(send_j) = collab:participant = msg_from in R2
        return sum(1 for j in range(i + 1, n)
                   if evs[j]["elem"] == _ELEM_SEND and evs[j]["participant"] == param)

    if key == "OB-P":
        return any(evs[j]["participant"] == param for j in range(i + 1, n))

    if key == "OB-M":
        return any(evs[j]["elem"] == _ELEM_SEND and evs[j]["activity"] == param
                   for j in range(i + 1, n))

    raise ValueError(f"No source-level Θ_τ^L implementation for task {key}")


def _compute_source_rows(xes_path: str, task, param, bottom: str) -> List[Row]:
    """Compute Θ_τ^L: source-level label rows from the collaborative XES log.

    Reads the XES directly using source accessors without building an intermediate
    ObjectCentricLog. Returns rows (case_id, event_id, k, label) aligned by
    (case_id, k) with the R2 rows from labels.compute_label_rows.
    """
    rows: List[Row] = []
    for case_id, evs in _read_xes_cases(xes_path).items():
        n = len(evs)
        rho: Dict[int, int] = {}   # only computed for tasks that need M4
        for i in range(n - 1):
            y = _src_label(task.key, evs, i, param, bottom, rho)
            rows.append((case_id, f"{case_id}::{i}", i + 1, y))
    return rows


# ---------------------------------------------------------------------------
# RQ2 orchestration
# ---------------------------------------------------------------------------

def _params_for(task, ocel_log, cfg):
    """Relevant parameter values for a parameterized task (resolved from R2 log)."""
    if task.param == "activity":
        acts = sorted({e.activity for ex in ocel_log for e in ex.events if e.is_send})
        return acts or [None]
    if task.param == "participant":
        ps = sorted({e.actor for ex in ocel_log for e in ex.events if e.actor})
        return ps or [None]
    return [None]


def run_one_log(spec: LogSpec, cfg: ExperimentConfig) -> List[dict]:
    r2_log = read_ocel2_labels(spec.ocel_path, cfg.schema)
    ctx2 = TL.build_context(r2_log, cfg.bottom)
    out: List[dict] = []

    for key in EQUIVALENCE_TASKS:
        task = TASKS[key]
        for param in _params_for(task, r2_log, cfg):
            r1_rows = _compute_source_rows(spec.xes_path, task, param, cfg.bottom)
            r2_rows = TL.compute_label_rows(r2_log, task, param, ctx2, drop_bottom=False)
            res = FID.compare_equivalence(r1_rows, r2_rows, task, cfg.bottom, cfg.numeric_tol)
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


def run_rq2(cfg: Optional[ExperimentConfig] = None) -> pd.DataFrame:
    cfg = cfg or ExperimentConfig()
    os.makedirs(cfg.out_dir, exist_ok=True)
    out: List[dict] = []
    for spec in cfg.logs:
        print(f"\n===== RQ2 LOG: {spec.name} =====")
        try:
            out.extend(run_one_log(spec, cfg))
        except Exception as ex:                       # noqa: BLE001
            print(f"[{spec.name}] ERROR: {ex}")
            out.append({"log": spec.name, "error": str(ex)})
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(cfg.out_dir, "rq2_fidelity.csv"), index=False)
    print("[ok] wrote rq2_fidelity.csv")
    return df
