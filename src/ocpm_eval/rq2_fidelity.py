"""
RQ2 — label fidelity of the reformulated tasks (predictor-independent).

For the 14 tasks the check is label EQUIVALENCE between:
  R1 — Θ_τ^L: labels computed directly from the collaborative XES log using the
       source accessors act/time_L/part/elem (the source log's total event
       order, prec_L). No intermediate ObjectCentricLog is built; each task
       definition is evaluated on the sorted per-case event sequence from the
       XES file.
  R2 — Θ_τ: labels computed from the OCEL 2.0 SQLite via the OCEL accessors
       evtype/time/pa/snd/rcv/msg/Msgs/pos, using labels.compute_label_rows.
The mapping preserves prec_L and label semantics by construction, so
Θ_τ = Θ_τ^L; empirical agreement=1.0 is expected.
"""
import os
from datetime import datetime
from typing import Optional, List, Dict, Tuple
import pandas as pd

from ocpm_tasks.catalog import TASKS, EQUIVALENCE_TASKS
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
    from mapping.collab_xes_to_ocel import _correct_utc_timestamps
    log = pm4py.read_xes(path)
    df = pm4py.convert_to_dataframe(log)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    df = df.reset_index(drop=True)
    # B11: pm4py's ISO8601 parser mislabels a non-zero UTC offset without
    # shifting the clock (see collab_xes_to_ocel._correct_utc_timestamps for
    # the root cause), so raw pm4py timestamps are wrong whenever a source
    # log's offset is non-zero. A log with ONE fixed offset throughout still
    # gives correct label VALUES here, because every timestamp carries the
    # same additive error, which cancels out in every difference the 14
    # tasks compute (R1 never reads an absolute timestamp, only diffs) --
    # this covers the 4 baseline logs and ToyCollab. It does NOT cancel when
    # a single collaboration case spans more than one offset, e.g. a DST
    # transition: BPI Challenge 2013 mixes +01:00/+02:00 within 254 of its
    # 7,554 cases, and the resulting hour-scale errors do not cancel in
    # diffs that straddle the transition. Confirmed empirically before this
    # fix: thousands of false RQ2 mismatches (NV-PrT/NV-PaT/NV-TNE/NV-TNM)
    # against R2, which already reads the corrected copy. Re-deriving R1's
    # timestamps the same way R2's are derived keeps both readers on the
    # identical clock regardless of a log's offset structure.
    df["time:timestamp"] = _correct_utc_timestamps(path, len(df))

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


def _src_label(key: str, evs: List[dict], i: int, param, bottom: str) -> object:
    """Θ_τ^L for one cut point i in one case, computed on the source total event order (prec_L)."""
    n = len(evs)

    if key == "NE-NEPr":
        return evs[i + 1]["activity"] if i + 1 < n else bottom

    if key == "NE-NPaA":
        return evs[i + 1]["participant"] if i + 1 < n else bottom

    if key == "NE-NEPa":
        # Pair (evtype, pa) of the next event -- must match labels._NE_NEPa's
        # representation (B13: a concatenated string is not injective).
        if i + 1 < n:
            return (evs[i + 1]["activity"], evs[i + 1]["participant"])
        return bottom

    if key == "NE-NPaM":
        # part(e_j) via 'from' (send) / 'to' (receive) O2O = collab:participant
        # of the message event in direction `param` ("send" or "receive",
        # default "send"); well-formedness (i)/(ii) makes this equal from(e)
        # for a send and to(e) for a receive.
        want = _ELEM_SEND if (param or "send") == "send" else _ELEM_RECV
        for j in range(i + 1, n):
            if evs[j]["elem"] == want:
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
        # Direction `param` ("send" or "receive", default "send").
        want = _ELEM_SEND if (param or "send") == "send" else _ELEM_RECV
        for j in range(i + 1, n):
            if evs[j]["elem"] == want:
                return (evs[j]["timestamp"] - evs[i]["timestamp"]).total_seconds()
        return bottom

    if key == "NV-NMPr":
        # Delgado et al. 2025, Table 2: "(send/receive)" -- both directions.
        return sum(1 for j in range(i + 1, n)
                   if evs[j]["elem"] in (_ELEM_SEND, _ELEM_RECV))

    if key == "NV-NMPa":
        # part(e_j) = collab:participant, for either direction (see NV-NMPr).
        return sum(1 for j in range(i + 1, n)
                   if evs[j]["elem"] in (_ELEM_SEND, _ELEM_RECV)
                   and evs[j]["participant"] == param)

    if key == "OB-P":
        return any(evs[j]["participant"] == param for j in range(i + 1, n))

    if key == "OB-M":
        # Delgado et al. 2025, Table 2: "sent/received" -- both directions.
        return any(evs[j]["elem"] in (_ELEM_SEND, _ELEM_RECV) and evs[j]["activity"] == param
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
        for i in range(n - 1):
            y = _src_label(task.key, evs, i, param, bottom)
            rows.append((case_id, f"{case_id}::{i}", i + 1, y))
    return rows


# ---------------------------------------------------------------------------
# RQ2 orchestration
# ---------------------------------------------------------------------------

# NE-NPaM and NV-TNM are parameterized by message direction d in {send, receive};
# the evaluation's RQ3 pipeline only instantiates send (matching Predict-Collab,
# a deliberate empirical scope decision), but RQ2's label-equivalence check
# tests both directions, since neither should go unverified.
_DIRECTION_TASKS = {"NE-NPaM", "NV-TNM"}


def _params_for(task, ocel_log, cfg):
    """Relevant parameter values for a parameterized task (resolved from R2 log)."""
    if task.key in _DIRECTION_TASKS:
        return ["send", "receive"]
    if task.param == "activity":
        # OB-M quantifies hat{a} over the FULL activity alphabet A_L, not only
        # message activities: a non-message hat{a} is a well-defined parameter
        # value for which Theta_OBM is always False (no message ever has that
        # evtype). Sweeping the full alphabet (E28) tests that boundary
        # instead of silently excluding it from RQ2's coverage.
        acts = sorted({e.activity for ex in ocel_log for e in ex.events})
        return acts or [None]
    if task.param == "participant":
        # P_L = part(E_L) u ran(from) u ran(to): a participant that is only
        # ever a message counterparty (never an event's own actor) is still in
        # P_L and is a valid parameter value for NE-NMPa/NV-PaT/NV-NMPa/OB-P.
        # Sourcing the domain from e.actor alone omits such endpoint-only
        # participants (E28); union in msg_from/msg_to.
        ps = {e.actor for ex in ocel_log for e in ex.events if e.actor}
        ps |= {e.msg_from for ex in ocel_log for e in ex.events if e.msg_from}
        ps |= {e.msg_to for ex in ocel_log for e in ex.events if e.msg_to}
        return sorted(ps) or [None]
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
    return out


def run_rq2(cfg: Optional[ExperimentConfig] = None,
           out_name: str = "rq2_fidelity.csv") -> pd.DataFrame:
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
    df.to_csv(os.path.join(cfg.out_dir, out_name), index=False)
    print(f"[ok] wrote {out_name}")
    return df
