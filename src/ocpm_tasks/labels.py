"""
Label (target) functions ℓ for the reformulated tasks, over the neutral
object-centric model. Pure functions: no ML, no I/O, no OCEL library.

Cut-point semantics. A target is evaluated at a *cut point* in a collaboration
instance. The cut point is an index ``i`` (0-based) in the instance's global trace
(its events ordered by (timestamp, event_id)); the prefix hd^k has k=i+1 and the
target is a function of the events AFTER the cut. This linear global trace is the
basis the paper uses to DEFINE the targets (so they match the case-centric baseline)
and is independent of how the observable prefix is ENCODED: with OCPA the observable
prefix is the object-centric execution graph and the features are graph-based, not a
linear-prefix encoding. These functions only produce the ground-truth value at each
cut point; pairing it with a representation's per-event features is the caller's job.

A ``LabelContext`` is built once per log; it carries the BOTTOM symbol and the
cross-case index needed by X-PaL (per participant, the time spans of its local cases
across all collaboration instances). Every label function has the uniform signature
``fn(ctx, ex, i, param) -> value | BOTTOM``. Rows are emitted for k in [1, n-1]
(i in [0, n-2]); the final, complete execution has no prefix to predict from.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable

from .model import ObjectCentricLog, Execution
from .catalog import Task, TASKS

BOTTOM = "__BOTTOM__"


@dataclass
class LabelContext:
    log: ObjectCentricLog
    bottom: str = BOTTOM
    # X-PaL: participant -> list of (case_id, first_ts, last_ts) over all CIs.
    participant_spans: Dict[str, List[Tuple[str, object, object]]] = \
        field(default_factory=dict)


def build_context(log: ObjectCentricLog, bottom: str = BOTTOM) -> LabelContext:
    spans: Dict[str, List[Tuple[str, object, object]]] = {}
    for ex in log:
        per_p: Dict[str, List] = {}
        for e in ex.events:
            if e.actor:
                per_p.setdefault(e.actor, []).append(e.timestamp)
        for p, ts in per_p.items():
            spans.setdefault(p, []).append((ex.case_id, min(ts), max(ts)))
    return LabelContext(log=log, bottom=bottom, participant_spans=spans)


# --- 14 single-case-counterpart tasks ----------------------------------------
def _NE_NEPr(ctx, ex, i, p):
    return ex.events[i + 1].activity if i + 1 < ex.n else ctx.bottom

def _NE_NPaA(ctx, ex, i, p):
    return ex.events[i + 1].actor if i + 1 < ex.n else ctx.bottom

def _NE_NEPa(ctx, ex, i, p):
    if i + 1 < ex.n:
        e = ex.events[i + 1]
        return f"{e.activity}||{e.actor}"
    return ctx.bottom

def _NE_NPaM(ctx, ex, i, p):
    for j in range(i + 1, ex.n):
        if ex.events[j].is_send:
            return ex.events[j].msg_from or ex.events[j].actor
    return ctx.bottom

def _NE_NMPa(ctx, ex, i, p):
    if p is None:
        raise ValueError("NE-NMPa requires param = participant.")
    for j in range(i + 1, ex.n):
        if ex.events[j].is_msg and ex.events[j].actor == p:
            return ex.events[j].activity
    return ctx.bottom

def _NE_NMPr(ctx, ex, i, p):
    for j in range(i + 1, ex.n):
        if ex.events[j].is_msg:
            return ex.events[j].activity
    return ctx.bottom

def _NV_PrT(ctx, ex, i, p):
    return (ex.events[-1].timestamp - ex.events[i].timestamp).total_seconds()

def _NV_PaT(ctx, ex, i, p):
    if p is None:
        raise ValueError("NV-PaT requires param = participant.")
    zs = [j for j in range(ex.n) if ex.events[j].actor == p]
    if zs:
        z = max(zs)
        return ((ex.events[z].timestamp - ex.events[i].timestamp).total_seconds()
                if z > i else 0.0)
    return 0.0

def _NV_TNE(ctx, ex, i, p):
    if i + 1 < ex.n:
        return (ex.events[i + 1].timestamp - ex.events[i].timestamp).total_seconds()
    return ctx.bottom

def _NV_TNM(ctx, ex, i, p):
    for j in range(i + 1, ex.n):
        if ex.events[j].is_send:
            return (ex.events[j].timestamp - ex.events[i].timestamp).total_seconds()
    return ctx.bottom

def _NV_NMPr(ctx, ex, i, p):
    return sum(1 for j in range(i + 1, ex.n) if ex.events[j].is_send)

def _NV_NMPa(ctx, ex, i, p):
    if p is None:
        raise ValueError("NV-NMPa requires param = participant.")
    return sum(1 for j in range(i + 1, ex.n)
               if ex.events[j].is_send
               and (ex.events[j].msg_from or ex.events[j].actor) == p)

def _OB_P(ctx, ex, i, p):
    if p is None:
        raise ValueError("OB-P requires param = participant.")
    return any(ex.events[j].actor == p for j in range(i + 1, ex.n))

def _OB_M(ctx, ex, i, p):
    if p is None:
        raise ValueError("OB-M requires param = send-activity label (a_hat).")
    return any(ex.events[j].is_send and ex.events[j].activity == p
               for j in range(i + 1, ex.n))


# --- object-enabled extensions ------------------------------------------------
def _X_MSt(ctx, ex, i, p):
    """Synchronization (pooling) time of the next message: t(receive) - t(send).
    BOTTOM if there is no next send or the message has no receive (Def. X-MSt)."""
    for j in range(i + 1, ex.n):
        if ex.events[j].is_send:
            s = ex.events[j]
            if s.msg_id is None:
                return ctx.bottom
            for r in ex.events:
                if r.is_receive and r.msg_id == s.msg_id:
                    return (r.timestamp - s.timestamp).total_seconds()
            return ctx.bottom
    return ctx.bottom

def _X_PaL(ctx, ex, i, p):
    """Cross-case participant load (Def. X-PaL): number of OTHER collaboration
    instances in which participant p is active at the time p next acts in this case;
    0 if p does not act again. Uses the cross-case index in the context."""
    if p is None:
        raise ValueError("X-PaL requires param = participant.")
    j = next((k for k in range(i + 1, ex.n) if ex.events[k].actor == p), None)
    if j is None:
        return 0
    t = ex.events[j].timestamp
    spans = ctx.participant_spans.get(p, [])
    return sum(1 for (cid, first, last) in spans
               if cid != ex.case_id and first <= t <= last)


LABEL_FNS: Dict[str, Callable] = {
    "NE-NEPr": _NE_NEPr, "NE-NPaA": _NE_NPaA, "NE-NEPa": _NE_NEPa,
    "NE-NPaM": _NE_NPaM, "NE-NMPa": _NE_NMPa, "NE-NMPr": _NE_NMPr,
    "NV-PrT": _NV_PrT, "NV-PaT": _NV_PaT, "NV-TNE": _NV_TNE,
    "NV-TNM": _NV_TNM, "NV-NMPr": _NV_NMPr, "NV-NMPa": _NV_NMPa,
    "OB-P": _OB_P, "OB-M": _OB_M,
    "X-PaL": _X_PaL, "X-MSt": _X_MSt,
}


def label_value(task: Task, ctx: LabelContext, ex: Execution, i: int, param):
    try:
        fn = LABEL_FNS[task.key]
    except KeyError:
        raise ValueError(f"No label function for task {task.key}")
    return fn(ctx, ex, i, param)


def compute_label_rows(log: ObjectCentricLog, task: Task, param=None,
                       ctx: Optional[LabelContext] = None,
                       drop_bottom: bool = True
                       ) -> List[Tuple[str, str, int, object]]:
    """Rows (case_id, event_id, k, y). For categorical/numeric tasks the BOTTOM
    (undefined) rows are dropped by default."""
    ctx = ctx or build_context(log)
    rows = []
    for ex in log:
        for i in range(ex.n - 1):
            y = label_value(task, ctx, ex, i, param)
            if drop_bottom and task.kind in ("categorical", "numeric") \
                    and y == ctx.bottom:
                continue
            rows.append((ex.case_id, ex.events[i].event_id, i + 1, y))
    return rows
