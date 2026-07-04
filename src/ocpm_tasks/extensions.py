"""
Object-enabled EXTENSION tasks (X-Inf, X-MSt) — targets with "no single-case
counterpart" that tasks.tex / discussion.tex (Sect. "object-enabled prediction
targets") outline. They are deliberately kept OUT of the main catalog
(``catalog.TASKS`` / ``EQUIVALENCE_TASKS`` / ``RQ3_SUBSET``) so they are never
mixed into the reformulation evaluation (RQ2 label-equivalence, RQ3 subset/full).
They are exercised only by a dedicated toy-log test (``tests/test_extensions_toy.py``).

Two extensions with intentionally different correlation requirements:

X-Inf — in-flight message backlog (CollaborationCase anchor, count regression).
    Needs NO send/receive correlation: the in-flight level after an event is a
    pure aggregate ``#send observations - #receive observations`` over the Message
    objects (M4). This is object-enabled because the receive observations belong to
    OTHER participants' projections, so no single participant-local case trace sees
    them. Target at cut ``i``: the PEAK in-flight backlog over the remainder of the
    case (``max`` of the running backlog over events after the cut).

X-MSt — message synchronization time (Message anchor, time regression).
    PRESUPPOSES a send<->receive correspondence that the core mapping deliberately
    does NOT establish (M4). It is defined over an ENRICHED model in which each
    communication event carries a native correlation id (``Event.corr_id``), from
    which the pairing is recovered. Target at cut ``i``: the latency
    ``receive_time - send_time`` of the NEXT send after the cut. Returns BOTTOM when
    that send has no matching reception (an unmatched / still-in-flight send) or when
    no correlation id is available — i.e. the task is undefined without enrichment.

Both label functions share the uniform signature ``fn(ctx, ex, i, param)`` used in
``labels.py`` and reuse its ``LabelContext`` / BOTTOM sentinel.
"""
from typing import Dict, List, Optional, Tuple, Callable

from .model import ObjectCentricLog, Execution, Event
from .catalog import Task, CC, MESSAGE, COUNT, REG_TIME
from .labels import LabelContext, build_context


# --- task metadata (separate from catalog.TASKS on purpose) ------------------
X_INF = Task("X-Inf", CC,      COUNT,    "numeric", object_enabled=True)
X_MST = Task("X-MSt", MESSAGE, REG_TIME, "numeric", object_enabled=True)

EXT_TASKS: Dict[str, Task] = {"X-Inf": X_INF, "X-MSt": X_MST}


# --- helpers -----------------------------------------------------------------
def in_flight_trajectory(ex: Execution) -> List[int]:
    """Running in-flight backlog AFTER each event: +1 per send observation, -1 per
    receive observation (clamped at 0). Aggregate/count-based, no correlation."""
    traj, running = [], 0
    for e in ex.events:
        if e.is_send:
            running += 1
        elif e.is_receive:
            running = max(0, running - 1)
        traj.append(running)
    return traj


def _receive_by_corr(ex: Execution) -> Dict[str, Event]:
    """Enrichment: index receive observations by their native correlation id."""
    return {e.corr_id: e for e in ex.events
            if e.is_receive and e.corr_id is not None}


# --- label functions ---------------------------------------------------------
def _X_Inf(ctx: LabelContext, ex: Execution, i: int, p) -> int:
    """Peak in-flight backlog over the events after the cut ``i`` (no correlation)."""
    future = in_flight_trajectory(ex)[i + 1:]
    return max(future) if future else 0


def _X_MSt(ctx: LabelContext, ex: Execution, i: int, p):
    """Synchronization latency (s) of the next send after the cut to its matching
    receive; BOTTOM if unmatched/in-flight or if no correlation id is available."""
    recv = _receive_by_corr(ex)
    for j in range(i + 1, ex.n):
        e = ex.events[j]
        if e.is_send:
            if e.corr_id is None:
                return ctx.bottom            # undefined without enrichment
            r = recv.get(e.corr_id)
            if r is None or r.timestamp < e.timestamp:
                return ctx.bottom            # unmatched / still in flight
            return (r.timestamp - e.timestamp).total_seconds()
    return ctx.bottom


EXT_LABEL_FNS: Dict[str, Callable] = {"X-Inf": _X_Inf, "X-MSt": _X_MSt}


def compute_ext_label_rows(log: ObjectCentricLog, task: Task, param=None,
                           ctx: Optional[LabelContext] = None,
                           drop_bottom: bool = True
                           ) -> List[Tuple[str, str, int, object]]:
    """Rows (case_id, event_id, k, y) for an extension task, mirroring
    ``labels.compute_label_rows``. Numeric BOTTOM rows are dropped by default."""
    ctx = ctx or build_context(log)
    try:
        fn = EXT_LABEL_FNS[task.key]
    except KeyError:
        raise ValueError(f"No extension label function for task {task.key}")
    rows = []
    for ex in log:
        for i in range(ex.n - 1):
            y = fn(ctx, ex, i, param)
            if drop_bottom and task.kind == "numeric" and y == ctx.bottom:
                continue
            rows.append((ex.case_id, ex.events[i].event_id, i + 1, y))
    return rows
