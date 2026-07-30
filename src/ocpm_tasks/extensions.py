"""
Object-enabled EXTENSION tasks (X-Inf, X-MSt) — targets outside the fourteen-type
taxonomy of Delgado et al., naturally expressed over object-centric relations
rather than over a single flattened case trace. This is NOT the same as having
no single-case counterpart: a case-centric trace that retained the same
attributes (direction, endpoints, a correlation id) could state equivalent
targets with additional preprocessing -- neither target strictly requires
object identity. They are deliberately kept OUT of the main catalog (``catalog.TASKS``
/ ``EQUIVALENCE_TASKS`` / ``RQ3_SUBSET``) so they are never mixed into the
reformulation evaluation (RQ2 label-equivalence, RQ3 subset/full). They are
exercised only by a dedicated toy-log test (``tests/test_extensions_toy.py``).

Two extensions with intentionally different correlation requirements:

X-Inf — in-flight message backlog (CollaborationCase anchor, count regression).
    Needs NO send/receive correlation: the in-flight level after an event is the
    RUNNING send/receive balance up to that event (+1 per send observation, -1
    per receive observation, clamped at 0 as it runs -- not a plain
    ``#send - #receive`` difference; see ``in_flight_trajectory``) over the Message
    objects (M4). This is object-enabled because the receive observations belong to
    OTHER participants' projections, so no single participant-local case trace sees
    them. Target at cut ``i``: the PEAK in-flight backlog over the remainder of the
    case (``max`` of the running backlog over events after the cut).

X-MSt — message synchronization time (Message anchor, time regression).
    PRESUPPOSES a send<->receive correspondence that the core mapping deliberately
    does NOT establish (M4). It is defined over an ENRICHED model in which each
    communication event carries a native correlation id (``Event.corr_id``), from
    which the pairing is recovered. ``Event.corr_id`` is populated by
    ``adapters.build_from_relations``, from either of two independent sources
    (see that function): preferably the mapping's own correlation refinement
    layer (C1's ``correlated_with`` O2O relation, already checked by PC.1), or
    a raw ``corr_attr`` residual attribute as a fallback -- X-MSt itself is
    agnostic to which one supplied the id. Target at cut ``i``: the latency
    ``receive_time - send_time`` of the NEXT send after the cut. A send matches a
    receive only if it shares the send's
    correlation id AND its message endpoints (``msg_from``/``msg_to``) AND occurs
    strictly after it in the case's positional order (ties on timestamp are
    resolved by source order, not treated as simultaneous — see ``_match_receive``).
    Returns BOTTOM when that send has no matching reception under all four
    conditions (unmatched, still in flight, endpoint mismatch, or a same-corr-id
    receive that precedes it) or when no correlation id is available — i.e. the
    task is undefined without enrichment.

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


def _match_receive(ex: Execution, j: int) -> Optional[Event]:
    """match(eps_j): the receive event, if
    any, that (2) shares the send's correlation id, (3) shares its message
    endpoints (``msg_from``/``msg_to``), and (4) occurs strictly after it in
    the case's positional order -- position, not merely timestamp, since
    ties on timestamp are resolved by source order (P1.2/prec_L), not
    treated as simultaneous. Condition 1 (same case) holds automatically:
    ``ex.events`` is already one collaboration case.

    Conditions 2+3 make match(eps_j) unique whenever ``corr_id`` is
    injective among same-case, same-endpoint receive events, an invariant
    the enrichment is assumed to satisfy. If that invariant is violated
    (e.g. a duplicated correlation id on two same-endpoint receives), the
    earliest qualifying candidate by position is returned deterministically,
    rather than silently keeping whichever receive a corr_id happened to
    collide with last (as a ``{corr_id: receive}`` index would).

    Both endpoints must be POSITIVELY known and equal, not merely
    structurally equal: ``send.msg_from == r.msg_from`` alone would accept
    two events that both have an undefined endpoint (``None == None``) as
    a "match" on that side, when in fact neither endpoint is known -- the
    condition-3 check would then be vacuous instead of a real guard."""
    send = ex.events[j]
    if send.msg_from is None or send.msg_to is None:
        return None            # counterparty endpoint unrecorded: cannot verify condition 3
    for r in ex.events[j + 1:]:
        if (r.is_receive and r.corr_id == send.corr_id
                and r.msg_from == send.msg_from and r.msg_to == send.msg_to):
            return r
    return None


# --- label functions ---------------------------------------------------------
def _X_Inf(ctx: LabelContext, ex: Execution, i: int, p) -> int:
    """Peak in-flight backlog over the events after the cut ``i`` (no correlation)."""
    future = in_flight_trajectory(ex)[i + 1:]
    return max(future) if future else 0


def _X_MSt(ctx: LabelContext, ex: Execution, i: int, p):
    """Synchronization latency (s) of the next send after the cut to its matching
    receive (see ``_match_receive`` for the endpoint/position conditions); BOTTOM
    if unmatched/in-flight, endpoint-mismatched, or if no correlation id is
    available."""
    for j in range(i + 1, ex.n):
        e = ex.events[j]
        if e.is_send:
            if e.corr_id is None:
                return ctx.bottom            # undefined without enrichment
            r = _match_receive(ex, j)
            if r is None:
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
