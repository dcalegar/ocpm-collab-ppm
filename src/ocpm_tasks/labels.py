"""
Label (target) functions ℓ for the reformulated tasks, over the neutral
object-centric model. Pure functions: no ML, no I/O, no OCEL library.

Cut-point semantics. A target is evaluated at a *cut point* in a collaboration
instance. The cut point is an index ``i`` (0-based) in the instance's global trace
(its events ordered by timestamp, ties broken by source/insertion order -- see
Execution.__post_init__); the prefix hd^k has k=i+1 and the
target is a function of the events AFTER the cut. This linear global trace is the
basis the paper uses to DEFINE the targets (so they match the case-centric baseline)
and is independent of how the observable prefix is ENCODED: with OCPA the observable
prefix is the object-centric execution graph and the features are graph-based, not a
linear-prefix encoding. These functions only produce the ground-truth value at each
cut point; pairing it with a representation's per-event features is the caller's job.

A ``LabelContext`` is built once per log; it carries the BOTTOM symbol used by
categorical/numeric targets when no further occurrence exists. Every label function
has the uniform signature ``fn(ctx, ex, i, param) -> value | BOTTOM``. Rows are
emitted for k in [1, n-1] (i in [0, n-2]); the final, complete execution has no
prefix to predict from.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable

from .model import ObjectCentricLog, Execution
from .catalog import Task, TASKS

BOTTOM = "__BOTTOM__"


@dataclass
class LabelContext:
    log: ObjectCentricLog
    bottom: str = BOTTOM


def build_context(log: ObjectCentricLog, bottom: str = BOTTOM) -> LabelContext:
    return LabelContext(log=log, bottom=bottom)


# --- 14 single-case-counterpart tasks ----------------------------------------
def _NE_NEPr(ctx, ex, i, p):
    return ex.events[i + 1].activity if i + 1 < ex.n else ctx.bottom

def _NE_NPaA(ctx, ex, i, p):
    return ex.events[i + 1].actor if i + 1 < ex.n else ctx.bottom

def _NE_NEPa(ctx, ex, i, p):
    # Returns the pair (evtype, pa), per appendixTasks.tex Def. NE-NEPa. A
    # concatenated string ("act||actor") is not injective: ("A||B","C") and
    # ("A","B||C") both encode to "A||B||C" -- confirmed collision (B13).
    if i + 1 < ex.n:
        e = ex.events[i + 1]
        return (e.activity, e.actor)
    return ctx.bottom

def _NE_NPaM(ctx, ex, i, p):
    """Next participant to send (p="send", default) or receive (p="receive") a
    message: the message-object endpoint, `from` for send / `to` for receive
    (tasks.tex, NE-NPaM), read via msg_from/msg_to. No fallback to the event's
    own actor: `from` on a send event (resp. `to` on a receive event) is
    always the event's own side, so under a correct construction this is
    never undefined (P1.3). If it ever is, that signals a construction
    defect (e.g. a lost O2O edge) that R2 should surface, not silently mask
    with a same-valued-by-P1.3-but-not-actually-the-relation substitute."""
    want_send = (p or "send") == "send"
    for j in range(i + 1, ex.n):
        e = ex.events[j]
        if e.is_send if want_send else e.is_receive:
            return e.msg_from if want_send else e.msg_to
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
    """Time until the next message in direction p ("send", default, or
    "receive"; tasks.tex, NV-TNM)."""
    want_send = (p or "send") == "send"
    for j in range(i + 1, ex.n):
        e = ex.events[j]
        if e.is_send if want_send else e.is_receive:
            return (e.timestamp - ex.events[i].timestamp).total_seconds()
    return ctx.bottom

def _NV_NMPr(ctx, ex, i, p):
    # Delgado et al. 2025, Table 2: "Number of remaining/total messages
    # (send/receive)" -- both directions count, not sends only.
    # tasks.tex quantifies over Msgs(c) via pos(m)>k; counting communication
    # events with is_msg here is equivalent, not an approximation, because
    # M4 creates exactly one Message object per send/receive event (a
    # bijection between E^snd_L u E^rcv_L and Msgs(c)) -- there is no
    # separate exchanged_in traversal that could disagree with this count.
    return sum(1 for j in range(i + 1, ex.n) if ex.events[j].is_msg)

def _NV_NMPa(ctx, ex, i, p):
    if p is None:
        raise ValueError("NV-NMPa requires param = participant.")
    # Both directions count (see _NV_NMPr); a message's owning participant
    # is its event's actor regardless of send/receive direction.
    return sum(1 for j in range(i + 1, ex.n)
               if ex.events[j].is_msg and ex.events[j].actor == p)

def _OB_P(ctx, ex, i, p):
    if p is None:
        raise ValueError("OB-P requires param = participant.")
    return any(ex.events[j].actor == p for j in range(i + 1, ex.n))

def _OB_M(ctx, ex, i, p):
    if p is None:
        raise ValueError("OB-M requires param = message-activity label (a_hat).")
    # Delgado et al. 2025, Table 2: "If a particular message will be
    # sent/received" -- both directions count, not sends only. Same
    # events-for-objects equivalence as _NV_NMPr (M4 bijection).
    return any(ex.events[j].is_msg and ex.events[j].activity == p
               for j in range(i + 1, ex.n))


LABEL_FNS: Dict[str, Callable] = {
    "NE-NEPr": _NE_NEPr, "NE-NPaA": _NE_NPaA, "NE-NEPa": _NE_NEPa,
    "NE-NPaM": _NE_NPaM, "NE-NMPa": _NE_NMPa, "NE-NMPr": _NE_NMPr,
    "NV-PrT": _NV_PrT, "NV-PaT": _NV_PaT, "NV-TNE": _NV_TNE,
    "NV-TNM": _NV_TNM, "NV-NMPr": _NV_NMPr, "NV-NMPa": _NV_NMPa,
    "OB-P": _OB_P, "OB-M": _OB_M,
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
