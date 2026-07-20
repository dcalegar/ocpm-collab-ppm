"""
Neutral object-centric model that the prediction tasks operate on. It is independent
of any concrete OCEL library: adapters (see ``adapters.py``) build it from a pm4py or
an OCPA OCEL. A task definition reads only this model, never the underlying log.

One ``Execution`` corresponds to one collaboration case (the CollaborationCase
viewpoint). The "kind" of a message is the activity (event type) of its event (M4).
"""
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict


@dataclass
class Event:
    event_id: str
    activity: str
    timestamp: datetime
    actor: str                       # participant (R2: in_projection -> for_participant; direct participant edge, P1.6)
    is_send: bool = False
    is_receive: bool = False
    msg_id: Optional[str] = None     # Message OBSERVATION identity (unique per send/receive; M4, no correlation)
    msg_type: Optional[str] = None   # = activity of the message event
    msg_from: Optional[str] = None   # sender (from relation)
    msg_to: Optional[str] = None     # receiver (to relation)
    corr_id: Optional[str] = None    # native correlation id pairing a send with its receive;
                                     # populated only by an ENRICHMENT step (not by the core
                                     # mapping M4). Used by the X-MSt extension task.

    @property
    def is_msg(self) -> bool:
        return self.is_send or self.is_receive


@dataclass
class Execution:
    """A collaboration case under the *global-trace viewpoint*: its events as a
    linear sequence ordered by timestamp (a stable sort; source/insertion order
    breaks ties -- see below and __post_init__, no secondary key). This linear order is the basis
    on which prefixes hd^k and the prediction targets are defined (it mirrors
    the case-centric baseline). It does NOT constrain how the observable prefix is
    encoded: in R2 the observable prefix is the object-centric execution graph, and
    the features (e.g. via OCPA) are graph-based, not a linear-prefix encoding. The
    linear order here is used only to define ground-truth targets at each cut point.
    Ties on timestamp keep the adapter's original (source/insertion) order --
    a stable sort on timestamp alone, no secondary key -- so cut points are
    deterministic PROVIDED that order already agrees with prec_L (B9).
    ``from_ocel2_sqlite`` (RQ2/R2) guarantees this directly, with an explicit
    ``ORDER BY ocel_id`` (ocel_id sorts consistently with prec_L once created
    with zero-padded indices, mapping.collab_xes_to_ocel._event_id).
    ``from_ocpa`` (RQ3/RQ-EXT) does not sort by ocel_id itself and inherits
    whatever row order OCPA's importer used -- but by construction it never
    receives a log with within-case timestamp ties to begin with:
    ``ocpm_eval.io_ocel._break_timestamp_ties`` (D23) nudges every tied
    within-case timestamp to a strictly increasing value in ocel_id/prec_L
    order before OCPA ever imports the file, so there is nothing left for
    OCPA's unverified row order to get wrong. Verified empirically on
    BPIC2013 (the only evaluated log with real ties, 4,051 same-instant
    pairs): 0 mismatches between this sort's output order and prec_L across
    all 7,554 executions. This guarantee holds only as long as every
    ``from_ocpa`` caller goes through ``load_ocpa_ocel`` (which always
    applies ``_break_timestamp_ties``); a caller that imports via OCPA
    directly, bypassing that helper, would reintroduce B9's original risk.
    """
    case_id: str
    events: List[Event]

    def __post_init__(self):
        # Stable sort: ties on timestamp preserve the order `events` was
        # built in (each adapter appends per case in source/insertion
        # order -- see adapters.build_from_relations). A secondary sort
        # key of str(event_id) was used before, but string comparison
        # does not agree with the events' true (numeric/insertion) order
        # once ids differ in digit count, e.g. "e::case::10" sorts before
        # "e::case::9" -- which silently reversed a synthesized
        # SendTask/ReceiveTask pair sharing one timestamp (they are
        # created as consecutive indices, so this triggers whenever the
        # pair straddles a power-of-10 boundary).
        self.events = sorted(self.events, key=lambda e: e.timestamp)

    @property
    def n(self) -> int:
        return len(self.events)


class ObjectCentricLog:
    """A collection of collaboration-case executions."""
    def __init__(self, executions: List[Execution]):
        self.executions = executions
        self._by_id: Dict[str, Execution] = {e.case_id: e for e in executions}

    def __iter__(self):
        return iter(self.executions)

    def __len__(self):
        return len(self.executions)

    def get(self, case_id: str) -> Optional[Execution]:
        return self._by_id.get(case_id)

    @property
    def case_ids(self) -> List[str]:
        return list(self._by_id.keys())
