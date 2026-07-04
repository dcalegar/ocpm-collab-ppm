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
    linear sequence ordered by (timestamp, event_id). This linear order is the basis
    on which the paper defines prefixes hd^k and the prediction targets (it mirrors
    the case-centric baseline). It does NOT constrain how the observable prefix is
    encoded: in R2 the observable prefix is the object-centric execution graph, and
    the features (e.g. via OCPA) are graph-based, not a linear-prefix encoding. The
    linear order here is used only to define ground-truth targets at each cut point.
    Ties on timestamp are broken by event_id so cut points are deterministic.
    """
    case_id: str
    events: List[Event]

    def __post_init__(self):
        self.events = sorted(self.events,
                             key=lambda e: (e.timestamp, str(e.event_id)))

    @property
    def n(self) -> int:
        return len(self.events)

    @property
    def messages(self) -> List[Event]:
        return [e for e in self.events if e.is_send]


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
