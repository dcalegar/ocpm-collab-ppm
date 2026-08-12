"""
Mapping vocabulary (object types, qualifiers, attributes) shared by the adapters
that turn an OCEL into the neutral object-centric model. Mirrors the mapping rules
M1-M8. This is the only place that names OCEL artefacts; the task definitions are
expressed over the neutral model and never touch these names.

Rule M2 declares ONE OBJECT TYPE PER PARTICIPANT IDENTIFIER, so there is no single
participant type to name here. Participant objects are instead identified the way
the mapping itself characterizes them: they are the objects reached by the
`in_participant`, `for_participant`, `from` and `to` qualifiers, i.e. every object
whose type is none of the structural types below. ``is_participant_type`` is that
test; the qualifier vocabulary is fixed, so it works whatever T_Pa the log declares.
"""
from dataclasses import dataclass


@dataclass
class Schema:
    # Structural object types (M1, M3, M4). Participant types are log-dependent.
    ot_cc: str = "CollaborationCase"
    ot_oc: str = "OrchestrationCase"
    ot_message: str = "Message"

    # E2O qualifiers (M6): in_collaboration, in_orchestration, send, receive,
    # in_participant.
    q_in_collaboration: str = "in_collaboration"
    q_in_orchestration: str = "in_orchestration"
    q_send: str = "send"
    q_receive: str = "receive"
    q_in_participant: str = "in_participant"

    # O2O qualifiers (M7): part_of, for_participant, from, to, exchanged_in.
    q_part_of: str = "part_of"
    q_for_participant: str = "for_participant"
    q_from: str = "from"
    q_to: str = "to"
    q_exchanged_in: str = "exchanged_in"

    # Object attributes.
    oa_caseid: str = "caseId"
    oa_name: str = "name"                 # participant objects
    oa_participant: str = "participant"   # OrchestrationCase (= participant name, P1.4)
    oa_sender: str = "sender"             # Message (M4)
    oa_receiver: str = "receiver"         # Message (M4)

    def is_participant_type(self, otype) -> bool:
        """True for the object types rule M2 derives from participant
        identifiers, i.e. everything that is not a structural type of the
        mapping."""
        return bool(otype) and otype not in (
            self.ot_cc, self.ot_oc, self.ot_message)
