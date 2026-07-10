"""
Mapping vocabulary (object types, qualifiers, attributes) shared by the adapters
that turn an OCEL into the neutral object-centric model. Mirrors the mapping rules
M1-M8. This is the only place that names OCEL artefacts; the task definitions are
expressed over the neutral model and never touch these names.
"""
from dataclasses import dataclass


@dataclass
class Schema:
    # Object types (M1).
    ot_cc: str = "CollaborationCase"
    ot_pp: str = "ParticipantProjection"
    ot_participant: str = "Participant"
    ot_message: str = "Message"
    ot_artifact: str = "BusinessArtifact"

    # E2O qualifiers (M6): within, in_projection, send, receive, participant.
    q_within: str = "within"
    q_in_projection: str = "in_projection"
    q_send: str = "send"
    q_receive: str = "receive"
    q_participant: str = "participant"

    # O2O qualifiers (M7): projection_of, for_participant, from, to, exchanged_in.
    q_projection_of: str = "projection_of"
    q_for_participant: str = "for_participant"
    q_from: str = "from"
    q_to: str = "to"
    q_exchanged_in: str = "exchanged_in"

    # Object attributes.
    oa_caseid: str = "caseId"
    oa_name: str = "name"                 # Participant
    oa_participant: str = "participant"   # ParticipantProjection (= participant name, P1.4)
    oa_sender: str = "sender"             # Message (M4)
    oa_receiver: str = "receiver"         # Message (M4)

    # Preserved event attribute (M6); used only as a fallback for send/receive.
    ea_elemtype: str = "elemType"
    elem_task: str = "task"
    elem_send: str = "SendTask"
    elem_receive: str = "ReceiveTask"
