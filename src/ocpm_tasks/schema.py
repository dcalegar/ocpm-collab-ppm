"""
Mapping vocabulary (object types, qualifiers, attributes) shared by the adapters
that turn an OCEL into the neutral object-centric model. Mirrors mapping.tex
(M1-M8). This is the only place that names OCEL artefacts; the task definitions are
expressed over the neutral model and never touch these names.
"""
from dataclasses import dataclass


@dataclass
class Schema:
    # Object types (M1).
    ot_ci: str = "CollaborationInstance"
    ot_lc: str = "LocalCase"
    ot_participant: str = "Participant"
    ot_message: str = "Message"
    ot_artifact: str = "BusinessArtifact"

    # E2O qualifiers (M7): within, local, send, receive. (No `actor`.)
    q_within: str = "within"
    q_local: str = "local"
    q_send: str = "send"
    q_receive: str = "receive"

    # O2O qualifiers (M8): part_of, executed_by, from, to, exchanged_in.
    q_part_of: str = "part_of"
    q_executed_by: str = "executed_by"
    q_from: str = "from"
    q_to: str = "to"
    q_exchanged_in: str = "exchanged_in"

    # Object attributes.
    oa_caseid: str = "caseId"
    oa_name: str = "name"                 # Participant
    oa_participant: str = "participant"   # LocalCase (= participant name, P1.4)
    oa_sender: str = "sender"             # Message (M4)
    oa_receiver: str = "receiver"         # Message (M4)

    # Preserved event attribute (M6); used only as a fallback for send/receive.
    ea_elemtype: str = "elemType"
    elem_task: str = "task"
    elem_send: str = "SendTask"
    elem_receive: str = "ReceiveTask"
