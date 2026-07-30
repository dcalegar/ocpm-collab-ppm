"""
Catalog of the 14 reformulated prediction tasks (the object-centric reformulation of
the collaborative baseline of Delgado et al.). Each task declares its anchor object
type, problem type, value kind, and parameterization.

Beyond this taxonomy, further exploratory extension directions (X-PaL, X-Inf,
X-Cmp, X-MSt, X-Lag) have been outlined; only X-Inf and X-MSt are implemented,
in ``extensions.py``, kept out of this catalog (see that module).

This module is pure metadata; the actual label computation lives in ``labels.py``.
"""
from dataclasses import dataclass
from typing import Optional, Dict, List

# Target anchors. CC, ORCHESTRATION_CASE and MESSAGE are object types of the
# mapping; PARTICIPANT is the anchor ROLE of the tasks about participants, not
# an object type -- rule M2 gives each participant identifier a type of its own,
# so there is no single "Participant" type in the OCEL (see ocpm_tasks/schema.py).
# These strings are reporting metadata only: no label function reads Task.anchor.
CC = "CollaborationCase"
PARTICIPANT = "Participant"
ORCHESTRATION_CASE = "OrchestrationCase"
MESSAGE = "Message"

# Problem types (as in the evaluation's RQ3 subset table).
MULTICLASS = "Multiclass classification"
BINARY = "Binary classification"
REG_TIME = "Regression (time)"
COUNT = "Count regression"


@dataclass(frozen=True)
class Task:
    key: str
    anchor: str
    problem_type: str
    kind: str                     # "categorical" | "numeric" | "binary"
    object_enabled: bool = False
    param: Optional[str] = None   # "participant" | "activity" | None


TASKS: Dict[str, Task] = {
    # --- Next event (categorical) ---
    "NE-NEPr": Task("NE-NEPr", CC,          MULTICLASS, "categorical"),
    "NE-NPaA": Task("NE-NPaA", PARTICIPANT, MULTICLASS, "categorical"),
    "NE-NEPa": Task("NE-NEPa", PARTICIPANT, MULTICLASS, "categorical"),
    "NE-NPaM": Task("NE-NPaM", PARTICIPANT, MULTICLASS, "categorical"),
    "NE-NMPa": Task("NE-NMPa", ORCHESTRATION_CASE, MULTICLASS, "categorical", param="participant"),
    "NE-NMPr": Task("NE-NMPr", CC,          MULTICLASS, "categorical"),
    # --- Numeric value ---
    "NV-PrT":  Task("NV-PrT",  CC,          REG_TIME, "numeric"),
    "NV-PaT":  Task("NV-PaT",  ORCHESTRATION_CASE, REG_TIME, "numeric", param="participant"),
    "NV-TNE":  Task("NV-TNE",  CC,          REG_TIME, "numeric"),
    "NV-TNM":  Task("NV-TNM",  MESSAGE,     REG_TIME, "numeric"),
    "NV-NMPr": Task("NV-NMPr", MESSAGE,     COUNT, "numeric", object_enabled=True),
    "NV-NMPa": Task("NV-NMPa", MESSAGE,     COUNT, "numeric", object_enabled=True,
                    param="participant"),
    # --- Outcome-based ---
    "OB-P":    Task("OB-P",    ORCHESTRATION_CASE, BINARY, "binary", param="participant"),
    "OB-M":    Task("OB-M",    MESSAGE,     BINARY, "binary", object_enabled=True,
                    param="activity"),
}

# The 14 tasks, used for RQ2's label-equivalence check.
EQUIVALENCE_TASKS: List[str] = list(TASKS.keys())

# Representative subset for the end-to-end demonstration (RQ3), one per
# anchor x problem-type combination.
RQ3_SUBSET: List[str] = ["NE-NPaA", "NE-NMPr", "NV-PrT", "NV-PaT", "NV-NMPr", "OB-M"]
