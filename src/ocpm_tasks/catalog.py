"""
Catalog of the reformulated prediction tasks (tasks.tex): the 14 tasks of Delgado et
al. plus the two object-enabled extensions X-PaL and X-MSt (16 total). Each task
declares its anchor object type, problem type, value kind, parameterization, whether
it is object-enabled, and whether it has a single-case counterpart.

This module is pure metadata; the actual label computation lives in ``labels.py``.
"""
from dataclasses import dataclass
from typing import Optional, Dict, List

# Anchor object types.
CI = "CollaborationInstance"
PARTICIPANT = "Participant"
LOCALCASE = "LocalCase"
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
    single_case_counterpart: bool = True


TASKS: Dict[str, Task] = {
    # --- Next event (categorical) ---
    "NE-NEPr": Task("NE-NEPr", CI,          MULTICLASS, "categorical"),
    "NE-NPaA": Task("NE-NPaA", PARTICIPANT, MULTICLASS, "categorical"),
    "NE-NEPa": Task("NE-NEPa", PARTICIPANT, MULTICLASS, "categorical"),
    "NE-NPaM": Task("NE-NPaM", PARTICIPANT, MULTICLASS, "categorical"),
    "NE-NMPa": Task("NE-NMPa", LOCALCASE,   MULTICLASS, "categorical", param="participant"),
    "NE-NMPr": Task("NE-NMPr", MESSAGE,     MULTICLASS, "categorical"),
    # --- Numeric value ---
    "NV-PrT":  Task("NV-PrT",  CI,          REG_TIME, "numeric"),
    "NV-PaT":  Task("NV-PaT",  LOCALCASE,   REG_TIME, "numeric", param="participant"),
    "NV-TNE":  Task("NV-TNE",  CI,          REG_TIME, "numeric"),
    "NV-TNM":  Task("NV-TNM",  MESSAGE,     REG_TIME, "numeric"),
    "NV-NMPr": Task("NV-NMPr", CI,          COUNT, "numeric", object_enabled=True),
    "NV-NMPa": Task("NV-NMPa", MESSAGE,     COUNT, "numeric", object_enabled=True,
                    param="participant"),
    # --- Outcome-based ---
    "OB-P":    Task("OB-P",    LOCALCASE,   BINARY, "binary", param="participant"),
    "OB-M":    Task("OB-M",    MESSAGE,     BINARY, "binary", object_enabled=True,
                    param="activity"),
    # --- Object-enabled extensions (no single-case counterpart) ---
    "X-PaL":   Task("X-PaL",   PARTICIPANT, COUNT, "numeric", object_enabled=True,
                    param="participant", single_case_counterpart=False),
    "X-MSt":   Task("X-MSt",   MESSAGE,     REG_TIME, "numeric", object_enabled=True,
                    single_case_counterpart=False),
}

# The 14 tasks with a single-case counterpart (RQ2 label-equivalence check).
EQUIVALENCE_TASKS: List[str] = [k for k, t in TASKS.items()
                                if t.single_case_counterpart]
# The 2 object-enabled extensions (RQ2 internal-consistency check).
CONSISTENCY_TASKS: List[str] = [k for k, t in TASKS.items()
                                if not t.single_case_counterpart]

# Representative subset for the end-to-end demonstration (RQ3), one per
# anchor x problem-type combination (evaluation Table tab:rq3subset).
RQ3_SUBSET: List[str] = ["NE-NPaA", "NE-NMPr", "NV-PrT", "NV-PaT", "NV-NMPr", "OB-M"]
