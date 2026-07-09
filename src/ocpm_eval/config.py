"""
Configuration of the evaluation stages. Prediction tasks live in the decoupled
``ocpm_tasks`` library; this module configures how the evaluation runs (log registry,
CV folds, learner, output paths). Inputs are OCEL 2.0 SQLite files (the format OCPA
imports natively).
"""
from dataclasses import dataclass, field
from typing import List, Optional

from ocpm_tasks.schema import Schema
from ocpm_tasks.catalog import RQ3_SUBSET


@dataclass
class LogSpec:
    name: str
    ocel_path: str   # OCEL 2.0 SQLite (.sqlite), produced by the converter (R2)
    xes_path: str    # extended collaborative XES (.xes), original source log (R1)


def predictcollab_ocel_logs() -> List[LogSpec]:
    return [
        LogSpec("Healthcare",  "data/logs/Predict-Collab/collectivelog_healthcare_collab.sqlite",
                               "data/logs/Predict-Collab/collectivelog_healthcare_collab.xes"),
        LogSpec("Artificial1", "data/logs/Predict-Collab/collectivelog_artificial1_collab.sqlite",
                               "data/logs/Predict-Collab/collectivelog_artificial1_collab.xes"),
        LogSpec("Artificial5", "data/logs/Predict-Collab/collectivelog_artificial5_collab.sqlite",
                               "data/logs/Predict-Collab/collectivelog_artificial5_collab.xes"),
        LogSpec("Real4",       "data/logs/Predict-Collab/collectivelog_real4_collab.sqlite",
                               "data/logs/Predict-Collab/collectivelog_real4_collab.xes"),
    ]


@dataclass
class ExperimentConfig:
    logs: List[LogSpec] = field(default_factory=predictcollab_ocel_logs)
    schema: Schema = field(default_factory=Schema)

    rq3_tasks: List[str] = field(default_factory=lambda: list(RQ3_SUBSET))

    # RQ3 protocol: 5-fold CV grouped by CollaborationCase, fixed RandomForest.
    n_folds: int = 5
    random_state: int = 3395
    rf_n_estimators: int = 200
    rf_max_depth: Optional[int] = None

    # Parameterized targets: resolved per log if None (most frequent).
    obm_target_activity: Optional[str] = None
    obp_target_participant: Optional[str] = None

    out_dir: str = "data/results"
    bottom: str = "__BOTTOM__"
    numeric_tol: float = 1.0     # seconds, for RQ2 temporal equivalence


# ---------------------------------------------------------------------------
# Object-enabled EXTENSION tasks (X-Inf, X-MSt) -- demo config, kept separate
# from ExperimentConfig/delgado_ocel_logs so it never mixes into RQ2/RQ3.
# ---------------------------------------------------------------------------
def toy_ext_log() -> LogSpec:
    """Hand-built toy log (src/mapping/support/build_toy_collab_log.py) for
    testing X-Inf and X-MSt extensions; NOT one of the four study logs. 100
    cases, 3 participants, 1,132 events, designed to exercise both targets:
    variable in-flight backlogs (X-Inf) and send/receive pairs with explicit
    msgId correlation (X-MSt), tied to case participant count (2 vs. 3) so
    both targets carry a genuine, prefix-observable signal rather than one
    drawn i.i.d. of the observed prefix. Demonstrates that both extensions
    are functionally correct, and exploitable by a generic object-centric
    feature set, when data has the needed semantic properties (send/receive
    events + correlation ids)."""
    return LogSpec("ToyCollab", "data/logs/ToyCollab/toy_collab.sqlite",
                                "data/logs/ToyCollab/toy_collab.xes")


@dataclass
class ExtExperimentConfig:
    logs: List[LogSpec] = field(default_factory=lambda: [toy_ext_log()])
    schema: Schema = field(default_factory=Schema)

    ext_tasks: List[str] = field(default_factory=lambda: ["X-Inf", "X-MSt"])
    # Residual event attribute carrying a native message-correlation id
    # (see build_toy_collab_log.py); enables X-MSt. None would leave X-MSt
    # undefined everywhere, as in the core (unenriched) mapping.
    corr_attr: Optional[str] = "msgId"

    # Same 5-fold protocol as ExperimentConfig (RQ3), grouped by CollaborationCase.
    n_folds: int = 5
    random_state: int = 3395
    rf_n_estimators: int = 200
    rf_max_depth: Optional[int] = None

    out_dir: str = "data/results"
    bottom: str = "__BOTTOM__"
