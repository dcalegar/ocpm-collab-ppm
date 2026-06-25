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
    ocel_path: str              # OCEL 2.0 SQLite (.sqlite), produced by the converter


def delgado_ocel_logs() -> List[LogSpec]:
    return [
        LogSpec("Healthcare",  "data/logs/healthcare.sqlite"),
        LogSpec("Artificial1", "data/logs/artificial1.sqlite"),
        LogSpec("Artificial5", "data/logs/artificial5.sqlite"),
        LogSpec("Real4",       "data/logs/real4.sqlite"),
    ]


@dataclass
class ExperimentConfig:
    logs: List[LogSpec] = field(default_factory=delgado_ocel_logs)
    schema: Schema = field(default_factory=Schema)

    rq3_tasks: List[str] = field(default_factory=lambda: list(RQ3_SUBSET))

    # RQ3 protocol: 5-fold CV grouped by CollaborationInstance, fixed RandomForest.
    n_folds: int = 5
    random_state: int = 3395
    rf_n_estimators: int = 200
    rf_max_depth: Optional[int] = None

    # Parameterized targets: resolved per log if None (most frequent).
    obm_target_activity: Optional[str] = None
    obp_target_participant: Optional[str] = None

    out_dir: str = "results"
    bottom: str = "__BOTTOM__"
    numeric_tol: float = 1.0     # seconds, for RQ2 temporal equivalence
