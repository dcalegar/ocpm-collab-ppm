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


def real_world_ocel_logs() -> List[LogSpec]:
    """BPI Challenge 2013 (incidents, collaborative). Real-world validation log,
    kept out of predictcollab_ocel_logs() since it does not share provenance
    with the four study logs reused from Delgado et al. (2025); run as a
    separate stage/config (see run_evaluation.py)."""
    return [
        LogSpec("BPI2013", "data/logs/BPIChallenge2013/BPI2013_incidents_collaborative.sqlite",
                            "data/logs/BPIChallenge2013/BPI2013_incidents_collaborative.xes"),
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

    # XGBoost hyperparameters (fixed across grouped folds).
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_reg_lambda: float = 1.0

    # LSTM hyperparameters
    lstm_units: int = 64
    lstm_epochs: int = 20
    lstm_batch_size: int = 32
    lstm_learning_rate: float = 0.001

    # Transformer hyperparameters
    transformer_units: int = 64
    transformer_epochs: int = 20
    transformer_batch_size: int = 32
    transformer_learning_rate: float = 0.001
    transformer_heads: int = 4
    transformer_layers: int = 2

    # Direct event-graph GNN (OCPA graph extraction + DGL GraphConv).
    gnn_hidden_dim: int = 64
    gnn_epochs: int = 100
    gnn_batch_size: int = 32
    gnn_learning_rate: float = 0.001
    gnn_early_stopping_patience: int = 10
    gnn_early_stopping_min_delta: float = 0.0001
    gnn_huber_delta: float = 1.0
    gnn_device: str = "auto"  # "auto", "cpu", or "cuda"
    gnn_verbose: bool = True
    gnn_log_every: int = 5
    # Maximum subgraph node counts (cut included); short prefixes are retained.
    gnn_k_values: tuple = (4, 8, 16)

    # Key into predictors.dispatch.PREDICTOR_REGISTRY -- which fit_and_score_fold
    # implementation run_rq3 uses. Only "random_forest" exists today;
    # more predictors (xgboost, lstm, transformer, gnn) register there as added.
    predictor: str = "random_forest"

    # Parameterized targets: resolved per log if None (most frequent).
    obm_target_activity: Optional[str] = None
    obp_target_participant: Optional[str] = None

    out_dir: str = "data/results"
    bottom: str = "__BOTTOM__"
    numeric_tol: float = 1.0     # seconds, for RQ2 temporal equivalence
