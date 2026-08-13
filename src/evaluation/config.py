"""
Configuration of the evaluation stages. Prediction tasks live in the decoupled
``tasks`` library; this module configures how the evaluation runs (log registry,
CV folds, learner, output paths). Inputs are OCEL 2.0 SQLite files (the format OCPA
imports natively).
"""
from dataclasses import dataclass, field
from typing import List, Optional

from tasks.schema import Schema
from tasks.catalog import RQ3_SUBSET


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

    # RQ3 CV protocol (not model-specific): 5-fold CV grouped by CollaborationCase.
    # The actual case_id -> fold assignment is persisted per log (see
    # rq3_pipeline.load_or_generate_folds) under folds_dir, computed once from
    # random_state and reused unchanged across every task and predictor run
    # against that log; changing n_folds or random_state invalidates any
    # already-persisted file (it raises rather than silently regenerating --
    # pass regenerate_folds=True or delete the file to rebuild it).
    n_folds: int = 5
    random_state: int = 3395
    folds_dir: str = "data/folds"
    regenerate_folds: bool = False

    # Torch device for every GPU-capable predictor (gnn, lstm_torch,
    # transformer -- random_forest and xgboost are CPU-only regardless).
    # "auto" picks CUDA when available, but that is NOT the default: measured
    # on this workload, CUDA's per-batch/per-graph dispatch overhead made
    # training slower wall-clock than plain CPU, both for GNN's tiny
    # per-sample subgraphs (gnn_k_values up to 16 nodes) and for LSTM's many
    # short, variable-length per-case sequences (see lstm_torch.py's device
    # comment for the earlier, narrower version of this finding re: MPS).
    # See predictors.common.resolve_device, used by all three.
    device: str = "cpu"  # "auto", "cpu", or "cuda"

    # ---- Predictor hyperparameters -----------------------------------------
    # One block per predictors.dispatch.PREDICTOR_REGISTRY entry, read by that
    # predictor's fit_and_score_fold via cfg.<prefix>_*; unrelated to whichever
    # predictor is actually selected by `predictor` below.

    # RandomForest hyperparameters.
    rf_n_estimators: int = 200
    rf_max_depth: Optional[int] = None

    # XGBoost hyperparameters (fixed across grouped folds).
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_reg_lambda: float = 1.0

    # LSTM hyperparameters (shared by lstm.py and lstm_torch.py). No early
    # stopping is implemented for this predictor, so these are tuned to avoid
    # underfitting within the fixed epoch budget rather than to converge exactly:
    # at the previous defaults (units=64, epochs=20, batch=32, lr=0.001), lstm_torch
    # trailed random_forest/xgboost/gnn by a wide margin on several RQ3 tasks
    # (e.g. Healthcare/NE-NMPr macro-F1 0.385 vs. 0.705-0.715). More epochs, a
    # smaller batch (more gradient updates per epoch on these small per-fold
    # training sets) and a higher learning rate closed most of that gap without
    # regressing tasks that were already fine.
    lstm_units: int = 32
    lstm_epochs: int = 60
    lstm_batch_size: int = 16
    lstm_learning_rate: float = 0.005

    # Transformer hyperparameters; transformer.py falls back to the LSTM ones
    # above for units/epochs/batch_size/learning_rate if these are left unset.
    # Unlike LSTM, transformer_epochs=20 was not badly underfitting (baseline
    # macro-F1/MAE were already close to random_forest/xgboost on the tasks
    # checked); reusing the LSTM bump (units/batch/lr) here gave mixed results
    # (small regressions on some tasks) for ~2-3x the training time, so only
    # the epoch budget was raised.
    transformer_units: int = 64
    transformer_epochs: int = 35
    transformer_batch_size: int = 32
    transformer_learning_rate: float = 0.001
    transformer_heads: int = 4
    transformer_layers: int = 2

    # GNN hyperparameters (direct event-graph predictor: OCPA graph extraction
    # + DGL GraphConv).
    gnn_hidden_dim: int = 64
    gnn_epochs: int = 100
    gnn_batch_size: int = 32
    gnn_learning_rate: float = 0.001
    gnn_early_stopping_patience: int = 10
    gnn_early_stopping_min_delta: float = 0.0001
    gnn_huber_delta: float = 1.0
    # Maximum subgraph node counts (cut included); short prefixes are retained.
    gnn_k_values: tuple = (4, 8, 16)

    # GNN runtime/logging knobs -- not hyperparameters, don't affect the fitted model.
    # Device selection moved to the shared `device` field above.
    gnn_verbose: bool = True
    gnn_log_every: int = 5
    # -------------------------------------------------------------------------

    # Key into predictors.dispatch.PREDICTOR_REGISTRY -- selects which
    # fit_and_score_fold implementation (and which hyperparameter block above)
    # run_rq3 uses: "random_forest" (default), "xgboost", "lstm", "lstm_torch",
    # "transformer", or "gnn".
    predictor: str = "random_forest"

    # Parameterized targets: resolved per log if None (most frequent).
    obm_target_activity: Optional[str] = None
    obp_target_participant: Optional[str] = None

    out_dir: str = "data/results"
    bottom: str = "__BOTTOM__"
    numeric_tol: float = 1.0     # seconds, for RQ2 temporal equivalence

    # RQ3 only. When True, run_rq3 also writes a per-stage wall-clock + RSS
    # memory profile to rq3_profile_{predictor}*.csv (see profiling.py),
    # kept separate from rq3_results_*.csv (V4: no computation times there).
    # Off by default: profiling spawns a background sampling thread per
    # stage, so it adds a small but nonzero overhead.
    profile: bool = False
