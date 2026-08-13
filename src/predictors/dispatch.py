"""Predictor registry: maps a config-selectable predictor key to its
fit_and_score_fold-shaped function. New predictors (xgboost, lstm, transformer,
gnn) get added here as they're implemented; nothing else needs to change to
select among them (see ExperimentConfig.predictor)."""
from typing import Callable, Dict

from .random_forest import fit_and_score_fold as fit_and_score_fold_rf
from .lstm_torch import fit_and_score_fold as fit_and_score_fold_lstm_torch
from .gnn import fit_and_score_fold as fit_and_score_fold_gnn
from .xgboost import fit_and_score_fold as fit_and_score_fold_xgboost
from .transformer import fit_and_score_fold as fit_and_score_fold_transformer

# lstm.py (TensorFlow/Keras) is disabled here: on this evaluation setup it ran
# ~3x slower than lstm_torch per fold (repeated tf.function retracing as
# feature shapes change across tasks) for the same architecture/hyperparameters,
# so lstm_torch is the sole LSTM baseline. The Keras implementation is left in
# place in lstm.py, just not registered.
PREDICTOR_REGISTRY: Dict[str, Callable] = {
    "random_forest": fit_and_score_fold_rf,
    "lstm_torch": fit_and_score_fold_lstm_torch,
    "gnn": fit_and_score_fold_gnn,
    "xgboost": fit_and_score_fold_xgboost,
    "transformer": fit_and_score_fold_transformer,
}



def resolve(predictor: str) -> Callable:
    try:
        return PREDICTOR_REGISTRY[predictor]
    except KeyError:
        valid = ", ".join(sorted(PREDICTOR_REGISTRY))
        raise KeyError(f"Unknown predictor {predictor!r}; valid options: {valid}") from None
