"""Predictor registry: maps a config-selectable predictor key to its
fit_and_score_fold-shaped function. New predictors (xgboost, lstm, transformer,
gnn) get added here as they're implemented; nothing else needs to change to
select among them (see ExperimentConfig.predictor / ExtExperimentConfig.predictor)."""
from typing import Callable, Dict

from .random_forest import fit_and_score_fold as fit_and_score_fold_rf
from .lstm import fit_and_score_fold as fit_and_score_fold_lstm
from .lstm_torch import fit_and_score_fold as fit_and_score_fold_lstm_torch

PREDICTOR_REGISTRY: Dict[str, Callable] = {
    "random_forest": fit_and_score_fold_rf,
    "lstm": fit_and_score_fold_lstm,
    "lstm_torch": fit_and_score_fold_lstm_torch,
}



def resolve(predictor: str) -> Callable:
    try:
        return PREDICTOR_REGISTRY[predictor]
    except KeyError:
        valid = ", ".join(sorted(PREDICTOR_REGISTRY))
        raise KeyError(f"Unknown predictor {predictor!r}; valid options: {valid}") from None
