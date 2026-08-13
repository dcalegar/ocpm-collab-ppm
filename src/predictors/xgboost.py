"""Per-fold XGBoost predictor using the shared tabular predictor contract."""
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from tasks.catalog import Task
from .common import NullStageTimer, xy_split


def _common_params(cfg):
    return {
        "n_estimators": cfg.xgb_n_estimators,
        "max_depth": cfg.xgb_max_depth,
        "learning_rate": cfg.xgb_learning_rate,
        "subsample": cfg.xgb_subsample,
        "colsample_bytree": cfg.xgb_colsample_bytree,
        "reg_lambda": cfg.xgb_reg_lambda,
        "tree_method": "hist",
        "random_state": cfg.random_state,
        "n_jobs": -1,
    }


def fit_and_score_fold(feats: dict, tt: pd.DataFrame, y_col: str,
                       task: Task, train_mask, test_mask, cfg,
                       timer=None) -> Dict[str, float]:
    """Fit on the grouped training fold and score its untouched test fold."""
    from xgboost import XGBClassifier, XGBRegressor

    timer = timer or NullStageTimer()
    X_tr, X_te, y_tr, y_te = xy_split(
        tt, feats["feature_cols"], y_col, train_mask, test_mask)
    if len(y_tr) == 0 or len(y_te) == 0:
        return {}

    params = _common_params(cfg)
    if task.kind in ("categorical", "binary"):
        y_tr_s, y_te_s = y_tr.astype(str), y_te.astype(str)
        encoder = LabelEncoder().fit(y_tr_s)
        y_tr_encoded = encoder.transform(y_tr_s)
        model = XGBClassifier(**params, eval_metric="logloss")
        with timer.stage("fit"):
            model.fit(
                X_tr,
                y_tr_encoded,
                sample_weight=compute_sample_weight("balanced", y_tr_encoded),
            )
        with timer.stage("predict"):
            prediction = encoder.inverse_transform(model.predict(X_te).astype(int))
        majority = y_tr_s.mode().iloc[0]
        return {
            "metric": float(f1_score(
                y_te_s, prediction, average="macro", zero_division=0)),
            "baseline": float(f1_score(
                y_te_s, [majority] * len(y_te_s),
                average="macro", zero_division=0)),
            "n_test": int(len(y_te_s)),
        }

    y_tr_float, y_te_float = y_tr.astype(float), y_te.astype(float)
    model = XGBRegressor(**params, objective="reg:squarederror", eval_metric="mae")
    with timer.stage("fit"):
        model.fit(X_tr, y_tr_float)
    with timer.stage("predict"):
        prediction = model.predict(X_te)
    median = float(np.median(y_tr_float))
    return {
        "metric": float(mean_absolute_error(y_te_float, prediction)),
        "baseline": float(mean_absolute_error(
            y_te_float, [median] * len(y_te_float))),
        "n_test": int(len(y_te_float)),
    }
