"""
Per-fold training and DESCRIPTIVE scoring (V4: no comparison, no tuning). A single
fixed RandomForest is used per problem type, with a trivial baseline (majority class
for classification; median for regression) as a sanity reference.
Categorical/binary -> macro F1; numeric -> MAE.
"""
from typing import Dict, List
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, mean_absolute_error

from ocpa.util.util import LinearRegression as OcpaLinearRegression
from ocpm_tasks.catalog import Task


def _xy(table: pd.DataFrame, feature_cols: List[str], y_col: str):
    X = table[feature_cols].copy()
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype).startswith("category"):
            X[c] = X[c].astype("category").cat.codes
    return X.fillna(0.0), table[y_col]


def fit_and_score_fold(table: pd.DataFrame, feature_cols: List[str], y_col: str,
                       task: Task, train_mask, test_mask, cfg) -> Dict[str, float]:
    X, y = _xy(table, feature_cols, y_col)
    X_tr, X_te, y_tr, y_te = X[train_mask], X[test_mask], y[train_mask], y[test_mask]
    if len(y_tr) == 0 or len(y_te) == 0:
        return {}
    if task.kind in ("categorical", "binary"):
        # Cast to str so sklearn's type_of_target sees discrete string labels.
        # Without this, Python booleans in an object-dtype Series cause
        # type_of_target to return "unknown" instead of "binary".
        y_tr_s, y_te_s = y_tr.astype(str), y_te.astype(str)
        clf = RandomForestClassifier(
            n_estimators=cfg.rf_n_estimators, max_depth=cfg.rf_max_depth,
            random_state=cfg.random_state, n_jobs=-1, class_weight="balanced")
        clf.fit(X_tr, y_tr_s)
        pred = clf.predict(X_te)
        maj = pd.Series(y_tr_s).mode().iloc[0]
        return {
            "metric": float(f1_score(y_te_s, pred, average="macro", zero_division=0)),
            "baseline": float(f1_score(y_te_s, [maj] * len(y_te_s),
                                       average="macro", zero_division=0)),
            "n_test": int(len(y_te_s)),
        }
    reg = RandomForestRegressor(
        n_estimators=cfg.rf_n_estimators, max_depth=cfg.rf_max_depth,
        random_state=cfg.random_state, n_jobs=-1)
    reg.fit(X_tr, y_tr.astype(float))
    pred = reg.predict(X_te)
    median = float(np.median(y_tr.astype(float)))
    return {
        "metric": float(mean_absolute_error(y_te.astype(float), pred)),
        "baseline": float(mean_absolute_error(y_te.astype(float),
                                              [median] * len(y_te))),
        "n_test": int(len(y_te)),
    }


def fit_and_score_fold_ocpa_lr(table: pd.DataFrame, feature_cols: List[str], y_col: str,
                               task: Task, train_mask, test_mask, cfg) -> Dict[str, float]:
    """Same protocol as fit_and_score_fold, but with OCPA's own regressor
    (ocpa.util.util.LinearRegression) in place of the RandomForest, for comparison.
    OCPA ships no classifier, so categorical/binary tasks are skipped (empty dict)."""
    if task.kind in ("categorical", "binary"):
        return {}
    X, y = _xy(table, feature_cols, y_col)
    X_tr, X_te, y_tr, y_te = X[train_mask], X[test_mask], y[train_mask], y[test_mask]
    if len(y_tr) == 0 or len(y_te) == 0:
        return {}
    reg = OcpaLinearRegression()
    try:
        reg.fit(X_tr.to_numpy(dtype=float), y_tr.astype(float).to_numpy())
    except np.linalg.LinAlgError:
        # normal-equation solve is singular (e.g. collinear/constant feature columns)
        return {}
    pred = reg.predict(X_te.to_numpy(dtype=float))
    median = float(np.median(y_tr.astype(float)))
    return {
        "metric": float(mean_absolute_error(y_te.astype(float), pred)),
        "baseline": float(mean_absolute_error(y_te.astype(float),
                                              [median] * len(y_te))),
        "n_test": int(len(y_te)),
    }
