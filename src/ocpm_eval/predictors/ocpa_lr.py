"""OCPA's own regressor (ocpa.util.util.LinearRegression), for comparison against
fit_and_score_fold (RandomForest). Not currently wired into any pipeline/registry."""
from typing import Dict, List
import numpy as np
import pandas as pd

from ocpa.util.util import LinearRegression as OcpaLinearRegression
from ocpm_tasks.catalog import Task
from .common import xy_split


def fit_and_score_fold_ocpa_lr(table: pd.DataFrame, feature_cols: List[str], y_col: str,
                               task: Task, train_mask, test_mask, cfg) -> Dict[str, float]:
    """Same protocol as random_forest.fit_and_score_fold, but with OCPA's own regressor
    in place of the RandomForest. OCPA ships no classifier, so categorical/binary
    tasks are skipped (empty dict)."""
    if task.kind in ("categorical", "binary"):
        return {}
    X_tr, X_te, y_tr, y_te = xy_split(table, feature_cols, y_col, train_mask, test_mask)
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
