"""
Smoke test for the predictors/ subpackage introduced when RandomForest logic was
moved out of the old ocpm_eval/models.py and behind predictors.dispatch.PREDICTOR_REGISTRY
(see .claude/planPredictores.md, "Modularization"). No OCEL log needed -- exercises
PREDICTOR_REGISTRY["random_forest"] directly on a tiny synthetic table, for one
classification and one regression task, so a future predictor addition/refactor has a
cheap regression check that doesn't need the full evaluation pipeline.

Run:  python tests/test_predictors_registry.py    (also pytest-compatible)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd                                                       # noqa: E402
from ocpm_eval.config import ExperimentConfig                             # noqa: E402
from predictors.dispatch import PREDICTOR_REGISTRY, resolve               # noqa: E402
from ocpm_tasks.catalog import Task, CC, BINARY, REG_TIME                 # noqa: E402

FEATURE_COLS = ["x1", "x2"]

_TABLE = pd.DataFrame({
    "case_id": ["c1", "c1", "c1", "c2", "c2", "c2", "c3", "c3"],
    "x1": [0.0, 1.0, 0.2, 0.9, 0.1, 1.1, 0.3, 0.8],
    "x2": [1.0, 0.0, 0.9, 0.2, 0.8, 0.1, 0.7, 0.3],
    "_y_clf": ["a", "b", "a", "b", "a", "b", "a", "b"],
    "_y_reg": [1.0, 5.0, 1.2, 4.8, 0.9, 5.1, 1.1, 4.9],
})
_TRAIN = pd.Series([True, True, True, True, True, True, False, False])
_TEST = ~_TRAIN

_CLF_TASK = Task("SMOKE-CLF", CC, BINARY, "binary")
_REG_TASK = Task("SMOKE-REG", CC, REG_TIME, "numeric")


def test_registry_has_random_forest():
    assert "random_forest" in PREDICTOR_REGISTRY
    assert resolve("random_forest") is PREDICTOR_REGISTRY["random_forest"]


def test_registry_has_lstm():
    assert "lstm" in PREDICTOR_REGISTRY
    assert resolve("lstm") is PREDICTOR_REGISTRY["lstm"]


def test_registry_has_lstm_torch():
    assert "lstm_torch" in PREDICTOR_REGISTRY
    assert resolve("lstm_torch") is PREDICTOR_REGISTRY["lstm_torch"]


def test_registry_has_gnn():
    assert "gnn" in PREDICTOR_REGISTRY
    assert resolve("gnn") is PREDICTOR_REGISTRY["gnn"]


def test_registry_has_xgboost():
    assert "xgboost" in PREDICTOR_REGISTRY
    assert resolve("xgboost") is PREDICTOR_REGISTRY["xgboost"]


def test_resolve_unknown_predictor_raises():
    try:
        resolve("does_not_exist")
    except KeyError:
        return
    raise AssertionError("resolve() should raise KeyError for an unregistered predictor")


def test_random_forest_classification():
    cfg = ExperimentConfig()
    fit_fn = PREDICTOR_REGISTRY["random_forest"]
    feats = {"feature_cols": FEATURE_COLS}
    result = fit_fn(feats, _TABLE, "_y_clf", _CLF_TASK, _TRAIN, _TEST, cfg)
    assert set(result) == {"metric", "baseline", "n_test"}
    assert result["n_test"] == 2


def test_random_forest_regression():
    cfg = ExperimentConfig()
    fit_fn = PREDICTOR_REGISTRY["random_forest"]
    feats = {"feature_cols": FEATURE_COLS}
    result = fit_fn(feats, _TABLE, "_y_reg", _REG_TASK, _TRAIN, _TEST, cfg)
    assert set(result) == {"metric", "baseline", "n_test"}
    assert result["n_test"] == 2


def test_xgboost_classification():
    cfg = ExperimentConfig(xgb_n_estimators=2, xgb_max_depth=2)
    result = PREDICTOR_REGISTRY["xgboost"](
        {"feature_cols": FEATURE_COLS}, _TABLE, "_y_clf",
        _CLF_TASK, _TRAIN, _TEST, cfg)
    assert set(result) == {"metric", "baseline", "n_test"}
    assert result["n_test"] == 2


def test_xgboost_regression():
    cfg = ExperimentConfig(xgb_n_estimators=2, xgb_max_depth=2)
    result = PREDICTOR_REGISTRY["xgboost"](
        {"feature_cols": FEATURE_COLS}, _TABLE, "_y_reg",
        _REG_TASK, _TRAIN, _TEST, cfg)
    assert set(result) == {"metric", "baseline", "n_test"}
    assert result["n_test"] == 2


def test_lstm_classification():
    cfg = ExperimentConfig(lstm_epochs=2, lstm_units=8)
    fit_fn = PREDICTOR_REGISTRY["lstm"]
    feats = {"feature_cols": FEATURE_COLS}
    result = fit_fn(feats, _TABLE, "_y_clf", _CLF_TASK, _TRAIN, _TEST, cfg)
    assert set(result) == {"metric", "baseline", "n_test"}
    assert result["n_test"] == 2


def test_lstm_regression():
    cfg = ExperimentConfig(lstm_epochs=2, lstm_units=8)
    fit_fn = PREDICTOR_REGISTRY["lstm"]
    feats = {"feature_cols": FEATURE_COLS}
    result = fit_fn(feats, _TABLE, "_y_reg", _REG_TASK, _TRAIN, _TEST, cfg)
    assert set(result) == {"metric", "baseline", "n_test"}
    assert result["n_test"] == 2


def test_lstm_torch_classification():
    cfg = ExperimentConfig(lstm_epochs=2, lstm_units=8)
    fit_fn = PREDICTOR_REGISTRY["lstm_torch"]
    feats = {"feature_cols": FEATURE_COLS}
    result = fit_fn(feats, _TABLE, "_y_clf", _CLF_TASK, _TRAIN, _TEST, cfg)
    assert set(result) == {"metric", "baseline", "n_test"}
    assert result["n_test"] == 2


def test_lstm_torch_regression():
    cfg = ExperimentConfig(lstm_epochs=2, lstm_units=8)
    fit_fn = PREDICTOR_REGISTRY["lstm_torch"]
    feats = {"feature_cols": FEATURE_COLS}
    result = fit_fn(feats, _TABLE, "_y_reg", _REG_TASK, _TRAIN, _TEST, cfg)
    assert set(result) == {"metric", "baseline", "n_test"}
    assert result["n_test"] == 2


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n[ok] {len(fns)} predictor registry tests passed.")


if __name__ == "__main__":
    _run_all()
