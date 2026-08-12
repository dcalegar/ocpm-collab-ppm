# predictors — per-learner fit/score modules

One fit-and-score module per learner, behind a single registry, so
[`ocpm_eval`](../ocpm_eval/README.md)'s RQ3 pipeline can select among them
without any pipeline code changing. Decoupled from `ocpm_eval`: the only internal
dependency is [`ocpm_tasks.catalog.Task`](../ocpm_tasks/README.md) (for `task.kind`),
so this package is reusable by any pipeline that can hand it a feature table in the
shared tabular contract below.

## Modules

| Module | Purpose |
|---|---|
| `common.py` | `xy_split` — shared train/test split with train-fit categorical encoding |
| `dispatch.py` | `PREDICTOR_REGISTRY` / `resolve` — maps a config-selectable predictor key to its `fit_and_score_fold` |
| `random_forest.py` | fixed `RandomForestClassifier`/`Regressor`, the default predictor |
| `xgboost.py` | `XGBClassifier`/`Regressor` |
| `lstm.py` | TensorFlow/Keras LSTM |
| `lstm_torch.py` | PyTorch LSTM |
| `gnn.py` | direct object-centric event-graph predictor over OCPA's `feature_storage` graphs |
| `ocpa_lr.py` | OCPA's own `LinearRegression`, for comparison; not wired into `dispatch`'s registry |

## The `fit_and_score_fold` contract

Every registered predictor (except `ocpa_lr.py`, kept out of the registry — see
above) implements the same signature:

```python
def fit_and_score_fold(feats: dict, tt: pd.DataFrame, y_col: str,
                        task: Task, train_mask, test_mask, cfg) -> Dict[str, float]:
    ...
```

- `feats["feature_cols"]` names the feature columns of `tt` (a `features.ocpa
  .extract_feature_table` output table, or any table shaped the same way).
- `task.kind` (`"categorical"` / `"binary"` / a numeric kind) selects macro-F1
  (classification) or MAE (regression) scoring.
- `train_mask`/`test_mask` are boolean masks over `tt` (one `GroupKFold` fold).
- Returns a metrics dict, or `{}` when a fold is empty or (for `ocpa_lr.py`) the
  task is not a regression task OCPA's regressor can handle.

## Usage

```python
from predictors.dispatch import resolve

fit_and_score_fold = resolve("random_forest")   # or "xgboost"/"lstm"/"lstm_torch"/"gnn"
metrics = fit_and_score_fold(feats, table, y_col, task, train_mask, test_mask, cfg)
```

## Tests

`tests/test_predictors_registry.py` is a synthetic-table smoke test per registry
entry (`random_forest`, `lstm`, `lstm_torch`, `xgboost`) plus a registry-presence
check for `gnn`, which needs OCPA's `feature_storage` graphs and so isn't exercised
against the synthetic table.
