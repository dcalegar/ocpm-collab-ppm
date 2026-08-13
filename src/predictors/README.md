# predictors — per-learner fit/score modules

One fit-and-score module per learner, behind a single registry, so
[`evaluation`](../evaluation/README.md)'s RQ3 pipeline (see
[Research questions](../../README.md#research-questions)) can select among
them without any pipeline code changing. Decoupled from `evaluation`: the only internal
dependency is [`tasks.catalog.Task`](../tasks/README.md) (for `task.kind`),
so this package is reusable by any pipeline that can hand it a feature table in the
shared tabular contract below.

## Modules

| Module | Purpose |
|---|---|
| `common.py` | `xy_split` — shared train/test split with train-fit categorical encoding; `resolve_device` — shared CPU/CUDA device resolution for `gnn.py`/`lstm_torch.py`/`transformer.py` |
| `dispatch.py` | `PREDICTOR_REGISTRY` / `resolve` — maps a config-selectable predictor key to its `fit_and_score_fold` |
| `random_forest.py` | fixed `RandomForestClassifier`/`Regressor`, the default predictor |
| `xgboost.py` | `XGBClassifier`/`Regressor` |
| `lstm.py` | TensorFlow/Keras LSTM |
| `lstm_torch.py` | PyTorch LSTM |
| `transformer.py` | PyTorch Transformer encoder (causal-masked, positional encoding) |
| `gnn.py` | direct object-centric event-graph predictor over OCPA's `feature_storage` graphs |

## The `fit_and_score_fold` contract

Every registered predictor implements the same signature:

```python
def fit_and_score_fold(feats: dict, tt: pd.DataFrame, y_col: str,
                        task: Task, train_mask, test_mask, cfg,
                        timer=None) -> Dict[str, float]:
    ...
```

- `feats["feature_cols"]` names the feature columns of `tt` (a `features.ocpa
  .extract_feature_table` output table, or any table shaped the same way).
- `task.kind` (`"categorical"` / `"binary"` / a numeric kind) selects macro-F1
  (classification) or MAE (regression) scoring.
- `train_mask`/`test_mask` are boolean masks over `tt` (one fold of the
  persisted, per-log `CollaborationCase`-grouped split -- see
  `evaluation/README.md#rq3-protocol`).
- `timer` (optional): an object exposing `timer.stage(name)` as a context
  manager, used to separately time/profile `"fit"` and `"predict"` within
  this call (e.g. `with timer.stage("fit"): clf.fit(...)`). `evaluation`
  passes a bound `evaluation.profiling.StageTimer` when `cfg.profile=True`;
  any caller that doesn't pass one (including a direct unit-test call, see
  `tests/test_predictors_registry.py`) gets `predictors.common
  .NullStageTimer`, a local no-op that duck-types the same protocol --
  `predictors` never imports `evaluation` to get this, keeping the
  decoupling above intact.
- Returns a metrics dict, or `{}` when a fold is empty.

**Sequence order precondition (`lstm.py`/`lstm_torch.py`/`transformer.py`
only).** These three group `tt`'s rows by `case_id` (`pandas.groupby`,
preserving each row's position within its group) to build one sequence per
collaboration instance, and feed it to the model in that row order. Pairing
each prediction back to its true label is safe regardless of `tt`'s row
order -- both are sliced from `tt` by the same boolean mask, so they stay
positionally consistent with each other. What is **not** guarded is whether
that per-case row order matches the true cut-point order (`event_id`
ascending within a case): the LSTM's forward pass and the Transformer's
causal mask are only meaningful if sequence position tracks real time. `tt`
arrives pre-sorted by `(case_id, event_id)` only because
`features.ocpa.extract_feature_table` sorts it explicitly (locked in by
`tests/test_features_leakage.py::test_feature_table_row_order_is_deterministic`);
a caller assembling its own `tt` for these three predictors (e.g. the
"Bringing your own OCEL" pattern in
[tasks/README.md](../tasks/README.md#bringing-your-own-ocel)) must preserve
that order itself, since nothing here re-sorts or checks it -- an unsorted
`tt` trains and predicts without error, just against a temporally scrambled
sequence. `random_forest.py`/`xgboost.py`/`gnn.py` score each row
independently and are unaffected.

**Device selection (`gnn.py`/`lstm_torch.py`/`transformer.py` only).** All
three resolve `cfg.device` (`"auto"`/`"cpu"`/`"cuda"`, see
`ExperimentConfig.device`) via `predictors.common.resolve_device`.
`random_forest.py`/`xgboost.py` are CPU-only regardless (`xgboost.py` uses
`tree_method="hist"`, not a GPU histogram method). The default is `"cpu"`,
not `"auto"`: measured on this workload, CUDA's per-batch/per-graph dispatch
overhead made training slower wall-clock than plain CPU, both for GNN's tiny
per-sample subgraphs (`gnn_k_values` up to 16 nodes, batch size 32 -- GPU
utilization sampled at ~4-37% during training) and for LSTM's many short,
variable-length per-case sequences (where the same effect was already found
with MPS on Apple Silicon). Pass `--device cuda`/`--device auto` (CLI) or
`ExperimentConfig(device=...)` (programmatic) to override.

## Usage

```python
from predictors.dispatch import resolve

fit_and_score_fold = resolve("random_forest")   # or "xgboost"/"lstm"/"lstm_torch"/"transformer"/"gnn"
metrics = fit_and_score_fold(feats, table, y_col, task, train_mask, test_mask, cfg)
```

## Tests

`tests/test_predictors_registry.py` is a synthetic-table smoke test per registry
entry (`random_forest`, `lstm`, `lstm_torch`, `xgboost`, `transformer`) plus a
registry-presence check for `gnn`, which needs OCPA's `feature_storage` graphs
and so isn't exercised against the synthetic table.
