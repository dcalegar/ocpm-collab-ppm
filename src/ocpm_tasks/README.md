# ocpm_tasks — object-centric collaborative prediction-task library

Standalone library for the 16 reformulated prediction tasks over collaborative
process logs: task definitions, ground-truth label functions, and a neutral
object-centric model they operate on. It has **no dependency on `mapping` or
`ocpm_eval`** — only `pandas` at import time (`pm4py`/`ocpa` are imported lazily,
only if you use the corresponding adapter).

## Install

Copy or vendor the `ocpm_tasks` package into your project (it's a plain
directory with no package-specific dependencies beyond pandas), or depend on
this repo and import it as `ocpm_tasks`.

```bash
pip install pandas
```

## What this library does (and does not)

`ocpm_tasks` only supplies the **ground-truth / task-definition side** of a
prediction pipeline — it never trains or runs a model:

* `catalog.TASKS` — metadata for each of the 16 tasks (anchor object,
  problem type, value kind, parameterization).
* `labels.compute_label_rows` — for every cut point `k` in a collaboration
  instance, deterministically derives the target value `y` by looking
  *forward* from that cut point (next activity, time to next message,
  message count, etc.). Pure functions: no ML, no I/O, no OCEL library.
* `fidelity` — RQ2-style label-fidelity checks (agreement with a
  single-case reference; internal well-definedness of the object-enabled
  tasks).

There is no notion here of a feature vector, a training loop, or a model.
To actually predict something you still need, from elsewhere: (1) a
**feature extractor** that turns the observable prefix at each cut point
into `X` (e.g. OCPA's `predictive_monitoring` module, pm4py, or your own),
and (2) a **learner** (e.g. scikit-learn) that fits `X -> y` and evaluates
it. See "Connecting to a concrete prediction with OCPA" below for how the
pieces line up.

## Modules

| Module | Purpose |
|---|---|
| `schema` | `Schema` — object types / E2O / O2O qualifier names, overridable if your OCEL uses different vocabulary |
| `model` | Neutral structures the tasks read: `Event`, `Execution`, `ObjectCentricLog` |
| `catalog` | `TASKS` — the 16 `Task` definitions (anchor object, problem type, value kind) |
| `labels` | `LabelContext`, `build_context`, `compute_label_rows` — ground-truth label computation |
| `fidelity` | `compare_equivalence` / `check_consistency` — label-fidelity comparators (optional, for validating a mapping) |
| `adapters` | `from_pm4py`, `from_ocpa`, `from_ocel2_sqlite`, `build_from_relations` — build an `ObjectCentricLog` from a concrete OCEL |

## Usage

Build the neutral model with one of the adapters, then compute labels for any task:

```python
from ocpm_tasks.adapters import from_ocel2_sqlite
from ocpm_tasks.catalog import TASKS
from ocpm_tasks.labels import build_context, compute_label_rows

log = from_ocel2_sqlite("my_log.sqlite")   # or from_pm4py(...) / from_ocpa(...)
ctx = build_context(log)

task = TASKS["NE-NMPr"]
rows = compute_label_rows(log, task, ctx=ctx)
# rows: List[(case_id, event_id, k, y)] — the ground-truth y at each cut point k
```

`param` (participant name or activity label) is required by parameterized tasks
(`NE-NMPa`, `NV-PaT`, `NV-NMPa`, `OB-P`, `OB-M`, `X-PaL`); pass it via
`compute_label_rows(log, task, param=..., ctx=ctx)`.

If your OCEL uses different object-type/qualifier names, pass a custom
`Schema(...)` to the adapter instead of editing this library.

## Connecting to a concrete prediction with OCPA

`ocpm_tasks` produces the target `y`; OCPA can supply both the parsed log
and the per-cut-point feature vector `X`. The two line up on `event_id`:

```python
from ocpa.objects.log.importer.ocel2.sqlite import factory as ocel_import
from ocpa.algo.predictive_monitoring import factory as predictive_monitoring, tabular

from ocpm_tasks.adapters import from_ocpa
from ocpm_tasks.catalog import TASKS
from ocpm_tasks.labels import build_context, compute_label_rows

# 1. Parse once with OCPA (native OCEL 2.0 import).
ocpa_ocel = ocel_import.apply("my_log.sqlite")

# 2. Build the neutral ocpm_tasks model from the SAME parsed log, so
#    case/event ids line up with OCPA's.
log = from_ocpa(ocpa_ocel)
ctx = build_context(log)

# 3. Features (X): OCPA's predictive_monitoring computes one row per event.
#    Use only PAST-relative features (elapsed time, preceding activities,
#    previously-seen object counts) -- anything execution-based would leak
#    the future past the cut point.
feature_set = [
    (predictive_monitoring.EVENT_ELAPSED_TIME, ()),
    (predictive_monitoring.EVENT_PRECEDING_ACTIVITIES, ("Send",)),
]
feature_storage = predictive_monitoring.apply(ocpa_ocel, feature_set, [])
table = tabular.construct_table(feature_storage)
table["event_id"] = [str(n.event_id) for fg in feature_storage.feature_graphs
                      for n in fg.nodes]
case_by_event_id = {str(e.event_id): ex.case_id for ex in log for e in ex.events}
table["case_id"] = table["event_id"].map(case_by_event_id)

# 4. Target (y): ocpm_tasks labels, joined by event_id.
task = TASKS["NE-NMPr"]
labels = {str(eid): y for (_case, eid, _k, y)
          in compute_label_rows(log, task, ctx=ctx)}
table["y"] = table["event_id"].map(labels)
table = table.dropna(subset=["y"])

# 5. Any learner works from here -- table's feature columns are X,
#    table["y"] is the target. Split by case_id (not by row) so prefixes of
#    the same collaboration instance don't leak across train/test.
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

case_ids = table["case_id"].unique()
train_cases = set(case_ids[: int(0.8 * len(case_ids))])
train = table[table["case_id"].isin(train_cases)]
test = table[~table["case_id"].isin(train_cases)]

feature_cols = [c for c in table.columns if c not in ("event_id", "case_id", "y")]
clf = RandomForestClassifier().fit(train[feature_cols], train["y"])

pred = clf.predict(test[feature_cols])          # <-- the actual prediction
print(f1_score(test["y"], pred, average="macro"))
```

For a fuller worked version of this pattern — including the alignment
oracle (checking OCPA's own remaining-time feature against the NV-PrT
label to catch id/partitioning mismatches), grouped cross-validation by
collaboration instance, and a trivial baseline — see
`ocpm_eval/features_ocpa.py`, `ocpm_eval/models.py` and
`ocpm_eval/rq3_pipeline.py` in this repo. They are *consumers* of
`ocpm_tasks` (not part of it) and show the full RQ3 pipeline end to end.

## Bringing your own OCEL

No adapter fits your source? Build a `ObjectCentricLog` directly with
`adapters.build_from_relations`, or construct `model.Event`/`model.Execution`
objects yourself — the label functions only depend on the `model` module, not
on any OCEL library.
