# ocpm_tasks — object-centric collaborative prediction-task library

Standalone library for the 14 reformulated prediction tasks over collaborative
process logs: task definitions, ground-truth label functions, and a neutral
object-centric model they operate on. It has **no dependency on `mapping` or
`ocpm_eval`** — only `pandas` at import time (`pm4py`/`ocpa` are imported lazily,
only if you use the corresponding adapter).

Beyond this 14-task taxonomy, two further "object-enabled" extension tasks,
**`X-Inf`** (in-flight message backlog) and **`X-MSt`** (message synchronization
time), are formalized in `extensions.py` — see "Object-enabled extension tasks"
below. Other exploratory directions (cross-case participant load, message-
convergence completion, inter-participant progress lag, intra-projection
orchestration concurrency) remain unimplemented. `X-MSt` presupposes a
correspondence between individual send and receive observations that the core
mapping (rule M4) deliberately does not establish; `extensions.py`/`adapters.py`
recover it via an explicit, opt-in enrichment step, not a change to the core
mapping.

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

* `catalog.TASKS` — metadata for each of the 14 tasks (anchor object,
  problem type, value kind, parameterization).
* `labels.compute_label_rows` — for every cut point `k` in a collaboration
  instance, deterministically derives the target value `y` by looking
  *forward* from that cut point (next activity, time to next message,
  message count, etc.). Pure functions: no ML, no I/O, no OCEL library.
* `fidelity` — RQ2-style label-equivalence check (agreement with a
  single-case reference).

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
| `catalog` | `TASKS` — the 14 `Task` definitions (anchor object, problem type, value kind) |
| `labels` | `LabelContext`, `build_context`, `compute_label_rows` — ground-truth label computation |
| `extensions` | `EXT_TASKS` (`X-Inf`, `X-MSt`), `compute_ext_label_rows` — object-enabled extension tasks, kept out of `catalog.TASKS`/`EQUIVALENCE_TASKS`/`RQ3_SUBSET` |
| `fidelity` | `compare_equivalence` — label-equivalence comparator (optional, for validating a mapping) |
| `adapters` | `from_pm4py`, `from_ocpa`, `from_ocel2_sqlite`, `build_from_relations` — build an `ObjectCentricLog` from a concrete OCEL; optional `corr_attr` enrichment for `X-MSt` |

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
(`NE-NMPa`, `NV-PaT`, `NV-NMPa`, `OB-P`, `OB-M`); pass it via
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

## Object-enabled extension tasks (`extensions.py`)

`X-Inf` and `X-MSt` — two "object-enabled" extension targets beyond the 14-task
taxonomy — are formalized here as label functions over the same `model.Execution`,
using the same `LabelContext`/BOTTOM convention as `labels.py`.
They are deliberately **not** part of `catalog.TASKS`/`EQUIVALENCE_TASKS`/
`RQ3_SUBSET`, so importing/using them never mixes into the 14-task RQ2/RQ3
evaluation:

* **`X-Inf`** (`CollaborationCase` anchor, count regression) — the peak, over the
  case's remainder after the cut, of the *running* send/receive balance (+1 per
  send, −1 per receive, clamped at 0 as it runs — not a plain `#send − #receive`
  difference; see `in_flight_trajectory`). Needs **no** send/receive correlation.
* **`X-MSt`** (`Message` anchor, time regression) — the latency between the next
  send after the cut and its matching receive; BOTTOM if that send is unmatched
  (still in flight) or if no correlation id is available. Needs `Event.corr_id`,
  which only an explicit **enrichment** populates (see `adapters.corr_attr`
  below) — the core mapping (M4) never infers it.

```python
from ocpm_tasks.adapters import from_ocel2_sqlite
from ocpm_tasks.extensions import EXT_TASKS, compute_ext_label_rows

# corr_attr: opt-in enrichment. A residual event attribute (e.g. "msgId") that
# carries a native message-correlation id; None (default) leaves every
# Event.corr_id unset, exactly like the unenriched core mapping.
log = from_ocel2_sqlite("data/logs/ToyCollab/toy_collab.sqlite", corr_attr="msgId")

for key in ("X-Inf", "X-MSt"):
    rows = compute_ext_label_rows(log, EXT_TASKS[key])
    # rows: List[(case_id, event_id, k, y)], BOTTOM rows dropped by default
```

Both extensions are demonstrated end to end — including the same OCPA feature
extraction and grouped cross-validation used for the 14 reformulated tasks — on a
**synthetic toy log** (`src/mapping/support/build_toy_collab_log.py` → 
`data/logs/ToyCollab/toy_collab.xes`: 100 cases, 1,132 events, 3 participants, designed to exercise 
both targets with variable in-flight backlogs and explicit `msgId` correlation ids, tied to
case participant count so the targets are genuinely learnable from the observed prefix), 
converted with the same `collab_xes_to_ocel.py` converter as the four study logs, via
`ocpm_eval/rq_ext_pipeline.py` (results in `data/results/rq_ext_results_toy.csv`).
A dedicated pure-Python unit test, `tests/test_extensions_toy.py`, verifies label logic 
by hand on specific patterns with intentional in-flight and unmatched sends.
