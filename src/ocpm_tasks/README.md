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

## Bringing your own OCEL

No adapter fits your source? Build a `ObjectCentricLog` directly with
`adapters.build_from_relations`, or construct `model.Event`/`model.Execution`
objects yourself — the label functions only depend on the `model` module, not
on any OCEL library.
