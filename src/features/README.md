# features — object-centric feature extraction

Reads an OCEL 2.0 SQLite log and turns it into the native OCPA feature table used
by RQ3 (see [Research questions](../../README.md#research-questions)).
Decoupled from [`evaluation`](../evaluation/README.md): it depends only
on [`tasks`](../tasks/README.md) (for the neutral model and `Schema`) and
OCPA itself, so it can be reused by any pipeline that needs OCPA features or the
same OCEL-reading path, not just the RQ2/RQ3 orchestrator.

## Modules

| Module | Purpose |
|---|---|
| `io_ocel.py` | `load_ocpa_ocel` / `read_ocel2_labels` — OCEL 2.0 SQLite → OCPA object (features) and → neutral model (labels), sharing one read path |
| `ocpa.py` | `extract_feature_table` — native OCPA past-relative event features (RQ3), with the event-id alignment oracle |

## Reading path

Both sides read the **same** OCEL 2.0 SQLite file, through two different readers,
because OCPA pins `pm4py==2.2.32`, which predates OCEL 2.0 and cannot read it:

- **Label side** (`io_ocel.read_ocel2_labels`): `tasks.adapters
  .from_ocel2_sqlite` (stdlib `sqlite3`) by default, or `from_ocpa` when
  an already-loaded OCPA object is passed in (to avoid re-parsing).
- **Feature side** (`io_ocel.load_ocpa_ocel`): OCPA's native
  `ocpa.objects.log.importer.ocel2.sqlite` importer, with **leading-type**
  process-execution extraction (`leading_type=CollaborationCase`) so
  each execution is one collaboration case — the default
  "connected components" extraction would merge instances that share a
  participant object. Before handing the file to OCPA,
  `_strip_participant_e2o` thins out the E2O edges whose target is a
  log-wide object shared across collaboration cases: the direct
  `in_participant` edge the `mapping` converter adds (a genuine M6
  relation, see [mapping's README](../mapping/README.md#mapping-rules-m1m8))
  and the D25 export witnesses, which reuse the `from`/`to` qualifiers at
  the E2O level. Left in place, OCPA's pairwise E2O connection would merge
  nearly every execution in the log; one witness row per object is kept so
  OCPA's importer still registers the object at all. Then
  `_break_timestamp_ties` nudges any within-case timestamp tie by whole
  microseconds, since OCPA's positional features cut the prefix on
  `timestamp <=` rather than on the source order.
- **Alignment**: OCPA feature rows are matched to `tasks` label rows
  by `event_id` (`feature_storage.feature_graphs → node.event_id`),
  validated by a remaining-time oracle (OCPA's own remaining-time feature
  must equal the `NV-PrT` label) that raises on any mismatch —
  `ocpa.py`.

## Usage

```python
from tasks.schema import Schema
from features.io_ocel import load_ocpa_ocel, read_ocel2_labels
from features.ocpa import extract_feature_table

schema = Schema()
ocpa_ocel = load_ocpa_ocel(schema, "log.sqlite")
ocel_log = read_ocel2_labels("log.sqlite", schema, ocpa_ocel=ocpa_ocel)
feats = extract_feature_table("log", schema, "log.sqlite", ocel_log, ocpa_ocel=ocpa_ocel)
# feats["table"], feats["feature_cols"], feats["feature_storage"]
```

Consumed by `evaluation.rq3_pipeline`, which pairs
`feats["table"]` with a `predictors.dispatch` predictor and a persisted,
per-log `CollaborationCase`-grouped fold split (see
`evaluation/README.md#rq3-protocol`).

## Tests

`tests/test_io_ocel_strip.py`, `tests/test_io_ocel_ties.py` (pure stdlib `sqlite3`,
no OCPA needed) and `tests/test_features_leakage.py` (needs the OCPA `.venv`, see
its module docstring for the exact invocation) cover this package directly.
