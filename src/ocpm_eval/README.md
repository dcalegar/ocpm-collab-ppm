# ocpm_eval — evaluation stages

Modular evaluation that answers RQ2–RQ4 of the study. Task definitions and
ground-truth labels come from the decoupled
[`ocpm_tasks`](../ocpm_tasks/README.md) library; `ocpm_eval` is a
*consumer* of that library, not part of it — it adds feature extraction,
model fitting, and the descriptive metrics/CSVs. Inputs are **OCEL 2.0
SQLite** logs (the format OCPA imports natively), produced by the
[`mapping`](../mapping/README.md) converter (RQ1, out of scope here).

## What this is (and isn't)

`ocpm_eval` is where an actual prediction happens — `ocpm_tasks` only
supplies the target `y`; here OCPA supplies the feature vector `X` and
scikit-learn fits/predicts/scores it. See
["Connecting to a concrete prediction with OCPA"](../ocpm_tasks/README.md#connecting-to-a-concrete-prediction-with-ocpa)
in the `ocpm_tasks` README for the minimal version of this join; this
package is the fuller, cross-validated version of the same pattern.

## Modules

| Module | Purpose |
|---|---|
| `config.py` | `ExperimentConfig`, `LogSpec` — log registry, CV/learner hyperparameters, output dir |
| `io_ocel.py` | `load_ocpa_ocel` / `read_ocel2_labels` — OCEL 2.0 SQLite → OCPA object (features) and → neutral model (labels), sharing one read path |
| `features_ocpa.py` | `extract_feature_table` — native OCPA past-relative features (RQ3), with the event-id alignment oracle |
| `models.py` | `fit_and_score_fold` — one fixed RandomForest per problem type, fit + **predict** + score, plus a trivial baseline |
| `rq2_fidelity.py` | RQ2 — label-fidelity: R1 (source XES) vs R2 (OCEL) equivalence for the 14 single-case tasks; internal consistency for X-PaL/X-MSt |
| `rq3_pipeline.py` | RQ3 — end-to-end feasibility: features + labels joined, 5-fold `GroupKFold` CV grouped by `CollaborationInstance` |
| `rq3_extensions_example.py` | worked example for X-PaL/X-MSt (object-enabled extensions, no single-case counterpart) — exploratory, kept out of the core coverage claim |
| `rq4_structure.py` | RQ4 — structural measures (object/relation counts, size, structural ratio) read directly via `sqlite3`, plus a fixed expressiveness matrix |
| `run_evaluation.py` | orchestrator — runs RQ2 → RQ3 (subset + full catalog) → RQ4 and writes all CSVs |

| Stage | Module | Output |
|---|---|---|
| **RQ2** label fidelity | `rq2_fidelity.py` | `results/rq2_fidelity.csv` |
| **RQ3** end-to-end feasibility (representative subset, in-paper) | `rq3_pipeline.py` | `results/rq3_results.csv` |
| **RQ3** full catalog (supplementary coverage, 14 tasks × 4 logs) | `rq3_pipeline.py` via `run_evaluation.py` | `results/rq3_results_full.csv` |
| **RQ3** X-PaL/X-MSt worked example (exploratory) | `rq3_extensions_example.py` | `results/rq3_extensions_example.csv` |
| **RQ4** structure & expressiveness | `rq4_structure.py` | `results/rq4_structure.csv`, `results/rq4_expressiveness.csv` |
| RQ1 transformation + P1 | — | [`mapping`](../mapping/README.md) (separate tool) |

## Reading path

Both sides read the **same** OCEL 2.0 SQLite file, through two different
readers, because OCPA pins `pm4py==2.2.32`, which predates OCEL 2.0 and
cannot read it:

- **Label side** (`io_ocel.read_ocel2_labels`): `ocpm_tasks.adapters
  .from_ocel2_sqlite` (stdlib `sqlite3`) by default, or `from_ocpa` when
  an already-loaded OCPA object is passed in (to avoid re-parsing).
- **Feature side** (`io_ocel.load_ocpa_ocel`): OCPA's native
  `ocpa.objects.log.importer.ocel2.sqlite` importer, with **leading-type**
  process-execution extraction (`leading_type=CollaborationInstance`) so
  each execution is one collaboration instance — the default
  "connected components" extraction would merge instances that share a
  `Participant` object. Before handing the file to OCPA,
  `_strip_actor_e2o` removes the redundant `actor` E2O edge that the
  `mapping` converter adds for export-compatibility (see
  [mapping's README](../mapping/README.md#mapping-rules-m1m8)); left in
  place, that edge would merge nearly every execution in the log.
- **Alignment**: OCPA feature rows are matched to `ocpm_tasks` label rows
  by `event_id` (`feature_storage.feature_graphs → node.event_id`),
  validated by a remaining-time oracle (OCPA's own remaining-time feature
  must equal the `NV-PrT` label) that raises on any mismatch —
  `features_ocpa.py`.

## Usage

```bash
# full evaluation (RQ2 + RQ3 + RQ4) — RQ3 requires OCPA installed; run from
# the repo root with .venv active. Writes CSVs to results/.
python -m ocpm_eval.run_evaluation
```

Run a single stage, or drive it programmatically:

```python
from ocpm_eval.config import ExperimentConfig
from ocpm_eval.rq3_pipeline import run_rq3

cfg = ExperimentConfig()          # edit logs/hyperparameters here, or pass overrides
df = run_rq3(cfg)                 # -> pandas.DataFrame, also written to results/rq3_results.csv
```

There is currently no automated test suite for `ocpm_tasks`/`ocpm_eval`; validate a
change by running the evaluation above and inspecting the `results/*.csv` outputs
(RQ2 fidelity should be ~1.0 on the consistency checks; RQ3 rows should have
`ran_end_to_end=True`).

Point the evaluation at your own logs by editing the registry in
`config.py` (`LogSpec(name, ocel_path, xes_path)` — `ocel_path` is the
OCEL 2.0 `.sqlite` from `mapping`, `xes_path` is only needed for RQ2's R1
comparison).

## RQ3 protocol

Representative subset (`ocpm_tasks.catalog.RQ3_SUBSET`): `NE-NPaA`,
`NE-NMPr`, `NV-PrT`, `NV-PaT`, `NV-NMPr`, `OB-M` — one task per anchor ×
problem-type combination. For each log × task: extract OCPA features,
compute R2 labels, join by `event_id`, then 5-fold `GroupKFold` CV
**grouped by `CollaborationInstance`** (all prefixes of one case stay in a
single fold, so no prefix leaks between train and test). Reports macro-F1
(classification) or MAE (regression) as mean ± sd over folds, alongside a
trivial baseline (majority class / median) computed on the same folds. No
computation times are reported (V4 profile).

`run_rq3` is task-agnostic (it just loops `cfg.rq3_tasks`), so the same
protocol extends to the full catalog without any pipeline changes —
`run_evaluation.main` also runs it over `ocpm_tasks.catalog.EQUIVALENCE_TASKS`
(the 14 single-case-counterpart tasks) and writes `rq3_results_full.csv`,
intended as supplementary material rather than an in-paper table: it
confirms all (task, log) combinations run end-to-end, but a few of the
non-curated tasks (e.g. `NV-TNE`, `NV-TNM`) land close to or slightly worse
than their trivial baseline in some logs — unlike the subset, which was
picked to show clear separation, the full catalog is a coverage check, not
a predictive-quality claim. The two object-enabled extensions (`X-PaL`,
`X-MSt`, no single-case counterpart) are deliberately left out of this run
and demonstrated separately in `rq3_extensions_example.py`, since the
bundled logs barely exercise asynchrony and would make `X-MSt` (and
especially `X-PaL`, which came out exactly `0.00±0.00` MAE in all four
logs — no cross-instance participant overlap in these logs) close to
degenerate.

## RQ2 protocol

For the 14 single-case-counterpart tasks: compare Θ_τ^L (labels computed
directly from the source XES via `rq2_fidelity._src_label`, no
intermediate object-centric model) against Θ_τ (labels computed from the
OCEL 2.0 log via `ocpm_tasks.labels.compute_label_rows`), row-aligned by
`(case_id, k)`. Proposition P2 (see the paper) guarantees these are equal;
empirical agreement ≈ 1.0 is the expected/verifying outcome, not a
tunable metric. For X-PaL/X-MSt (no single-case counterpart) the check is
internal well-definedness of the OCEL-side labels only (non-negative
counts/times, BOTTOM only where the definition allows it).

RQ2 *equivalence* needs the original XES (R1); pass `xes_path` in each
`LogSpec`. The bundled example logs in `data/logs/` include both the
`.xes` source and its converted `.sqlite`, so this runs out of the box.
