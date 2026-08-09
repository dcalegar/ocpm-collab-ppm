# ocpm_eval — evaluation stages

Modular evaluation that answers RQ2–RQ3 of the study. Task definitions and
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
| `predictors/` | one fit-and-score module per learner (`random_forest.py`, `lstm.py`, etc.), behind `dispatch.PREDICTOR_REGISTRY` keyed by `ExperimentConfig.predictor` — fit + **predict** + score, plus a trivial baseline |
| `rq2_fidelity.py` | RQ2 — label-fidelity: R1 (source XES) vs R2 (OCEL) equivalence for the 14 tasks |
| `rq3_pipeline.py` | RQ3 — end-to-end feasibility: features + labels joined, 5-fold `GroupKFold` CV grouped by `CollaborationCase` |
| `run_evaluation.py` | orchestrator — runs the requested (rq, log_group, rq3_scope) combinations and writes all CSVs |

| Stage | Module | Output |
|---|---|---|
| **RQ2** label fidelity | `rq2_fidelity.py` | `results/rq2_fidelity.csv` |
| **RQ3** end-to-end feasibility (representative subset) | `rq3_pipeline.py` | `results/rq3_results_{predictor}.csv` (e.g. `rq3_results_random_forest.csv`) |
| **RQ3** full catalog (supplementary coverage, 14 tasks × 4 logs) | `rq3_pipeline.py` via `run_evaluation.py` (`rq3_scopes=("partial","full")`) | `results/rq3_results_{predictor}_full.csv` |
| RQ1 transformation + P1 | — | [`mapping`](../mapping/README.md) (separate tool) |
| **RQ2/RQ3 on BPI2013** (opt-in real-world validation, see below) | `rq2_fidelity.py`/`rq3_pipeline.py` via `run_evaluation.py` (`log_groups=("bpi2013",)`) | `results/rq2_fidelity_bpi2013.csv`, `results/rq3_results_{predictor}_bpi2013.csv` |

## Reading path

Both sides read the **same** OCEL 2.0 SQLite file, through two different
readers, because OCPA pins `pm4py==2.2.32`, which predates OCEL 2.0 and
cannot read it:

- **Label side** (`io_ocel.read_ocel2_labels`): `ocpm_tasks.adapters
  .from_ocel2_sqlite` (stdlib `sqlite3`) by default, or `from_ocpa` when
  an already-loaded OCPA object is passed in (to avoid re-parsing).
- **Feature side** (`io_ocel.load_ocpa_ocel`): OCPA's native
  `ocpa.objects.log.importer.ocel2.sqlite` importer, with **leading-type**
  process-execution extraction (`leading_type=CollaborationCase`) so
  each execution is one collaboration case — the default
  "connected components" extraction would merge instances that share a
  `Participant` object. Before handing the file to OCPA,
  `_strip_participant_e2o` removes the direct `participant` E2O edge that
  the `mapping` converter adds (a genuine M6 relation, see
  [mapping's README](../mapping/README.md#mapping-rules-m1m8)); left in
  place, OCPA's pairwise E2O connection would merge nearly every
  execution in the log.
- **Alignment**: OCPA feature rows are matched to `ocpm_tasks` label rows
  by `event_id` (`feature_storage.feature_graphs → node.event_id`),
  validated by a remaining-time oracle (OCPA's own remaining-time feature
  must equal the `NV-PrT` label) that raises on any mismatch —
  `features_ocpa.py`.

## Usage

```bash
# default evaluation: RQ2 + RQ3 (partial scope) on the four Predict-Collab
# study logs, + EXT on the toy log — WITHOUT BPI2013 or the RQ3 full
# catalog. RQ3 requires OCPA installed; run from the repo root with .venv
# active. Writes CSVs to results/.
python -m ocpm_eval.run_evaluation
```

Same command on Windows, once `.venv` is active (PowerShell:
`.venv\Scripts\Activate.ps1`; `cmd.exe`: `.venv\Scripts\activate.bat`). See
the root [README.md](../../README.md#setup-virtual-environments) for the
full macOS/Linux + Windows setup of `.venv`.

Run a single stage, or drive it programmatically:

```python
from ocpm_eval.config import ExperimentConfig
from ocpm_eval.rq3_pipeline import run_rq3

cfg = ExperimentConfig()          # edit logs/hyperparameters here, or pass overrides
df = run_rq3(cfg)                 # -> pandas.DataFrame, also written to results/rq3_results_random_forest.csv
```

`ocpm_tasks` has automated regression tests in the repo's `tests/` directory
(`test_extensions_toy.py` for the X-Inf/X-MSt extension tasks, plus
`test_mapping_checks.py` for the converter's consistency checks); run them
directly with `python tests/<file>.py` (also pytest-compatible). `ocpm_eval`
itself has no dedicated unit tests; validate a change here by running the
evaluation above and inspecting the `results/*.csv` outputs (RQ2 fidelity's
`agreement` column should be ~1.0; RQ3 rows should have `ran_end_to_end=True`).

Point the evaluation at your own logs by editing the registry in
`config.py` (`LogSpec(name, ocel_path, xes_path)` — `ocel_path` is the
OCEL 2.0 `.sqlite` from `mapping`, `xes_path` is only needed for RQ2's R1
comparison).

### Selecting stages individually

`run_evaluation.main` exposes three independent axes plus one unrelated flag:

- `rqs: Iterable["RQ2"|"RQ3"] = ("RQ2", "RQ3")`
- `log_groups: Iterable["predictcollab"|"bpi2013"] = ("predictcollab",)`
- `rq3_scopes: Iterable["partial"|"full"] = ("partial",)` — RQ3 only, ignored for RQ2
- `predictor: Optional[str] = None` — key into `predictors.dispatch.PREDICTOR_REGISTRY`
  (`"random_forest"`, `"xgboost"`, `"lstm"`, `"lstm_torch"`, or `"gnn"`);
  `None` leaves
  `cfg.predictor` (also default `"random_forest"`) untouched, so it only
  overrides when explicitly passed. Feeds the `rq3_results_{predictor}*.csv`
  filenames.
- `run_rq_ext: bool = True` — EXT on the toy log; independent of the three axes above (see below)

There is no CLI flag for this yet — enable/disable stages by calling `main`
programmatically:

```python
from ocpm_eval.run_evaluation import main

main()                                                    # default: Predict-Collab, RQ2+RQ3 partial, + EXT
main(rq3_scopes=("partial", "full"))                      # + RQ3 full catalog on Predict-Collab
main(log_groups=("predictcollab", "bpi2013"))              # + RQ2/RQ3 on BPI2013 as well
main(log_groups=("bpi2013",), run_rq_ext=False)            # BPI2013 only (skips the four study logs)
main(log_groups=("bpi2013",), rq3_scopes=("partial", "full"),
     run_rq_ext=False)                                     # BPI2013 (subset + full catalog) only
```

### With BPI2013 (opt-in real-world validation)

BPI2013 (`data/logs/BPIChallenge2013/`, registered in
`config.py::real_world_ocel_logs()`) is ~29.5x larger than the largest study log
by events (7,554 cases / 69,584 OCEL events vs. Artificial5's 100 cases /
2,360 events), so its
OCPA feature extraction + RandomForest fitting time is significantly longer;
`"bpi2013"` is opt-in (not in `log_groups`'s default) for that reason. It does
not share provenance with the four study logs reused from Delgado et al.
(2025), so it always runs as a separate stage against a separate config
(`replace(cfg, logs=real_world_ocel_logs())`) and writes to separate CSVs
(`rq2_fidelity_bpi2013.csv`, `rq3_results_random_forest_bpi2013.csv`), never
mixing into `rq2_fidelity.csv`/`rq3_results*.csv`.

See [data/logs/README.md](../../data/logs/README.md#bpichallenge2013--real-life-application-log)
for the log's provenance and the synthesized-`SendTask` design, and
`data/logs/BPIChallenge2013/collab_convert.py` for the full conversion logic.

## RQ3 protocol

Representative subset (`ocpm_tasks.catalog.RQ3_SUBSET`): `NE-NPaA`,
`NE-NMPr`, `NV-PrT`, `NV-PaT`, `NV-NMPr`, `OB-M` — one task per anchor ×
problem-type combination. For each log × task: extract OCPA features,
compute R2 labels, join by `event_id`, then 5-fold `GroupKFold` CV
**grouped by `CollaborationCase`** (all prefixes of one case stay in a
single fold, so no prefix leaks between train and test). Reports macro-F1
(classification) or MAE (regression) as mean ± sd over folds, alongside a
trivial baseline (majority class / median) computed on the same folds. No
computation times are reported (V4 profile).

`run_rq3` is task-agnostic (it just loops `cfg.rq3_tasks`), so the same
protocol extends to the full catalog without any pipeline changes —
passing `rq3_scopes=("partial", "full")` to `run_evaluation.main` also runs it
over `ocpm_tasks.catalog.EQUIVALENCE_TASKS` (all 14 tasks) and writes
`rq3_results_random_forest_full.csv`, intended as supplementary coverage rather than a
curated result: it confirms all (task, log)
combinations run end-to-end, but a few of the non-curated tasks (e.g.
`NV-TNE`, `NV-TNM`) land close to or slightly worse than their trivial
baseline in some logs — unlike the subset, which was picked to show clear
separation, the full catalog is a coverage check, not a predictive-quality
claim. Beyond the 14-task catalog, `ocpm_tasks/extensions.py` implements two
further object-enabled extension tasks (X-Inf, X-MSt) kept out of
`catalog.TASKS` — see the EXT section below. X-MSt in particular
presupposes a send/receive correspondence that the core mapping (rule M4)
deliberately does not establish, so it needs an enrichment step beyond the
current converter.

## RQ2 protocol

For the 14 tasks: compare Θ_τ^L (labels computed directly from the source
XES via `rq2_fidelity._src_label`, no intermediate object-centric model)
against Θ_τ (labels computed from the OCEL 2.0 log via
`ocpm_tasks.labels.compute_label_rows`), row-aligned by `(case_id, k)`.
The mapping preserves event order and label semantics by construction, so
these are expected to be equal; empirical agreement ≈ 1.0 is the
expected/verifying outcome, not a tunable metric.

RQ2 *equivalence* needs the original XES (R1); pass `xes_path` in each
`LogSpec`. The bundled example logs in `data/logs/` include both the
`.xes` source and its converted `.sqlite`, so this runs out of the box.
