# ocpm_eval — evaluation stages

Modular evaluation that answers RQ2–RQ3 of the study. Task definitions and
ground-truth labels come from the decoupled
[`ocpm_tasks`](../ocpm_tasks/README.md) library; feature extraction and OCEL
reading come from the decoupled [`features`](../features/README.md) library;
model fitting comes from the decoupled [`predictors`](../predictors/README.md)
library. `ocpm_eval` is a *consumer* of all three, not part of them — it wires
them together into the RQ2/RQ3 pipelines and writes the descriptive
metrics/CSVs. Inputs are **OCEL 2.0 SQLite** logs (the format OCPA imports
natively), produced by the [`mapping`](../mapping/README.md) converter (RQ1,
out of scope here).

## What this is (and isn't)

`ocpm_eval` is where an actual prediction happens — `ocpm_tasks` only
supplies the target `y`; `features` supplies the feature vector `X` via OCPA,
and `predictors` fits/predicts/scores it. See
["Connecting to a concrete prediction with OCPA"](../ocpm_tasks/README.md#connecting-to-a-concrete-prediction-with-ocpa)
in the `ocpm_tasks` README for the minimal version of this join; this
package is the fuller, cross-validated version of the same pattern.

## Modules

| Module | Purpose |
|---|---|
| `config.py` | `ExperimentConfig`, `LogSpec` — log registry, CV/learner hyperparameters, output dir |
| `rq2_fidelity.py` | RQ2 — label-fidelity: R1 (source XES) vs R2 (OCEL) equivalence for the 14 tasks |
| `rq3_pipeline.py` | RQ3 — end-to-end feasibility: `features.ocpa` features + `ocpm_tasks` labels joined, 5-fold `GroupKFold` CV grouped by `CollaborationCase`, scored by a `predictors` predictor |
| `run_evaluation.py` | orchestrator — runs the requested (rq, log_group, rq3_scope) combinations and writes all CSVs |

Feature extraction (`load_ocpa_ocel`/`read_ocel2_labels`/`extract_feature_table`)
and predictor selection (`PREDICTOR_REGISTRY`/`resolve`) now live in
[`features`](../features/README.md) and [`predictors`](../predictors/README.md)
respectively — see those packages' READMEs for their modules, contracts and
reading path. They are reusable independently of this orchestrator.

| Stage | Module | Output |
|---|---|---|
| **RQ2** label fidelity | `rq2_fidelity.py` | `results/rq2_fidelity_predictcollab.csv` |
| **RQ3** end-to-end feasibility (representative subset) | `rq3_pipeline.py` | `results/rq3_results_{predictor}_predictcollab.csv` (e.g. `rq3_results_random_forest_predictcollab.csv`) |
| **RQ3** full catalog (supplementary coverage, 14 tasks × 4 logs) | `rq3_pipeline.py` via `run_evaluation.py` (`rq3_scopes=("partial","full")`) | `results/rq3_results_{predictor}_predictcollab_full.csv` |
| RQ1 transformation + P1 | — | [`mapping`](../mapping/README.md) (separate tool) |
| **RQ2/RQ3 on BPI2013** (opt-in real-world validation, see below) | `rq2_fidelity.py`/`rq3_pipeline.py` via `run_evaluation.py` (`log_groups=("bpi2013",)`) | `results/rq2_fidelity_bpi2013.csv`, `results/rq3_results_{predictor}_bpi2013.csv` |

The log group (`predictcollab`/`bpi2013`) always appears in the output
filename, so results stay self-describing as more predictors are added — no
predictcollab file is left nameless just because it's the default group.

## Reading path

Both the label side and the feature side read the **same** OCEL 2.0 SQLite
file, through two different readers (OCPA pins `pm4py==2.2.32`, which
predates OCEL 2.0 and cannot read it), aligned by `event_id` and validated by
a remaining-time oracle — see [`features`](../features/README.md#reading-path)
for the full explanation (`load_ocpa_ocel`/`read_ocel2_labels`/
`extract_feature_table`).

## Usage

```bash
# default evaluation: RQ2 + RQ3 (partial scope) on the four Predict-Collab
# study logs — WITHOUT BPI2013 or the RQ3 full catalog. RQ3 requires OCPA
# installed; run from the repo root with .venv active. Writes CSVs to results/.
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
df = run_rq3(cfg)                 # -> pandas.DataFrame, also written to results/rq3_results_random_forest_predictcollab.csv
```

`ocpm_tasks` has automated regression tests in the repo's `tests/` directory
(`test_mapping_checks.py` for the converter's consistency checks); run them
directly with `python tests/<file>.py` (also pytest-compatible). `ocpm_eval`
has `test_predictors_registry.py`, a synthetic-table smoke test per
`predictors.dispatch.PREDICTOR_REGISTRY` entry, but no end-to-end pipeline
tests; validate a pipeline change by running the evaluation above and
inspecting the `results/*.csv` outputs (RQ2 fidelity's `agreement` column
should be ~1.0; RQ3 rows should have `ran_end_to_end=True`).

Point the evaluation at your own logs by editing the registry in
`config.py` (`LogSpec(name, ocel_path, xes_path)` — `ocel_path` is the
OCEL 2.0 `.sqlite` from `mapping`, `xes_path` is only needed for RQ2's R1
comparison).

### Selecting stages individually

`run_evaluation.main` exposes four independent axes:

- `rqs: Iterable["RQ2"|"RQ3"] = ("RQ2", "RQ3")`
- `log_groups: Iterable["predictcollab"|"bpi2013"] = ("predictcollab",)`
- `rq3_scopes: Iterable["partial"|"full"] = ("partial",)` — RQ3 only, ignored for RQ2
- `predictors: Iterable[str] = ("random_forest",)` — keys into
  `predictors.dispatch.PREDICTOR_REGISTRY` (`"random_forest"`, `"xgboost"`,
  `"lstm"`, `"lstm_torch"`, `"gnn"`); RQ3 only (RQ2 has no predictor). RQ3 runs
  once per `(log_group, predictor, scope)` combination, each to its own
  `rq3_results_{predictor}*.csv`; RQ2 runs once per `log_group` regardless of
  how many predictors are requested.

Every axis is also a CLI flag, so a slow combination — a heavy predictor on a
large log group — can be run on its own without editing code:

```bash
python -m ocpm_eval.run_evaluation                                    # default: Predict-Collab, RQ2+RQ3 partial, random_forest
python -m ocpm_eval.run_evaluation --rq3-scopes partial full           # + RQ3 full catalog on Predict-Collab
python -m ocpm_eval.run_evaluation --log-groups predictcollab bpi2013  # + RQ2/RQ3 on BPI2013 as well
python -m ocpm_eval.run_evaluation --log-groups bpi2013                # BPI2013 only (skips the four study logs)
python -m ocpm_eval.run_evaluation --rqs RQ3 --predictors gnn xgboost  # RQ3 only, two predictors, Predict-Collab
python -m ocpm_eval.run_evaluation --help                              # full flag list
```

or programmatically:

```python
from ocpm_eval.run_evaluation import main

main()                                                    # default: Predict-Collab, RQ2+RQ3 partial, random_forest
main(rq3_scopes=("partial", "full"))                      # + RQ3 full catalog on Predict-Collab
main(log_groups=("predictcollab", "bpi2013"))              # + RQ2/RQ3 on BPI2013 as well
main(log_groups=("bpi2013",))                              # BPI2013 only (skips the four study logs)
main(rqs=("RQ3",), predictors=("gnn", "xgboost"))           # RQ3 only, two predictors, Predict-Collab
main(log_groups=("bpi2013",), rq3_scopes=("partial", "full"))  # BPI2013 (subset + full catalog) only
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
mixing into `rq2_fidelity_predictcollab.csv`/`rq3_results*_predictcollab*.csv`.

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
`rq3_results_random_forest_predictcollab_full.csv`, intended as supplementary coverage rather than a
curated result: it confirms all (task, log)
combinations run end-to-end, but a few of the non-curated tasks (e.g.
`NV-TNE`, `NV-TNM`) land close to or slightly worse than their trivial
baseline in some logs — unlike the subset, which was picked to show clear
separation, the full catalog is a coverage check, not a predictive-quality
claim.

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
