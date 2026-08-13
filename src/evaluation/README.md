# evaluation — evaluation stages

Modular evaluation that answers RQ2–RQ3 of the study (see
[Research questions](../../README.md#research-questions)). Task definitions and
ground-truth labels come from the decoupled
[`tasks`](../tasks/README.md) library; feature extraction and OCEL
reading come from the decoupled [`features`](../features/README.md) library;
model fitting comes from the decoupled [`predictors`](../predictors/README.md)
library. `evaluation` is a *consumer* of all three, not part of them — it wires
them together into the RQ2/RQ3 pipelines and writes the descriptive
metrics/CSVs. Inputs are **OCEL 2.0 SQLite** logs (the format OCPA imports
natively), produced by the [`mapping`](../mapping/README.md) converter (RQ1,
out of scope here).

## What this is (and isn't)

`evaluation` is where an actual prediction happens — `tasks` only
supplies the target `y`; `features` supplies the feature vector `X` via OCPA,
and `predictors` fits/predicts/scores it. See
["Connecting to a concrete prediction with OCPA"](../tasks/README.md#connecting-to-a-concrete-prediction-with-ocpa)
in the `tasks` README for the minimal version of this join; this
package is the fuller, cross-validated version of the same pattern.

## Modules

| Module | Purpose |
|---|---|
| `config.py` | `ExperimentConfig`, `LogSpec` — log registry, CV/learner hyperparameters, output dir |
| `rq2_fidelity.py` | RQ2 — label-fidelity: R1 (source XES) vs R2 (OCEL) equivalence for the 14 tasks |
| `rq3_pipeline.py` | RQ3 — end-to-end feasibility: `features.ocpa` features + `tasks` labels joined, 5-fold CV grouped by `CollaborationCase` against a persisted per-log fold assignment, scored by a `predictors` predictor |
| `run_evaluation.py` | orchestrator — runs the requested (rq, log_group, rq3_scope, predictor) combinations and writes all CSVs |
| `profiling.py` | opt-in per-stage wall-clock + peak-RSS profiling (`--profile`), written to its own `rq3_profile_*.csv` so no computation times enter `rq3_results_*.csv` |
| `plot_rq3_metrics.py` | standalone reporting script — grouped bar charts from `rq3_results_*.csv` and, when present, `rq3_profile_*.csv` |

Feature extraction (`load_ocpa_ocel`/`read_ocel2_labels`/`extract_feature_table`)
and predictor selection (`PREDICTOR_REGISTRY`/`resolve`) now live in
[`features`](../features/README.md) and [`predictors`](../predictors/README.md)
respectively — see those packages' READMEs for their modules, contracts and
reading path. They are reusable independently of this orchestrator.

| Stage | Module | Output |
|---|---|---|
| **RQ2** label fidelity | `rq2_fidelity.py` | `data/results/rq2_fidelity_predictcollab.csv` |
| **RQ3** end-to-end feasibility (representative subset) | `rq3_pipeline.py` | `data/results/rq3_results_{predictor}_predictcollab.csv` (e.g. `rq3_results_random_forest_predictcollab.csv`) |
| **RQ3** full catalog (supplementary coverage, 14 tasks × 4 logs) | `rq3_pipeline.py` via `run_evaluation.py` (`rq3_scopes=("partial","full")`) | `data/results/rq3_results_{predictor}_predictcollab_full.csv` |
| RQ1 transformation + P1 | — | [`mapping`](../mapping/README.md) (separate tool) |
| **RQ2/RQ3 on BPI2013** (opt-in real-world validation, see below) | `rq2_fidelity.py`/`rq3_pipeline.py` via `run_evaluation.py` (`log_groups=("bpi2013",)`) | `data/results/rq2_fidelity_bpi2013.csv`, `data/results/rq3_results_{predictor}_bpi2013.csv` |

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
# installed; run from the repo root with .venv active. Writes CSVs to data/results/.
python -m evaluation.run_evaluation
```

Same command on Windows, once `.venv` is active (PowerShell:
`.venv\Scripts\Activate.ps1`; `cmd.exe`: `.venv\Scripts\activate.bat`). See
the root [README.md](../../README.md#setup-virtual-environments) for the
full macOS/Linux + Windows setup of `.venv`.

Run a single stage, or drive it programmatically:

```python
from evaluation.config import ExperimentConfig
from evaluation.rq3_pipeline import run_rq3

cfg = ExperimentConfig()          # edit logs/hyperparameters here, or pass overrides
df = run_rq3(cfg)                 # -> pandas.DataFrame, also written to data/results/rq3_results.csv
```

Note the output name: `run_rq3` defaults to `rq3_results.csv`, while `main()`
derives the self-describing `rq3_results_{predictor}_{log_group}[_full].csv`
per combination. Pass `out_name=` if you want the latter from a direct call.

The repo's `tests/` directory holds the regression tests — see the
[root README](../../README.md#usage) for the full table of what each file
covers. Relevant here: `test_predictors_registry.py` (a synthetic-table smoke
test per `predictors.dispatch.PREDICTOR_REGISTRY` entry) and
`test_features_leakage.py` (the no-leakage and deterministic-row-order
guarantees the RQ3 protocol below rests on). There are no end-to-end pipeline
tests, and neither the persisted fold assignment nor `profiling.py` is
covered; validate a pipeline change by running the evaluation above and
inspecting the `data/results/*.csv` outputs (RQ2 fidelity's `agreement` column
should be ~1.0 and `full_equivalence` True; RQ3 rows should have
`ran_end_to_end=True`).

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
  `"lstm"`, `"lstm_torch"`, `"transformer"`, `"gnn"`); RQ3 only (RQ2 has no
  predictor). RQ3 runs
  once per `(log_group, predictor, scope)` combination, each to its own
  `rq3_results_{predictor}*.csv`; RQ2 runs once per `log_group` regardless of
  how many predictors are requested.

Every axis is also a CLI flag, so a slow combination — a heavy predictor on a
large log group — can be run on its own without editing code:

```bash
python -m evaluation.run_evaluation                                    # default: Predict-Collab, RQ2+RQ3 partial, random_forest
python -m evaluation.run_evaluation --rq3-scopes partial full           # + RQ3 full catalog on Predict-Collab
python -m evaluation.run_evaluation --log-groups predictcollab bpi2013  # + RQ2/RQ3 on BPI2013 as well
python -m evaluation.run_evaluation --log-groups bpi2013                # BPI2013 only (skips the four study logs)
python -m evaluation.run_evaluation --rqs RQ3 --predictors gnn xgboost  # RQ3 only, two predictors, Predict-Collab
python -m evaluation.run_evaluation --help                              # full flag list
```

Four further flags are path/run knobs rather than axes:

| Flag | Effect |
|---|---|
| `--out-dir DIR` | override `ExperimentConfig.out_dir` (default `data/results`) |
| `--folds-dir DIR` | override `ExperimentConfig.folds_dir` (default `data/folds`) |
| `--regenerate-folds` | rebuild each log's persisted `CollaborationCase -> fold` assignment instead of reusing it (see [RQ3 protocol](#rq3-protocol)) |
| `--profile` | RQ3 only: also write a per-stage wall-clock + peak-RSS profile to `rq3_profile_{predictor}*.csv`, alongside — never inside — the results CSV |

or programmatically:

```python
from evaluation.run_evaluation import main

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

Representative subset (`tasks.catalog.RQ3_SUBSET`): `NE-NPaA`,
`NE-NMPr`, `NV-PrT`, `NV-PaT`, `NV-NMPr`, `OB-M` — one task per anchor ×
problem-type combination. For each log × task: extract OCPA features,
compute R2 labels, join by `event_id`, then 5-fold CV **grouped by
`CollaborationCase`** (all prefixes of one case stay in a single fold, so no
prefix leaks between train and test). Reports macro-F1 (classification) or
MAE (regression) as mean ± sd over folds, alongside a trivial baseline
(majority class / median) computed on the same folds. No computation times
are reported (V4 profile).

The `CollaborationCase -> fold` assignment is **persisted per log**, under
`data/folds/{log_name}_folds.csv` (+ a `.meta.json` recording the `n_folds`/
`random_state` it was generated with) — see
`rq3_pipeline.generate_folds`/`load_or_generate_folds`. It is loaded once per
log and reused unchanged across every task and every predictor run against
that log, so results are comparable fold-for-fold not only between
predictors (already guaranteed before, since `cfg.predictor` is never read
while building the per-task feature/label table) but also between tasks. The
first run against a log generates and writes the file; later runs reuse it.
Passing `--regenerate-folds` (or `ExperimentConfig(regenerate_folds=True)`)
forces a fresh partition; changing `n_folds`/`random_state` without
regenerating raises rather than silently reusing a stale file. This is a
task/predictor-agnostic, case-level partition (not row-count-balanced per
task the way `sklearn.model_selection.GroupKFold` is), which is safe because
feature values never leak across `CollaborationCase` boundaries even when
two cases share a log-wide object such as a `Participant` (regression-tested
in `tests/test_features_leakage.py`, CROSS-CASE direction) — partitioning
the already-extracted feature table by `case_id` is therefore equivalent to
having extracted features on physically separate per-fold logs.

Because the partition is fixed at the case level and not re-balanced per
task, a (log, task) pair whose labelled rows survive for very few
`CollaborationCase`s (rows get dropped by `tasks.labels.compute_label_rows`,
e.g. BOTTOM rows) can hit degenerate folds:
- fewer than 2 cases with labelled rows for that task → the task is skipped
  entirely for that log, `note: "too few collaboration instances for CV"`,
  no CV attempted;
- 2+ cases, but the fixed partition happens to put all/none of them in a
  given fold → that individual fold is skipped (empty train or test mask)
  rather than raising. Not observed on the four Predict-Collab logs or
  BPI2013 at the default `n_folds=5`, but no longer prevented by
  construction the way the old per-task `GroupKFold` was.

Both are handled in `run_one_log`; the output `folds` column counts only the
folds that actually ran, so a value below `n_folds` signals one of these.

`run_rq3` is task-agnostic (it just loops `cfg.rq3_tasks`), so the same
protocol extends to the full catalog without any pipeline changes —
passing `rq3_scopes=("partial", "full")` to `run_evaluation.main` also runs it
over `tasks.catalog.EQUIVALENCE_TASKS` (all 14 tasks) and writes
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
`tasks.labels.compute_label_rows`), row-aligned by `(case_id, k)`.
The mapping preserves event order and label semantics by construction, so
these are expected to be equal; empirical agreement ≈ 1.0 is the
expected/verifying outcome, not a tunable metric.

RQ2 *equivalence* needs the original XES (R1); pass `xes_path` in each
`LogSpec`. The bundled example logs in `data/logs/` include both the
`.xes` source and its converted `.sqlite`, so this runs out of the box.

## Reporting (`plot_rq3_metrics.py`)

Standalone script, run after `run_evaluation.py`; it computes nothing new,
only aggregates and charts what's already in the CSVs:

- **Accuracy** — reads `rq3_results_{predictor}_{log_group}[_full].csv` per
  predictor, one grouped bar chart per `metric_name` (macro-F1, MAE, ...),
  bars = predictors, x-axis = `log | task`, error bars = `metric_sd` across
  CV folds.
- **Profiling (optional)** — if `rq3_profile_{predictor}_{log_group}[_full].csv`
  files are present (RQ3 run with `--profile`), aggregates `seconds` and
  `rss_peak_mb` to mean ± sd per (model, log, task) for each pipeline stage
  (`ocel_load`, `feature_extraction`, `labeling`, `fit`, `predict`) and draws
  two charts per stage (time, peak memory). Missing profile CSVs are skipped,
  not an error.

Per-predictor CSV paths default to the naming convention above, derived from
`--log-group`/`--scope`/`--results-dir`; a default path that doesn't exist is
skipped (not every predictor has been run against every log group/scope), but
an explicitly-passed override (e.g. `--xgboost path.csv`) still raises if
missing. Charts are saved as PNGs under `--output-dir` (default
`data/results/plots/rq3_metrics`), one file per metric or per stage-metric
combination; each model keeps the same color across every chart.

```bash
python -m evaluation.plot_rq3_metrics --log-group predictcollab --scope full
```
