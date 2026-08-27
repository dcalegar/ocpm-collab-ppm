# A Framework for Object-Centric Predictive Monitoring of Collaborative Processes

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

<!-- Replace XXXXXXX with the Zenodo concept DOI once the first GitHub release is archived. -->

Reproducibility code for the study that reformulates collaborative predictive process monitoring (PPM) tasks over a **rich object-centric representation (OCEL 2.0)** and demonstrates them end-to-end with **native object-centric features (OCPA)**.

The code is split into a reusable **prediction-task library** and a modular
**evaluation** that consumes it. Inputs are **OCEL 2.0 (SQLite)** event logs.

### Citation

A preprint version of the accompanying paper is available on arXiv:

> Daniel Calegari, Andrea Delgado, Leonel Peña, Martín Rubio.
> *A Framework for Object-Centric Predictive Monitoring of Collaborative
> Processes.* Preprint, arXiv:XXXX.XXXXX, 2026.
> <https://arxiv.org/abs/XXXX.XXXXX> *(identifier to be added once assigned)*

```bibtex
@misc{calegari2026ocpm,
  title        = {A Framework for Object-Centric Predictive Monitoring of Collaborative Processes},
  author       = {Calegari, Daniel and Delgado, Andrea and Pe\~{n}a, Leonel and Rubio, Mart\'{i}n},
  year         = {2026},
  eprint       = {XXXX.XXXXX},
  archiveprefix = {arXiv},
  primaryclass = {cs.SE},
  howpublished = {arXiv preprint arXiv:XXXX.XXXXX},
  url          = {https://arxiv.org/abs/XXXX.XXXXX}
}
```

### Scope: this repository reproduces exactly what the paper reports

The published claims are backed by the converter and its consistency checks, the
14 reformulated prediction tasks with their label-fidelity comparison against the
collaborative baseline, and the end-to-end run of those tasks on the study logs
(and, opt-in, on BPI Challenge 2013) — stages RQ1, RQ2 and RQ3 below.

Object-enabled prediction targets outside the 14-task taxonomy, and the
resource/correlation refinement layers, were part of an earlier draft of the
paper and are preserved for a follow-up paper on the `future/object-enabled-extensions`
branch (see `docs/future_object_enabled_extensions.md` there) rather than in this
codebase's main line.

---

## Research questions

The paper poses four research questions; this repository implements the
evaluation for the first three (RQ4 is a qualitative discussion, not a
pipeline stage, so it has no corresponding code):

- **RQ1 — Mapping.** How can an extended collaborative event log (XES) be
  transformed into a semantically consistent object-centric representation
  (OCEL 2.0)? Addressed by the converter in
  [src/mapping/](src/mapping/README.md): mapping rules M1–M8 and consistency
  checks P1.1–P1.7 (P1.7 is an implementation-only export check, not part of
  the P1.1–P1.6 criteria formalized in the paper).
- **RQ2 — Task reformulation.** How can the 14 collaborative prediction
  tasks be reformulated as object-centric tasks without loss of label
  information? Addressed by comparing each reformulated task's labels
  against the original collaborative log
  ([src/tasks/fidelity.py](src/tasks/fidelity.py),
  [src/evaluation/rq2_fidelity.py](src/evaluation/rq2_fidelity.py)).
- **RQ3 — Feasibility.** Can the reformulated tasks be trained and scored
  end-to-end in an object-centric PPM tool? Addressed by fitting the tasks
  on native OCPA features with 5-fold cross-validation grouped by
  collaboration instance, against a per-log fold assignment persisted under
  `data/folds/` and reused across every task and predictor
  ([src/evaluation/rq3_pipeline.py](src/evaluation/rq3_pipeline.py)).

This section is the canonical definition of RQ1–RQ3; every other README in
this repository links back here on first mention instead of restating it.

---

## Tools

| Tool | Role | Notes |
|---|---|---|
| [OCPA](https://github.com/ocpm/ocpa) | native object-centric feature extraction (RQ3) and OCEL 2.0 import | GPL-3.0; pins `pm4py==2.2.32`; v1.3.3, pinned to commit `de056e02` (see [Setup](#setup-virtual-environments)) |
| [pm4py](https://processintelligence.solutions/pm4py/installation) | dependency of OCPA | `==2.2.32` (pulled by OCPA). It does **not** read OCEL 2.0, so it is **not** used for reading here |
| [scikit-learn](https://scikit-learn.org) | RandomForest learner + shared metrics | macro-F1 / MAE, used by every predictor |
| [XGBoost](https://xgboost.readthedocs.io), [PyTorch](https://pytorch.org), [TensorFlow/Keras](https://keras.io), [DGL](https://www.dgl.ai) | the remaining predictors — see [src/predictors/](src/predictors/README.md) | XGBoost; LSTM (Torch) / Transformer / GNN; LSTM (Keras). Optional: only the selected predictor's library is needed |
| OCEL 2.0 (SQLite) | input event-log format | read natively by OCPA; read for labels with the stdlib `sqlite3` |

**Why SQLite and not pm4py for reading?** OCPA pins `pm4py==2.2.32`, which predates
OCEL 2.0 and cannot read it. OCPA's own OCEL 2.0 importer reads the **SQLite** format.
Therefore, both sides read OCEL 2.0 SQLite: OCPA for features, and a small stdlib
`sqlite3` reader (`tasks.adapters.from_ocel2_sqlite`) for the labels. 
No OCEL 1.0 is used anywhere.

**Known upstream warning.** Importing the pm4py version pinned by OCPA emits a
`DeprecationWarning` for `scipy.optimize.optimize`. It originates in
pm4py 2.2.32 rather than this codebase and does not affect the current test
results. The warning is intentionally left visible because a future SciPy
release may require upgrading OCPA/pm4py together.

---

## Documentation map

Each package has its own README with the detail that belongs to it; this
top-level file covers setup and how they fit together. The external format
specifications the codebase targets are kept as reference PDFs under
[docs/](docs/): the [OCEL 2.0 specification](docs/ocel20_specification.pdf)
(Berti et al.), the [OCED Call for Action](docs/OCED_CFA2024.pdf), and
[Fahland et al., "Towards a Simple and Extensible Standard for Object-Centric
Event Data (OCED)"](docs/2410.14495v1.pdf).

| README | Covers |
|---|---|
| [README.md](README.md) | this file — tools, setup, repository structure, usage, design notes |
| [src/mapping/README.md](src/mapping/README.md) | RQ1 — the XES→OCEL 2.0 converter: mapping rules M1–M8, consistency checks P1.1–P1.7, CLI usage |
| [src/tasks/README.md](src/tasks/README.md) | the 14 prediction tasks, ground-truth label functions, the neutral object-centric model, and how to connect the library to a concrete OCPA-based prediction |
| [src/features/README.md](src/features/README.md) | OCEL 2.0 reading + native OCPA feature extraction, decoupled from the evaluation orchestrator |
| [src/predictors/README.md](src/predictors/README.md) | the per-learner fit/score registry (RandomForest, XGBoost, LSTM, Transformer, GNN, …), decoupled from the evaluation orchestrator |
| [src/evaluation/README.md](src/evaluation/README.md) | RQ2–RQ3 — the evaluation stages that wire `tasks` + `features` + `predictors` together: fidelity/feasibility/structure metrics |
| [src/evaluation/RQ3_EXECUTION_PLAN.md](src/evaluation/RQ3_EXECUTION_PLAN.md) | ordered command list to run the full RQ3 sweep — all log groups × scopes × predictors, with profiling and reporting |

## Repository structure

```
ocpm-collab-ppm/
├── README.md                  # this file
├── LICENSE                    # GPL-3.0 notice (OCPA dependency)
├── requirements.txt           # pinned evaluation deps; OCPA installed separately (pinned commit)
├── pyproject.toml             # makes src/ packages importable (pip install -e .)
├── src/
│   ├── mapping/                       # MAPPING TOOL — extended XES → OCEL 2.0 (RQ1)
│   │   ├── README.md                  #   mapping rules M1-M8, checks P1.1-P1.7, usage
│   │   ├── collab_xes_to_ocel.py      #   transformation + checks
│   │   └── support/                   #   supporting files
│   │       ├── collab.xesext          #     collaborative XES extension definition
│   │       ├── ocel20-schema-json.json  #   OCEL 2.0 JSON schema (draft-07)
│   │       ├── build_leakage_test_logs.py  # builds tests/fixtures/leak_*.sqlite
│   │       └── printOCEL.py           #     debug helper to inspect OCEL objects and print images
│   ├── tasks/            # PREDICTION TASKS — reusable library (decoupled)
│   │   ├── README.md          #   modules, usage, connecting to OCPA prediction
│   │   ├── schema.py          #   mapping vocabulary (object types/qualifiers/attrs)
│   │   ├── model.py           #   neutral object-centric model (Event/Execution/Log)
│   │   ├── catalog.py         #   the 14 tasks + RQ2/RQ3 subsets
│   │   ├── labels.py          #   label functions ℓ
│   │   ├── fidelity.py        #   RQ2 comparator (label equivalence)
│   │   └── adapters.py        #   from_ocel2_sqlite (default) / from_pm4py / from_ocpa
│   ├── features/               # FEATURE EXTRACTION — reusable library (decoupled)
│   │   ├── README.md          #   modules, reading path, usage
│   │   ├── io_ocel.py         #   OCEL 2.0 SQLite -> OCPA object (features) / neutral model (labels)
│   │   └── ocpa.py            #   native OCPA features + alignment oracle (RQ3)
│   ├── predictors/             # MODEL FITTING — reusable library (decoupled)
│   │   ├── README.md          #   fit_and_score_fold contract, registry, usage
│   │   ├── common.py          #   xy_split — shared train/test split
│   │   ├── dispatch.py        #   PREDICTOR_REGISTRY / resolve
│   │   └── random_forest.py, xgboost.py, lstm.py, lstm_torch.py, transformer.py, gnn.py
│   └── evaluation/             # EXPERIMENTATION — evaluation stages (wires tasks + features + predictors)
│       ├── README.md          #   modules, RQ2/RQ3 protocols, usage
│       ├── config.py          #   log registry, CV/learner config
│       ├── rq2_fidelity.py    #   RQ2 — label fidelity
│       ├── rq3_pipeline.py    #   RQ3 — end-to-end, 5-fold CV grouped by CI
│       ├── profiling.py       #   opt-in per-stage wall-clock + RSS profiling (--profile)
│       ├── plot_rq3_metrics.py #  charts from rq3_results_*.csv / rq3_profile_*.csv
│       └── run_evaluation.py  #   orchestrator (RQ2/RQ3)
├── data/
│   ├── logs/
│   │   ├── Predict-Collab/      # the four study logs
│   │   └── BPIChallenge2013/    # real-world validation log (opt-in stage, see below)
│   ├── folds/                   # persisted per-log CollaborationCase -> fold assignment
│   └── results/                 # evaluation outputs
└── docs/                       # reference specifications (OCEL 2.0, OCED) — not code
    ├── ocel20_specification.pdf
    ├── OCED_CFA2024.pdf
    └── 2410.14495v1.pdf
```

The directories the project revolves around: **example logs** (`data/logs/`),
**prediction tasks** ([src/tasks/](src/tasks/README.md)), and
**experimentation** ([src/evaluation/](src/evaluation/README.md)) — which wires in
the decoupled [src/features/](src/features/README.md) (feature extraction) and
[src/predictors/](src/predictors/README.md) (model fitting) libraries — fed by
the **converter** ([src/mapping/](src/mapping/README.md)).

---

## Setup (virtual environments)

`mapping` and `evaluation`/`tasks`/`features`/`predictors` require **different
versions of pm4py** and must run in **separate virtual environments**:

| Environment | Used by | pm4py |
|---|---|---|
| `.venv` | `evaluation`, `tasks`, `features`, `predictors` | `==2.2.32` (pinned by OCPA) |
| `.venv-mapping` | `src/mapping/` | `==2.7.22.5` (OCEL 2.0 write support) |

Recommended Python: **3.10**. Below: macOS/Linux, then Windows. Both follow the
same four steps — only venv creation/activation syntax differs.

> **Windows note.** `src/mapping/support/` (converter helper scripts) used to be
> named `aux/`; `aux` is a reserved DOS device name (like `con`, `prn`, `nul`)
> and silently breaks `git clone`/checkout on Windows. The repository already
> uses `support/` — if you have an old checkout, zip, or backup with an `aux`
> folder under `src/mapping/`, discard it and re-clone instead of merging it in.

> **Apple Silicon (arm64) note.** Both venvs must be built with an **arm64-native**
> Python 3.10 (e.g. `brew install python@3.10` under an arm64 Homebrew at
> `/opt/homebrew`, or `python.org`'s universal installer invoked as
> `arch -arm64 python3.10 -m venv .venv`). If `python3.10 -m venv` picks up an
> Intel-only interpreter (e.g. one under `/usr/local/bin`, the Intel Homebrew
> prefix), every `numpy`/`pandas`/`pm4py` import fails with the misleading error
> `ImportError: ... you should not try to import numpy from its source
> directory`. Check with `file "$(readlink -f .venv/bin/python3.10)"`; if it says
> `x86_64` only, recreate the venv with an arm64 interpreter, or always invoke it
> as `arch -x86_64 .venv/bin/python ...` (Rosetta) instead. This applies to
> **both** `.venv` and `.venv-mapping` independently — fixing one does not fix
> the other. The same symptom (a wrong-architecture wheel under a
> right-architecture interpreter) can also show up after `pip install
> --upgrade`/`--force-reinstall` pulls an x86_64 wheel from a stale cache; the
> fix is the same: `pip install --force-reinstall --no-cache-dir <package>`
> for the affected package (check with `file` on its `*.so` files under
> `site-packages/`).

Optional system dependency for OCPA visualization (not used by this
pipeline): Graphviz — `brew install graphviz` (macOS), your distro's package
manager (Linux), or see the Windows section below.

### macOS / Linux

#### Environment 1 — evaluation (`evaluation` + `tasks` + `features` + `predictors`)

```bash
# 1) clone
git clone <your-fork-url> ocpm-collab-ppm
cd ocpm-collab-ppm

# 2) create the evaluation environment
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 3) install OCPA FIRST (it resolves pm4py==2.2.32 and the OCEL 2.0 importer).
#    Pinned to the exact commit (v1.3.3) used to produce the published results;
#    do not install from @main, which is a moving branch.
pip install "git+https://github.com/ocpm/ocpa.git@de056e0203a3fa4a9bbc19a95e001eada323074a"      # GPL-3.0

# 4) install remaining deps and make local packages importable
pip install -r requirements.txt
pip install -e .
```

> If `pip install -e .` re-resolves `pandas`/`numpy` in a way that conflicts with
> OCPA, prefer OCPA's versions (reinstall OCPA last).

#### Environment 2 — mapping (`src/mapping/`)

```bash
# In a separate terminal (or after deactivating .venv)
python3.10 -m venv .venv-mapping
source .venv-mapping/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-mapping.txt
pip install -e .
```

### Windows

Use the `py` launcher (bundled with the python.org installer) to pin Python
3.10 — a bare `python3.10` command typically doesn't exist on Windows.
Commands below are for **PowerShell**; the `cmd.exe` equivalent for
activation is noted inline.

> If `Activate.ps1` is blocked ("running scripts is disabled on this
> system"), allow it for the current session:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

Optional system dependency for OCPA visualization (not used by this
pipeline): install [Graphviz for Windows](https://graphviz.org/download/)
and add its `bin\` folder to `PATH` (or `choco install graphviz` with
Chocolatey), then restart the terminal.

#### Environment 1 — evaluation (`evaluation` + `tasks` + `features` + `predictors`)

```powershell
# 1) clone
git clone <your-fork-url> ocpm-collab-ppm
cd ocpm-collab-ppm

# 2) create the evaluation environment
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1        # cmd.exe: .venv\Scripts\activate.bat
python -m pip install --upgrade pip

# 3) install OCPA FIRST (it resolves pm4py==2.2.32 and the OCEL 2.0 importer).
#    Pinned to the exact commit (v1.3.3) used to produce the published results;
#    do not install from @main, which is a moving branch.
pip install "git+https://github.com/ocpm/ocpa.git@de056e0203a3fa4a9bbc19a95e001eada323074a"      # GPL-3.0

# 4) install remaining deps and make local packages importable
pip install -r requirements.txt
pip install -e .
```

> If `pip install -e .` re-resolves `pandas`/`numpy` in a way that conflicts with
> OCPA, prefer OCPA's versions (reinstall OCPA last).

#### Environment 2 — mapping (`src/mapping/`)

```powershell
# In a separate terminal (or after deactivating .venv)
py -3.10 -m venv .venv-mapping
.venv-mapping\Scripts\Activate.ps1   # cmd.exe: .venv-mapping\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements-mapping.txt
pip install -e .
```

---

## Visual Studio Code

`src/` is on the analysis path. Recommended extensions: Python + Pylance.

- For `src/evaluation` / `src/tasks` / `src/features` / `src/predictors`
  files → select the `.venv` interpreter.
- For `src/mapping/` files → select `.venv-mapping` interpreter
  (use **Python: Select Interpreter** per file, or set it per workspace folder).
  On Windows the interpreter path is `.venv\Scripts\python.exe` /
  `.venv-mapping\Scripts\python.exe`.

---

## Usage

Full details, module-by-module, live in each package's own README (see the
[documentation map](#documentation-map) above); this section is the quick
path to running things.

### Evaluation ([`evaluation`](src/evaluation/README.md) + [`tasks`](src/tasks/README.md) + [`features`](src/features/README.md) + [`predictors`](src/predictors/README.md))

```bash
# full evaluation (RQ2 + RQ3, partial scope, on the four Predict-Collab study
# logs) — RQ3 requires OCPA installed; run from the repo root with .venv
# active. Writes CSVs to data/results/ (RQ3 under a per-stage subdirectory,
# e.g. data/results/predictcollab_partial/ -- see table below).
python -m evaluation.run_evaluation
```

Same command on Windows, once `.venv` is active (`.venv\Scripts\Activate.ps1`
in PowerShell, or `.venv\Scripts\activate.bat` in `cmd.exe` — see
[Setup](#setup-virtual-environments)).

The command above does **not** run the RQ3 full-catalog scope
(`rq3_scopes=("partial", "full")`) or the opt-in BPI2013 real-world validation
log group (`log_groups=("bpi2013",)`, ~29.5x larger than the study logs by
events) — see [evaluation's README](src/evaluation/README.md#usage) for how to
enable them.

Automated regression tests live in [`tests/`](tests/):

| Test file | Covers | Needs |
|---|---|---|
| `test_mapping_checks.py` | the P1.1–P1.7 consistency checks of the converter, incl. deliberate-corruption scenarios | `.venv-mapping` |
| `test_predictors_registry.py` | a synthetic-table classification/regression smoke test per `predictors.dispatch.PREDICTOR_REGISTRY` entry (`random_forest`, `xgboost`, `lstm`, `lstm_torch`, `transformer`), plus a registry-presence check for `gnn`, which needs OCPA's `feature_storage` graphs and so isn't exercised against the synthetic table | `.venv` |
| `test_fidelity.py` | `tasks.fidelity.compare_equivalence` — `agreement` vs `exact_agreement` vs `full_equivalence` | none (pure) |
| `test_io_ocel_strip.py` | `features.io_ocel._strip_participant_e2o` — the cross-case E2O edge strip | stdlib `sqlite3` |
| `test_io_ocel_ties.py` | `features.io_ocel._break_timestamp_ties` — within-case timestamp tie-breaking | stdlib `sqlite3` |
| `test_labels_golden.py` | expected-value tests for the fourteen label functions of `tasks.labels`, against a hand-computed toy log (the only non-agreement check of label correctness in the repository) | none (pure) |
| `test_task_anchors.py` | the target anchor of the fourteen tasks, kept identical across `tasks.catalog`, the `anchor` column of every `rq3_results_*.csv`, and the Anchor column of the paper's RQ3 tables (the paper checks skip when the paper sources are not checked out alongside; override the location with `PAPER_DIR`) | none (pure) |
| `test_features_leakage.py` | temporal and cross-case leakage of every OCPA feature family, plus the deterministic row order | `.venv` (OCPA) |

Run the whole suite with `python -m pytest tests/ -q`, or a single file
directly — e.g. `python tests/test_mapping_checks.py`. On macOS the two
XGBoost tests need the OpenMP runtime; see the `xgboost` note in
`requirements.txt` for the one-line workaround if `brew install libomp`
isn't an option.

Beyond these, the evaluation pipelines have no end-to-end unit tests (the
persisted fold assignment and `evaluation.profiling` in particular are
untested); validate a pipeline change by running the evaluation above and
inspecting the `data/results/*.csv` (RQ2) / `data/results/{log_group}_{scope}
/*.csv` (RQ3) outputs (RQ2 fidelity's `agreement` column should be ~1.0 and
`full_equivalence` True; RQ3 rows should have `ran_end_to_end=True`).

Point the evaluation at your own logs by editing the registry in
`src/evaluation/config.py` — `LogSpec(name, ocel_path, xes_path)`, where
`ocel_path` is the OCEL 2.0 `.sqlite` produced by `mapping` and `xes_path` is
the original XES, needed only for RQ2's R1 side.

### Mapping tool ([`src/mapping/`](src/mapping/README.md))

`collab_xes_to_ocel.py` implements the model-to-model transformation
**μ: extended collaborative XES → OCEL 2.0** (mapping rules M1–M8), producing 
`output.jsonocel` and `output.sqlite` files conformant with the OCEL 2.0 JSON schema (Berti et al. 2023,
Definition 2).

The object types of an exported log are `CollaborationCase`,
`OrchestrationCase`, `Message`, and **one type per participant identifier**
(rule M2) — `Hospital`, `Laboratory`, … for Healthcare. The identifier is also
kept as the `name` object attribute, and participant objects stay addressable
without naming their types, through the fixed qualifiers `in_participant`
(E2O), `for_participant`, `from` and `to` (O2O).

```bash
# Convert a collaborative XES log to OCEL 2.0
python src/mapping/collab_xes_to_ocel.py input.xes output

# Abort if any consistency check or schema validation fails
python src/mapping/collab_xes_to_ocel.py input.xes output --strict

# Skip OCEL 2.0 schema validation of the output
python src/mapping/collab_xes_to_ocel.py input.xes output --no-validate

# Verbose logging
python src/mapping/collab_xes_to_ocel.py input.xes output -v
```

Same commands on Windows, once `.venv-mapping` is active
(`.venv-mapping\Scripts\Activate.ps1` in PowerShell, or
`.venv-mapping\Scripts\activate.bat` in `cmd.exe`); forward slashes in
`input.xes`/`output` paths work fine on Windows too — no need to switch to
backslashes.

The extended XES source must use the `collab` extension attributes defined in
`src/mapping/support/collab.xesext`: `collab:elemType` (`task` / `SendTask` /
`ReceiveTask`), `collab:participant`, `collab:fromParticipant`, and
`collab:toParticipant`.


---

## Evaluation stages

Maps each [research question](#research-questions) to its concrete artifacts.

| Stage | What | Where | Output |
|---|---|---|---|
| RQ1 | XES→OCEL transformation + checks | **converter (separate tool)** — out of scope here | — |
| RQ2 | label fidelity (equivalence for the 14 tasks) | `evaluation/rq2_fidelity.py` | `data/results/rq2_fidelity_predictcollab.csv` |
| RQ3 | end-to-end feasibility on native OCPA features, 5-fold CV grouped by collaboration instance | `evaluation/rq3_pipeline.py` | `data/results/predictcollab_partial/rq3_results_random_forest_predictcollab.csv` |
| RQ3 profiling (opt-in, `--profile`) | per-stage wall-clock + peak RSS, kept out of the results CSV | `evaluation/profiling.py` | `data/results/predictcollab_partial/rq3_profile_random_forest_predictcollab.csv` |
| RQ2/RQ3 (BPI2013) | real-world validation on BPI Challenge 2013 (incidents, collaborative) — **opt-in**, not one of the four study logs | `evaluation/run_evaluation.py` (`log_groups=("bpi2013",)`) | `data/results/rq2_fidelity_bpi2013.csv`, `data/results/bpi2013_partial/rq3_results_random_forest_bpi2013.csv` |

RQ3 outputs (results, profile, progress log, and once generated, `plots/`)
live under a per-(log_group, scope) stage subdirectory of `data/results/`,
e.g. `data/results/predictcollab_full/` — see
[`evaluation/RQ3_EXECUTION_PLAN.md`](src/evaluation/RQ3_EXECUTION_PLAN.md)
for the full ordered command list (all log groups × scopes × predictors)
and [`evaluation/README.md`](src/evaluation/README.md#output) for the
underlying convention. RQ2 output stays flat in `data/results/` (no scope
axis).

RQ2 *equivalence* compares the 14 tasks against the original collaborative log
(R1). Both sides are read by the evaluation itself — R1 straight from the
`.xes` via `rq2_fidelity._src_label`, R2 from the converted `.sqlite` — so it
needs no extra input beyond the `xes_path` of each `LogSpec`, which the bundled
logs already ship alongside their `.sqlite`.

---

## Design notes

- **Prediction tasks as a library.** `tasks` is decoupled from the
  experimentation and can be used inside an OCPA- or pm4py-based pipeline: build the
  neutral model with an adapter, then call the label functions. It depends only on
  pandas (OCPA/pm4py are optional, lazy).
- **14 tasks** = the object-centric reformulation of the 14 tasks of the
  collaborative baseline (`tasks/catalog.py`). The RQ3 representative subset
  is six tasks covering the anchor × problem-type combinations.
- **Targets vs features.** Targets are defined over each collaboration instance's
  global trace (linear cut points, deterministic order); features are object-centric
  and past-relative (prefix-respecting) so per-event rows do not leak the future.
- **Alignment.** OCPA rows are matched to labels by `event_id`
  (`feature_storage.feature_graphs` → `node.event_id`) and validated by a
  remaining-time oracle that aborts on any mismatch.
- **OCPA compatibility and concurrency.** `load_ocpa_ocel` and
  `extract_feature_table` temporarily replace selected pandas and `random`
  methods to work around bugs in the pinned OCPA 1.3.3 revision, restoring the
  original methods in `finally` blocks. These replacements are process-global
  and therefore not thread-safe: do not invoke either function concurrently in
  the same Python process. Run feature extraction sequentially or isolate
  concurrent runs in separate processes.
