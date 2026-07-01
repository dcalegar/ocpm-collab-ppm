# Object-Centric Predictive Monitoring of Collaborative Processes

Reproducibility code for the study that reformulates collaborative predictive process
monitoring (PPM) tasks over a **rich object-centric representation (OCEL 2.0)** and
demonstrates them end-to-end with **native object-centric features (OCPA)**.

The code is split into a reusable **prediction-task library** and a modular
**evaluation** that consumes it. Inputs are **OCEL 2.0 (SQLite)** event logs.

---

## Tools

| Tool | Role | Notes |
|---|---|---|
| [OCPA](https://github.com/ocpm/ocpa) | native object-centric feature extraction (RQ3) and OCEL 2.0 import | GPL-3.0; pins `pm4py==2.2.32`; OCEL 2.0 importer on `main` |
| [pm4py](https://processintelligence.solutions/pm4py/installation) | dependency of OCPA | `==2.2.32` (pulled by OCPA). It does **not** read OCEL 2.0, so it is **not** used for reading here |
| [scikit-learn](https://scikit-learn.org) | RandomForest learner + metrics | macro-F1 / MAE |
| OCEL 2.0 (SQLite) | input event-log format | read natively by OCPA; read for labels with the stdlib `sqlite3` |

**Why SQLite and not pm4py for reading?** OCPA pins `pm4py==2.2.32`, which predates
OCEL 2.0 and cannot read it. OCPA's own OCEL 2.0 importer reads the **SQLite** format.
Therefore, both sides read OCEL 2.0 SQLite: OCPA for features, and a small stdlib
`sqlite3` reader (`ocpm_tasks.adapters.from_ocel2_sqlite`) for the labels. 
No OCEL 1.0 is used anywhere.

---

## Documentation map

Each package has its own README with the detail that belongs to it; this
top-level file covers setup and how they fit together.

| README | Covers |
|---|---|
| [README.md](README.md) | this file — tools, setup, repository structure, usage, design notes |
| [src/mapping/README.md](src/mapping/README.md) | RQ1 — the XES→OCEL 2.0 converter: mapping rules M1–M8, consistency checks P1.1–P1.5, CLI usage |
| [src/ocpm_tasks/README.md](src/ocpm_tasks/README.md) | the 16 prediction tasks, ground-truth label functions, the neutral object-centric model, and how to connect the library to a concrete OCPA-based prediction |
| [src/ocpm_eval/README.md](src/ocpm_eval/README.md) | RQ2–RQ4 — the evaluation stages that consume `ocpm_tasks`: feature extraction, model fitting, fidelity/feasibility/structure metrics |

## Repository structure

```
ocpm-collab-ppm/
├── README.md                  # this file
├── LICENSE                    # GPL-3.0 notice (OCPA dependency)
├── requirements.txt           # learner dep; OCPA installed separately
├── pyproject.toml             # makes src/ packages importable (pip install -e .)
├── src/
│   ├── mapping/                       # MAPPING TOOL — extended XES → OCEL 2.0 (RQ1)
│   │   ├── README.md                  #   mapping rules M1-M8, checks P1.1-P1.5, usage
│   │   ├── collab_xes_to_ocel.py      #   transformation + checks
│   │   └── aux/                       #   supporting files
│   │       ├── collab.xesext          #     collaborative XES extension definition
│   │       ├── ocel20-schema-json.json  #   OCEL 2.0 JSON schema (draft-07)
│   │       └── printOCEL.py           #     debug helper to inspect OCEL objects and print images
│   ├── ocpm_tasks/            # PREDICTION TASKS — reusable library (decoupled)
│   │   ├── README.md          #   modules, usage, connecting to OCPA prediction
│   │   ├── schema.py          #   mapping vocabulary (object types/qualifiers/attrs)
│   │   ├── model.py           #   neutral object-centric model (Event/Execution/Log)
│   │   ├── catalog.py         #   the 16 tasks + RQ2/RQ3 subsets
│   │   ├── labels.py          #   label functions ℓ (incl. cross-case X-PaL, X-MSt)
│   │   ├── fidelity.py        #   RQ2 comparators (equivalence; consistency)
│   │   └── adapters.py        #   from_ocel2_sqlite (default) / from_pm4py / from_ocpa
│   └── ocpm_eval/             # EXPERIMENTATION — evaluation stages (use ocpm_tasks)
│       ├── README.md          #   modules, reading path, RQ2/RQ3 protocols, usage
│       ├── config.py          #   log registry, CV/learner config
│       ├── io_ocel.py         #   OCEL 2.0 SQLite -> neutral model (labels)
│       ├── features_ocpa.py   #   native OCPA features + alignment oracle (RQ3)
│       ├── models.py          #   per-fold training + descriptive metrics
│       ├── rq2_fidelity.py    #   RQ2 — label fidelity
│       ├── rq3_pipeline.py    #   RQ3 — end-to-end, 5-fold CV grouped by CI
│       ├── rq4_structure.py   #   RQ4 — structural measures + expressiveness
│       └── run_evaluation.py  #   orchestrator (RQ2/RQ3/RQ4)
├── data/
│   └── logs/                  # EXAMPLE LOGS (extended XES + converted OCEL 2.0 JSON)
└── results/                   # evaluation outputs
```

The three directories the project revolves around: **example logs** (`data/logs/`),
**prediction tasks** ([src/ocpm_tasks/](src/ocpm_tasks/README.md)), and
**experimentation** ([src/ocpm_eval/](src/ocpm_eval/README.md)), fed by the
**converter** ([src/mapping/](src/mapping/README.md)).

---

## Setup (macOS, virtual environments)

`mapping` and `ocpm_eval`/`ocpm_tasks` require **different versions of pm4py** and
must run in **separate virtual environments**:

| Environment | Used by | pm4py |
|---|---|---|
| `.venv` | `ocpm_eval`, `ocpm_tasks` | `==2.2.32` (pinned by OCPA) |
| `.venv-mapping` | `src/mapping/` | `>=2.7.16` (OCEL 2.0 write support) |

Recommended Python: **3.10**. Optional system dependency for OCPA visualization
(not used by this pipeline): `brew install graphviz`.

> **Apple Silicon (arm64) note.** Both venvs must be built with an **arm64-native**
> Python 3.10 (e.g. `brew install python@3.10` under an arm64 Homebrew at
> `/opt/homebrew`, or `python.org`'s universal installer invoked as
> `arch -arm64 python3.10 -m venv .venv`). If `python3.10 -m venv` picks up an
> Intel-only interpreter (e.g. one under `/usr/local/bin`, the Intel Homebrew
> prefix), every `numpy`/`pandas`/`pm4py` import fails with the misleading error
> `ImportError: ... you should not try to import numpy from its source
> directory`. Check with `file "$(readlink -f .venv/bin/python3.10)"`; if it says
> `x86_64` only, recreate the venv with an arm64 interpreter, or always invoke it
> as `arch -x86_64 .venv/bin/python ...` (Rosetta) instead.

### Environment 1 — evaluation (`ocpm_eval` + `ocpm_tasks`)

```bash
# 1) clone
git clone <your-fork-url> ocpm-collab-ppm
cd ocpm-collab-ppm

# 2) create the evaluation environment
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 3) install OCPA FIRST (it resolves pm4py==2.2.32 and the OCEL 2.0 importer)
pip install "git+https://github.com/ocpm/ocpa.git@main"      # GPL-3.0

# 4) install remaining deps and make local packages importable
pip install -r requirements.txt
pip install -e .
```

> If `pip install -e .` re-resolves `pandas`/`numpy` in a way that conflicts with
> OCPA, prefer OCPA's versions (reinstall OCPA last).

### Environment 2 — mapping (`src/mapping/`)

```bash
# In a separate terminal (or after deactivating .venv)
python3.10 -m venv .venv-mapping
source .venv-mapping/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-mapping.txt
pip install -e .
```

---

## Visual Studio Code

`src/` is on the analysis path. Recommended extensions: Python + Pylance.

- For `src/ocpm_eval` / `src/ocpm_tasks` files → select the `.venv` interpreter.
- For `src/mapping/` files → select `.venv-mapping` interpreter
  (use **Python: Select Interpreter** per file, or set it per workspace folder).

---

## Usage

Full details, module-by-module, live in each package's own README (see the
[documentation map](#documentation-map) above); this section is the quick
path to running things.

### Evaluation ([`ocpm_eval`](src/ocpm_eval/README.md) + [`ocpm_tasks`](src/ocpm_tasks/README.md))

```bash
# full evaluation (RQ2 + RQ3 + RQ4) — RQ3 requires OCPA installed; run from
# the repo root with .venv active. Writes CSVs to results/ (see table below).
python -m ocpm_eval.run_evaluation
```

There is currently no automated test suite for `ocpm_tasks`/`ocpm_eval`; validate a
change by running the evaluation above and inspecting the `results/*.csv` outputs
(RQ2 fidelity should be ~1.0 on the consistency checks; RQ3 rows should have
`ran_end_to_end=True`).

Point the evaluation at your own logs by editing the registry in
`src/ocpm_eval/config.py` (`LogSpec(name, ocel_path)` with OCEL 2.0 `.sqlite` files).

### Mapping tool ([`src/mapping/`](src/mapping/README.md))

`collab_xes_to_ocel.py` implements the model-to-model transformation
**μ: extended collaborative XES → OCEL 2.0** (mapping rules M1–M8), producing 
`output.jsonocel` and `output.sqlite` files conformant with the OCEL 2.0 JSON schema (Berti et al. 2023,
Definition 2).

```bash
# Convert a collaborative XES log to OCEL 2.0
python src/mapping/collab_xes_to_ocel.py input.xes output

# Abort if any P1 check or schema validation fails
python src/mapping/collab_xes_to_ocel.py input.xes output --strict

# Skip OCEL 2.0 schema validation of the output
python src/mapping/collab_xes_to_ocel.py input.xes output --no-validate

# Verbose logging
python src/mapping/collab_xes_to_ocel.py input.xes output -v
```

The extended XES source must use the `collab` extension attributes defined in
`src/mapping/aux/collab.xesext`: `collab:elemType` (`task` / `SendTask` /
`ReceiveTask`), `collab:participant`, `collab:fromParticipant`, and
`collab:toParticipant`.


---

## Evaluation stages

| RQ | What | Where | Output |
|---|---|---|---|
| RQ1 | XES→OCEL transformation + checks | **converter (separate tool)** — out of scope here | — |
| RQ2 | label fidelity (equivalence for the 14 tasks; consistency for X-PaL/X-MSt) | `ocpm_eval/rq2_fidelity.py` | `results/rq2_fidelity.csv` |
| RQ3 | end-to-end feasibility on native OCPA features, 5-fold CV grouped by collaboration instance | `ocpm_eval/rq3_pipeline.py` | `results/rq3_results.csv` |
| RQ4 | structural measures + expressiveness matrix | `ocpm_eval/rq4_structure.py` | `results/rq4_structure.csv`, `results/rq4_expressiveness.csv` |

RQ2 *equivalence* (the 14 tasks vs the original collaborative log, R1) needs the
converter's R1 reader; pass `r1_logs={name: ObjectCentricLog}` to
`ocpm_eval.run_evaluation.main`. The included toy only exercises the *consistency*
side (X-PaL, X-MSt).

---

## Design notes

- **Prediction tasks as a library.** `ocpm_tasks` is decoupled from the
  experimentation and can be used inside an OCPA- or pm4py-based pipeline: build the
  neutral model with an adapter, then call the label functions. It depends only on
  pandas (OCPA/pm4py are optional, lazy).
- **16 tasks** = the 14 of the collaborative baseline + two object-enabled extensions
  (X-PaL cross-case participant load; X-MSt message synchronization time). The RQ3
  representative subset is six tasks covering the anchor × problem-type combinations.
- **Targets vs features.** Targets are defined over each collaboration instance's
  global trace (linear cut points, deterministic order); features are object-centric
  and past-relative (prefix-respecting) so per-event rows do not leak the future.
- **Alignment.** OCPA rows are matched to labels by `event_id`
  (`feature_storage.feature_graphs` → `node.event_id`) and validated by a
  remaining-time oracle that aborts on any mismatch.
