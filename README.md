# ocpm-collab-ppm
Object-Centric Predictive Monitoring of Collaborative Processes

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
| [scikit-learn](https://scikit-learn.org) | RandomForest learner + metrics (RQ3) | macro-F1 / MAE |
| OCEL 2.0 (SQLite) | input event-log format | read natively by OCPA; read for labels with the stdlib `sqlite3` |

**Why SQLite and not pm4py for reading?** OCPA pins `pm4py==2.2.32`, which predates
OCEL 2.0 and cannot read it. OCPA's own OCEL 2.0 importer reads the **SQLite** format.
Therefore both sides read OCEL 2.0 SQLite: OCPA for features, and a small stdlib
`sqlite3` reader (`ocpm_tasks.adapters.from_ocel2_sqlite`) for the labels. No OCEL 1.0
is used anywhere.

---

## Repository structure

```
ocpm-collab-ppm/
├── README.md                  # this file
├── LICENSE                    # GPL-3.0 notice (OCPA dependency)
├── requirements.txt           # learner dep; OCPA installed separately
├── pyproject.toml             # makes src/ packages importable (pip install -e .)
├── src/
│   ├── mapping/               # MAPPING TOOL — from extended XES to OCED 2.0
│   ├── ocpm_tasks/            # PREDICTION TASKS — reusable library (decoupled)
│   │   ├── schema.py          #   mapping vocabulary (object types/qualifiers/attrs)
│   │   ├── model.py           #   neutral object-centric model (Event/Execution/Log)
│   │   ├── catalog.py         #   the 16 tasks + RQ2/RQ3 subsets
│   │   ├── labels.py          #   label functions ℓ (incl. cross-case X-PaL, X-MSt)
│   │   ├── fidelity.py        #   RQ2 comparators (equivalence; consistency)
│   │   └── adapters.py        #   from_ocel2_sqlite (default) / from_pm4py / from_ocpa
│   └── ocpm_eval/             # EXPERIMENTATION — evaluation stages (use ocpm_tasks)
│       ├── config.py          #   log registry, CV/learner config
│       ├── io_ocel.py         #   OCEL 2.0 SQLite -> neutral model (labels)
│       ├── features_ocpa.py   #   native OCPA features + alignment oracle (RQ3)
│       ├── models.py          #   per-fold training + descriptive metrics
│       ├── rq2_fidelity.py    #   RQ2 — label fidelity
│       ├── rq3_pipeline.py    #   RQ3 — end-to-end, 5-fold CV grouped by CI
│       ├── rq4_structure.py   #   RQ4 — structural measures + expressiveness
│       └── run_evaluation.py  #   orchestrator (RQ2/RQ3/RQ4)
├── data/
│   └── logs/                  # EXAMPLE LOGS (OCEL 2.0 SQLite)
└── results/                   # evaluation outputs
```

The three directories the project revolves around: **example logs** (`data/logs/`),
**prediction tasks** (`src/ocpm_tasks/`), and **experimentation** (`src/ocpm_eval/`).

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

- For `ocpm_eval` / `ocpm_tasks` files → select the `.venv` interpreter.
- For `src/mapping/` files → select `.venv-mapping` interpreter
  (use **Python: Select Interpreter** per file, or set it per workspace folder).

---

## Usage

```bash
export PYTHONPATH=src   # or rely on `pip install -e .`

# A) prediction-task library on the toy log — runs WITHOUT OCPA/pm4py
python scripts/example_tasks.py data/logs/toy_collab.sqlite

# B) full evaluation on the toy log — RQ3 requires OCPA installed
python scripts/run_evaluation.py --toy

# C) run the offline task tests
pytest -q
```

Point the evaluation at your own logs by editing the registry in
`src/ocpm_eval/config.py` (`LogSpec(name, ocel_path)` with OCEL 2.0 `.sqlite` files).

---

## Evaluation stages

| RQ | What | Where | Output |
|---|---|---|---|
| RQ1 | XES→OCEL transformation + properties P1 | **converter (separate tool)** — out of scope here | — |
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
