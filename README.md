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
| [scikit-learn](https://scikit-learn.org) | RandomForest learner + metrics | macro-F1 / MAE |
| OCEL 2.0 (SQLite) | input event-log format | read natively by OCPA; read for labels with the stdlib `sqlite3` |

**Why SQLite and not pm4py for reading?** OCPA pins `pm4py==2.2.32`, which predates
OCEL 2.0 and cannot read it. OCPA's own OCEL 2.0 importer reads the **SQLite** format.
Therefore, both sides read OCEL 2.0 SQLite: OCPA for features, and a small stdlib
`sqlite3` reader (`ocpm_tasks.adapters.from_ocel2_sqlite`) for the labels. 
No OCEL 1.0 is used anywhere.

---

## Repository structure

```
ocpm-collab-ppm/
├── README.md                  # this file
├── LICENSE                    # GPL-3.0 notice (OCPA dependency)
├── requirements.txt           # learner dep; OCPA installed separately
├── pyproject.toml             # makes src/ packages importable (pip install -e .)
├── src/
│   ├── mapping/               # MAPPING TOOL — extended XES → OCEL 2.0
│   │   ├── collab_xes_to_ocel.py  #   transformation + checks
│   │   └── aux/               #   supporting files
│   │       ├── collab.xesext  #     collaborative XES extension definition
│   │       ├── ocel20-schema-json.json  #   OCEL 2.0 JSON schema (draft-07)
│   │       └── printOCEL.py   #     debug helper to inspect OCEL objects and print images
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
│   └── logs/                  # EXAMPLE LOGS (extended XES + converted OCEL 2.0 JSON)
└── results/                   # evaluation outputs
```

The three directories the project revolves around: **example logs** (`data/logs/`),
**prediction tasks** (`src/ocpm_tasks/`), and **experimentation** (`src/ocpm_eval/`).

---

## Mapping tool (`src/mapping/`)

`src/mapping/collab_xes_to_ocel.py` implements the model-to-model transformation
**μ: extended collaborative XES → OCEL 2.0** (mapping rules M1–M8), producing
`.jsonocel` and SQLite files conformant with the OCEL 2.0 JSON schema (Berti et al. 2023,
Definition 2). This is the converter referenced as RQ1 in the evaluation stages table.

### What it does

| Rule | Effect |
|---|---|
| M1 | One `CollaborationInstance` object per global case |
| M2 | One `Participant` object per orchestration/pool (log-level scope) |
| M3 | One `LocalCase` object per (case, participant) pair |
| M4 | One `Message` object per `SendTask`; heuristic send/receive correlation |
| M5 | Each source event → OCEL event (activity = event type; `elemType` attribute) |
| M6 | E2O relations: `within` (→ CI), `local` (→ LocalCase), `send`/`receive` (→ Message) |
| M7 | O2O relations: `part_of`, `executed_by`, `from`, `to`, `exchanged_in` |
| M8 | Residual source attributes forwarded unchanged to the OCEL events |

After transformation, consistency properties **P1.1–P1.5** are checked automatically
(totality, per-case partition, message well-formedness, local-case coherence, no orphan
objects) and reported before export.

### Requirements

```
pm4py >= 2.7.16   # OCEL 2.0 JSON and SQLite exporters
pandas            # pulled in by pm4py
jsonschema        # optional; enables full draft-07 schema validation
```

The pure-Python transformation and consistency checks (`transform`,
`run_consistency_checks`) are importable without pm4py; only the I/O helpers
(`read_collaborative_xes`, `write_ocel2_json`, `write_ocel2_sqlite`) require it.

### Usage

The second argument is a **base path without extension**. Two output files are
always written: `<output_base>.jsonocel` and `<output_base>.sqlite`.

```bash
# Convert a collaborative XES log to OCEL 2.0 (JSON + SQLite)
python src/mapping/collab_xes_to_ocel.py input.xes output/my_log
# → output/my_log.jsonocel
# → output/my_log.sqlite

# Abort if any P1 check or schema validation fails
python src/mapping/collab_xes_to_ocel.py input.xes output/my_log --strict

# Skip OCEL 2.0 JSON schema validation of the output
python src/mapping/collab_xes_to_ocel.py input.xes output/my_log --no-validate

# Verbose logging
python src/mapping/collab_xes_to_ocel.py input.xes output/my_log -v
```

The extended XES source must use the `collab` extension attributes defined in
`src/mapping/aux/collab.xesext`: `collab:elemType` (`task` / `SendTask` /
`ReceiveTask`), `collab:participant`, `collab:fromParticipant`, and
`collab:toParticipant`.

---

## Setup (macOS, virtual environment)

Recommended Python: **3.10** (a version compatible with OCPA's pinned `pm4py==2.2.32`;
verify against OCPA's `requirements.txt`). Optional system dependency for OCPA
visualization (not used by this pipeline): `brew install graphviz`.

```bash
# 1) clone
git clone <your-fork-url> ocpm-collab-ppm
cd ocpm-collab-ppm

# 2) virtual environment (isolates everything from the system Python)
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 3) install OCPA FIRST (it resolves pm4py==2.2.32 and the OCEL 2.0 importer)
pip install "git+https://github.com/ocpm/ocpa.git@main"      # GPL-3.0

# 4) install the remaining deps and make the local packages importable
pip install -r requirements.txt
pip install -e .

# 5) (optional) regenerate the toy log
python scripts/make_toy_log.py 8 data/logs/toy_collab.sqlite
```

> If `pip install -e .` re-resolves `pandas`/`numpy` in a way that conflicts with
> OCPA, prefer OCPA's versions (reinstall OCPA last) — installation follows what OCPA
> requires.

---

## Visual Studio Code

Open the folder in VS Code and select the `.venv` interpreter (the workspace already
points to `${workspaceFolder}/.venv/bin/python`). `src/` is on the analysis path, and
two ready-to-run debug configurations are provided (Run ▸ "Example: prediction tasks"
and "Run evaluation (toy)"). Recommended extensions: Python + Pylance.

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
