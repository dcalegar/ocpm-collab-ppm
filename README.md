# A Framework for Object-Centric Predictive Monitoring of Collaborative Processes

Reproducibility code for the study that reformulates collaborative predictive process
monitoring (PPM) tasks over a **rich object-centric representation (OCEL 2.0)** and
demonstrates them end-to-end with **native object-centric features (OCPA)**.

The code is split into a reusable **prediction-task library** and a modular
**evaluation** that consumes it. Inputs are **OCEL 2.0 (SQLite)** event logs.

### Scope: this repository implements more than the paper reports

The published claims are backed by the converter and its consistency checks, the
14 reformulated prediction tasks with their label-fidelity comparison against the
collaborative baseline, and the end-to-end run of those tasks on the study logs
(and, opt-in, on BPI Challenge 2013) — stages RQ1, RQ2 and RQ3 below.

Two further targets outside the 14-task taxonomy — the **in-flight message
backlog** and the **message synchronization time** — are implemented, unit-tested,
and demonstrated here on a purpose-built toy log, together with the correlation
enrichment the second one presupposes. They are **continuation work**: no results
of theirs are reported in the paper, which only discusses such targets as
directions the object-centric representation opens. The artifacts they produce
(`data/results/rq_ext_results_toy.csv`) therefore correspond to no published
number. They are deliberately kept out of `catalog.TASKS`, `EQUIVALENCE_TASKS`
and `RQ3_SUBSET`, so they cannot affect the reproduction of the published
results.

---

## Tools

| Tool | Role | Notes |
|---|---|---|
| [OCPA](https://github.com/ocpm/ocpa) | native object-centric feature extraction (RQ3) and OCEL 2.0 import | GPL-3.0; pins `pm4py==2.2.32`; v1.3.3, pinned to commit `de056e02` (see [Setup](#setup-virtual-environments)) |
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
| [src/mapping/README.md](src/mapping/README.md) | RQ1 — the XES→OCEL 2.0 converter: mapping rules M1–M8, consistency checks P1.1–P1.6, CLI usage |
| [src/ocpm_tasks/README.md](src/ocpm_tasks/README.md) | the 14 prediction tasks, ground-truth label functions, the neutral object-centric model, and how to connect the library to a concrete OCPA-based prediction |
| [src/ocpm_eval/README.md](src/ocpm_eval/README.md) | RQ2–RQ3 — the evaluation stages that consume `ocpm_tasks`: feature extraction, model fitting, fidelity/feasibility/structure metrics |

## Repository structure

```
ocpm-collab-ppm/
├── README.md                  # this file
├── LICENSE                    # GPL-3.0 notice (OCPA dependency)
├── requirements.txt           # pinned evaluation deps; OCPA installed separately (pinned commit)
├── pyproject.toml             # makes src/ packages importable (pip install -e .)
├── src/
│   ├── mapping/                       # MAPPING TOOL — extended XES → OCEL 2.0 (RQ1)
│   │   ├── README.md                  #   mapping rules M1-M8, checks P1.1-P1.6, usage
│   │   ├── collab_xes_to_ocel.py      #   transformation + checks
│   │   └── support/                   #   supporting files
│   │       ├── collab.xesext          #     collaborative XES extension definition
│   │       ├── ocel20-schema-json.json  #   OCEL 2.0 JSON schema (draft-07)
│   │       └── printOCEL.py           #     debug helper to inspect OCEL objects and print images
│   ├── ocpm_tasks/            # PREDICTION TASKS — reusable library (decoupled)
│   │   ├── README.md          #   modules, usage, connecting to OCPA prediction
│   │   ├── schema.py          #   mapping vocabulary (object types/qualifiers/attrs)
│   │   ├── model.py           #   neutral object-centric model (Event/Execution/Log)
│   │   ├── catalog.py         #   the 14 tasks + RQ2/RQ3 subsets
│   │   ├── labels.py          #   label functions ℓ
│   │   ├── extensions.py      #   object-enabled EXTENSION tasks (X-Inf, X-MSt) — not in catalog.TASKS
│   │   ├── fidelity.py        #   RQ2 comparator (label equivalence)
│   │   └── adapters.py        #   from_ocel2_sqlite (default) / from_pm4py / from_ocpa
│   └── ocpm_eval/             # EXPERIMENTATION — evaluation stages (use ocpm_tasks)
│       ├── README.md          #   modules, reading path, RQ2/RQ3 protocols, usage
│       ├── config.py          #   log registry, CV/learner config
│       ├── io_ocel.py         #   OCEL 2.0 SQLite -> neutral model (labels)
│       ├── features_ocpa.py   #   native OCPA features + alignment oracle (RQ3)
│       ├── models.py          #   per-fold training + descriptive metrics
│       ├── rq2_fidelity.py    #   RQ2 — label fidelity
│       ├── rq3_pipeline.py    #   RQ3 — end-to-end, 5-fold CV grouped by CI
│       ├── rq_ext_pipeline.py #   EXT — X-Inf/X-MSt demo on the TOY log only
│       └── run_evaluation.py  #   orchestrator (RQ2/RQ3/EXT)
└── data/
    ├── logs/
    │   ├── Predict-Collab/      # the four study logs
    │   ├── BPIChallenge2013/    # real-world validation log (opt-in stage, see below)
    │   └── ToyCollab/           # toy_collab.* (X-Inf/X-MSt demo log)
    └── results/                 # evaluation outputs, incl. rq_ext_results_toy.csv (EXT demo)
```

The three directories the project revolves around: **example logs** (`data/logs/`),
**prediction tasks** ([src/ocpm_tasks/](src/ocpm_tasks/README.md)), and
**experimentation** ([src/ocpm_eval/](src/ocpm_eval/README.md)), fed by the
**converter** ([src/mapping/](src/mapping/README.md)).

---

## Setup (virtual environments)

`mapping` and `ocpm_eval`/`ocpm_tasks` require **different versions of pm4py** and
must run in **separate virtual environments**:

| Environment | Used by | pm4py |
|---|---|---|
| `.venv` | `ocpm_eval`, `ocpm_tasks` | `==2.2.32` (pinned by OCPA) |
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

#### Environment 1 — evaluation (`ocpm_eval` + `ocpm_tasks`)

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

#### Environment 1 — evaluation (`ocpm_eval` + `ocpm_tasks`)

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

- For `src/ocpm_eval` / `src/ocpm_tasks` files → select the `.venv` interpreter.
- For `src/mapping/` files → select `.venv-mapping` interpreter
  (use **Python: Select Interpreter** per file, or set it per workspace folder).
  On Windows the interpreter path is `.venv\Scripts\python.exe` /
  `.venv-mapping\Scripts\python.exe`.

---

## Usage

Full details, module-by-module, live in each package's own README (see the
[documentation map](#documentation-map) above); this section is the quick
path to running things.

### Evaluation ([`ocpm_eval`](src/ocpm_eval/README.md) + [`ocpm_tasks`](src/ocpm_tasks/README.md))

```bash
# full evaluation (RQ2 + RQ3, partial scope, on the four Predict-Collab study
# logs) — RQ3 requires OCPA installed; run from the repo root with .venv
# active. Writes CSVs to results/ (see table below).
python -m ocpm_eval.run_evaluation
```

Same command on Windows, once `.venv` is active (`.venv\Scripts\Activate.ps1`
in PowerShell, or `.venv\Scripts\activate.bat` in `cmd.exe` — see
[Setup](#setup-virtual-environments)).

The command above does **not** run the RQ3 full-catalog scope
(`rq3_scopes=("partial", "full")`) or the opt-in BPI2013 real-world validation
log group (`log_groups=("bpi2013",)`, ~29.5x larger than the study logs by
events) — see [ocpm_eval's README](src/ocpm_eval/README.md#usage) for how to
enable them.

Automated regression tests live in [`tests/`](tests/): `test_mapping_checks.py`
(the P1.1–P1.6 consistency checks of the converter, incl. deliberate-corruption
scenarios) and `test_extensions_toy.py` (the X-Inf/X-MSt extension tasks of
`ocpm_tasks/extensions.py`). Run them directly — `python tests/test_mapping_checks.py`
and `python tests/test_extensions_toy.py` (also pytest-compatible; pytest is not
installed in the provided venvs). The evaluation pipelines of `ocpm_eval` have no
dedicated unit tests; validate a change there by running the evaluation above and
inspecting the `results/*.csv` outputs (RQ2 fidelity's `agreement` column should
be ~1.0; RQ3 rows should have `ran_end_to_end=True`).

Point the evaluation at your own logs by editing the registry in
`src/ocpm_eval/config.py` (`LogSpec(name, ocel_path)` with OCEL 2.0 `.sqlite` files).

### Mapping tool ([`src/mapping/`](src/mapping/README.md))

`collab_xes_to_ocel.py` implements the model-to-model transformation
**μ: extended collaborative XES → OCEL 2.0** (mapping rules M1–M8), producing 
`output.jsonocel` and `output.sqlite` files conformant with the OCEL 2.0 JSON schema (Berti et al. 2023,
Definition 2). Two optional refinement layers can be applied on top, each only
when the source log records the identifier it needs: the resource layer
(R1–R3) and the correlation layer (C1). Both only add objects and relations, so
the core representation and every prediction label are unchanged.

The object types of an exported log are `CollaborationCase`,
`OrchestrationCase`, `Message`, and **one type per participant identifier**
(rule M2) — `Hospital`, `Laboratory`, … for Healthcare. The identifier is also
kept as the `name` object attribute, and participant objects stay addressable
without naming their types, through the fixed qualifiers `participant`,
`for_participant`, `from` and `to`.

```bash
# Convert a collaborative XES log to OCEL 2.0
python src/mapping/collab_xes_to_ocel.py input.xes output

# Abort if any consistency check or schema validation fails
python src/mapping/collab_xes_to_ocel.py input.xes output --strict

# Skip OCEL 2.0 schema validation of the output
python src/mapping/collab_xes_to_ocel.py input.xes output --no-validate

# Apply the resource layer (R1-R3) over the actor attribute; checks PR.1/PR.2
python src/mapping/collab_xes_to_ocel.py input.xes output --resource-attr org:resource

# Apply the correlation layer (C1) over a native message id; checks PC.1
python src/mapping/collab_xes_to_ocel.py input.xes output --correlation-attr msgInstanceId

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

| Stage | What | Where | Output |
|---|---|---|---|
| RQ1 | XES→OCEL transformation + checks | **converter (separate tool)** — out of scope here | — |
| RQ2 | label fidelity (equivalence for the 14 tasks) | `ocpm_eval/rq2_fidelity.py` | `results/rq2_fidelity.csv` |
| RQ3 | end-to-end feasibility on native OCPA features, 5-fold CV grouped by collaboration instance | `ocpm_eval/rq3_pipeline.py` | `results/rq3_results_random_forest.csv` |
| EXT | object-enabled EXTENSION tasks (X-Inf, X-MSt) — **follow-up work, not reported in the paper**; feasibility demo on a toy log, not one of the four study logs, kept separate from the RQ2/RQ3 results | `ocpm_eval/rq_ext_pipeline.py` | `results/rq_ext_results_toy.csv` |
| RQ2/RQ3 (BPI2013) | real-world validation on BPI Challenge 2013 (incidents, collaborative) — **opt-in**, not one of the four study logs | `ocpm_eval/run_evaluation.py` (`log_groups=("bpi2013",)`) | `results/rq2_fidelity_bpi2013.csv`, `results/rq3_results_random_forest_bpi2013.csv` |

RQ2 *equivalence* (the 14 tasks vs the original collaborative log, R1) needs the
converter's R1 reader; pass `r1_logs={name: ObjectCentricLog}` to
`ocpm_eval.run_evaluation.main`.

### EXT — object-enabled extensions (X-Inf, X-MSt), follow-up work

`ocpm_tasks/extensions.py` formalizes two "object-enabled" targets as label functions
over the same neutral model as the 14 reformulated tasks, but keeps them **out of
`catalog.TASKS`/`EQUIVALENCE_TASKS`/`RQ3_SUBSET`** so they never mix into RQ2/RQ3:

- **X-Inf** (in-flight backlog, `CollaborationCase` anchor, count regression) — peak,
  over the case's remainder after the cut, of the *running* send/receive balance
  (+1 per send, −1 per receive, clamped at 0 as it runs — not a plain
  `#send − #receive` difference). Needs **no** send/receive correlation.
- **X-MSt** (message synchronization time, `Message` anchor, time regression) —
  latency of the next send's matching receive after the cut. Needs a native
  correlation id, which the core mapping (M4) deliberately does not infer; an opt-in
  `corr_attr` enrichment (`ocpm_tasks/adapters.py`, default `None`) supplies it from a
  residual event attribute (e.g. `"msgId"`) without changing the core mapping.

Both are demonstrated on a **synthetic toy log**
(`src/mapping/support/build_toy_collab_log.py` → `data/logs/ToyCollab/toy_collab.xes`: 100 cases, 1,132 events,
3 participants, designed to exercise both targets with variable in-flight backlogs and explicit `msgId`
correlation ids on send/receive events; backlog and latency are tied to a prefix-observable structural
property, the case's participant count, so the targets carry a genuine, learnable signal instead of one
drawn i.i.d. of the observed prefix). This is converted with the same
`collab_xes_to_ocel.py` converter as the study logs and run through the *same* OCPA 
feature-extraction/CV/RandomForest machinery as RQ3 (`ocpm_eval/rq_ext_pipeline.py`), 
demonstrating that the extensions work end-to-end in the native object-centric pipeline
on a dataset comparable in size to the study logs.
A dedicated pure-Python unit test (`tests/test_extensions_toy.py`) verifies label logic 
by hand on specific synthetic patterns.

---

## Design notes

- **Prediction tasks as a library.** `ocpm_tasks` is decoupled from the
  experimentation and can be used inside an OCPA- or pm4py-based pipeline: build the
  neutral model with an adapter, then call the label functions. It depends only on
  pandas (OCPA/pm4py are optional, lazy).
- **14 tasks** = the object-centric reformulation of the 14 tasks of the
  collaborative baseline (`ocpm_tasks/catalog.py`). The RQ3 representative subset
  is six tasks covering the anchor × problem-type combinations.
- **Object-enabled extensions.** Two extension tasks outside the 14-task taxonomy,
  X-Inf and X-MSt, **are** implemented in `ocpm_tasks/extensions.py` but kept out
  of `catalog.TASKS` — see the EXT section above. X-MSt presupposes a
  correspondence between individual send and receive observations that the core
  mapping (rule M4) deliberately does not establish; it is supplied by the opt-in
  `corr_attr` enrichment in `ocpm_tasks/adapters.py`, not by the converter. Further
  extension directions (cross-case participant load, message-convergence
  completion, inter-participant progress lag, intra-orchestration-case
  concurrency, cross-case resource load) are outlined as future work and are
  **not** implemented here.
- **Targets vs features.** Targets are defined over each collaboration instance's
  global trace (linear cut points, deterministic order); features are object-centric
  and past-relative (prefix-respecting) so per-event rows do not leak the future.
- **Alignment.** OCPA rows are matched to labels by `event_id`
  (`feature_storage.feature_graphs` → `node.event_id`) and validated by a
  remaining-time oracle that aborts on any mismatch.
