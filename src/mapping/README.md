# mapping — collaborative XES → OCEL 2.0 converter

Model-to-model transformation **μ: extended collaborative XES → OCEL 2.0**
(Berti et al. 2023, Definition 2), implementing mapping rules **M1–M8** and
machine-checking consistency properties **P1.1–P1.5**. This is RQ1 of the
study: the converter that produces the rich object-centric logs consumed by
[`ocpm_tasks`](../ocpm_tasks/README.md) and [`ocpm_eval`](../ocpm_eval/README.md).

It is a standalone tool: pure pandas/Python for the transformation and
checks (unit-testable without pm4py), with pm4py used only at the two I/O
endpoints (reading XES, writing OCEL 2.0).

## What it does

Given an extended collaborative XES log, `collab_xes_to_ocel.py`:

1. **Reads** the XES file (`pm4py.read_xes`), grouping events by case and
   ordering each trace by `(timestamp, source order)`.
2. **Transforms** it into four OCEL 2.0 tables (events, objects, E2O
   relations, O2O relations) per rules M1–M8 (see below).
3. **Checks** the result against consistency properties P1.1–P1.5 and
   prints a pass/fail report with transformation stats (event/object/
   relation counts, message counts, unmatched receives).
4. **Exports** `<output>.jsonocel` and `<output>.sqlite`.
5. **Validates** the exported `.jsonocel` against the embedded OCEL 2.0
   JSON schema (draft-07) — full validation via `jsonschema` if installed,
   otherwise a dependency-free structural fallback.

## Mapping rules (M1–M8)

| Rule | Produces |
|---|---|
| M1 | `CollaborationInstance` object per global case |
| M2 | `Participant` object per orchestration/pool (BPMN terminology — a pool like "Laboratory", not a role or person; a global object reused across cases) |
| M3 | `LocalCase` object per (case, participant) — the execution of a participant's orchestration within one global case |
| M4 | `Message` object per `SendTask`, correlated to the earliest unmatched later `ReceiveTask` whose (from, to) mirror the send's (participant, to) — a declared ETL heuristic, since XES has no native message identity |
| M5 | `Event` per source event; activity = XES activity; `elemType` (`task`/`SendTask`/`ReceiveTask`) preserved as an attribute |
| M6 | E2O relations: `within` (event→CI), `local` (event→LocalCase), `send`/`receive` (event→Message), plus a redundant `actor` edge (event→Participant) — see note below |
| M7 | O2O relations: `part_of` (LocalCase→CI), `executed_by` (LocalCase→Participant), `from`/`to` (Message→Participant), `exchanged_in` (Message→CI) |
| M8 | Residual source event attributes (not consumed by M1–M7) are carried over unchanged |

**Note on the `actor` E2O edge.** Conceptually a participant is reached via
`local -> executed_by`, keeping the orchestration (Participant) distinct
from its per-case projection (LocalCase). But pm4py's OCEL 2.0 exporters
drop any object reachable only via O2O, which would silently lose every
`Participant` object on export. The converter therefore adds a redundant
`actor` E2O edge purely as an export-compatibility workaround — it is not
part of the conceptual model. Consumers that read the exported SQLite for
object-centric feature extraction (OCPA) must strip this edge first, since
OCPA's default execution-extraction connects all E2O-related objects of an
event pairwise and this edge would merge every `CollaborationInstance`
that shares a `Participant`; see `ocpm_eval/io_ocel.py`'s
`_strip_actor_e2o`.

## Consistency checks (P1.1–P1.5)

Machine-checked guards against implementation defects (the mapping's
correctness argument is by construction; these checks catch bugs):

| Check | Verifies |
|---|---|
| P1.1 Totality | one OCEL event per source event; timestamps preserved |
| P1.2 Per-case partition | each `CollaborationInstance`'s `within`-related events == that case's source events, no dangling edges |
| P1.3 Message well-formedness | exactly one `send`, at most one `receive` per Message; `from`/`to` agree with sender/receiver attributes; receive timestamp ≥ send timestamp |
| P1.4 Local-case coherence | `local`/`part_of` agree with `within`; each `LocalCase` is `executed_by` exactly one `Participant`, name-consistent |
| P1.5 No orphan objects | every object is referenced by at least one E2O or O2O relation |

## Usage

```bash
# Convert a collaborative XES log to OCEL 2.0 (.jsonocel + .sqlite)
python src/mapping/collab_xes_to_ocel.py input.xes output

# Abort if any P1 check or schema validation fails
python src/mapping/collab_xes_to_ocel.py input.xes output --strict

# Skip OCEL 2.0 schema validation of the output
python src/mapping/collab_xes_to_ocel.py input.xes output --no-validate

# Verbose logging
python src/mapping/collab_xes_to_ocel.py input.xes output -v
```

`output` is a base path/name; the tool appends `.jsonocel` and `.sqlite`
(any of `.jsonocel`/`.sqlite`/`.json` passed in `output` is stripped first).

### Source format

The extended XES must carry the `collab` extension attributes defined in
`aux/collab.xesext`, plus the standard XES case/activity/timestamp keys:

| XES key | Meaning |
|---|---|
| `case:concept:name` | global case id |
| `concept:name` | activity |
| `time:timestamp` | event timestamp |
| `collab:elemType` | `task` \| `SendTask` \| `ReceiveTask` |
| `collab:participant` | participant executing the event |
| `collab:fromParticipant` | sender (Send/Receive events) |
| `collab:toParticipant` | receiver (Send/Receive events) |

## Module layout

```
mapping/
├── collab_xes_to_ocel.py   # the transformation (M1-M8), checks (P1.1-P1.5), I/O, CLI
└── aux/
    ├── collab.xesext           # collab XES extension definition (source vocabulary)
    ├── ocel20-schema-json.json # OCEL 2.0 JSON schema (draft-07), reference copy
    └── printOCEL.py             # debug script: load a .jsonocel with pm4py, discover
                                  #   an OC-Petri-net / OC-DFG, save PNGs (not part of
                                  #   the conversion pipeline; requires graphviz)
```

## Setup

Requires **pm4py >= 2.7.16** (OCEL 2.0 write support), which conflicts with
the `pm4py==2.2.32` pinned by OCPA in the evaluation environment — run this
tool from a **separate virtual environment**. See the root
[README.md](../../README.md#setup-macos-virtual-environments) for the full
two-environment setup (`.venv-mapping` here, `.venv` for
`ocpm_eval`/`ocpm_tasks`).

```bash
python3.10 -m venv .venv-mapping
source .venv-mapping/bin/activate
pip install -r requirements-mapping.txt
```

## Where the output goes

The converted `.sqlite`/`.jsonocel` files are the **R2** input consumed
downstream:
- `ocpm_tasks.adapters.from_ocel2_sqlite` / `from_ocpa` build the neutral
  object-centric model from them for label computation.
- `ocpm_eval` reads the same files for both labels (`io_ocel.py`) and
  OCPA feature extraction (`features_ocpa.py`).
- RQ2 fidelity also reads the *original* `.xes` directly (R1, source-level
  labels) to check agreement against the R2 labels — see
  `ocpm_eval/rq2_fidelity.py`.

See [`data/logs/`](../../data/logs/) for example XES sources and their
converted OCEL 2.0 outputs.
