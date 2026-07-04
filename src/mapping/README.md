# mapping — collaborative XES → OCEL 2.0 converter

Model-to-model transformation **μ: extended collaborative XES → OCEL 2.0**
(Berti et al. 2023, Definition 2), implementing mapping rules **M1–M8** and
machine-checking consistency properties **P1.1–P1.6**. This is RQ1 of the
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
3. **Checks** the result against consistency properties P1.1–P1.6 and
   prints a pass/fail report with transformation stats (event/object/
   relation counts, message counts).
4. **Exports** `<output>.jsonocel` and `<output>.sqlite`.
5. **Validates** the exported `.jsonocel` against the embedded OCEL 2.0
   JSON schema (draft-07) — full validation via `jsonschema` if installed,
   otherwise a dependency-free structural fallback.

## Mapping rules (M1–M8)

| Rule | Produces |
|---|---|
| M1 | `CollaborationCase` object per collaboration case |
| M2 | `Participant` object per orchestration/pool (BPMN terminology — a pool like "Laboratory", not a role or person; a global object reused across cases) |
| M3 | `ParticipantProjection` object per (case, participant) — the execution of a participant's orchestration within one collaboration case |
| M4 | `Message` object per `SendTask` **and** per `ReceiveTask` — one per communication *observation*, not a correlated message instance. The source logs do not guarantee a message identifier or other correlation information, so the core mapping does **not** infer any correspondence between a send observation and a receive observation |
| M5 | `Event` per source event; activity = XES activity; `elemType` (`task`/`SendTask`/`ReceiveTask`) preserved as an attribute |
| M6 | E2O relations: `within` (event→CC), `in_projection` (event→ParticipantProjection), `send`/`receive` (event→its own Message), plus the direct `participant` edge (event→Participant) — see note below |
| M7 | O2O relations: `projection_of` (ParticipantProjection→CC), `for_participant` (ParticipantProjection→Participant), `from`/`to` (Message→Participant), `exchanged_in` (Message→CC) |
| M8 | Residual source event attributes (not consumed by M1–M7) are carried over unchanged. `collab:participant`, `collab:elemType`, `fromParticipant`, and `toParticipant` are additionally retained under fixed attribute names (`participant`, `elemType`, `fromParticipant`, `toParticipant`) even though they are also materialized by E2O/O2O relations |

**Note on the `participant` E2O edge.** A participant is reachable two
ways: directly, via the `participant` edge (event→Participant), and
indirectly via `in_projection -> for_participant`, keeping the
orchestration (Participant) distinct from its per-case projection
(ParticipantProjection). Both are part of the conceptual model (rule M6);
their agreement is machine-checked by P1.6. The direct edge also has a
practical motivation: pm4py's OCEL 2.0 exporters drop any object
reachable only via O2O, which would silently lose every `Participant`
object on export without it. Consumers that read the exported SQLite for
object-centric feature extraction (OCPA) must still strip this edge
before import, since OCPA's default execution-extraction connects all
E2O-related objects of an event pairwise and this edge would merge every
`CollaborationCase` that shares a `Participant`; see
`ocpm_eval/io_ocel.py`'s `_strip_participant_e2o`.

## Event order preservation (criterion P1.2)

The per-case event order of the source XES log is preserved during transformation through **insertion order**: events are emitted in source order (sorted by timestamp, ties broken by appearance order), and both OCEL 2.0 serializations (JSON and SQLite) maintain this insertion order. 

**Important precondition for trace reconstruction:**
- Consumers reading the OCEL output must preserve insertion order when accessing events
- Any reordering, sorting, or manipulation of events will lose the original trace order
- The mapping does not emit an explicit order attribute; order is an implicit property of the serialization

This design keeps the transformation lightweight while relying on standard serialization semantics. Tools like OCPA that read OCEL from SQLite in insertion order will reconstruct traces correctly; tools that reorder events (e.g., by timestamp, activity, or ID) may produce different outputs.

## Consistency checks (P1.1–P1.6)

Machine-checked guards against implementation defects (the mapping's
correctness argument is by construction; these checks catch bugs):

| Check | Verifies |
|---|---|
| P1.1 Totality | one OCEL event per source event; timestamps preserved |
| P1.2 Per-case partition | each `CollaborationCase`'s `within`-related events == that case's source events, no dangling edges |
| P1.3 Message well-formedness | every Message is related to exactly one event, by `send` **xor** `receive` (never both); `from`/`to` O2O relations agree with the Message's sender/receiver attributes and with the related event's preserved `fromParticipant`/`toParticipant` attributes; the related event's participant is the sender (send) or the receiver (receive) |
| P1.4 Participant-projection coherence | `in_projection`/`projection_of` agree with `within`; each `ParticipantProjection` is `for_participant` exactly one `Participant`, name-consistent |
| P1.5 No orphan objects | every object is referenced by at least one E2O or O2O relation |
| P1.6 Participant coherence | for every event, the `Participant` reached by the direct `participant` edge equals the one reached via `in_projection -> for_participant` |

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
├── collab_xes_to_ocel.py   # the transformation (M1-M8), checks (P1.1-P1.6), I/O, CLI
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
