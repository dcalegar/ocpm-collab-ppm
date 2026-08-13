# mapping — collaborative XES → OCEL 2.0 converter

Model-to-model transformation **μ: extended collaborative XES → OCEL 2.0**
(Berti et al. 2023, Definition 2), implementing mapping rules **M1–M8** and
machine-checking consistency properties **P1.1–P1.7**. This is RQ1 of the
study (see [Research questions](../../README.md#research-questions)): the
converter that produces the rich object-centric logs consumed by
[`tasks`](../tasks/README.md), [`features`](../features/README.md),
and [`evaluation`](../evaluation/README.md).

It is a standalone tool: pure pandas/Python for the transformation and
checks (unit-testable without pm4py), with pm4py used only at the two I/O
endpoints (reading XES, writing OCEL 2.0).

## What it does

Given an extended collaborative XES log, `collab_xes_to_ocel.py`:

1. **Reads** the XES file (`pm4py.read_xes`), grouping events by case and
   ordering each trace by `(timestamp, source order)`.
2. **Transforms** it into four OCEL 2.0 tables (events, objects, E2O
   relations, O2O relations) per rules M1–M8 (see below).
3. **Checks** the result against consistency properties P1.1–P1.7 and
   prints a pass/fail report with transformation stats (event/object/
   relation counts, message counts). P1.1–P1.6 run over the pre-export
   `TransformResult`; P1.7 runs after the export witnesses are added, over
   the E2O table actually handed to the exporters (see below).
4. **Exports** `<output>.jsonocel` and `<output>.sqlite`.
5. **Validates** the exported `.jsonocel` against the embedded OCEL 2.0
   JSON schema (draft-07) — full validation via `jsonschema` if installed,
   otherwise a dependency-free structural fallback.

## Mapping rules (M1–M8)

| Rule | Produces |
|---|---|
| M1 | `CollaborationCase` object per collaboration case |
| M2 | one participant object per participant identifier, **typed by the identifier itself** — a participant is an orchestration/pool (BPMN terminology, a pool like "Laboratory", not a role or person) and a global object reused across cases. There is no single `Participant` type; the participants of a log are declared as object types (see *Participant object types* below) |
| M3 | `OrchestrationCase` object per (case, participant) — the participant's local execution within one collaboration case, under the assumption that each participant executes its local process at most once per collaboration case |
| M4 | `Message` object per `SendTask` **and** per `ReceiveTask` — one per communication *observation*, not a correlated message instance. The source logs do not guarantee a message identifier or other correlation information, so the core mapping does **not** infer any correspondence between a send observation and a receive observation |
| M5 | `Event` per source event; activity = XES activity; `elemType` (`task`/`SendTask`/`ReceiveTask`) preserved as an attribute |
| M6 | E2O relations: `in_collaboration` (event→CC), `in_orchestration` (event→OrchestrationCase), `send`/`receive` (event→its own Message), plus the direct `in_participant` edge (event→participant object) — see note below |
| M7 | O2O relations: `part_of` (OrchestrationCase→CC), `for_participant` (OrchestrationCase→participant object), `from`/`to` (Message→participant object), `exchanged_in` (Message→CC) |
| M8 | Residual source event attributes (not consumed by M1–M7) are carried over unchanged. `collab:participant`, `collab:elemType`, `fromParticipant`, and `toParticipant` are additionally retained under fixed attribute names (`participant`, `elemType`, `fromParticipant`, `toParticipant`) even though they are also materialized by E2O/O2O relations |

**Note on the `in_participant` E2O edge.** A participant is reachable two
ways: directly, via the `in_participant` edge (event→participant object), and
indirectly via `in_orchestration -> for_participant`, keeping the
participant distinct from its per-case execution
(`OrchestrationCase`). Both are part of the conceptual model (rule M6);
their agreement is machine-checked by P1.6. The direct edge also has a
practical motivation: pm4py's OCEL 2.0 exporters drop any object
reachable only via O2O, which would silently lose every participant
object on export without it. Consumers that read the exported SQLite for
object-centric feature extraction (OCPA) must still strip this edge
before import, since OCPA's default execution-extraction connects all
E2O-related objects of an event pairwise and this edge would merge every
`CollaborationCase` that shares a participant; see
`features/io_ocel.py`'s `_strip_participant_e2o`.

## Participant object types

Rule M2 makes the object type of a participant object the participant
identifier itself, so the object types of an exported log are
`CollaborationCase`, `OrchestrationCase`, `Message`, and one type per
participant (`Hospital`, `Laboratory`, … for Healthcare; `OrgLineA2`, … for
BPIC 2013). The identifier is *also* kept as the `name` object attribute, so it
remains readable as a value; P1.4 requires the two encodings to agree.

The type is an **injective, identifier-safe encoding** of the participant name
rather than the raw string: runs of non-alphanumeric characters are dropped and
each following part is capitalized (`Org line A2` → `OrgLineA2`); an
alphanumeric identifier is a fixed point, so the common case reads as the
participant name itself. Two properties are enforced when the registry is built,
and the converter raises rather than emitting a log that violates them:

- **Identifier-safety.** OCPA's OCEL 2.0 importer materializes one DataFrame
  column per object type and resolves it with `getattr(row, object_type)` over
  `itertuples`, silently dropping any type whose name is not a valid Python
  identifier.
- **Injectivity, including under pm4py's name stripping.** pm4py's SQLite
  exporter writes one physical `object_<stripped name>` table per object type,
  so two participants whose *stripped* names coincide would collide on one
  table. A collision appends a numeric suffix until both images are free.

The qualifier vocabulary is unaffected and stays fixed, so participant objects
remain addressable without naming any of their types — through `in_participant`,
`for_participant`, `from` and `to`. That is how the downstream reader
identifies them (`tasks/schema.py::Schema.is_participant_type`), rather
than by comparing against a single type name.

## Event order preservation (criterion P1.2)

The per-case event order of the source XES log (timestamp order, ties broken by
appearance order — the total order ≺_L) is encoded in the **event
identifier itself**: ids are created as `e::<case>::<idx>` with the per-case
index **zero-padded** to the width needed for that case, so lexicographic
identifier order agrees exactly with ≺_L (μ_E is an order-embedding; check
P1.2b verifies this on every export).

**Important precondition for trace reconstruction:**
- Consumers must enumerate the events of a case **explicitly in identifier
  order** (e.g. an `ORDER BY` on the event id when querying the SQLite
  serialization, or sorting the id strings after reading the JSON)
- Neither OCEL 2.0 serialization guarantees a canonical enumeration order for a
  set of events on its own; **do not rely on insertion/row/array order** of an
  unordered query or scan (SQLite gives no such guarantee without `ORDER BY`)
- The mapping emits no separate order attribute; the zero-padded rank inside
  the id is the order carrier

The project's own reader (`tasks/adapters.py::from_ocel2_sqlite`) issues
`ORDER BY ocel_id` accordingly. Third-party importers that scan tables
unordered (e.g. pm4py's OCEL2-SQLite reader) should have their events re-sorted
by event id after reading.

## Consistency checks (P1.1–P1.7)

Machine-checked guards against implementation defects (the mapping's
correctness argument is by construction; these checks catch bugs):

P1.1–P1.6 compare the constructed output against a **recomputed
expectation** from the source log (a fresh call to `_sorted_case_events`),
not just aggregate counts read off the transform's own output. This
recomputation is independent of `transform()`'s in-memory state, but not of
`_sorted_case_events` itself (both routes call the same function), so a
systematic defect in that function's ordering/normalization logic would
reproduce on both sides and go undetected. P1.1–P1.6 also run over
`TransformResult` (the pre-export DataFrames), not the artifact actually
written to disk; P1.7 closes that second gap, running over the E2O table
handed to the OCEL exporters after the export-only reachability witnesses
are added (see `_add_export_reachability_witnesses`).

| Check | Verifies |
|---|---|
| P1.1 Totality | one OCEL event per source event, with the *same activity, timestamp, and preserved `collab:participant`/`collab:elemType`* checked per event id (not just matching totals) |
| P1.2 Per-case partition | each `CollaborationCase`'s `in_collaboration`-related events equal the *exact set* of that case's source event ids (not just a matching count), no dangling edges |
| P1.2b Per-case order | the identifier order reproduces `prec_L` exactly, timestamp *and* its tie-break by source order (not just a non-decreasing-timestamp check, which a tie inversion would still pass) |
| P1.3 Message well-formedness | every Message is related to exactly one event, by `send` **xor** `receive` (never both), and conversely every source send/receive event is related to *exactly one* Message (which events count as communication events is derived from the *source* log, not from the output's own `elemType`); `from`/`to` O2O relations agree with the Message's sender/receiver attributes and with the related event's preserved `fromParticipant`/`toParticipant` attributes whenever both are defined; the related event's participant is the sender (send) or the receiver (receive). Counterparty endpoints the source never recorded are reported explicitly (`n_messages_missing_sender`/`_receiver`) rather than silently skipped or treated as a disagreement; an attribute/relation pair that disagrees on whether it is defined at all *does* fail the check |
| P1.4 OrchestrationCase coherence | `in_orchestration`/`part_of` agree with `in_collaboration`; each `OrchestrationCase` is `for_participant` exactly one participant object, whose `name` attribute equals the `OrchestrationCase`'s own `participant` attribute and whose object **type** equals *tau*(that same attribute) -- not the identifier itself, since M2's object type is the injective encoding tau(participant), not the raw identifier string |
| P1.5 No orphan objects | every object is referenced by at least one E2O or O2O relation |
| P1.6 Participant coherence | for every event, the participant object reached by the direct `in_participant` edge equals the one reached via `in_orchestration -> for_participant` |
| P1.7 Export reachability | every object materialized by `transform()` is reachable in the E2O relations table actually handed to the OCEL exporters (`export_relations_df`, after `_add_export_reachability_witnesses`), not merely in `TransformResult` |

**Counterparty endpoint completeness.** The source format may leave a
send/receive event's own side implicit (backfilled from
`collab:participant`) but may also leave the *counterparty* side
(`toParticipant` of a Send, `fromParticipant` of a Receive) unrecorded. That
side is never guessed: `from`/`to` stay undefined for that Message, with no
fabricated O2O relation or `sender`/`receiver` attribute. This occurs in
practice: 26 of 550 `SendTask` events in Healthcare and 44 of 530 in
Artificial5 lack `toParticipant` (Artificial1 and Real4: none).

## Usage

```bash
# Convert a collaborative XES log to OCEL 2.0 (.jsonocel + .sqlite)
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
`.venv-mapping\Scripts\activate.bat` in `cmd.exe`); `input.xes`/`output` with
forward slashes work as-is, no need to switch to backslashes.

`output` is a base path/name; the tool appends `.jsonocel` and `.sqlite`
(any of `.jsonocel`/`.sqlite`/`.json` passed in `output` is stripped first).

### Source format

The extended XES must carry the `collab` extension attributes defined in
`support/collab.xesext`, plus the standard XES case/activity/timestamp keys:

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
├── collab_xes_to_ocel.py   # the transformation (M1-M8), checks (P1.1-P1.7), I/O, CLI
└── support/
    ├── collab.xesext           # collab XES extension definition (source vocabulary)
    ├── ocel20-schema-json.json # OCEL 2.0 JSON schema (draft-07), reference copy
    ├── build_leakage_test_logs.py # builds the tests/fixtures/leak_*.sqlite logs
                                  #   used by tests/test_features_leakage.py (needs
                                  #   pm4py>=2.7, i.e. this venv, not the OCPA one)
    └── printOCEL.py             # debug script: load a .jsonocel with pm4py, discover
                                  #   an OC-Petri-net / OC-DFG, save PNGs (not part of
                                  #   the conversion pipeline; requires graphviz)
```

> This directory used to be named `aux/`; it was renamed to `support/` because
> `aux` is a reserved DOS device name on Windows (like `con`, `prn`, `nul`),
> which silently broke `git clone`/checkout for Windows users.

## Setup

Requires **pm4py >= 2.7.16** (OCEL 2.0 write support), which conflicts with
the `pm4py==2.2.32` pinned by OCPA in the evaluation environment — run this
tool from a **separate virtual environment**. See the root
[README.md](../../README.md#setup-virtual-environments) for the full
two-environment setup (`.venv-mapping` here, `.venv` for
`evaluation`/`tasks`), including the Apple Silicon and Windows notes.

**macOS / Linux:**

```bash
python3.10 -m venv .venv-mapping
source .venv-mapping/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-mapping.txt
pip install -e .
```

**Windows (PowerShell):**

```powershell
py -3.10 -m venv .venv-mapping
.venv-mapping\Scripts\Activate.ps1   # cmd.exe: .venv-mapping\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements-mapping.txt
pip install -e .
```

## Where the output goes

The converted `.sqlite`/`.jsonocel` files are the **R2** input consumed
downstream:
- `tasks.adapters.from_ocel2_sqlite` / `from_ocpa` build the neutral
  object-centric model from them for label computation.
- `features` reads the same files for both labels (`io_ocel.py`) and
  OCPA feature extraction (`ocpa.py`).
- RQ2 fidelity also reads the *original* `.xes` directly (R1, source-level
  labels) to check agreement against the R2 labels — see
  `evaluation/rq2_fidelity.py`.

See [`data/logs/`](../../data/logs/) for example XES sources and their
converted OCEL 2.0 outputs.
