# ocpm_eval — evaluation stages (V4 profile)

Modular evaluation; task definitions/labels come from the decoupled `ocpm_tasks`
library. Inputs are **OCEL 2.0 SQLite** logs (the format OCPA imports natively). One
module per research question.

| Stage | Module | Output |
|---|---|---|
| **RQ2** label fidelity | `rq2_fidelity.py` | `rq2_fidelity.csv` |
| **RQ3** end-to-end feasibility | `rq3_pipeline.py` | `rq3_results.csv` |
| **RQ4** structure & expressiveness | `rq4_structure.py` | `rq4_structure.csv`, `rq4_expressiveness.csv` |
| RQ1 transformation + P1 | — | converter (separate tool) |

Reading: the **label** side reads the OCEL 2.0 SQLite with the stdlib reader
`ocpm_tasks.adapters.from_ocel2_sqlite` (OCPA pins pm4py==2.2.32, which cannot read
OCEL 2.0, so pm4py is not used for reading). The **feature** side reads the same file
through OCPA's native importer `ocpa.objects.log.importer.ocel2.sqlite`, with
leading-type process-execution extraction (one execution per CollaborationInstance);
the feature↔label alignment is validated by a remaining-time oracle.

RQ3 representative subset (`ocpm_tasks.catalog.RQ3_SUBSET`): NE-NPaA, NE-NMPr, NV-PrT,
NV-PaT, NV-NMPr, OB-M. 5-fold CV grouped by CollaborationInstance; macro-F1 / MAE as
mean±sd plus a trivial baseline. No times reported.
