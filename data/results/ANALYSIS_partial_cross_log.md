# RQ3 Analysis — Partial Catalog, Predict-Collab vs BPI2013

Both `partial` stages: the same 6-task representative subset (`RQ3_SUBSET`) ×
the same 5 predictors × identical hyperparameters, 5-fold CV grouped by
`CollaborationCase`, `--profile` on.

| Stage | Logs | Cells | Samples / cell | Total samples |
|---|---|---|---|---|
| [`predictcollab_partial/`](predictcollab_partial/) | 4 | 24 | 549 – 2,260 | 35,617 |
| [`bpi2013_partial/`](bpi2013_partial/) | 1 | 6 | 18,481 – 62,030 | 328,631 |

All 150 rows `ran_end_to_end=True`; `evaluation.audit_stage` reports no
problems on either stage. Because the task set and the model configuration are
held fixed, **data scale is the only variable** — this document reports the
per-stage findings and then what changes between them.

This replicates the method of
[`predictcollab_full/ANALYSIS.md`](predictcollab_full/ANALYSIS.md) on the
partial catalog. The `full` catalog is not covered: BPI2013's `full` stage has
GNN results for only 6 of its 14 tasks, so a like-for-like 5-predictor
comparison is not available there.

## Environment

Identical machine for every number below, so timings are comparable to each
other but not portable:

| | |
|---|---|
| CPU | Intel Core i5-13400F — 10 physical / 16 logical cores, 2.5 GHz base |
| RAM | 15.8 GB |
| GPU | NVIDIA GTX 1660, 6 GB — present and CUDA-visible, **deliberately unused** |
| OS | Windows 10 Pro Education, build 19045 |
| Python | 3.10.11 |
| Key libs | torch 2.13.0+cu130, dgl 1.1.2+cu118, scikit-learn 1.7.2, xgboost 3.2.0, ocpa 1.3.3, pandas 2.3.3 |

Every predictor ran on **CPU** (`ExperimentConfig.device="cpu"`). That is a
measured choice: CUDA's per-batch dispatch overhead made the graph and sequence
models *slower* wall-clock, and re-testing on BPI2013 did not change the answer
(250.5 s CPU vs 324.9 s CUDA for one GNN fold, identical metrics) because a
larger log adds batches rather than work per batch.

One methodological difference between the stages, which affects **wall-clock
bookkeeping only, not any metric**: GNN's BPI2013 run was executed one task per
process (see [Operational note](#operational-note-why-gnn-ran-per-task)). Its
per-log stages were therefore measured once per task; the merged profile keeps
a single copy, matching what a single run would have recorded.

## Highlights

- **Scale decides whether model choice matters at all.** On Predict-Collab the
  five models sit within 0.010 macro-F1 of one another against a mean fold SD
  of 0.024 — the entire spread fits inside one model's own noise. On BPI2013
  the spread widens 5x to 0.051 while absolute fold SD *halves* to 0.011. The
  median winner's margin grows from 0.5% to 2.1% of the winning score. The
  ranking does not become reliable, but it stops being meaningless.
- **GNN goes from also-ran to dominant.** Outright wins flip from 3 of 21
  (last place, Predict-Collab) to **4 of 5** (BPI2013), including all three
  regressions by clear margins — `NV-PrT` MAE 1.155e6 against Random Forest's
  1.624e6, a 29% reduction. The structural model appears to need data volume
  before its inductive bias pays; on small logs it merely costs more.
- **GNN is the only model whose cost is genuinely predictable.** Its fit time
  per 1k samples is **13.72 s on Predict-Collab and 13.52 s on BPI2013** — a
  1.5% difference across a 9.2x change in data volume, with r(fit, samples) =
  0.99 and 1.00. Total fit scales 9.09x against 9.23x more data: almost exactly
  linear. Nothing else in the table behaves this well.
- **The tree ensembles' fixed budget finally shows.** Random Forest's fit grew
  only 2.0x and XGBoost's 1.4x for 9.2x the data, so their cost *per sample*
  **fell** 4.5x and 6.4x. `predictcollab_full`'s analysis predicted exactly
  this ("they would only start to separate on much larger logs"); this is the
  larger log, and they do.
- **LSTM scales worst — 32.7x cost for 9.2x data.** Its per-sample cost rises
  3.6x (0.70 → 2.53 s/1k) where the Transformer's rises only 1.2x. The likely
  cause is BPI2013's longer cases: the LSTM's advantage on Predict-Collab was
  cheap short sequences, and that advantage does not survive.
- **At scale, a cheap learner spends more time loading data than learning.**
  XGBoost's `fit` share drops from 72.4% to **37.5%** while `ocel_load` rises
  to **41.5%**; Random Forest's `fit` falls to 46.0% with `ocel_load` at 34.6%.
  Preprocessing is a fixed ~5.9 s on Predict-Collab and ~40 s on BPI2013 — it
  grows with the log, but the tree models' training does not, so it takes over.
- **Memory becomes a differentiator, and inverts the ranking.** Peak RSS spans
  only 1.27x across models on Predict-Collab (838–1064 MB) but 1.65x on BPI2013
  (1903–3147 MB) — and **Random Forest becomes the heaviest model of the five**
  (3147 MB, up 3.8x), overtaking GNN (2748 MB, up 2.6x). The cheapest model to
  train is the most expensive to hold in memory, because it fits 200 trees over
  the entire table at once while GNN streams mini-batches.
- **`NE-NMPr` is degenerate on BPI2013 but a real task on Predict-Collab.**
  Constant target (`n_labels=1`) on BPI2013, so all five models tie at 1.0;
  8 labels and ~0.72 macro-F1 on Predict-Collab. Degeneracy is a property of a
  (log, task) pair, never of a task alone.
- **Saturation ties are a small-log artefact.** Predict-Collab has 3 tied cells
  of 24 (two of them 5-way, only one degenerate); BPI2013 has 1 of 6, and it is
  the degenerate one. Near-deterministic targets that every model solves
  perfectly are a feature of small curated logs.
- **Absolute scores collapse, but that is cardinality, not failure.** Mean
  macro-F1 falls from ~0.81 to ~0.39, while `NE-NPaA` goes from 3 labels to 25.
  Lift over baseline stays clearly positive (+0.14 to +0.19), so the pipeline
  is still learning — macro-F1 over 25 classes simply cannot reach the values a
  3-class target shows.

## Accuracy

Degenerate cells excluded throughout (`Healthcare/OB-M`,
`BPI2013/NE-NMPr`); "outright wins" counts only cells with a unique best model.

| | Predict-Collab | BPI2013 |
|---|---|---|
| Mean macro-F1, best → worst | XGBoost 0.812 / GNN 0.811 / Transformer 0.809 / LSTM 0.803 / RF 0.801 | LSTM 0.411 / GNN 0.404 / XGBoost 0.380 / Transformer 0.376 / RF 0.361 |
| Best-to-worst spread | 0.010 | **0.051** |
| Mean fold SD | 0.024 | **0.011** |
| Relative fold SD | 3.4% | 6.0% |
| Median winner margin (relative) | 0.5% | 2.1% |
| Margin < winner's own fold SD | 8/9 clf, 10/12 reg | 2/2 clf, 3/3 reg |
| Outright wins | LSTM 6, RF 4, Transformer 4, XGBoost 4, **GNN 3** (of 21) | **GNN 4**, LSTM 1 (of 5) |
| Tied cells | 3 of 24 | 1 of 6 |

**The spread grows 5x while the noise halves**, so differences that were
invisible on the small logs become an order of magnitude larger relative to
fold variance. They are still inside the noise floor by the strict test — but
with only 5 decided cells on BPI2013 that test has little power, which is a
limitation of the partial catalog, not evidence of equivalence.

Per-task scores, same 6 tasks in both stages:

| Task | Metric | Predict-Collab (mean of 5 models) | BPI2013 (mean of 5 models) |
|---|---|---|---|
| `NE-NPaA` | macro-F1 ↑ | 0.771 | 0.090 |
| `NE-NMPr` | macro-F1 ↑ | 0.725 | *degenerate* (1.0) |
| `OB-M` | macro-F1 ↑ | 0.964 | 0.683 |
| `NV-PrT` | MAE ↓ | 9.21 | 1.38e6 |
| `NV-PaT` | MAE ↓ | 7.23 | 7.25e5 |
| `NV-NMPr` | MAE ↓ | 0.82 | 1.61 |

MAE is not comparable across stages (different units and target ranges); only
the classification column supports a direct reading.

## Cost

Total wall-clock seconds, all logs/tasks/folds, `note`-tagged rows excluded:

| Model | PC `fit` | BPI `fit` | **fit ×** | PC total | BPI total | PC s/1k | BPI s/1k | r(fit, samples) PC → BPI |
|---|---|---|---|---|---|---|---|---|
| GNN | 2442.5 | 22208.4 | **9.1** | 2474.0 | 22456.9 | 13.72 | **13.52** | 0.99 → 1.00 |
| Transformer | 368.5 | 3995.8 | 10.8 | 375.0 | 4043.2 | 2.15 | 2.58 | 1.00 → n/a |
| LSTM | 119.9 | 3926.0 | **32.7** | 126.0 | 3965.8 | 0.70 | 2.53 | 0.72 → n/a |
| Random Forest | 17.8 | 36.3 | **2.0** | 27.4 | 78.8 | 0.100 | 0.022 | 0.63 → 0.90 |
| XGBoost | 17.2 | 24.2 | **1.4** | 23.7 | 64.4 | 0.097 | 0.015 | 0.20 → 0.23 |

Reference point: **9.23x more samples** (35,617 → 328,631).

Read the `fit ×` column against that 9.23x:

- **GNN 9.1x — linear.** Its per-sample cost is effectively a constant of the
  architecture (13.7 vs 13.5 s/1k). Expensive, but you can price a new log
  from its event count alone.
- **Transformer 10.8x — mildly superlinear**, consistent with O(seq_len²)
  attention as cases lengthen.
- **LSTM 32.7x — badly superlinear**, 3.5x worse than the data growth. On the
  small logs LSTM was 3x cheaper than the Transformer; at scale they converge
  (2.53 vs 2.58 s/1k) and LSTM's advantage disappears entirely.
- **RF 2.0x and XGBoost 1.4x — sublinear**, because 200 trees is a fixed budget
  that barely notices more rows. Their r(fit, samples) stays low (0.90, 0.23).

The `r` values are `n/a` for LSTM and Transformer on BPI2013 for a structural
reason: that stage has one log, and five of its six tasks share exactly 62,030
samples. With the degenerate sixth task short-circuited by those two models,
their remaining points have zero variance in `samples`, so the correlation is
undefined rather than absent.

### Which phase dominates

Share of each model's own total (%):

| Model | | `ocel_load` | `feature_extraction` | `labeling` | `fit` | `predict` |
|---|---|---|---|---|---|---|
| GNN | PC → BPI | 0.2 → 0.1 | 0.1 → 0.1 | 0.0 → 0.0 | 98.7 → 98.9 | 1.0 → 0.9 |
| Transformer | PC → BPI | 1.2 → 0.7 | 0.4 → 0.3 | 0.0 → 0.0 | 98.3 → 98.8 | 0.1 → 0.1 |
| LSTM | PC → BPI | 3.5 → 0.7 | 1.2 → 0.3 | 0.1 → 0.0 | 95.1 → 99.0 | 0.1 → 0.0 |
| Random Forest | PC → BPI | 15.4 → **34.6** | 6.2 → 15.4 | 0.3 → 0.7 | 64.8 → **46.0** | 13.3 → 3.3 |
| XGBoost | PC → BPI | 18.5 → **41.5** | 6.7 → 18.8 | 0.5 → 1.1 | 72.4 → **37.5** | 1.9 → 1.1 |

Preprocessing is model-independent and rises from ~5.9 s to ~40 s per log
(`ocel_load` 1.09 → 27.27 s, `feature_extraction` 0.40 → 12.52 s). For the
neural models that is still invisible. For the tree models it **overtakes
training**: XGBoost now spends more time loading the OCEL than fitting.

This is the concrete answer to a question the earlier analysis could only
speculate about — optimising OCEL loading is worthless for the expensive models
at any scale, and becomes the single largest lever for the cheap ones as logs
grow.

### Memory

Peak RSS (MB):

| Model | Predict-Collab | BPI2013 | growth |
|---|---|---|---|
| Random Forest | 838 (lowest) | **3147 (highest)** | 3.8x |
| GNN | 1064 (highest) | 2748 | 2.6x |
| Transformer | 1003 | 2191 | 2.2x |
| LSTM | 958 | 2007 | 2.1x |
| XGBoost | 930 | 1903 (lowest) | 2.0x |

Spread widens from 1.27x to 1.65x, and the ordering **inverts**: the model with
the smallest footprint on the small logs has the largest on the big one. Random
Forest materialises the full feature table and 200 fully-grown trees
simultaneously, whereas the neural models stream mini-batches and hold only
their parameters. On a 15.8 GB machine this is the constraint most likely to
bind first if the log grows another order of magnitude — and it points at the
opposite model from the one the runtime table would suggest.

## What changes with scale — summary

| Property | Small logs (Predict-Collab) | Large log (BPI2013) |
|---|---|---|
| Does model choice matter? | No — spread inside one model's noise | Marginally — spread 5x wider, noise halved |
| Best model | none distinguishable (XGBoost 0.812 nominal) | GNN wins 4 of 5 decided cells |
| Cheapest to train | XGBoost / RF, ~17 s | XGBoost / RF, ~25–36 s (barely moved) |
| Most predictable cost | GNN (r=0.99) | GNN (r=1.00, same s/1k) |
| Bottleneck for tree models | training (65–72%) | **data loading** (35–42%) |
| Memory-critical model | GNN | **Random Forest** |
| Ties | 3 of 24 (saturation) | 1 of 6 (degenerate only) |

The practical reading: **cost rankings established on small logs do not
transfer, and two of them invert.** LSTM looks 3x cheaper than the Transformer
on small logs and is not at scale; Random Forest looks the lightest on memory
and becomes the heaviest. Only GNN's cost model transfers unchanged — it is the
most expensive option by a wide margin, but the only one you can extrapolate.

Conversely, the *accuracy* conclusion strengthens rather than transfers: model
choice bought nothing on the small logs, and buys something real, though still
noise-adjacent, at scale — and what it buys favours the structural model that
looked least attractive on the small logs.

## Operational note: why GNN ran per-task

GNN's BPI2013 stage was run one task per process rather than as a single
6-task run. A first monolithic attempt stalled: after 27 of 30 folds at a
stable ~849 s each, one fold ran past 125 minutes (8.8x) while consuming 11
cores at 53% kernel time, with 8.7 GB RAM free and no competing process. It was
killed, and since `run_rq3` writes its CSVs only after every task completes,
all 7.7 h of completed folds were lost with it.

Re-run one task per process, all 6 tasks completed with a maximum fold-time
deviation of **1.02x**. The same task (`OB-M`) that stalled ran clean in
841–856 s.

`OB-M` is the last entry in the task catalog, so it is always the task running
when a monolithic process has been alive longest — and it is the task on which
every severe timing outlier in this project has landed, across three different
predictors. Process lifetime, not that task's data, is the common factor. The
per-task structure was adopted for checkpointing but appears to prevent the
condition as well.

Bookkeeping consequence: each per-task run repeats `ocel_load` and
`feature_extraction`, so the merged profile keeps one copy of each rather than
six. Without that, GNN's totals would be inflated against the four predictors
that ran monolithically. No metric is affected.

## Reproducing this analysis

Run `python -m evaluation.audit_stage --all` first — it reports degenerate
cells, tied optima, unfitted cells, fold coverage and timing outliers, and
exits non-zero on a problem. Then apply the eleven checks documented in
[`predictcollab_full/ANALYSIS.md`](predictcollab_full/ANALYSIS.md#reproducing-this-study-on-another-stage),
which transfer unchanged to any stage directory.

For a cross-stage comparison, three additional rules:

1. **Hold the task set fixed.** Comparing `partial` against `full` would
   confound scale with task mix. Both stages here run the same 6 tasks.
2. **Normalise cost by sample count, not by wall clock.** The `fit ×` column
   is only interpretable against the sample-count ratio (9.23x here); seconds
   per 1k samples is what exposes super- and sub-linear scaling.
3. **Check degeneracy per (log, task), never per task.** `NE-NMPr` is
   degenerate on one stage and a normal 8-class target on the other.

Known limitation: scale and provenance are confounded. BPI2013 is both ~9x
larger *and* a real-world log, against four small curated study logs, and it
contributes a single log against four. Effects should be read as "what changes
between small curated logs and a large real one", not as a controlled
size sweep.
