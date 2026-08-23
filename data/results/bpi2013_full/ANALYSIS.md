# RQ3 Analysis — BPI Challenge 2013, Full Catalog

BPI Challenge 2013 (incidents, collaborative — the real-world validation log,
not one of the four Predict-Collab study logs) × all 14 tasks × 5 predictors
(Random Forest, XGBoost, LSTM, Transformer, GNN), 5-fold CV grouped by
`CollaborationCase`, `--profile` on. 14 (log, task) rows per predictor, 70 in
total, all `ran_end_to_end=True`, no errors. Source: `rq3_results_*.csv` /
`rq3_profile_*.csv` in this directory; charts in [`plots/`](plots/).

This replicates, on the largest available log, the method of
[`../predictcollab_full/ANALYSIS.md`](../predictcollab_full/ANALYSIS.md)
(same 11 checks, listed at the end of that document). The cross-stage
comparison lives in
[`../ANALYSIS_full_cross_log.md`](../ANALYSIS_full_cross_log.md).

## Environment

All timings below come from this single machine, so they are comparable to
each other but not portable as absolute figures:

| | |
|---|---|
| CPU | Intel Core i5-13400F — 10 physical / 16 logical cores (6 P-cores + 4 E-cores), 2.5 GHz base |
| RAM | 15.8 GB |
| GPU | NVIDIA GTX 1660, 6 GB — **present and CUDA-visible, but deliberately unused** |
| OS | Windows 10 Pro Education, build 19045 |
| Python | 3.10.11 |
| Key libs | torch 2.13.0+cu130 (10 intra-op threads), dgl 1.1.2+cu118, scikit-learn 1.7.2, xgboost 3.2.0, ocpa 1.3.3, numpy 2.2.6, pandas 2.3.3 |

Every predictor ran on **CPU** (`ExperimentConfig.device="cpu"`) — a measured
choice, re-tested on this log specifically (one GNN fold: 250.5 s CPU vs
324.9 s CUDA, identical metrics). Fixed epoch budgets, no early stopping
anywhere: `lstm_epochs=60` (batch 16), `transformer_epochs=35` (batch 32),
`gnn_epochs=50` (batch 32, hidden 64, k=8).

### Provenance of the GNN rows

The GNN's 14 rows come from **two runs**, using the reuse shortcut documented
in [`RQ3_EXECUTION_PLAN.md`](../../../src/evaluation/RQ3_EXECUTION_PLAN.md)
("Optional: reusing `partial` results inside `full`"): the 6 `RQ3_SUBSET`
tasks are the [`../bpi2013_partial/`](../bpi2013_partial/) stage's own rows,
and the remaining 8 were run with `--tasks` (24,270.7 s) and merged. Combined
GNN wall-clock for this stage: 46,943 s (13.0 h). The merged CSVs are
content-equivalent to a single `full` run but are not the output of one
command; the `ocel_load`/`feature_extraction` rows kept in the profile are the
8-task run's, not the partial stage's.

## Highlights

1. **The accuracy signal clears fold noise by ~3x more than on Predict-Collab.**
   Stage-mean macro-F1 spread across models vs mean per-cell fold SD: 0.0503
   vs 0.0133 here (**3.8x**), against 0.0318 vs 0.0245 on Predict-Collab
   (1.3x) — the spread widens *and* the fold SDs roughly halve. Per cell, the
   best-to-worst gap exceeds the mean fold SD in **12 of 12** decided cells
   (Predict-Collab: 41 of 49). The ranking here carries signal; it did not
   there.
2. **GNN takes 7 of the 12 decided cells** (lstm_torch 3, transformer 2), and
   the win structure is cleanly by problem type: GNN wins **all 4 time
   regressions**, lstm_torch wins **3 of 4 multiclass** tasks, GNN wins 0 of
   the 3 `Participant`-anchored (high-cardinality) tasks.
3. **Every predictor learns the classification tasks; none of them learns the
   regression tasks.** All 5 beat the baseline in all 6 classification cells
   (mean macro-F1 gain 0.11–0.17). On the 6 regression cells, four of the five
   are *worse than the trivial baseline on every single cell* — Random Forest
   by 42.8% on average, XGBoost 30.8%, LSTM 27.2%, Transformer 21.5%.
4. **GNN is the only model that does not lose to the baseline on regression —
   but it mostly ties it.** It beats the baseline in 4 of 6 cells, and in
   three of those by 0.15%, 0.42% and 0.54%. Only `NV-PaT` (−6.3% MAE) is a
   real gain. Read as "GNN degrades to predicting the mean where the others
   degrade below it", not as "GNN solves time regression on BPI2013".
5. **GNN's cost is exactly linear in sample count**: r = +0.9997 between mean
   per-fold fit seconds and `samples`, 13.46 s per 1,000 samples. No other
   predictor comes close (lstm_torch r = 0.71, RF 0.64, transformer 0.27,
   XGBoost 0.19). GNN is the one model here you can budget from row count
   alone.
6. **The memory risk at this scale is Random Forest, not the GNN.** RF peaks
   at 6,409 MB (fit and predict) — 2.5x the GNN's 2,559 MB and ~40% of this
   machine's RAM.
7. **Two tasks are degenerate by construction** (`NE-NMPa`, `NE-NMPr`:
   `n_labels == 1`), tied at macro-F1 1.0 for all five models. They are
   excluded from every win count and cost total below.

## Accuracy

### Classification — macro-F1 (higher is better)

| Task | Baseline | GNN | LSTM | RF | Transformer | XGBoost |
|---|---|---|---|---|---|---|
| NE-NEPa (73 labels) | 0.0091 | 0.0683 | **0.0770** | 0.0446 | 0.0718 | 0.0338 |
| NE-NEPr (4) | 0.1878 | 0.4430 | **0.4534** | 0.3769 | 0.4341 | 0.3850 |
| NE-NPaA (25) | 0.0371 | 0.0926 | **0.1171** | 0.0731 | 0.1156 | 0.0524 |
| NE-NPaM (25) | 0.0379 | 0.0866 | 0.1280 | 0.0734 | **0.1330** | 0.0606 |
| OB-M (2) | 0.4125 | **0.7144** | 0.7057 | 0.6480 | 0.6371 | 0.7080 |
| OB-P (2) | 0.4172 | **0.6463** | 0.6101 | 0.5735 | 0.6324 | 0.6386 |
| *mean gain over baseline* | — | 0.158 | **0.165** | 0.115 | 0.154 | 0.130 |

Absolute macro-F1 is low on the next-event tasks (0.03–0.13) because label
cardinality is high — 73 distinct next events on `NE-NEPa`. Against a 0.0091
baseline, LSTM's 0.0770 is an 8.5x lift, so the pipeline *is* learning; the
low absolute number is the target, not the model.

### Regression — MAE (lower is better; **bold** = better than baseline)

| Task | Baseline | GNN | LSTM | RF | Transformer | XGBoost |
|---|---|---|---|---|---|---|
| NV-PrT | 1,156,741 | **1,155,014** | 1,414,592 | 1,623,818 | 1,261,690 | 1,445,739 |
| NV-PaT | 679,188 | **636,370** | 739,033 | 826,165 | 688,620 | 736,745 |
| NV-TNE | 126,967 | **126,278** | 181,151 | 210,870 | 220,661 | 193,364 |
| NV-TNM | 246,049 | **245,024** | 348,709 | 373,119 | 321,899 | 342,556 |
| NV-NMPa | 0.5088 | 0.5958 | 0.6631 | 0.7350 | 0.5578 | 0.6917 |
| NV-NMPr | 1.3920 | 1.4179 | 1.6334 | 1.8435 | 1.4471 | 1.7218 |
| *mean rel. gain* | — | **−1.9%** | −27.2% | −42.8% | −21.5% | −30.8% |

This is the stage's most important negative result and it should be reported
as one: **the RQ3 pipeline does not demonstrate useful numeric prediction on
BPI2013.** Predicting the training mean beats four of the five learners on
every regression target, by up to 73.8% (Transformer on `NV-TNE`). The
Predict-Collab stage gives the opposite picture (every model beats the
baseline on 21–24 of 24 regression cells, mean relative gain ~46%), which
makes this a property of the log, not of the implementation.

### Noise floor

Winner-vs-second margin against the winner's own fold SD: **8 of 12 winners
sit inside their own noise**. The four that do not:

| Task | Winner | Margin | Winner SD | Margin / SD |
|---|---|---|---|---|
| NV-TNE | GNN | 54,873 | 6,785 | 8.1x |
| NV-TNM | GNN | 76,875 | 59,155 | 1.3x |
| NE-NEPa | LSTM | 0.0052 | 0.0043 | 1.2x |
| OB-P | GNN | 0.0078 | 0.0059 | 1.3x |

So GNN's headline "7 of 12" is really *three* results that survive fold
variance plus four coin-flips. That caveat is not new — the same check gives
40 of 49 winners inside their own SD on Predict-Collab — but it is milder
here (67% vs 82% of winners).

What *is* robust at the stage level is the **spread**: in all 12 cells the
best-to-worst gap exceeds the mean fold SD (on Predict-Collab, 41 of 49), so
picking the worst model rather than the best is a real loss even where picking
between the top two is not.

### Ties and degenerate targets

2 of 14 cells are degenerate (`degenerate == True`, `n_labels == 1`):
`NE-NMPa` (14,538 samples) and `NE-NMPr` (18,481). All five models score
macro-F1 1.0 = baseline 1.0. These are the only ties in the stage, and both
are excluded from the 12-cell denominator used above.

Note a cost asymmetry they create: LSTM and Transformer **short-circuit** a
constant training target and never enter `fit`, while GNN, RF and XGBoost pay
for it (GNN: 2,192 s across the two cells). Any summed fit time must exclude
these cells or it compares different amounts of work — the totals below do.

## Cost / Profiling

All `fit` figures exclude the two degenerate cells.

| Stage | GNN | Transformer | LSTM | XGBoost | RF |
|---|---|---|---|---|---|
| `ocel_load` (s) | 30.1 | 60.1 | 26.8 | 26.3 | 26.6 |
| `feature_extraction` (s) | 14.0 | 25.1 | 12.2 | 12.2 | 12.7 |
| `labeling` (s) | 1.2 | 1.5 | 1.8 | 1.5 | 1.2 |
| **`fit` (s)** | **43,994** | **13,513** | **10,753** | **103** | **84** |
| `predict` (s, mean/fold) | 6.16 | 0.283 | 0.026 | 0.039 | 0.094 |
| peak RSS, `fit` (MB) | 2,559 | 2,192 | 1,995 | 2,024 | **6,409** |

- **Training**: GNN costs 3.3x the Transformer, 4.1x the LSTM, and 426–521x
  the two tabular models. In wall-clock: 12.2 h of fitting versus under 2 min.
- **Inference**: the ranking is different. GNN is 237x the LSTM per fold and
  22x the Transformer — it is expensive at serving time too, which the
  training-time ranking alone would not tell you.
- **Memory**: Random Forest is the outlier, at 2.5x the GNN's peak. On a
  15.8 GB machine its 6.4 GB is the first number that would fail on a larger
  log — nothing about the GNN is.
- **Preprocessing is model-independent** (`ocel_load`, `feature_extraction`),
  so the Transformer's ~2.3x on both is machine state, not a property of the
  model — see the caveat below.

### Does cost track data volume?

| Predictor | r(fit s/fold, samples) | fit s per 1k samples |
|---|---|---|
| GNN | **+0.9997** | 13.459 |
| LSTM | +0.7092 | 3.292 |
| RF | +0.6409 | 0.025 |
| Transformer | +0.2715 | 4.137 |
| XGBoost | +0.1879 | 0.030 |

GNN's near-perfect linearity follows directly from its configuration: fixed
50 epochs over fixed-size k=8 subgraphs, one per sample, so sample count is
the only free variable. The others build data-derived sequences or trees, and
their cost is driven by target structure instead — which is check 8:

| Normalised fit time (model's own mean = 1.0) | GNN | LSTM | RF | Transformer | XGBoost |
|---|---|---|---|---|---|
| Binary classification | 1.27 | 1.66 | 0.86 | 0.70 | 0.12 |
| Count regression | 1.26 | 1.49 | 1.01 | 0.70 | 0.12 |
| Multiclass classification | 0.80 | 0.72 | 1.00 | 1.72 | **2.18** |
| Regression (time) | 1.03 | 0.71 | 1.06 | 0.58 | 0.11 |
| *max/min spread across tasks* | 4.4x | 6.5x | 19.5x | 25.7x | **229.6x** |

XGBoost's 229x spread is entirely the multiclass column — it fits one booster
per class, so its budget is set by **label cardinality, not row count**. This
reproduces the same finding on Predict-Collab and is the practical rule for
sizing it.

### Caveat: an unexplained per-task cost anomaly in the sequence models

With fixed epochs and no early stopping, fit time should track
samples × epochs. It does for the GNN. It does not for the Transformer and
LSTM, in a way `audit_stage`'s outlier check cannot catch — that check
compares folds *within* a cell, and these deviations are uniform across all
5 folds of the affected task:

- **Transformer / `NE-NEPr`**: 1,192.7 s per fold (folds: 1196.7, 1191.7,
  1189.8, 1187.3, 1198.1) against ~156 s per fold on its other 62,030-sample
  tasks — **7.6x**, with the *same* sample count and *fewer* labels than
  `NE-NEPa` (4 vs 73), so neither explains it.
- **LSTM / `NV-NMPa`, `NV-NMPr`, `OB-M`, `OB-P`**: ~1,188–1,497 s per task
  against ~770 s for its other 62,030-sample tasks — ~1.9x.

The plausible mechanism is sequence padding: both models derive `max_len`
from the data (`lstm_torch.py:91`), and the LSTM packs sequences back to each
batch's own maximum (`lstm_torch.py:31-38`) while the Transformer does not, so
one long sequence would inflate every batch for the whole task. **This is a
hypothesis, not a measurement** — confirming it means logging per-task
`max_len`, which the profile CSV does not record.

Two consequences, both worth stating rather than smoothing over: the
Transformer's fit total carries ~5,200 s (38%) of unexplained cost, so the
GNN-vs-Transformer ratio of 3.3x is really "3.3x, or 5.3x if `NE-NEPr` is
normalised to its siblings"; and the Transformer's model-independent
`ocel_load` (60.1 s vs ~26.5 s for everyone else) says that run saw machine
contention, which caps how precisely any of its timings can be compared.

## Takeaway

Predict-Collab's conclusion was that accuracy could not decide anything and
cost had to. Here accuracy carries more signal — 3.8x the noise floor against
1.3x — but not enough to pick a single overall winner: two thirds of the
per-cell winners are still inside their own fold SD. What the data does
support is a choice *by task type*, and one that rules models out:

- **Sequence-shaped classification (next event / next participant)** — LSTM.
  Wins 3 of 4 multiclass tasks and the best mean lift over baseline (0.165) at
  10,753 s of training and the cheapest inference in the stage (0.026 s/fold).
- **Object-centric binary and numeric targets** — GNN, with the caveat that
  on regression it is buying "not worse than the baseline" rather than a real
  gain, for 43,994 s of fitting and 237x the LSTM's inference cost. Its win
  on `NV-TNE` (8.1x its own SD) and its `Message`/`OrchestrationCase` sweep
  are the concrete case for it; the global mean is not.
- **Offline / batch, cost-dominated** — XGBoost over Random Forest at this
  scale. Training cost is comparable (103 s vs 84 s) and the two split the
  classification cells 3–3, but XGBoost beats RF on **all 6** regression cells
  and peaks at 2.0 GB against RF's 6.4 GB. Memory, not accuracy, is what
  decides this pair here. Size XGBoost against label cardinality, not row
  count.
- **Do not ship any of them for time regression on this log.** That is a
  catalog-and-log finding, not a model-selection one, and it is the single
  result from this stage most worth carrying into RQ4.

Stage-level notes independent of predictor choice: `NE-NMPa` and `NE-NMPr`
are degenerate on BPI2013 (12 effective targets, not 14), and the four tasks
anchored on shorter object lifecycles (`NE-NPaM`, `NE-NMPa`, `NE-NMPr`,
`NV-TNM`: 14,538–18,481 samples against 62,030 for the other ten) are what
makes the partial/full runtime split uneven — see the runbook's reuse section.

## Reproducing this analysis

Run `python -m evaluation.audit_stage --log-group bpi2013 --scope full`
first (it automates checks 1–3 and 9–11 and exits non-zero on a problem),
then apply the 11-check table in
[`../predictcollab_full/ANALYSIS.md`](../predictcollab_full/ANALYSIS.md#reproducing-this-study-on-another-stage),
which this document follows unchanged. To repeat the run itself, see
[`src/evaluation/RQ3_EXECUTION_PLAN.md`](../../../src/evaluation/RQ3_EXECUTION_PLAN.md).
