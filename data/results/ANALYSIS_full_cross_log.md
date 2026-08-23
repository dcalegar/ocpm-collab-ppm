# RQ3 Analysis — Full Catalog, Predict-Collab vs BPI2013

Both `full` stages: the same 14 tasks (`EQUIVALENCE_TASKS`) × the same 5
predictors × identical hyperparameters, 5-fold CV grouped by
`CollaborationCase`, `--profile` on.

| Stage | Logs | Cells | Samples / cell | Total samples |
|---|---|---|---|---|
| [`predictcollab_full/`](predictcollab_full/) | 4 | 56 | 329 – 2,260 | 80,279 |
| [`bpi2013_full/`](bpi2013_full/) | 1 | 14 | 14,538 – 62,030 | 686,317 |

All 350 rows `ran_end_to_end=True`; `evaluation.audit_stage` reports no
problems on either stage. Because the task set and the model configuration
are held fixed, the logs are the only variable — this document reports what
changes between them and what does not.

This is the full-catalog counterpart of
[`ANALYSIS_partial_cross_log.md`](ANALYSIS_partial_cross_log.md) and draws on
the two per-stage documents:
[`predictcollab_full/ANALYSIS.md`](predictcollab_full/ANALYSIS.md) and
[`bpi2013_full/ANALYSIS.md`](bpi2013_full/ANALYSIS.md).

**One caveat on reading it as a scale study.** BPI2013 is 8.5x Predict-Collab
by labelled samples (~29.5x by OCEL events), but it is also a *different kind
of log*: one real-world incident-management process versus four study logs,
two of them artificial. Scale and provenance are confounded. Where a finding
plausibly follows from volume alone it is called out as such; where it does
not, it is reported as "differs between these logs", not "changes with scale".

## Environment

Identical machine for every number below, so timings are comparable to each
other but not portable:

| | |
|---|---|
| CPU | Intel Core i5-13400F — 10 physical / 16 logical cores, 2.5 GHz base |
| RAM | 15.8 GB |
| GPU | NVIDIA GTX 1660, 6 GB — present and CUDA-visible, deliberately unused |
| OS | Windows 10 Pro Education, build 19045 |
| Python | 3.10.11 |

All predictors on CPU; fixed epoch budgets, no early stopping
(`lstm_epochs=60`, `transformer_epochs=35`, `gnn_epochs=50`, `gnn_k=8`).
The BPI2013 GNN rows are merged from two runs — see that stage's
[Provenance](bpi2013_full/ANALYSIS.md#provenance-of-the-gnn-rows).

## Highlights

1. **The accuracy signal-to-noise ratio roughly triples.** Stage-mean
   macro-F1 spread across models against mean per-cell fold SD: 0.0318 / 0.0245
   = **1.3x** on Predict-Collab, 0.0503 / 0.0133 = **3.8x** on BPI2013 — the
   spread widens and the fold SDs halve. Per cell, the best-to-worst gap
   clears the mean fold SD in 41 of 49 cells there and 12 of 12 here. This is
   a sharpening, not a flip: on *both* stages the winner-vs-second margin
   still sits inside the winner's own SD in most cells (82% vs 67%).
2. **The win distribution goes from flat to concentrated.** Predict-Collab:
   lstm_torch 13, GNN 12, XGBoost 9, Transformer 8, RF 7 (of 49 decided) —
   i.e. noise. BPI2013: GNN 7, lstm_torch 3, Transformer 2 (of 12), split
   cleanly by problem type.
3. **Regression inverts completely.** On Predict-Collab every model beats the
   trivial baseline on 21–24 of 24 regression cells, mean relative MAE gain
   ~46%. On BPI2013 four of five models are worse than the baseline on
   *every* regression cell, and the fifth (GNN) merely ties it. This is the
   sharpest difference between the stages.
4. **Classification lift shrinks but never disappears**: mean macro-F1 gain
   over baseline falls from 0.55–0.58 to 0.11–0.17, and all five models still
   beat the baseline in every non-degenerate classification cell on both
   stages. The pipeline keeps learning; it just learns less of a much harder
   target.
5. **GNN's cost model is the only one that transfers.** Fit seconds per 1,000
   samples: 13.71 (Predict-Collab) vs 13.46 (BPI2013) — a ratio of **0.98**
   across an 8.5x change in data volume and a different log. Everyone else's
   per-sample cost moves by 1.8x–5.2x in one direction or the other.
6. **The tabular models get cheaper per sample at scale; the sequence models
   get more expensive.** RF 0.24x and XGBoost 0.19x per-sample cost on the
   bigger log (fixed overheads amortising), against LSTM 4.43x and Transformer
   1.84x.
7. **Random Forest is the memory risk, and only at scale**: peak RSS 898 MB →
   6,409 MB (7.1x), against GNN's 1,074 → 2,559 MB (2.4x). At Predict-Collab
   size RF looks like the cheap safe default; at BPI2013 size it is the one
   number that would fail first on a 15.8 GB machine.

## Accuracy

### Does the ranking mean anything?

| | Predict-Collab full | BPI2013 full |
|---|---|---|
| Decided cells (ties/degenerate excluded) | 49 of 56 | 12 of 14 |
| Stage-mean macro-F1 spread across models | 0.0318 | 0.0503 |
| Mean per-cell fold SD | 0.0245 | 0.0133 |
| Spread / SD | 1.3x | **3.8x** |
| Cells where model spread < mean fold SD | 8 of 49 (16%) | **0 of 12** |
| Winners inside their own SD | 40 of 49 (82%) | 8 of 12 (67%) |
| Outright wins | lstm 13, gnn 12, xgb 9, trf 8, rf 7 | gnn 7, lstm 3, trf 2 |

The Predict-Collab stage's own conclusion was that accuracy could not decide
anything and cost had to. The larger log weakens that conclusion without
overturning it. Fold SDs nearly halve (0.0245 → 0.0133) while the model
spread grows, so the aggregate signal goes from 1.3x the noise floor to 3.8x,
and no cell is left where the model spread is inside the noise. But the
*winner-vs-second* margin still fails to clear its own SD in two thirds of
cells — down from four fifths, not eliminated. The robust statement at both
scales remains "avoid the worst model", not "pick this one"; what BPI2013
adds is that the worst model is now reliably identifiable.

### Win structure by problem type (BPI2013, decided cells)

| Winner | Binary | Count reg. | Multiclass | Time reg. |
|---|---|---|---|---|
| GNN | 2 | 1 | 0 | **4** |
| LSTM | 0 | 0 | **3** | 0 |
| Transformer | 0 | 1 | 1 | 0 |

The specialisation only becomes visible at BPI2013 scale — on Predict-Collab
the equivalent crosstab is diffuse. GNN sweeps the object-centric numeric and
binary targets and wins **none** of the three `Participant`-anchored
high-cardinality multiclass tasks, which go to the sequence models.

### Lift over baseline

| | Predict-Collab full | BPI2013 full |
|---|---|---|
| Classification, mean macro-F1 gain | 0.553 – 0.585 | 0.115 – 0.165 |
| Classification, cells beating baseline | 29–30 of 30 | 6 of 6 |
| Regression, mean relative MAE gain | +0.448 – +0.471 | **−0.428 – −0.019** |
| Regression, cells beating baseline | 21–24 of 24 | GNN 4 of 6; all others **0 of 6** |

The regression row is the finding to carry into RQ4. On the study logs the
reformulated numeric tasks look solved; on the one real-world log they are
not merely harder — the learners are actively worse than predicting the
training mean, by 21.5% (Transformer) to 42.8% (Random Forest) on average.
GNN's "wins" there are three ties (0.15%, 0.42%, 0.54%) plus one genuine
−6.3% on `NV-PaT`.

Two readings are compatible with the data and this study cannot separate them:
BPI2013's time targets carry variance the 7-column native OCPA feature table
does not explain, or the study logs' generative structure makes their time
targets unusually predictable. Distinguishing them needs a second real-world
log, which is outside this evaluation's scope.

### Degenerate targets

| Stage | Degenerate cells | Effective targets |
|---|---|---|
| Predict-Collab full | 2 of 56 — Artificial1/`NE-NPaM`, Healthcare/`OB-M` | 54 |
| BPI2013 full | 2 of 14 — `NE-NMPa`, `NE-NMPr` | 12 |

Degeneracy is log-specific, not task-specific: no task is degenerate on both
stages. It is a property of how a given log instantiates a target, so it must
be re-checked per log rather than assumed from the catalog.

## Cost

### Per-1,000-sample cost — the scaling comparison

Totals are not comparable across stages (different cell counts and sizes), so
the honest unit is seconds per 1,000 labelled samples:

| Predictor | fit s/1k (PC) | fit s/1k (BPI) | ratio | predict s/1k (PC) | predict s/1k (BPI) | ratio |
|---|---|---|---|---|---|---|
| GNN | 13.714 | 13.459 | **0.98** | 0.1406 | 0.1257 | 0.89 |
| LSTM | 0.744 | 3.292 | **4.43** | 0.0009 | 0.0005 | 0.56 |
| Transformer | 2.254 | 4.137 | 1.84 | 0.0022 | 0.0052 | 2.41 |
| Random Forest | 0.104 | 0.025 | **0.24** | 0.0208 | 0.0019 | 0.09 |
| XGBoost | 0.156 | 0.030 | **0.19** | 0.0030 | 0.0008 | 0.26 |

Three distinct behaviours:

- **GNN — invariant.** Its per-sample cost is unchanged (0.98x) across both
  stages, matching its within-stage r = +0.9997 against sample count on
  BPI2013. This follows directly from the configuration: fixed 50 epochs over
  fixed-size k=8 subgraphs, one per sample, so nothing about the log can
  change the work per sample. It is the only predictor here whose runtime can
  be forecast for an unseen log from row count alone: **~13.5 s per 1,000
  samples per fold-set on this machine.**
- **Tabular — sub-linear.** RF and XGBoost cost 4–5x *less* per sample on the
  larger log: their per-fit fixed overheads dominate at 700-sample cells and
  amortise at 62,000-sample ones.
- **Sequence — super-linear.** LSTM 4.4x and Transformer 1.8x more expensive
  per sample. Part of this is the unexplained per-task anomaly documented in
  the BPI2013 stage ([caveat](bpi2013_full/ANALYSIS.md#caveat-an-unexplained-per-task-cost-anomaly-in-the-sequence-models):
  Transformer/`NE-NEPr` at 7.6x its siblings, LSTM at ~1.9x on four tasks),
  which alone accounts for ~5,200 s of the Transformer's 13,513 s. The
  LSTM's 4.4x is too large to be explained that way and is a real scaling
  cost.

### Memory

| Peak RSS (MB) | Predict-Collab full | BPI2013 full | ratio |
|---|---|---|---|
| Random Forest | 898 | **6,409** | **7.1** |
| GNN | 1,074 | 2,559 | 2.4 |
| Transformer | 1,034 | 2,192 | 2.1 |
| XGBoost | 964 | 2,024 | 2.1 |
| LSTM | 962 | 1,995 | 2.1 |

Four of the five predictors sit in a tight 2.1–2.4x band, tracking data
volume as expected. Random Forest does not: its ensemble grows with the
training set, so it goes from the *lowest*-memory model on the small stage to
2.5x the next-highest on the large one, at ~40% of the machine's RAM. This is
the clearest example in the study of a small-log conclusion that reverses at
scale, and it does not show up in accuracy or wall-clock at all.

### Which phase dominates

Preprocessing (`ocel_load` + `feature_extraction` + `labeling`) is
model-independent and, on both stages, negligible against `fit` for the three
neural models — but on Predict-Collab it is comparable to or larger than
`fit` for RF and XGBoost (~6 s of preprocessing against 42–63 s of fitting
spread over 280 folds). On BPI2013 preprocessing is ~40 s against 84–103 s of
fitting for the same two models. So for the tabular predictors the pipeline,
not the learner, is a first-order cost at both scales — which is exactly the
argument for the native-OCPA feature path being the thing to optimise if
those models are the target.

## What changes between these logs — summary

| Dimension | Predict-Collab full | BPI2013 full | Verdict |
|---|---|---|---|
| Accuracy signal / fold noise | 1.3x | 3.8x | sharpens with scale |
| Winner-vs-second inside own SD | 82% of cells | 67% of cells | improves, not resolved |
| Win distribution | flat across all 5 | concentrated, by problem type | changes with scale |
| Classification vs baseline | +0.55–0.58 macro-F1 | +0.11–0.17 | harder target, still learning |
| Regression vs baseline | +45–47% MAE | −43% to −2% | **differs between logs** |
| GNN per-sample cost | 13.71 s/1k | 13.46 s/1k | invariant |
| Tabular per-sample cost | baseline | 0.19–0.24x | amortises with scale |
| Sequence per-sample cost | baseline | 1.8–4.4x | grows with scale |
| RF peak memory | lowest of the five | highest, 6.4 GB | **reverses with scale** |
| Degenerate targets | 2 (Artificial1, Healthcare) | 2 (different tasks) | log-specific |

## Operational note: why the GNN ran in two pieces on BPI2013

The BPI2013 GNN stage was assembled from the `partial` stage's 6 tasks plus a
`--tasks` run of the remaining 8, per the runbook's documented reuse
shortcut — 24,271 s instead of the ~46,900 s a fresh 14-task run would have
cost. The shortcut is sound because a task's result does not depend on which
other tasks ran alongside it (`run_one_log` computes the fold map, context and
baselines before the task loop; every predictor re-seeds per fold), and it was
verified empirically: on the 6 shared (log, task) rows, `xgboost`,
`lstm_torch` and `transformer` are bit-identical between scopes and
`random_forest`/`gnn` differ by at most 1.5e-11.

The trade-off, stated plainly: `bpi2013_full/rq3_results_gnn_bpi2013_full.csv`
is content-equivalent to a single `full` run but is not the output of one
reproducible command.

## Reproducing this analysis

Per stage, run
`python -m evaluation.audit_stage --log-group <group> --scope full` (it
automates checks 1–3 and 9–11 and exits non-zero on a problem) and then the
11-check table in
[`predictcollab_full/ANALYSIS.md`](predictcollab_full/ANALYSIS.md#reproducing-this-study-on-another-stage).
The cross-stage figures above need nothing further: they are per-1,000-sample
normalisations of the same two CSV families, which is what makes stages with
different cell counts comparable at all. To repeat the runs, see
[`src/evaluation/RQ3_EXECUTION_PLAN.md`](../../src/evaluation/RQ3_EXECUTION_PLAN.md).
