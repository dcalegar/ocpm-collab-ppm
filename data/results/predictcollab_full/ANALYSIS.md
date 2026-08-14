# RQ3 Analysis — Predict-Collab, Full Catalog

Predict-Collab (4 study logs: Healthcare, Artificial1, Artificial5, Real4) ×
all 14 tasks × 5 predictors (Random Forest, XGBoost, LSTM, Transformer, GNN),
5-fold CV grouped by `CollaborationCase`, `--profile` on. 56 (log, task) rows
per predictor, all `ran_end_to_end=True`, no errors. Source: `rq3_results_*.csv`
/ `rq3_profile_*.csv` in this directory; charts in [`plots/`](plots/).

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

Every predictor ran on **CPU** (`ExperimentConfig.device="cpu"`). That is a
measured choice, not a limitation: on this workload CUDA's per-batch/per-graph
dispatch overhead made GNN and the sequence models *slower* wall-clock than
plain CPU, because the per-sample graphs (k=8 nodes) and per-case sequences
are far too small to amortise kernel-launch cost — GPU utilisation sampled at
only 4-37% during a CUDA GNN run. So the numbers here reflect a
CPU-appropriate workload, not GPU starvation.

Methodology notes: runs were sequential (one predictor at a time, never in
parallel, so no CPU contention between measured jobs), with
`PYTHONHASHSEED=0`. The hybrid P-core/E-core CPU means thread placement can
add some run-to-run variance on short stages.

## Highlights

- **The five models are statistically indistinguishable; their cost differs by
  ~100x.** GNN leads on mean macro-F1 (0.796) over Random Forest (0.767), but
  that 0.030 spread is barely above the mean fold-to-fold SD (0.023) of a
  *single* model. Task by task it dissolves: the winner's margin over second
  place is smaller than the winner's own fold SD in **21 of 32** classification
  tasks and **20 of 24** regression tasks, with a median margin of 0.003
  macro-F1. Meanwhile GNN costs 92.8 min of compute against Random Forest's
  56 s. *On this benchmark, model choice buys far less than it costs.*
- **Random Forest wins the most individual tasks (14/56) despite the lowest
  mean.** Consistent with the noise finding above — with margins this thin,
  "wins" are close to a coin flip, and they spread across all five (RF 14,
  LSTM 13, GNN 12, XGBoost 9, Transformer 8).
- **Cost is driven by training-unit granularity, not data volume.** LSTM and
  Transformer train on one example *per case* (80 cases per dev split);
  GNN trains on one induced subgraph *per labelled event* (~1080 per dev
  split). At the configured budgets that is **1700 gradient steps/fold for
  GNN vs 300 (LSTM) and 105 (Transformer)** — each GNN step additionally
  paying DGL graph batching plus two `GraphConv` layers and attention
  pooling. GNN's fit time correlates with sample count at r=0.99; Random
  Forest's at only r=0.65, because its fixed 200-tree budget dominates.
- **The cheapest model to *train* is the second most expensive to *predict*.**
  Random Forest fits in 0.15 s/fold but infers in 0.030 s/fold — 24x LSTM's
  0.0012 s. For offline batch evaluation training cost dominates; for online
  predictive monitoring (scoring running cases as events arrive) the ranking
  flips and the sequence models win. GNN is worst on both (162x LSTM at
  inference).
- **`fit` dominates every model, but only Random Forest leaves room for
  anything else.** Training is 98-99% of total time for GNN/LSTM/Transformer,
  so the shared pipeline (`ocel_load` + `feature_extraction` + `labeling`,
  ~5.9 s fixed) is invisible to them. For Random Forest that same fixed cost
  is **10.5%** of its total and `predict` another **14.9%** — i.e. with a
  cheap learner the pipeline stops being training-bound and becomes
  I/O-and-inference-bound. Optimising OCEL loading only pays off in that
  regime.
- **Task choice changes cost for exactly one model: XGBoost.** Its fit time
  swings 9.4x between binary (0.046 s/fold) and multiclass (0.431 s/fold),
  and within multiclass it tracks label cardinality almost linearly — on
  `NE-NEPr` it costs 0.30 s on Artificial1 (8 activities) but 1.58 s on
  Artificial5 (28), ~0.04-0.06 s per class. That is the signature of fitting
  one tree ensemble *per class*. Every other model is flat (ratio 0.8-1.2
  across the same tasks): their cost is fixed by architecture and epoch
  budget, not by the label space.
- **`NE-NEPr` and `NE-NEPa` are the same task on these logs.** Their scores are
  bit-identical for every model on Healthcare/Artificial1/Artificial5 (and
  agree to ~1e-4 on Real4). Cause is verifiable in the data: `NE-NEPa` labels
  the next event's *(activity, actor)* pair while `NE-NEPr` labels its
  *activity*, and **no activity in any of the four logs is performed by more
  than one actor** — so the pair is a bijective relabeling of the activity and
  induces the identical partition. The catalog effectively contributes 13
  distinct targets here, not 14.
- **Task family predicts the winner better than overall rank does.** Random
  Forest takes 8 of 16 `Message`-anchored tasks and 5 of 8 binary
  classifications — locally-decidable, low-cardinality targets that suit
  axis-aligned splits — yet wins **0 of 16** `Regression (time)` tasks, the
  sharpest specialisation in the table. GNN takes 6 of 12
  `OrchestrationCase`-anchored tasks, where the label depends on the
  surrounding interaction structure it explicitly models. The sequence models
  cover the time regressions (LSTM 6 + Transformer 4 of 16), where the ordered
  prefix is the signal.
- **The weakest results are a label-cardinality ceiling, not a modeling
  failure.** `Artificial5/NE-NEPr`+`NE-NEPa` average ~0.40 macro-F1 — the
  catalog's lowest — but that log has 28 distinct activities and a trivial
  baseline of 0.003, so every model lands ~130x above baseline. Macro-F1 over
  28 classes simply cannot reach the values a 2-class task shows.
- **`NV-PrT` is the consistently hardest regression target** (highest MAE on 3
  of 4 logs: Healthcare 15.07, Artificial5 10.57, Real4 6.99). It is the only
  time-regression task anchored on `CollaborationCase` without a participant
  or direction parameter — i.e. the least conditioned target, predicting
  across the whole collaboration rather than one participant's slice.

## Accuracy

Mean over all 56 (log, task) rows, by model:

| Model | Mean macro-F1 ↑ | Mean MAE ↓ | Mean lift over baseline (clf) | Mean lift over baseline (reg) | Task wins |
|---|---|---|---|---|---|
| GNN | **0.7965** | 3.966 | **+0.548** | 4.339 | 12 |
| Transformer | 0.7891 | 3.965 | +0.541 | 4.340 | 8 |
| LSTM | 0.7877 | **3.929** | +0.539 | **4.376** | 13 |
| XGBoost | 0.7774 | 4.077 | +0.529 | 4.229 | 9 |
| Random Forest | 0.7667 | 4.244 | +0.518 | 4.061 | **14** |

("lift" = metric − baseline for classification, baseline − metric for
regression; positive is always better than the trivial baseline. All five
models beat baseline on every task — see `rq3_comparison_f1_macro.png` /
`rq3_comparison_mae.png`.)

**Read the ranking with care.** Mean per-task fold SD is 0.023 while the
best-to-worst model spread is 0.030, and per-task cross-model spread is 0.053
(median 0.040). The column that actually separates the models is not accuracy
— it is cost, below.

**Per-log difficulty** (mean macro-F1 across all models/tasks):

| Log | Activities | Mean macro-F1 |
|---|---|---|
| Real4 | 19 | 0.903 |
| Healthcare | 21 | 0.834 |
| Artificial1 | 8 | 0.718 |
| Artificial5 | 28 | 0.679 |

Cardinality explains the *hardest tasks* (Artificial5's 28-way next-activity
targets) but not the whole log ordering — Real4 leads despite 19 activities
because several of its targets are near-deterministic (`NE-NPaM` 0.999,
`OB-P` 0.999, `NE-NMPa` 0.995).

## Cost / Profiling

Total wall-clock seconds, summed across all logs/tasks/folds, on the CPU-only
setup described under [Environment](#environment):

| Model | Total (s) | Total (min) | `fit` (s) | `predict` (s) | Peak RSS (MB) |
|---|---|---|---|---|---|
| GNN | 5567.3 | **92.8** | 5504.8 | 56.4 | 1074.4 |
| Transformer | 892.9 | 14.9 | 885.8 | 0.8 | 1034.5 |
| LSTM | 298.8 | 5.0 | 292.3 | 0.3 | 961.5 |
| XGBoost | 70.1 | 1.2 | 62.6 | 1.2 | 963.5 |
| Random Forest | **56.2** | **0.9** | 41.9 | 8.3 | 899.3 |

Peak RSS varies by only ~1.2x across models, so memory is not a
differentiator at this log size.

### Which phase dominates

Share of each model's own total time (%):

| Model | `ocel_load` | `feature_extraction` | `labeling` | `fit` | `predict` |
|---|---|---|---|---|---|
| GNN | 0.1 | 0.0 | 0.0 | **98.9** | 1.0 |
| Transformer | 0.5 | 0.2 | 0.0 | **99.2** | 0.1 |
| LSTM | 1.5 | 0.5 | 0.1 | **97.9** | 0.1 |
| XGBoost | 6.3 | 2.3 | 0.3 | **89.3** | 1.7 |
| Random Forest | 7.3 | 2.9 | 0.3 | **74.6** | 14.9 |

The three preprocessing stages are model-independent by construction and cost
a near-constant **~4.3 s + ~1.5 s + ~0.25 s ≈ 5.9 s** in total for every model
(per log: `ocel_load` 0.34-1.54 s, `feature_extraction` 0.09-0.72 s, both
tracking log size). Because that cost is fixed, its *share* is purely a
function of how expensive the learner is — negligible for the neural models,
but over a tenth of Random Forest's budget. `labeling` is never material
(≤0.3%, 0.0025-0.0062 s per task).

The other asymmetry is `predict`: 14.9% of Random Forest's total against
≤1.7% for everything else. Random Forest is the only model whose inference
cost is a visible fraction of its own pipeline.

### Which tasks cost more to fit

Fit time per fold, normalised by each model's own mean (1.00 = that model's
average task):

| Model | Binary clf | Multiclass clf | Count reg. | Time reg. | Spread |
|---|---|---|---|---|---|
| XGBoost | 0.29 | **2.75** | 0.47 | 0.48 | **9.4x** |
| Random Forest | 1.03 | 1.08 | 0.90 | 0.99 | 1.2x |
| LSTM | 0.98 | 1.04 | 1.00 | 0.99 | 1.1x |
| Transformer | 1.03 | 0.99 | 1.01 | 0.98 | 1.1x |
| GNN | 1.06 | 0.97 | 1.01 | 0.96 | 1.1x |

Only XGBoost's cost depends on what is being predicted. It fits one gradient-
boosted ensemble **per class**, so the 200-tree budget is effectively
multiplied by label cardinality — visible directly on `NE-NEPr` as fit time
rises with the log's activity count (Artificial1 8 → 0.30 s, Real4 19 →
0.76 s, Healthcare 21 → 0.84 s, Artificial5 28 → 1.58 s) while the binary
`OB-M` stays flat at 0.033-0.048 s on all four. The control models show
ratios of 0.82-1.20 on the same pair, confirming they are indifferent.

Aggregated over models, the two costliest tasks are `NE-NEPr` (1.62x) and
`NE-NEPa` (1.59x) — the high-cardinality next-activity targets — with every
other task in a narrow 0.79-1.15x band despite identical sample counts
(1502 events) for ten of them. **Label cardinality, not sample count, is what
makes a task expensive**, and only for the one model that scales with it.

### Where the time goes

| Model | Training unit | Grad. steps/fold¹ | Fit s/fold | Predict s/fold | Fit s per 1k samples | r(fit, samples) |
|---|---|---|---|---|---|---|
| GNN | event subgraph (k=8) | 1700 | 19.66 | 0.201 | 13.71 | 0.99 |
| Transformer | case sequence | 105 | 3.28 | 0.0031 | 2.25 | 1.00 |
| LSTM | case sequence | 300 | 1.08 | 0.0012 | 0.74 | 0.70 |
| XGBoost | row (200 trees) | — | 0.22 | 0.0043 | 0.16 | 0.30 |
| Random Forest | row (200 trees) | — | 0.15 | 0.0298 | 0.10 | 0.65 |

¹ For Healthcare's 80-case / ~1080-event development split at the configured
budgets (LSTM 60 epochs × batch 16; Transformer 35 × 32; GNN 50 × 32).

Three structural facts explain the whole table:

1. **Granularity.** GNN's example count equals the *event* count, the other
   neural models' equals the *case* count — a ~13x multiplier on steps before
   any per-step cost is considered. This is inherent to modelling each
   prefix as its own induced subgraph, not a tuning artefact.
2. **Per-step cost.** Transformer runs fewer steps than LSTM (105 vs 300) yet
   costs 3x more per fold: self-attention is O(seq_len²) per layer against the
   LSTM's O(seq_len) packed pass, so it pays quadratically as cases lengthen.
   Its near-perfect r=1.00 with sample count reflects that.
3. **Fixed budgets don't scale.** The tree ensembles fit a constant 200 trees
   regardless of log size, so their cost barely tracks sample count (r=0.30,
   0.65) — they are effectively flat across this benchmark, and would only
   start to separate on much larger logs.

## Takeaway

For this catalog the accuracy differences do not survive contact with
fold-to-fold variance, so the decision should be made on cost and operational
fit, not on the leaderboard:

- **Offline / batch evaluation** — Random Forest. Lowest total compute (<1
  min), most task wins, within 0.03 macro-F1 of the top model. Caveat: it
  wins no time-regression task at all, so pair it with a sequence model if
  the target mix is regression-heavy.
- **Online / streaming inference** — LSTM. 24x cheaper per prediction than
  Random Forest, best MAE, second-most task wins, 5 min total training.
- **GNN** buys +0.030 macro-F1 for ~131x the fit time and 162x the inference
  time. Its structural modelling does pay off on `OrchestrationCase`-anchored
  tasks specifically; that, not the global mean, is the case for using it.
- **XGBoost** is the only choice whose budget depends on the task mix: cheap
  on binary and regression targets, but scaling with label cardinality on
  multiclass ones. Size it against the number of classes, not the row count.

Two catalog-level notes independent of predictor choice: `NE-NEPr`/`NE-NEPa`
are label-equivalent on all four logs (13 effective targets, not 14), and the
Artificial5 next-activity tasks are bounded by 28-way label cardinality rather
than by model capacity.

## Reproducing this study on another stage

Every highlight above comes from one of nine checks run over the two CSV
families in this directory. None needs anything beyond `rq3_results_*.csv`
(accuracy) and `rq3_profile_*.csv` (timing, requires `--profile`), so the same
analysis transfers unchanged to any other stage directory — e.g.
`data/results/bpi2013_full/`. Load both families with the predictor key mapped
to a display name, then:

| # | Check | Question it answers | How |
|---|---|---|---|
| 1 | **Noise floor** | Are model differences real? | Per (log, task), sort by `metric_mean` (desc for `f1_macro`, asc for `mae`); compare the winner-vs-second margin against the winner's own `metric_sd`. Report the count where margin < SD. Also compare the best-to-worst spread of model means against the mean `metric_sd`. |
| 2 | **Task wins** | Does any model dominate? | Per (log, task) take the argmax/argmin of `metric_mean`; `value_counts()` the winner. |
| 3 | **Win structure** | *Which kind* of task suits each model? | Crosstab the per-task winner against `problem_type` and against `anchor`. Look for zero-cells (e.g. RF winning 0/16 time regressions) — they are stronger signals than the totals. |
| 4 | **Lift over baseline** | Is the pipeline learning at all? | `metric_mean − baseline_mean` for classification, `baseline_mean − metric_mean` for regression. Always inspect alongside the raw metric: a low absolute score over a high-cardinality target can still be a large lift, and a perfect score can be a degenerate baseline (check `baseline_mean == 1.0`). |
| 5 | **Cost ranking** | What does each model actually cost? | Sum `seconds` in the profile CSV grouped by model; break out by `stage`. Also take mean `seconds` per fold for `fit` and `predict` separately — the training and inference rankings differ. |
| 6 | **Phase shares** | Which phase dominates, and for whom? | Group profile `seconds` by (model, stage), divide each model's row by its own total. Preprocessing stages are model-independent, so their *share* is a proxy for how cheap the learner is. |
| 7 | **Scaling** | Does cost track data volume? | Join mean per-fold `fit` seconds to the results CSV's `samples` column on (log, task); compute Pearson r per model and seconds-per-1k-samples. High r ⇒ the model scales with data; low r ⇒ a fixed budget dominates. |
| 8 | **Task-dependent cost** | Does the target change the cost? | Normalise each model's per-task `fit` seconds by its own mean, then average by `problem_type`. A model with a flat profile (~1.0 everywhere) is task-insensitive; a large spread points to a per-class or per-label mechanism, which you confirm by regressing that model's fit time on the log's label cardinality. |
| 9 | **Duplicate targets** | Are two tasks secretly the same? | Look for (log, task) pairs whose `metric_mean` is bit-identical across *all* models. Confirm in the log itself, not just the metrics — here, checking that no activity is performed by more than one actor proved the (activity, actor) label is a bijective relabeling of activity. |

Two practical rules learned running this:

- **Read accuracy through check 1 before any other.** Ranking tables invite
  conclusions that the fold variance does not support; on this stage the
  entire model spread sat inside a single model's own noise.
- **Never report a timing without the machine and device.** The whole cost
  story here is CPU-specific and would invert on some workloads — see
  [Environment](#environment), and note that CUDA was measured *slower* for
  these graph/sequence sizes.

To repeat the run itself (rather than the analysis), see
[`src/evaluation/RQ3_EXECUTION_PLAN.md`](../../../src/evaluation/RQ3_EXECUTION_PLAN.md).
