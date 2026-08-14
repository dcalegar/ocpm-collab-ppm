# RQ3 execution plan — all predictors × all case studies

Runbook for the full RQ3 sweep: both log groups (Predict-Collab, BPI2013) ×
both task-catalog scopes (partial, full) × all five registered predictors,
each with profiling, followed by report generation. See
[`README.md`](README.md) for the underlying concepts (axes, output
convention, profiling, reporting); this file is just the ordered command
list.

## Ordering rationale

- **Log group**: `predictcollab` before `bpi2013` — BPI2013 is ~29.5x larger
  by OCEL events than the largest Predict-Collab log (see `run_evaluation.py`
  module docstring), so it is opt-in and runs last.
- **Scope**: `partial` (6-task representative subset) before `full` (all 14
  tasks) within each log group — `full` is roughly 2x the cost of `partial`.
- **Predictor**: `random_forest` → `xgboost` → `lstm_torch` → `transformer` →
  `gnn`, fastest to slowest. `gnn` is deliberately last in every stage, it remains the
  slowest predictor by a wide margin (CPU, direct OCPA event-graph + DGL
  GraphConv training).
- **Device**: every GPU-capable predictor (`gnn`, `lstm_torch`, `transformer`)
  defaults to `device="cpu"` (`ExperimentConfig.device`, see
  [`predictors/README.md`](../predictors/README.md#device-selection-gnnpylstm_torchpytransformerpy-only))
  — CUDA's per-batch/per-graph dispatch overhead measured *slower* than CPU
  for these workloads' small graphs/sequences. Re-tested on BPI2013 in case
  the larger log changed the picture: it does not (one GNN fold, 250.5 s CPU
  vs 324.9 s CUDA, identical metrics), because a bigger log adds *batches*,
  not work per batch, and the feature table is 7 columns wide. Don't pass
  `--device cuda` unless you have also changed `gnn_batch_size`/`gnn_k` or
  the feature width.

## Commands

Every RQ3 run below uses `--profile` (writes `rq3_profile_*.csv` alongside
`rq3_results_*.csv`) and `PYTHONHASHSEED=0` (the script re-execs itself with
this set if omitted — see its module docstring — so this is redundant but
explicit). Run predictors for a given stage one at a time, sequentially, not
in parallel: profiling captures wall-clock time, and two heavy jobs sharing a
CPU would skew both.

Each stage's outputs land in `data/results/{log_group}_{scope}/` (e.g.
`data/results/predictcollab_full/`) — results, profile and progress files
together, plots in a `plots/` subdirectory once generated.

The four axes below (`--rqs`, `--log-groups`, `--rq3-scopes`, `--predictors`)
select whole stages. Two further flags cut *below* a stage and are what you
want for re-running one cell rather than repeating hours of work:

| Flag | Effect |
|---|---|
| `--tasks TASK...` | run just these `tasks.catalog` keys instead of the whole scope catalog |
| `--logs LOG...` | run just these `LogSpec` names within the selected `--log-groups` |

A narrowed run writes to **suffixed** filenames inside the same stage
directory — `rq3_results_transformer_bpi2013_full__OB-M.csv` — so it can never
overwrite the stage's canonical CSVs; merging the rows back is a deliberate
step (see [Recovering from a bad timing](#recovering-from-a-single-bad-log-task-timing-outlier)).

### Stage 1 — Predict-Collab, partial

```bash
PYTHONHASHSEED=0 python -m evaluation.run_evaluation --rqs RQ3 --log-groups predictcollab --predictors random_forest --rq3-scopes partial --profile
PYTHONHASHSEED=0 python -m evaluation.run_evaluation --rqs RQ3 --log-groups predictcollab --predictors xgboost       --rq3-scopes partial --profile
PYTHONHASHSEED=0 python -m evaluation.run_evaluation --rqs RQ3 --log-groups predictcollab --predictors lstm_torch    --rq3-scopes partial --profile
PYTHONHASHSEED=0 python -m evaluation.run_evaluation --rqs RQ3 --log-groups predictcollab --predictors transformer   --rq3-scopes partial --profile
PYTHONHASHSEED=0 python -m evaluation.run_evaluation --rqs RQ3 --log-groups predictcollab --predictors gnn           --rq3-scopes partial --profile

python -m evaluation.audit_stage      --log-group predictcollab --scope partial
python -m evaluation.plot_rq3_metrics --log-group predictcollab --scope partial
```

### Stage 2 — Predict-Collab, full

```bash
PYTHONHASHSEED=0 python -m evaluation.run_evaluation --rqs RQ3 --log-groups predictcollab --predictors random_forest --rq3-scopes full --profile
PYTHONHASHSEED=0 python -m evaluation.run_evaluation --rqs RQ3 --log-groups predictcollab --predictors xgboost       --rq3-scopes full --profile
PYTHONHASHSEED=0 python -m evaluation.run_evaluation --rqs RQ3 --log-groups predictcollab --predictors lstm_torch    --rq3-scopes full --profile
PYTHONHASHSEED=0 python -m evaluation.run_evaluation --rqs RQ3 --log-groups predictcollab --predictors transformer   --rq3-scopes full --profile
PYTHONHASHSEED=0 python -m evaluation.run_evaluation --rqs RQ3 --log-groups predictcollab --predictors gnn           --rq3-scopes full --profile

python -m evaluation.audit_stage      --log-group predictcollab --scope full
python -m evaluation.plot_rq3_metrics --log-group predictcollab --scope full
```

### Stage 3 — BPIC2013, partial

```bash
PYTHONHASHSEED=0 python -m evaluation.run_evaluation --rqs RQ3 --log-groups bpi2013 --predictors random_forest --rq3-scopes partial --profile
PYTHONHASHSEED=0 python -m evaluation.run_evaluation --rqs RQ3 --log-groups bpi2013 --predictors xgboost       --rq3-scopes partial --profile
PYTHONHASHSEED=0 python -m evaluation.run_evaluation --rqs RQ3 --log-groups bpi2013 --predictors lstm_torch    --rq3-scopes partial --profile
PYTHONHASHSEED=0 python -m evaluation.run_evaluation --rqs RQ3 --log-groups bpi2013 --predictors transformer   --rq3-scopes partial --profile
PYTHONHASHSEED=0 python -m evaluation.run_evaluation --rqs RQ3 --log-groups bpi2013 --predictors gnn           --rq3-scopes partial --profile

python -m evaluation.audit_stage      --log-group bpi2013 --scope partial
python -m evaluation.plot_rq3_metrics --log-group bpi2013 --scope partial
```

### Stage 4 — BPIC2013, full

```bash
PYTHONHASHSEED=0 python -m evaluation.run_evaluation --rqs RQ3 --log-groups bpi2013 --predictors random_forest --rq3-scopes full --profile
PYTHONHASHSEED=0 python -m evaluation.run_evaluation --rqs RQ3 --log-groups bpi2013 --predictors xgboost       --rq3-scopes full --profile
PYTHONHASHSEED=0 python -m evaluation.run_evaluation --rqs RQ3 --log-groups bpi2013 --predictors lstm_torch    --rq3-scopes full --profile
PYTHONHASHSEED=0 python -m evaluation.run_evaluation --rqs RQ3 --log-groups bpi2013 --predictors transformer   --rq3-scopes full --profile
PYTHONHASHSEED=0 python -m evaluation.run_evaluation --rqs RQ3 --log-groups bpi2013 --predictors gnn           --rq3-scopes full --profile

python -m evaluation.audit_stage      --log-group bpi2013 --scope full
python -m evaluation.plot_rq3_metrics --log-group bpi2013 --scope full
```

## Optional: reusing `partial` results inside `full`

`partial`'s 6 tasks (`RQ3_SUBSET`) are a strict subset of `full`'s 14
(`EQUIVALENCE_TASKS`), and a task's result does not depend on which other
tasks ran alongside it: `run_one_log` computes `ocel_load`,
`feature_extraction`, the fold map, `ctx` and `a_hat`/`p_star` *before* the
task loop, each task builds its own `tt` from a copy of the shared table, and
every predictor re-seeds its RNG on entry to each fold. So for an expensive
predictor you can run `partial`, then run only `full`'s remaining 8 tasks,
and merge — instead of paying for those 6 tasks twice.

Verified empirically against the existing runs (the 6 shared (log, task) rows,
`partial` vs `full`): `xgboost`/`lstm_torch`/`transformer` are bit-identical
on both log groups; `random_forest` and `gnn` differ by at most 1.5e-11 —
float summation-order noise, ~12 orders of magnitude below reported precision.

```bash
# 1. the partial stage, as usual
PYTHONHASHSEED=0 python -m evaluation.run_evaluation --rqs RQ3 --log-groups bpi2013 \
    --predictors gnn --rq3-scopes partial --profile

# 2. only full's remaining 8 tasks, into a suffixed file
PYTHONHASHSEED=0 python -m evaluation.run_evaluation --rqs RQ3 --log-groups bpi2013 \
    --predictors gnn --rq3-scopes full --profile \
    --tasks NE-NEPa NE-NEPr NE-NMPa NE-NPaM NV-NMPa NV-TNE NV-TNM OB-P

# 3. concatenate partial's 6 rows with step 2's 8 rows into the 14-row full CSV
#    (same merge caveats as the outlier-patch section below)
```

**Weigh the saving before doing this.** It is *not* 6/14 of the runtime: task
cost is very uneven (on BPI2013 four tasks have 14-18k labelled rows against
62k for the other ten), and the two sets split the cheap ones. Measured share
of a full run spent on the 6 partial tasks: 46.6% (`lstm_torch`), 57.4%
(`transformer`), ~47.9% for `gnn` by sample-count proxy (its fit time tracks
sample count at r=0.99). So the realistic saving is **~30%**, worth it for
`gnn` on BPI2013 (~17-20 h down to ~11-13 h) and pointless for the tabular
predictors, which finish either scope in minutes.

Trade-off to accept knowingly: the merged `full` CSV is no longer the output
of one reproducible command, only content-equivalent to it.

## Following a long run live

`run_rq3` always writes a plain-text progress log (one line per fold/task/log
elapsed time, flushed immediately) to
`data/results/{log_group}_{scope}/rq3_progress_{predictor}_{log_group}[_full].log`
— tail it from any terminal, independent of how the process itself was
launched/redirected:

```powershell
# PowerShell (VS Code's default terminal on Windows)
Get-Content -Path data\results\predictcollab_full\rq3_progress_gnn_predictcollab_full.log -Wait -Tail 20
```

```bash
# Git Bash
tail -f data/results/predictcollab_full/rq3_progress_gnn_predictcollab_full.log
```

## Recovering from a single bad (log, task) timing outlier

If profiling shows an isolated outlier (e.g. one fold taking orders of
magnitude longer than its siblings — external machine interference, not a
code issue), it's cheaper to patch just that cell than to redo the whole
stage:

1. Re-run only that cell with `--tasks` (and `--logs` when the log group has
   more than one log), keeping every other flag identical to the stage's:

   ```bash
   PYTHONHASHSEED=0 python -m evaluation.run_evaluation --rqs RQ3 \
       --log-groups bpi2013 --predictors transformer --rq3-scopes full \
       --tasks OB-M --profile
   ```

   This writes `rq3_{results,profile}_transformer_bpi2013_full__OB-M.{csv,log}`
   into the stage directory. The `__…` suffix is automatic and exists so the
   patch can never overwrite the stage's own CSVs.
2. Replace the matching rows in the stage's `rq3_results_*.csv` (one row per
   (log, task)) and `rq3_profile_*.csv` (many rows per (log, task), one per
   stage/fold) with the patch run's rows.
3. Watch for `ocel_load`/`feature_extraction` duplicates in the profile CSV:
   those two stages are recorded once per log (not per task) and carry no
   `task` value, so a row-level filter keyed on (log, task) will not match
   them and the patch's copies get appended alongside the originals. Drop the
   patch's and keep the stage's.
4. Delete the `__…`-suffixed patch files.
