"""
RQ3 — end-to-end feasibility on a native OCPA pipeline.

For each log and each task in the representative subset: build the object-centric log
(via tasks adapters), extract OCPA features, compute ℓ^R2 labels (tasks),
join them by (case_id, k), and run 5-fold cross-validation GROUPED BY
CollaborationCase (all prefixes of a case stay in one fold). Reports the
descriptive metric (macro F1 / MAE) as mean +/- sd over folds, plus a trivial
baseline. Per-task/per-log elapsed time is printed to stdout for progress
monitoring only; it is not part of the persisted results (V4). Finer-grained
per-stage wall-clock + memory profiling (ocel_load, feature_extraction,
labeling, fit, predict -- the last two recorded by each predictors/*.py
module via a bound StageTimer, see profiling.py and predictors/README.md)
is available opt-in via cfg.profile=True, written to a separate
rq3_profile_*.csv rather than into rq3_results_*.csv, preserving V4.

Fold assignment (which CollaborationCase lands in which of the n_folds
buckets) is a PERSISTED, per-log partition -- see generate_folds/
load_or_generate_folds below -- computed once from the log's case_id set
alone, independent of task and predictor:
  * independent of predictor by construction even before persistence, since
    cfg.predictor is never read while building the per-task feature/label
    table `tt` or its case_id grouping;
  * independent of task only once persisted: the previous in-line
    `GroupKFold(...).split(idx, groups=tt["case_id"])` balanced folds by
    ROW COUNT per case, and a case's row count varies by task (categorical/
    numeric label rows are dropped when no further occurrence exists, see
    tasks.labels.compute_label_rows), so two tasks on the same log could get
    different fold boundaries under that scheme. Persisting a single
    case-level partition (data/folds/{log_name}_folds.csv) removes that
    -- verified empirically to introduce no empty test folds on any of the
    four Predict-Collab study logs or on BPI2013 at the default n_folds=5.
Because feature values are per-execution and never leak across
CollaborationCase boundaries (regression-tested in
tests/test_features_leakage.py, CROSS-CASE direction), partitioning the
already-extracted feature table by case_id this way is equivalent to having
extracted features on physically separate per-fold logs, at a fraction of
the cost.
"""
import json
import os
import time
import collections
from typing import Dict, Iterable, List, Optional
import numpy as np
import pandas as pd

from tasks.catalog import TASKS
from tasks import labels as TL
from .config import ExperimentConfig, LogSpec
from .profiling import NullProfileCollector, ProfileCollector, StageTimer
from features.io_ocel import load_ocpa_ocel, read_ocel2_labels
from features.ocpa import extract_feature_table
from predictors.dispatch import resolve as resolve_predictor


def _a_hat(log, cfg):
    if cfg.obm_target_activity:
        return cfg.obm_target_activity
    # OB-M's own definition triggers on a message sent OR received (is_msg,
    # see labels.py::_OB_M), so the candidate activities must include
    # receive-only ones, not just sends.
    c = collections.Counter(e.activity for ex in log for e in ex.events if e.is_msg)
    return c.most_common(1)[0][0] if c else None


def _p_star(log, cfg):
    if cfg.obp_target_participant:
        return cfg.obp_target_participant
    c = collections.Counter(e.actor for ex in log for e in ex.events if e.actor)
    return c.most_common(1)[0][0] if c else None


def _param(task, a_hat, p_star):
    return a_hat if task.param == "activity" else (
        p_star if task.param == "participant" else None)


def generate_folds(case_ids: Iterable[str], n_folds: int, seed: int) -> Dict[str, int]:
    """Deterministic partition of CollaborationCase ids into n_folds buckets,
    a pure function of the case_id SET (not of any task's label/feature
    rows), so the same assignment is valid for every task and predictor run
    on this log. Not row-count-balanced per task like the previous in-line
    GroupKFold call -- see the module docstring for why that's the point."""
    ids = sorted(set(map(str, case_ids)))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    return {ids[i]: int(rank % n_folds) for rank, i in enumerate(order)}


def _folds_paths(folds_dir: str, log_name: str) -> "tuple[str, str]":
    base = os.path.join(folds_dir, f"{log_name}_folds")
    return f"{base}.csv", f"{base}.meta.json"


def save_folds(fold_of: Dict[str, int], n_folds: int, seed: int,
               folds_dir: str, log_name: str) -> None:
    os.makedirs(folds_dir, exist_ok=True)
    csv_path, meta_path = _folds_paths(folds_dir, log_name)
    (pd.DataFrame({"case_id": list(fold_of), "fold": list(fold_of.values())})
     .sort_values("case_id")
     .to_csv(csv_path, index=False))
    with open(meta_path, "w") as f:
        json.dump({"n_folds": n_folds, "seed": seed, "n_cases": len(fold_of)},
                  f, indent=2)


def load_folds(folds_dir: str, log_name: str, n_folds: int, seed: int,
               case_ids: Iterable[str]) -> Optional[Dict[str, int]]:
    """Returns the persisted case_id -> fold map, or None if no file exists yet.
    Raises if a file exists but was generated under a different (n_folds,
    seed) or a different case_id set -- silently reusing a stale partition
    would defeat the point of pinning it (see module docstring); the fix is
    to delete the file or pass regenerate_folds=True."""
    csv_path, meta_path = _folds_paths(folds_dir, log_name)
    if not (os.path.exists(csv_path) and os.path.exists(meta_path)):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    if meta.get("n_folds") != n_folds or meta.get("seed") != seed:
        raise RuntimeError(
            f"[{log_name}] {csv_path} was generated with n_folds="
            f"{meta.get('n_folds')}, seed={meta.get('seed')}; current config "
            f"requests n_folds={n_folds}, seed={seed}. Delete the file(s) or "
            "pass regenerate_folds=True to regenerate.")
    df = pd.read_csv(csv_path, dtype={"case_id": str, "fold": int})
    fold_of = dict(zip(df["case_id"], df["fold"]))
    current_ids = set(map(str, case_ids))
    if set(fold_of) != current_ids:
        raise RuntimeError(
            f"[{log_name}] {csv_path}'s case_id set does not match the "
            f"current log ({len(fold_of)} vs {len(current_ids)} cases) -- "
            "the log file likely changed since the folds were generated. "
            "Delete the file(s) or pass regenerate_folds=True to regenerate.")
    return fold_of


def load_or_generate_folds(case_ids: Iterable[str], n_folds: int, seed: int,
                           folds_dir: str, log_name: str,
                           regenerate: bool = False) -> Dict[str, int]:
    case_ids = list(case_ids)
    if not regenerate:
        existing = load_folds(folds_dir, log_name, n_folds, seed, case_ids)
        if existing is not None:
            return existing
    fold_of = generate_folds(case_ids, n_folds, seed)
    save_folds(fold_of, n_folds, seed, folds_dir, log_name)
    return fold_of


def _log(msg: str, log_file=None) -> None:
    """Print AND (if given) append to a persistent per-run progress log file,
    flushed immediately -- so progress is visible via `tail -f` from outside
    whatever process/redirection launched this run, not just on its stdout."""
    print(msg)
    if log_file is not None:
        log_file.write(msg + "\n")
        log_file.flush()


def run_one_log(spec: LogSpec, cfg: ExperimentConfig,
                profiler: "Optional[ProfileCollector]" = None,
                log_file=None) -> List[dict]:
    profiler = profiler or NullProfileCollector()
    with profiler.stage("ocel_load", spec.name, cfg.predictor):
        ocpa_ocel = load_ocpa_ocel(cfg.schema, spec.ocel_path)
        ocel_log = read_ocel2_labels(spec.ocel_path, cfg.schema, ocpa_ocel=ocpa_ocel)
    ctx = TL.build_context(ocel_log, cfg.bottom)
    a_hat, p_star = _a_hat(ocel_log, cfg), _p_star(ocel_log, cfg)

    with profiler.stage("feature_extraction", spec.name, cfg.predictor):
        feats = extract_feature_table(spec.name, cfg.schema, spec.ocel_path, ocel_log,
                                      ocpa_ocel=ocpa_ocel)
    feats["log_name"] = spec.name
    table, feature_cols = feats["table"], feats["feature_cols"]
    fit_fn = resolve_predictor(cfg.predictor)

    # Persisted, task/predictor-independent CollaborationCase -> fold map (see
    # module docstring). Loaded/generated once here so every task in this log
    # reuses the exact same partition.
    fold_of = load_or_generate_folds(
        ocel_log.case_ids, cfg.n_folds, cfg.random_state,
        cfg.folds_dir, spec.name, regenerate=cfg.regenerate_folds)

    rows: List[dict] = []
    for key in cfg.rq3_tasks:
        t_task_start = time.perf_counter()
        task = TASKS[key]
        param = _param(task, a_hat, p_star)
        with profiler.stage("labeling", spec.name, cfg.predictor, task=key):
            # labels joined by event_id (the table carries event_id and case_id)
            lab = {str(e): y for (_c, e, _k, y)
                   in TL.compute_label_rows(ocel_log, task, param, ctx)}
            tt = table.copy()
            tt["_y"] = tt["event_id"].astype(str).map(lab)
            tt = tt[tt["_y"].notna()].reset_index(drop=True)

        rec = {"log": spec.name, "task": key, "anchor": task.anchor,
               "problem_type": task.problem_type, "predictor": cfg.predictor,
               "samples": int(len(tt)), "ran_end_to_end": len(tt) > 0}
        if task.param == "activity":
            rec["obm_activity"] = a_hat
        if task.param == "participant":
            rec["participant"] = p_star
        if len(tt) == 0:
            rec["note"] = "no labelled rows"
            rows.append(rec)
            continue

        n_groups = tt["case_id"].astype(str).nunique()
        metric_name = "f1_macro" if task.kind in ("categorical", "binary") else "mae"
        if n_groups < 2:
            rec["note"] = "too few collaboration instances for CV"
            rows.append(rec)
            continue

        tt_fold = tt["case_id"].astype(str).map(fold_of)

        # Recorded, not left to be inferred: a task whose training target is
        # constant cannot be learned, so every predictor scores exactly the
        # trivial baseline and the row is a tie by construction -- it must not
        # be read as "all models perfect", nor counted as a win for anyone.
        # `metric_mean == baseline_mean` is what degeneracy *causes*, not what
        # it *is* (a perfectly-predictable task looks identical), so the cause
        # is stored directly. Predictor-independent, hence computed here rather
        # than in predictors/*.py -- some of which short-circuit this case (see
        # predictors/README.md, "Degenerate folds") while others fit the
        # constant target normally.
        n_labels = int(tt["_y"].nunique())
        rec["n_labels"] = n_labels
        rec["degenerate"] = bool(
            all(tt.loc[tt_fold != f, "_y"].nunique() < 2
                for f in range(cfg.n_folds)
                if (tt_fold != f).sum() > 0))

        ms, bs = [], []
        for fold_no in range(cfg.n_folds):
            test_mask = tt_fold == fold_no
            train_mask = ~test_mask
            if test_mask.sum() == 0 or train_mask.sum() == 0:
                # Possible only when this task's labelled rows survive for
                # very few CollaborationCases (dropped BOTTOM rows, see
                # tasks.labels.compute_label_rows) and the fixed partition
                # happens to put all/none of them in this bucket -- not
                # observed on the study logs or BPI2013 at n_folds=5, but
                # the fixed partition (unlike the old per-task GroupKFold) no
                # longer prevents it by construction, so it must be handled.
                continue
            # Bound timer, not a "fit_predict" wrap: each predictor module
            # records its own "fit"/"predict" stages internally (see
            # predictors/README.md) so training and inference are broken out
            # separately rather than measured as one combined stage.
            timer = StageTimer(profiler, spec.name, cfg.predictor, task=key, fold=fold_no)
            t_fold_start = time.perf_counter()
            r = fit_fn(feats, tt, "_y", task,
                      train_mask, test_mask, cfg, timer)
            fold_elapsed = time.perf_counter() - t_fold_start
            _log(f"    [{spec.name}][{key}] fold {fold_no + 1}/{cfg.n_folds}: "
                f"{fold_elapsed:.1f}s", log_file)
            if r:
                ms.append(r["metric"]); bs.append(r["baseline"])
        elapsed = round(time.perf_counter() - t_task_start, 2)
        rec["metric_name"] = metric_name
        rec["metric_mean"] = float(np.mean(ms)) if ms else None
        rec["metric_sd"] = float(np.std(ms)) if ms else None
        rec["baseline_mean"] = float(np.mean(bs)) if bs else None
        rec["folds"] = len(ms)
        _log(f"  [{key}] {elapsed:.1f}s", log_file)
        rows.append(rec)
    return rows


def run_rq3(cfg: Optional[ExperimentConfig] = None,
           out_name: str = "rq3_results.csv") -> pd.DataFrame:
    cfg = cfg or ExperimentConfig()
    os.makedirs(cfg.out_dir, exist_ok=True)
    # Opt-in (cfg.profile): per-stage wall-clock + RSS profiling, written to
    # its own rq3_profile_*.csv sibling -- kept out of `out`/out_name so the
    # V4 "no computation times in the results" profile still holds for
    # rq3_results_*.csv itself (see module docstring).
    profiler = ProfileCollector() if cfg.profile else None
    # Always-on (unlike profiling): a plain-text progress log, one line per
    # per-fold/per-task/per-log elapsed-time message (see _log), so progress
    # on a long run is visible via `tail -f` from any terminal, independent
    # of whatever redirected (or didn't redirect) this process's stdout.
    progress_name = out_name.replace("rq3_results", "rq3_progress", 1).replace(".csv", ".log")
    if progress_name == out_name:
        progress_name = f"rq3_progress_{out_name}.log"
    progress_path = os.path.join(cfg.out_dir, progress_name)
    out: List[dict] = []
    with open(progress_path, "w") as log_file:
        for spec in cfg.logs:
            _log(f"\n===== RQ3 LOG: {spec.name} =====", log_file)
            t_log = time.perf_counter()
            try:
                out.extend(run_one_log(spec, cfg, profiler=profiler, log_file=log_file))
            except Exception as ex:                       # noqa: BLE001
                _log(f"[{spec.name}] ERROR: {ex}", log_file)
                out.append({"log": spec.name, "error": str(ex)})
            log_elapsed = time.perf_counter() - t_log
            _log(f"  [time] {spec.name} total: {log_elapsed:.1f}s", log_file)
    print(f"[ok] wrote {progress_name}")
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(cfg.out_dir, out_name), index=False)
    print(f"[ok] wrote {out_name}")
    if profiler is not None:
        # Swap the "rq3_results" prefix, NOT "rq3_results_": with the trailing
        # underscore the default out_name ("rq3_results.csv", used by any direct
        # run_rq3(cfg) call rather than by main(), which always passes
        # rq3_results_{predictor}_{group}*.csv) does not match, leaving
        # profile_name == out_name -- the profile frame then OVERWRITES the
        # results CSV it is meant to sit beside. The fallback below keeps any
        # future out_name that escapes the convention entirely from doing the
        # same, rather than relying on the prefix always being there.
        profile_name = out_name.replace("rq3_results", "rq3_profile", 1)
        if profile_name == out_name:
            profile_name = f"rq3_profile_{out_name}"
        profiler.to_frame().to_csv(os.path.join(cfg.out_dir, profile_name), index=False)
        print(f"[ok] wrote {profile_name}")
    return df
