"""
RQ3 — end-to-end feasibility on a native OCPA pipeline.

For each log and each task in the representative subset: build the object-centric log
(via ocpm_tasks adapters), extract OCPA features, compute ℓ^R2 labels (ocpm_tasks),
join them by (case_id, k), and run 5-fold cross-validation GROUPED BY
CollaborationInstance (all prefixes of a case stay in one fold). Reports the
descriptive metric (macro F1 / MAE) as mean +/- sd over folds, plus a trivial
baseline. No computation times are reported (V4).
"""
import os
import collections
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from ocpm_tasks.catalog import TASKS
from ocpm_tasks import labels as TL
from .config import ExperimentConfig, LogSpec
from .io_ocel import load_ocpa_ocel, read_ocel2_labels
from .features_ocpa import extract_feature_table
from .models import fit_and_score_fold


def _a_hat(log, cfg):
    if cfg.obm_target_activity:
        return cfg.obm_target_activity
    c = collections.Counter(e.activity for ex in log for e in ex.events if e.is_send)
    return c.most_common(1)[0][0] if c else None


def _p_star(log, cfg):
    if cfg.obp_target_participant:
        return cfg.obp_target_participant
    c = collections.Counter(e.actor for ex in log for e in ex.events if e.actor)
    return c.most_common(1)[0][0] if c else None


def _param(task, a_hat, p_star):
    return a_hat if task.param == "activity" else (
        p_star if task.param == "participant" else None)


def run_one_log(spec: LogSpec, cfg: ExperimentConfig) -> List[dict]:
    ocpa_ocel = load_ocpa_ocel(cfg.schema, spec.ocel_path)
    ocel_log = read_ocel2_labels(spec.ocel_path, cfg.schema, ocpa_ocel=ocpa_ocel)
    ctx = TL.build_context(ocel_log, cfg.bottom)
    a_hat, p_star = _a_hat(ocel_log, cfg), _p_star(ocel_log, cfg)

    feats = extract_feature_table(spec.name, cfg.schema, spec.ocel_path, ocel_log,
                                  ocpa_ocel=ocpa_ocel)
    table, feature_cols = feats["table"], feats["feature_cols"]

    rows: List[dict] = []
    for key in cfg.rq3_tasks:
        task = TASKS[key]
        param = _param(task, a_hat, p_star)
        # labels joined by event_id (the table carries event_id and case_id)
        lab = {str(e): y for (_c, e, _k, y)
               in TL.compute_label_rows(ocel_log, task, param, ctx)}
        tt = table.copy()
        tt["_y"] = tt["event_id"].astype(str).map(lab)
        tt = tt[tt["_y"].notna()].reset_index(drop=True)

        rec = {"log": spec.name, "task": key, "anchor": task.anchor,
               "problem_type": task.problem_type, "samples": int(len(tt)),
               "ran_end_to_end": len(tt) > 0}
        if task.param == "activity":
            rec["obm_activity"] = a_hat
        if task.param == "participant":
            rec["participant"] = p_star
        if len(tt) == 0:
            rec["note"] = "no labelled rows"
            rows.append(rec)
            continue

        groups = tt["case_id"].astype(str).values
        n_groups = len(set(groups))
        n_splits = min(cfg.n_folds, n_groups)
        metric_name = "f1_macro" if task.kind in ("categorical", "binary") else "mae"
        if n_splits < 2:
            rec["note"] = "too few collaboration instances for CV"
            rows.append(rec)
            continue

        gkf = GroupKFold(n_splits=n_splits)
        ms, bs = [], []
        idx = np.arange(len(tt))
        for tr, te in gkf.split(idx, groups=groups):
            train_mask = pd.Series(False, index=tt.index); train_mask.iloc[tr] = True
            test_mask = pd.Series(False, index=tt.index); test_mask.iloc[te] = True
            r = fit_and_score_fold(tt, feature_cols, "_y", task,
                                   train_mask, test_mask, cfg)
            if r:
                ms.append(r["metric"]); bs.append(r["baseline"])
        rec["metric_name"] = metric_name
        rec["metric_mean"] = float(np.mean(ms)) if ms else None
        rec["metric_sd"] = float(np.std(ms)) if ms else None
        rec["baseline_mean"] = float(np.mean(bs)) if bs else None
        rec["folds"] = len(ms)
        rows.append(rec)
    return rows


def run_rq3(cfg: Optional[ExperimentConfig] = None,
           out_name: str = "rq3_results.csv") -> pd.DataFrame:
    cfg = cfg or ExperimentConfig()
    os.makedirs(cfg.out_dir, exist_ok=True)
    out: List[dict] = []
    for spec in cfg.logs:
        print(f"\n===== RQ3 LOG: {spec.name} =====")
        try:
            out.extend(run_one_log(spec, cfg))
        except Exception as ex:                       # noqa: BLE001
            print(f"[{spec.name}] ERROR: {ex}")
            out.append({"log": spec.name, "error": str(ex)})
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(cfg.out_dir, out_name), index=False)
    print(f"[ok] wrote {out_name}")
    return df
