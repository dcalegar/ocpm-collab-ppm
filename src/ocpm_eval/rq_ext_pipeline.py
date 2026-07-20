"""
RQ-EXT — object-enabled EXTENSION tasks (X-Inf, X-MSt), demonstrated on a
hand-built toy log (data/logs/ToyCollab/toy_collab.*) designed to exercise both extensions,
never on the four study logs.

This mirrors rq3_pipeline.run_one_log/run_rq3 exactly (same OCPA feature
extraction, same grouped-by-CollaborationCase CV, same fixed RandomForest /
descriptive-metric protocol), but:
  * reads ocpm_tasks.extensions.EXT_TASKS / compute_ext_label_rows instead of
    catalog.TASKS / labels.compute_label_rows;
  * passes ``corr_attr`` through to the label side so X-MSt's send<->receive
    correspondence is available (an explicit ENRICHMENT, not part of the core
    M1-M8 mapping -- see ocpm_tasks/adapters.py); the toy log carries ``msgId``
    attributes on both send and receive events to enable X-MSt evaluation;
  * is invoked on ExtExperimentConfig's toy log registry, kept out of
    ExperimentConfig/rq3_results*.csv entirely.

Because it reuses the exact same feature-extraction/model-fitting building
blocks as RQ3, a task passing through here demonstrates it can run
in the same native OCPA pipeline -- it is not a different, easier code path.
"""
import os
from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from ocpm_tasks import extensions as EXT
from .config import ExtExperimentConfig, LogSpec
from .io_ocel import load_ocpa_ocel, read_ocel2_labels
from .features_ocpa import extract_feature_table
from .predictors.dispatch import resolve as resolve_predictor


def run_one_log(spec: LogSpec, cfg: ExtExperimentConfig) -> List[dict]:
    ocpa_ocel = load_ocpa_ocel(cfg.schema, spec.ocel_path)
    ocel_log = read_ocel2_labels(spec.ocel_path, cfg.schema, ocpa_ocel=ocpa_ocel,
                                 corr_attr=cfg.corr_attr)
    ctx = EXT.build_context(ocel_log, cfg.bottom)

    feats = extract_feature_table(spec.name, cfg.schema, spec.ocel_path, ocel_log,
                                  ocpa_ocel=ocpa_ocel)
    table, feature_cols = feats["table"], feats["feature_cols"]
    fit_fn = resolve_predictor(cfg.predictor)

    rows: List[dict] = []
    for key in cfg.ext_tasks:
        task = EXT.EXT_TASKS[key]
        lab = {str(e): y for (_c, e, _k, y)
               in EXT.compute_ext_label_rows(ocel_log, task, None, ctx)}
        tt = table.copy()
        tt["_y"] = tt["event_id"].astype(str).map(lab)
        tt = tt[tt["_y"].notna()].reset_index(drop=True)

        rec = {"log": spec.name, "task": key, "anchor": task.anchor,
               "problem_type": task.problem_type, "predictor": cfg.predictor,
               "samples": int(len(tt)), "ran_end_to_end": len(tt) > 0}
        if len(tt) == 0:
            rec["note"] = "no labelled rows (e.g. X-MSt with no corr_attr)"
            rows.append(rec)
            continue

        groups = tt["case_id"].astype(str).values
        n_groups = len(set(groups))
        n_splits = min(cfg.n_folds, n_groups)
        metric_name = "mae"   # both extensions are numeric (count / time)
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
            r = fit_fn(tt, feature_cols, "_y", task,
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


def run_rq_ext(cfg: "ExtExperimentConfig | None" = None,
              out_name: str = "rq_ext_results_toy.csv") -> pd.DataFrame:
    cfg = cfg or ExtExperimentConfig()
    os.makedirs(cfg.out_dir, exist_ok=True)
    out: List[dict] = []
    for spec in cfg.logs:
        print(f"\n===== RQ-EXT LOG (toy): {spec.name} =====")
        try:
            out.extend(run_one_log(spec, cfg))
        except Exception as ex:                       # noqa: BLE001
            print(f"[{spec.name}] ERROR: {ex}")
            out.append({"log": spec.name, "error": str(ex)})
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(cfg.out_dir, out_name), index=False)
    print(f"[ok] wrote {out_name}")
    return df
