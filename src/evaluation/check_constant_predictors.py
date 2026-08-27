"""Constant-predictor reference (read-only).

For each BPIC 2013 numeric target, reports the MAE of the
median constant (= the published trivial baseline, L1-optimal) against the MAE of
the mean constant (L2-optimal). If the four squared-error predictors land near the
mean constant and the Huber/L1 GNN lands near the median constant, the training
objective explains the published pattern without reference to the encoding.

Run (ocpa venv, ~30 s):  .venv/bin/python3.10 src/evaluation/check_constant_predictors.py
"""
import os, sys, collections
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
import numpy as np
from evaluation.config import ExperimentConfig, LogSpec
from evaluation.rq3_pipeline import load_or_generate_folds
from tasks.catalog import TASKS
from tasks import labels as TL
from features.io_ocel import load_ocpa_ocel, read_ocel2_labels
from features.ocpa import extract_feature_table

spec = LogSpec("BPI2013", "data/logs/BPIChallenge2013/BPI2013_incidents_collaborative.sqlite",
               "data/logs/BPIChallenge2013/BPI2013_incidents_collaborative.xes")
cfg = ExperimentConfig()
ocpa_ocel = load_ocpa_ocel(cfg.schema, spec.ocel_path)
ocel_log = read_ocel2_labels(spec.ocel_path, cfg.schema, ocpa_ocel=ocpa_ocel)
ctx = TL.build_context(ocel_log, cfg.bottom)
c = collections.Counter(e.actor for ex in ocel_log for e in ex.events if e.actor)
p_star = c.most_common(1)[0][0]
feats = extract_feature_table(spec.name, cfg.schema, spec.ocel_path, ocel_log, ocpa_ocel=ocpa_ocel)
table = feats["table"]
fold_of = load_or_generate_folds(ocel_log.case_ids, cfg.n_folds, cfg.random_state,
                                 cfg.folds_dir, spec.name, regenerate=False)
print(f"p* = {p_star}\n")
print(f"{'task':9s} {'samples':>8s} {'MAE(median)':>13s} {'MAE(mean)':>13s} {'mean/median':>12s}")
for key in ("NV-PrT", "NV-PaT", "NV-TNE", "NV-TNM", "NV-NMPr", "NV-NMPa"):
    task = TASKS[key]
    param = p_star if task.param == "participant" else None
    lab = {str(e): y for (_c, e, _k, y) in TL.compute_label_rows(ocel_log, task, param, ctx)}
    tt = table.copy()
    tt["_y"] = tt["event_id"].astype(str).map(lab)
    tt = tt[tt["_y"].notna()].reset_index(drop=True)
    folds = tt["case_id"].astype(str).map(fold_of)
    med, mea = [], []
    for f in sorted(folds.unique()):
        ytr = tt.loc[folds != f, "_y"].astype(float).to_numpy()
        yte = tt.loc[folds == f, "_y"].astype(float).to_numpy()
        if len(ytr) == 0 or len(yte) == 0:
            continue
        med.append(np.abs(yte - np.median(ytr)).mean())
        mea.append(np.abs(yte - ytr.mean()).mean())
    print(f"{key:9s} {len(tt):8d} {np.mean(med):13.3f} {np.mean(mea):13.3f} "
          f"{np.mean(mea)/np.mean(med):11.2f}x")
