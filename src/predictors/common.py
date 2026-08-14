"""Shared helpers reused across predictor modules."""
from contextlib import contextmanager
from typing import List
import pandas as pd


# Marker written to a profile row's `note` column when a predictor entered a
# stage but deliberately did no work, so the recorded ~0s is readable as "was
# not run" rather than "ran, very fast" -- the two are otherwise identical in
# the CSV. Defined here (not in evaluation.profiling) so predictors can label
# their rows without importing evaluation; see predictors/README.md.
DEGENERATE_CONSTANT_TARGET = "skipped:constant-target"


class NullStageTimer:
    """Default for fit_and_score_fold's optional `timer` parameter, used
    when the caller (a direct call, e.g. tests/test_predictors_registry.py,
    or any pipeline not doing profiling) doesn't pass one. Duck-types
    evaluation.profiling.StageTimer's `.stage(name, note=...)` context manager
    without importing evaluation -- this package stays decoupled from it (see
    predictors/README.md)."""

    @contextmanager
    def stage(self, name: str, note: str = None):
        yield


def resolve_device(cfg):
    """Resolve cfg.device ("auto"/"cpu"/"cuda", default "cpu") to a torch.device,
    shared by every GPU-capable predictor (gnn, lstm_torch, transformer --
    random_forest and xgboost are CPU-only regardless). "auto" picks CUDA
    when available; that is deliberately not the default -- see
    ExperimentConfig.device's comment for the measured dispatch-overhead
    regression this default avoids. torch is imported lazily so importing
    this module doesn't force a torch dependency on predictors that don't
    need it."""
    import torch
    requested = getattr(cfg, "device", "cpu").lower()
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda', but CUDA is not available")
    use_cuda = requested == "cuda" or (requested == "auto" and torch.cuda.is_available())
    return torch.device("cuda" if use_cuda else "cpu")


def xy_split(table: pd.DataFrame, feature_cols: List[str], y_col: str, train_mask, test_mask):
    """Split into train/test and encode categorical columns fit on train only.

    Unseen categories at test time (present in test but not in train) are
    encoded as -1, matching pandas' standard out-of-vocabulary convention,
    rather than being folded into a train+test-wide category mapping.
    """
    X_tr = table.loc[train_mask, feature_cols].copy()
    X_te = table.loc[test_mask, feature_cols].copy()
    for c in feature_cols:
        if X_tr[c].dtype == object or str(X_tr[c].dtype).startswith("category"):
            categories = pd.Categorical(X_tr[c]).categories
            X_tr[c] = pd.Categorical(X_tr[c], categories=categories).codes
            X_te[c] = pd.Categorical(X_te[c], categories=categories).codes
    y = table[y_col]
    return X_tr.fillna(0.0), X_te.fillna(0.0), y[train_mask], y[test_mask]
