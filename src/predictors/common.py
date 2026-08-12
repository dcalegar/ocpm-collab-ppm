"""Shared helpers reused across predictor modules."""
from typing import List
import pandas as pd


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
