"""
Per-fold training and scoring using a TensorFlow/Keras LSTM predictor.
Supports both classification (macro F1) and regression (MAE) tasks, following
the same predictor contract as random_forest.py.
"""
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder

import tensorflow as tf

from ocpm_tasks.catalog import Task
from .common import xy_split


def fit_and_score_fold(feats: dict, tt: pd.DataFrame, y_col: str,
                       task: Task, train_mask, test_mask, cfg) -> Dict[str, float]:
    feature_cols = feats["feature_cols"]
    X_tr, X_te, y_tr, y_te = xy_split(tt, feature_cols, y_col, train_mask, test_mask)
    if len(y_tr) == 0 or len(y_te) == 0:
        return {}

    # Set seeds for reproducibility
    seed = getattr(cfg, "random_state", 3395)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Scale continuous tabular features
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)

    # Group into sequences by case_id (GroupKFold ensures all events of a case are in the same fold)
    case_ids_tr = tt.loc[train_mask, "case_id"].values
    case_ids_te = tt.loc[test_mask, "case_id"].values

    def make_sequences(X, y, case_ids):
        seq_X, seq_y = [], []
        df_X = pd.DataFrame(X)
        df_X["case_id"] = case_ids
        df_y = pd.DataFrame({"y": y.values, "case_id": case_ids})
        for _, group in df_X.groupby("case_id", sort=False, dropna=False):
            seq_X.append(group.drop(columns=["case_id"]).values)
        for _, group in df_y.groupby("case_id", sort=False, dropna=False):
            seq_y.append(group["y"].values)
        return seq_X, seq_y

    if task.kind in ("categorical", "binary"):
        y_tr_used = y_tr.astype(str)
        y_te_used = y_te.astype(str)
    else:
        y_tr_used = y_tr
        y_te_used = y_te

    seq_X_tr, seq_y_tr = make_sequences(X_tr_scaled, y_tr_used, case_ids_tr)
    seq_X_te, seq_y_te = make_sequences(X_te_scaled, y_te_used, case_ids_te)

    from keras.utils import pad_sequences
    X_tr_pad = pad_sequences(seq_X_tr, padding='post', dtype='float32')
    X_te_pad = pad_sequences(seq_X_te, padding='post', dtype='float32')

    units = getattr(cfg, "lstm_units", 64)
    epochs = getattr(cfg, "lstm_epochs", 20)
    batch_size = getattr(cfg, "lstm_batch_size", 32)
    lr = getattr(cfg, "lstm_learning_rate", 0.001)

    if task.kind in ("categorical", "binary"):
        y_tr_s, y_te_s = y_tr_used, y_te_used
        le = LabelEncoder()
        le.fit(y_tr_s)
        num_classes = len(le.classes_)
        
        y_tr_enc = pad_sequences([le.transform(seq) for seq in seq_y_tr], padding='post', value=-1)
        
        # We need a sample weight to ignore padded values in loss computation
        sample_weight = (y_tr_enc != -1).astype('float32')
        y_tr_enc_valid = np.maximum(y_tr_enc, 0) # replace -1 with 0 for valid loss calc
        y_tr_enc_valid = np.expand_dims(y_tr_enc_valid, axis=-1)

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(None, X_tr_pad.shape[2])),
            tf.keras.layers.LSTM(units=units, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes if num_classes > 2 else 1, activation="softmax" if num_classes > 2 else "sigmoid")
        ])

        loss_fn = "sparse_categorical_crossentropy" if num_classes > 2 else "binary_crossentropy"
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=loss_fn)
        model.fit(X_tr_pad, y_tr_enc_valid, sample_weight=sample_weight, epochs=epochs, batch_size=batch_size, verbose=0)

        probs = model.predict(X_te_pad, verbose=0)
        # Extract only the valid (unpadded) predictions and flatten
        flat_probs = np.concatenate([probs[i, :len(seq_X_te[i])] for i in range(len(seq_X_te))])
        pred_enc = np.argmax(flat_probs, axis=1) if num_classes > 2 else (flat_probs > 0.5).astype(int).flatten()
        pred_labels = le.inverse_transform(pred_enc)

        maj = pd.Series(y_tr_s).mode().iloc[0]
        return {
            "metric": float(f1_score(y_te_s, pred_labels, average="macro", zero_division=0)),
            "baseline": float(f1_score(y_te_s, [maj] * len(y_te_s), average="macro", zero_division=0)),
            "n_test": int(len(y_te_s)),
        }
    else:
        y_tr_pad = pad_sequences([seq.astype(float) for seq in seq_y_tr], padding='post', dtype='float32', value=-9999.0)
        sample_weight = (y_tr_pad != -9999.0).astype('float32')
        y_tr_pad_valid = np.where(y_tr_pad == -9999.0, 0.0, y_tr_pad)
        y_tr_pad_valid = np.expand_dims(y_tr_pad_valid, axis=-1)

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(None, X_tr_pad.shape[2])),
            tf.keras.layers.LSTM(units=units, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1)
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
        model.fit(X_tr_pad, y_tr_pad_valid, sample_weight=sample_weight, epochs=epochs, batch_size=batch_size, verbose=0)

        pred = model.predict(X_te_pad, verbose=0)
        flat_pred = np.concatenate([pred[i, :len(seq_X_te[i])] for i in range(len(seq_X_te))]).flatten()
        
        y_te_float = y_te.astype(float).to_numpy()
        median = float(np.median(y_tr.astype(float).to_numpy()))

        return {
            "metric": float(mean_absolute_error(y_te_float, flat_pred)),
            "baseline": float(mean_absolute_error(y_te_float, [median] * len(y_te_float))),
            "n_test": int(len(y_te_float)),
        }
