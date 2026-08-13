"""
Per-fold training and scoring using a PyTorch LSTM predictor.
Supports both classification (macro F1) and regression (MAE) tasks, following
the same predictor contract as random_forest.py and lstm.py.
"""
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tasks.catalog import Task
from .common import NullStageTimer, xy_split


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, is_classification):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.is_classification = is_classification

    def forward(self, x, lengths):
        # Pack sequence to ignore padding during LSTM computation
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_x)
        # No total_length: pad back only to this call's own max(lengths), not
        # the padded input's full time axis. x is padded to the fold-wide max
        # sequence length (set once in fit_and_score_fold), so a mini-batch
        # that doesn't contain the fold's longest case would otherwise force
        # dropout+fc over padding far beyond anything this batch needs.
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        out = self.dropout(out)
        out = self.fc(out)
        # Activation (softmax/sigmoid) is handled implicitly by CrossEntropyLoss/BCEWithLogitsLoss
        return out


def fit_and_score_fold(feats: dict, tt: pd.DataFrame, y_col: str,
                       task: Task, train_mask, test_mask, cfg,
                       timer=None) -> Dict[str, float]:
    timer = timer or NullStageTimer()
    feature_cols = feats["feature_cols"]
    X_tr, X_te, y_tr, y_te = xy_split(tt, feature_cols, y_col, train_mask, test_mask)
    if len(y_tr) == 0 or len(y_te) == 0:
        return {}

    seed = getattr(cfg, "random_state", 3395)
    torch.manual_seed(seed)
    np.random.seed(seed)

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)

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
        y_scaler = None
    else:
        y_scaler = StandardScaler()
        y_tr_used = pd.Series(y_scaler.fit_transform(y_tr.astype(float).values.reshape(-1, 1)).flatten())
        y_te_used = pd.Series(y_scaler.transform(y_te.astype(float).values.reshape(-1, 1)).flatten())

    seq_X_tr, seq_y_tr = make_sequences(X_tr_scaled, y_tr_used, case_ids_tr)
    seq_X_te, seq_y_te = make_sequences(X_te_scaled, y_te_used, case_ids_te)

    def pad_sequences(seq_list, padding_value=0.0):
        lengths = [len(s) for s in seq_list]
        max_len = max(lengths)
        dim = seq_list[0].shape[-1] if seq_list[0].ndim > 1 else 1
        padded = np.full((len(seq_list), max_len, dim), padding_value, dtype=np.float32)
        for i, s in enumerate(seq_list):
            if s.ndim == 1:
                padded[i, :len(s), 0] = s
            else:
                padded[i, :len(s), :] = s
        return padded, lengths

    X_tr_pad, lengths_tr = pad_sequences(seq_X_tr)
    X_te_pad, lengths_te = pad_sequences(seq_X_te)
    input_dim = X_tr_pad.shape[2]

    # CPU/GPU device routing. MPS (Apple Silicon) was tried and reverted: for
    # this workload (many short, variable-length per-case sequences run
    # through pack_padded_sequence) MPS's per-op dispatch overhead made it
    # slower than CPU on both predictcollab and, far more severely, on the
    # larger BPIC2013 log -- see run_evaluation.py docstring's reproducibility
    # note for why device choice must stay fixed across a reported run.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    units = getattr(cfg, "lstm_units", 64)
    epochs = getattr(cfg, "lstm_epochs", 20)
    batch_size = getattr(cfg, "lstm_batch_size", 32)
    lr = getattr(cfg, "lstm_learning_rate", 0.001)

    X_tr_tens = torch.tensor(X_tr_pad, dtype=torch.float32).to(device)
    lengths_tr_tens = torch.tensor(lengths_tr, dtype=torch.int64) # Keep on CPU for pack_padded_sequence
    
    X_te_tens = torch.tensor(X_te_pad, dtype=torch.float32).to(device)
    lengths_te_tens = torch.tensor(lengths_te, dtype=torch.int64)

    if task.kind in ("categorical", "binary"):
        y_tr_s, y_te_s = y_tr_used, y_te_used
        le = LabelEncoder()
        le.fit(y_tr_s)
        num_classes = len(le.classes_)

        if num_classes < 2:
            # Degenerate fold: the training target is constant, so there is
            # nothing to learn; predicting that constant matches the trivial
            # baseline and avoids feeding a single-logit head a target with
            # no negative class, which can otherwise emit a class index the
            # encoder never saw (ValueError on inverse_transform below).
            const_label = le.classes_[0]
            pred_labels = [const_label] * len(y_te_s)
            score = float(f1_score(y_te_s, pred_labels, average="macro", zero_division=0))
            return {"metric": score, "baseline": score, "n_test": int(len(y_te_s))}

        y_tr_enc, _ = pad_sequences([le.transform(seq) for seq in seq_y_tr], padding_value=-1)
        y_tr_enc = y_tr_enc.squeeze(-1)
        
        mask_tr = y_tr_enc != -1
        y_tr_tens = torch.tensor(np.where(mask_tr, y_tr_enc, 0), dtype=torch.long if num_classes > 2 else torch.float32).to(device)
        mask_tr_tens = torch.tensor(mask_tr, dtype=torch.bool).to(device)
        
        output_dim = num_classes if num_classes > 2 else 1
        model = LSTMModel(input_dim, units, output_dim, is_classification=True).to(device)
        
        criterion = nn.CrossEntropyLoss(reduction='none') if num_classes > 2 else nn.BCEWithLogitsLoss(reduction='none')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        dataset = TensorDataset(X_tr_tens, y_tr_tens, lengths_tr_tens, mask_tr_tens)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        with timer.stage("fit"):
            model.train()
            for epoch in range(epochs):
                for bx, by, bl, bmask in loader:
                    # Match by/bmask's time axis to what forward() now
                    # returns (this batch's own max length, see forward()).
                    m = int(bl.max())
                    by, bmask = by[:, :m], bmask[:, :m]
                    optimizer.zero_grad()
                    out = model(bx, bl)
                    if num_classes > 2:
                        loss = criterion(out.view(-1, num_classes), by.view(-1))
                    else:
                        loss = criterion(out.view(-1), by.view(-1))

                    loss = (loss * bmask.view(-1)).sum() / bmask.sum().clamp(min=1)
                    loss.backward()
                    optimizer.step()

        with timer.stage("predict"):
            model.eval()
            with torch.no_grad():
                out_te = model(X_te_tens, lengths_te_tens)
                if num_classes > 2:
                    probs = torch.softmax(out_te, dim=-1).cpu().numpy()
                    pred_enc_all = np.argmax(probs, axis=-1)
                else:
                    probs = torch.sigmoid(out_te).cpu().numpy()
                    pred_enc_all = (probs > 0.5).astype(int).squeeze(-1)
                
        flat_pred = []
        for i, l in enumerate(lengths_te):
            flat_pred.extend(pred_enc_all[i, :l])
            
        pred_labels = le.inverse_transform(flat_pred)
        
        maj = pd.Series(y_tr_s).mode().iloc[0]
        return {
            "metric": float(f1_score(y_te_s, pred_labels, average="macro", zero_division=0)),
            "baseline": float(f1_score(y_te_s, [maj] * len(y_te_s), average="macro", zero_division=0)),
            "n_test": int(len(y_te_s)),
        }
    else:
        y_tr_pad, _ = pad_sequences([seq.astype(float) for seq in seq_y_tr], padding_value=-9999.0)
        y_tr_pad = y_tr_pad.squeeze(-1)
        
        mask_tr = y_tr_pad != -9999.0
        y_tr_tens = torch.tensor(np.where(mask_tr, y_tr_pad, 0.0), dtype=torch.float32).to(device)
        mask_tr_tens = torch.tensor(mask_tr, dtype=torch.bool).to(device)
        
        model = LSTMModel(input_dim, units, 1, is_classification=False).to(device)
        criterion = nn.MSELoss(reduction='none')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        dataset = TensorDataset(X_tr_tens, y_tr_tens, lengths_tr_tens, mask_tr_tens)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        with timer.stage("fit"):
            model.train()
            for epoch in range(epochs):
                for bx, by, bl, bmask in loader:
                    m = int(bl.max())
                    by, bmask = by[:, :m], bmask[:, :m]
                    optimizer.zero_grad()
                    out = model(bx, bl).squeeze(-1)
                    loss = criterion(out, by)
                    loss = (loss * bmask).sum() / bmask.sum().clamp(min=1)
                    loss.backward()
                    optimizer.step()

        with timer.stage("predict"):
            model.eval()
            with torch.no_grad():
                out_te = model(X_te_tens, lengths_te_tens).squeeze(-1).cpu().numpy()
            
        flat_pred = []
        for i, l in enumerate(lengths_te):
            flat_pred.extend(out_te[i, :l])
        flat_pred = np.array(flat_pred)
        flat_pred = y_scaler.inverse_transform(flat_pred.reshape(-1, 1)).flatten()
            
        y_te_float = y_te.astype(float).to_numpy()
        median = float(np.median(y_tr.astype(float).to_numpy()))

        return {
            "metric": float(mean_absolute_error(y_te_float, flat_pred)),
            "baseline": float(mean_absolute_error(y_te_float, [median] * len(y_te_float))),
            "n_test": int(len(y_te_float)),
        }
