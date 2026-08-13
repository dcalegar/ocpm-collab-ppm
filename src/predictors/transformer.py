"""
Per-fold training and scoring using a PyTorch Transformer predictor.
Supports both classification (macro F1) and regression (MAE) tasks.
"""
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder

import torch
import torch.nn as nn
import math

from tasks.catalog import Task
from .common import NullStageTimer, xy_split


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, is_classification, num_heads=4, num_layers=2):
        super(TransformerModel, self).__init__()
        self.is_classification = is_classification
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            batch_first=True,
            dropout=0.2
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self._causal_mask_cache = {}

    def _causal_mask(self, seq_len, device):
        key = (seq_len, device)
        mask = self._causal_mask_cache.get(key)
        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
            self._causal_mask_cache[key] = mask
        return mask

    def forward(self, x, src_key_padding_mask):
        # x is (batch, seq, input_dim)
        x = self.input_projection(x)
        x = self.pos_encoder(x)

        seq_len = x.size(1)
        causal_mask = self._causal_mask(seq_len, x.device)

        # transformer output is (batch, seq, hidden_dim)
        out = self.transformer_encoder(x, mask=causal_mask, src_key_padding_mask=src_key_padding_mask)

        out = self.dropout(out)
        out = self.fc(out)
        return out


def _pad_X(X_list: List[np.ndarray]):
    """Pad a batch of variable-length case sequences to THIS batch's own max
    length. Padding every batch to a dataset-wide max (the previous approach)
    forces every batch's self-attention -- O(seq_len^2) per layer, unlike the
    LSTM predictor's packed-sequence O(seq_len) -- to pay the cost of the
    single longest case in the whole train/test set on every batch of every
    epoch. On logs with a long-tailed case-length distribution (e.g. BPI2013:
    median case length 7, max 135) that turns a handful of outlier cases into
    a blanket ~(135/7)^2 ~= 370x attention-cost multiplier applied uniformly,
    which is what made the transformer predictor hang on BPI2013 while the
    LSTM predictor finished the same log in ~26 minutes."""
    lengths = [len(x) for x in X_list]
    max_len = max(lengths)
    dim = X_list[0].shape[-1]
    bx = np.zeros((len(X_list), max_len, dim), dtype=np.float32)
    for i, x in enumerate(X_list):
        bx[i, :lengths[i], :] = x
    return bx, np.asarray(lengths, dtype=np.int64), max_len


def _pad_y(y_list: List[np.ndarray], lengths, max_len, pad_value):
    by = np.full((len(y_list), max_len), pad_value, dtype=np.float32)
    for i, y in enumerate(y_list):
        by[i, :lengths[i]] = y
    return by


def _length_bucketed_batches(lengths: List[int], batch_size: int):
    """Group cases of similar length into the same batch (sort by length,
    chunk, then shuffle chunk ORDER) so a batch's padding target tracks its
    own members instead of a rare long outlier. Batch composition is
    deterministic given `lengths`; only the order batches are visited in
    varies epoch to epoch, via the module-level np.random seeding already
    done by the caller."""
    order = np.argsort(lengths, kind="stable")
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    np.random.shuffle(batches)
    return batches


def _create_padding_mask(lengths, max_len):
    mask = torch.ones((len(lengths), max_len), dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask[i, :l] = False
    return mask


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

    lengths_tr = [len(s) for s in seq_X_tr]
    lengths_te = [len(s) for s in seq_X_te]
    input_dim = seq_X_tr[0].shape[-1]

    # CPU/GPU Device routing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    units = getattr(cfg, "transformer_units", getattr(cfg, "lstm_units", 64))
    epochs = getattr(cfg, "transformer_epochs", getattr(cfg, "lstm_epochs", 20))
    batch_size = getattr(cfg, "transformer_batch_size", getattr(cfg, "lstm_batch_size", 32))
    lr = getattr(cfg, "transformer_learning_rate", getattr(cfg, "lstm_learning_rate", 0.001))
    num_heads = getattr(cfg, "transformer_heads", 4)
    num_layers = getattr(cfg, "transformer_layers", 2)

    # Check if units is divisible by num_heads (d_model must be divisible by nhead)
    if units % num_heads != 0:
        units = (units // num_heads) * num_heads
        if units == 0: units = num_heads

    def predict_in_batches(model, seq_X, lengths, eval_batch_size):
        # Sequential (order-preserving), locally-padded batches: avoids
        # padding the whole test set to its single longest case, and keeps
        # output rows aligned with `lengths` for the flat_pred reconstruction
        # below (no sort/unsort needed since order is never disturbed).
        outputs = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(seq_X), eval_batch_size):
                chunk_X = seq_X[i:i + eval_batch_size]
                bx, blen, _ = _pad_X(chunk_X)
                bx_tens = torch.tensor(bx, dtype=torch.float32).to(device)
                pad_mask = _create_padding_mask(blen, bx_tens.size(1)).to(device)
                out = model(bx_tens, pad_mask)
                outputs.append(out.cpu())
        return outputs

    if task.kind in ("categorical", "binary"):
        y_tr_s, y_te_s = y_tr_used, y_te_used
        le = LabelEncoder()
        le.fit(y_tr_s)
        num_classes = len(le.classes_)

        if num_classes < 2:
            const_label = le.classes_[0]
            pred_labels = [const_label] * len(y_te_s)
            score = float(f1_score(y_te_s, pred_labels, average="macro", zero_division=0))
            return {"metric": score, "baseline": score, "n_test": int(len(y_te_s))}

        y_enc_tr = [le.transform(seq).astype(np.float32) for seq in seq_y_tr]

        output_dim = num_classes if num_classes > 2 else 1
        model = TransformerModel(input_dim, units, output_dim, is_classification=True, num_heads=num_heads, num_layers=num_layers).to(device)

        criterion = nn.CrossEntropyLoss(reduction='none') if num_classes > 2 else nn.BCEWithLogitsLoss(reduction='none')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        with timer.stage("fit"):
            model.train()
            for epoch in range(epochs):
                for idx in _length_bucketed_batches(lengths_tr, batch_size):
                    bx, blen, max_len = _pad_X([seq_X_tr[i] for i in idx])
                    by_pad = _pad_y([y_enc_tr[i] for i in idx], blen, max_len, pad_value=-1.0)
                    bmask = by_pad != -1.0
                    by_used = np.where(bmask, by_pad, 0)

                    bx_tens = torch.tensor(bx, dtype=torch.float32).to(device)
                    by_tens = torch.tensor(by_used, dtype=torch.long if num_classes > 2 else torch.float32).to(device)
                    bmask_tens = torch.tensor(bmask, dtype=torch.bool).to(device)

                    optimizer.zero_grad()
                    src_key_padding_mask = _create_padding_mask(blen, max_len).to(device)
                    out = model(bx_tens, src_key_padding_mask)
                    if num_classes > 2:
                        loss = criterion(out.reshape(-1, num_classes), by_tens.reshape(-1))
                    else:
                        loss = criterion(out.reshape(-1), by_tens.reshape(-1))

                    loss = (loss * bmask_tens.reshape(-1)).sum() / bmask_tens.sum().clamp(min=1)
                    loss.backward()
                    optimizer.step()

        with timer.stage("predict"):
            outputs = predict_in_batches(model, seq_X_te, lengths_te, batch_size)
            if num_classes > 2:
                pred_chunks = [np.argmax(torch.softmax(o, dim=-1).numpy(), axis=-1) for o in outputs]
            else:
                pred_chunks = [(torch.sigmoid(o).numpy() > 0.5).astype(int).squeeze(-1) for o in outputs]

        flat_pred = []
        pos = 0
        for chunk in pred_chunks:
            for row in chunk:
                flat_pred.extend(row[:lengths_te[pos]])
                pos += 1

        pred_labels = le.inverse_transform(flat_pred)

        maj = pd.Series(y_tr_s).mode().iloc[0]
        return {
            "metric": float(f1_score(y_te_s, pred_labels, average="macro", zero_division=0)),
            "baseline": float(f1_score(y_te_s, [maj] * len(y_te_s), average="macro", zero_division=0)),
            "n_test": int(len(y_te_s)),
        }
    else:
        y_tr_float = [seq.astype(np.float32) for seq in seq_y_tr]

        model = TransformerModel(input_dim, units, 1, is_classification=False, num_heads=num_heads, num_layers=num_layers).to(device)
        criterion = nn.MSELoss(reduction='none')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        with timer.stage("fit"):
            model.train()
            for epoch in range(epochs):
                for idx in _length_bucketed_batches(lengths_tr, batch_size):
                    bx, blen, max_len = _pad_X([seq_X_tr[i] for i in idx])
                    by_pad = _pad_y([y_tr_float[i] for i in idx], blen, max_len, pad_value=-9999.0)
                    bmask = by_pad != -9999.0
                    by_used = np.where(bmask, by_pad, 0.0)

                    bx_tens = torch.tensor(bx, dtype=torch.float32).to(device)
                    by_tens = torch.tensor(by_used, dtype=torch.float32).to(device)
                    bmask_tens = torch.tensor(bmask, dtype=torch.bool).to(device)

                    optimizer.zero_grad()
                    src_key_padding_mask = _create_padding_mask(blen, max_len).to(device)
                    out = model(bx_tens, src_key_padding_mask).squeeze(-1)
                    loss = criterion(out, by_tens)
                    loss = (loss * bmask_tens).sum() / bmask_tens.sum().clamp(min=1)
                    loss.backward()
                    optimizer.step()

        with timer.stage("predict"):
            outputs = predict_in_batches(model, seq_X_te, lengths_te, batch_size)

        flat_pred = []
        pos = 0
        for o in outputs:
            arr = o.squeeze(-1).numpy()
            for row in arr:
                flat_pred.extend(row[:lengths_te[pos]])
                pos += 1
        flat_pred = np.array(flat_pred)
        flat_pred = y_scaler.inverse_transform(flat_pred.reshape(-1, 1)).flatten()

        y_te_float = y_te.astype(float).to_numpy()
        median = float(np.median(y_tr.astype(float).to_numpy()))

        return {
            "metric": float(mean_absolute_error(y_te_float, flat_pred)),
            "baseline": float(mean_absolute_error(y_te_float, [median] * len(y_te_float))),
            "n_test": int(len(y_te_float)),
        }
