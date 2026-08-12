"""
Per-fold training and scoring using a PyTorch Transformer predictor.
Supports both classification (macro F1) and regression (MAE) tasks.
"""
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder

import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader, TensorDataset

from ocpm_tasks.catalog import Task
from .common import xy_split


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

    def forward(self, x, src_key_padding_mask):
        # x is (batch, seq, input_dim)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        
        seq_len = x.size(1)
        # Create a boolean causal mask to match src_key_padding_mask type
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        
        # transformer output is (batch, seq, hidden_dim)
        out = self.transformer_encoder(x, mask=causal_mask, src_key_padding_mask=src_key_padding_mask)
        
        out = self.dropout(out)
        out = self.fc(out)
        return out


def fit_and_score_fold(feats: dict, tt: pd.DataFrame, y_col: str,
                       task: Task, train_mask, test_mask, cfg) -> Dict[str, float]:
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

    X_tr_tens = torch.tensor(X_tr_pad, dtype=torch.float32).to(device)
    lengths_tr_tens = torch.tensor(lengths_tr, dtype=torch.int64)
    
    X_te_tens = torch.tensor(X_te_pad, dtype=torch.float32).to(device)
    lengths_te_tens = torch.tensor(lengths_te, dtype=torch.int64)

    def create_padding_mask(lengths_tensor, max_len):
        mask = torch.ones((len(lengths_tensor), max_len), dtype=torch.bool)
        for i, l in enumerate(lengths_tensor):
            mask[i, :l] = False
        return mask

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

        y_tr_enc, _ = pad_sequences([le.transform(seq) for seq in seq_y_tr], padding_value=-1)
        y_tr_enc = y_tr_enc.squeeze(-1)
        
        mask_tr = y_tr_enc != -1
        y_tr_tens = torch.tensor(np.where(mask_tr, y_tr_enc, 0), dtype=torch.long if num_classes > 2 else torch.float32).to(device)
        mask_tr_tens = torch.tensor(mask_tr, dtype=torch.bool).to(device)
        
        output_dim = num_classes if num_classes > 2 else 1
        model = TransformerModel(input_dim, units, output_dim, is_classification=True, num_heads=num_heads, num_layers=num_layers).to(device)
        
        criterion = nn.CrossEntropyLoss(reduction='none') if num_classes > 2 else nn.BCEWithLogitsLoss(reduction='none')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        dataset = TensorDataset(X_tr_tens, y_tr_tens, lengths_tr_tens, mask_tr_tens)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        model.train()
        for epoch in range(epochs):
            for bx, by, bl, bmask in loader:
                optimizer.zero_grad()
                src_key_padding_mask = create_padding_mask(bl, bx.size(1)).to(device)
                out = model(bx, src_key_padding_mask)
                if num_classes > 2:
                    loss = criterion(out.view(-1, num_classes), by.view(-1))
                else:
                    loss = criterion(out.view(-1), by.view(-1))
                
                loss = (loss * bmask.view(-1)).sum() / bmask.sum().clamp(min=1)
                loss.backward()
                optimizer.step()
                
        model.eval()
        with torch.no_grad():
            src_key_padding_mask_te = create_padding_mask(lengths_te_tens, X_te_tens.size(1)).to(device)
            out_te = model(X_te_tens, src_key_padding_mask_te)
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
        
        model = TransformerModel(input_dim, units, 1, is_classification=False, num_heads=num_heads, num_layers=num_layers).to(device)
        criterion = nn.MSELoss(reduction='none')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        dataset = TensorDataset(X_tr_tens, y_tr_tens, lengths_tr_tens, mask_tr_tens)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        model.train()
        for epoch in range(epochs):
            for bx, by, bl, bmask in loader:
                optimizer.zero_grad()
                src_key_padding_mask = create_padding_mask(bl, bx.size(1)).to(device)
                out = model(bx, src_key_padding_mask).squeeze(-1)
                loss = criterion(out, by)
                loss = (loss * bmask).sum() / bmask.sum().clamp(min=1)
                loss.backward()
                optimizer.step()
                
        model.eval()
        with torch.no_grad():
            src_key_padding_mask_te = create_padding_mask(lengths_te_tens, X_te_tens.size(1)).to(device)
            out_te = model(X_te_tens, src_key_padding_mask_te).squeeze(-1).cpu().numpy()
            
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
