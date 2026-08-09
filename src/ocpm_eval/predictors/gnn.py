"""Direct object-centric event-graph predictor using OCPA graphs and DGL."""
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass(frozen=True)
class EventGraphSample:
    event_id: str
    node_ids: Tuple[str, ...]
    edges: Tuple[Tuple[int, int], ...]
    x: np.ndarray
    y: object


def execution_train_validation_split(case_ids: Sequence[str], seed: int):
    """Split the outer 80% development fold into 70% train / 10% validation."""
    ids = sorted(set(map(str, case_ids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) / 8))) if len(ids) > 1 else 0
    return set(ids[n_val:]), set(ids[:n_val])


def _graph_index(storage) -> Dict[str, Tuple[object, Dict[str, int]]]:
    result = {}
    for graph in storage.feature_graphs:
        order = {str(node.event_id): i for i, node in enumerate(graph.nodes)}
        result.update({event_id: (graph, order) for event_id in order})
    return result


def extract_k_prefix_graphs(feats: dict, rows: pd.DataFrame, y_col: str,
                            k: int) -> List[EventGraphSample]:
    """Extract fixed-size induced graphs from structural predecessors.

    ``k`` includes the cut event, as in prefix-based predictive monitoring.
    The other nodes are its closest ancestors in the OCPA event graph. Initial
    prefixes shorter than k are skipped rather than padded with invented events.
    """
    if not 2 <= k <= 8:
        raise ValueError("k must be between 2 and 8")
    if feats.get("feature_storage") is None:
        raise ValueError("GNN requires OCPA feature_storage")
    labelled = rows.copy()
    labelled.index = labelled["event_id"].astype(str)
    full = feats["table"].copy()
    full.index = full["event_id"].astype(str)
    cols, graph_by_event = feats["feature_cols"], _graph_index(feats["feature_storage"])
    samples = []
    for target_id in labelled.index:
        if target_id not in graph_by_event:
            continue
        graph, order = graph_by_event[target_id]
        predecessors, raw_edges = {event_id: [] for event_id in order}, []
        for edge in graph.edges:
            source, target = str(edge.source), str(edge.target)
            if source in order and target in order:
                predecessors[target].append(source)
                raw_edges.append((source, target))
        distances, queue = {target_id: 0}, [target_id]
        while queue:
            current = queue.pop(0)
            for parent in predecessors.get(current, ()):
                if parent not in distances:
                    distances[parent] = distances[current] + 1
                    queue.append(parent)
        chosen = sorted(distances, key=lambda e: (distances[e], -order[e]))[:k]
        if len(chosen) < k:
            continue
        chosen.sort(key=order.get)
        local = {event_id: i for i, event_id in enumerate(chosen)}
        edges = tuple((local[s], local[t]) for s, t in raw_edges
                      if s in local and t in local)
        edges += tuple((i, i) for i in range(k))
        samples.append(EventGraphSample(
            target_id, tuple(chosen), edges,
            full.loc[chosen, cols].to_numpy(dtype=np.float32),
            labelled.loc[target_id, y_col]))
    return samples


def fit_and_score_fold(feats, tt, y_col, task, train_mask, test_mask, cfg):
    """Select k=2..8 on whole validation executions, then evaluate test once."""
    import dgl
    import torch
    import torch.nn as nn
    from dgl.nn import GraphConv
    from torch.utils.data import DataLoader

    seed = getattr(cfg, "random_state", 3395)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    requested_device = getattr(cfg, "gnn_device", "auto").lower()
    if requested_device not in ("auto", "cpu", "cuda"):
        raise ValueError("gnn_device must be 'auto', 'cpu', or 'cuda'")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("gnn_device='cuda', but CUDA is not available")
    use_cuda = requested_device == "cuda" or (
        requested_device == "auto" and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    if device.type == "cuda":
        try:
            dgl.graph(([0], [0]), num_nodes=1).to(device)
        except Exception as exc:
            if requested_device == "cuda":
                raise RuntimeError(
                    "CUDA is available in PyTorch, but this DGL build cannot use it") from exc
            device = torch.device("cpu")
    development, test = tt.loc[train_mask], tt.loc[test_mask]
    train_cases, val_cases = execution_train_validation_split(
        development["case_id"].astype(str), seed)
    train = development[development["case_id"].astype(str).isin(train_cases)]
    validation = development[development["case_id"].astype(str).isin(val_cases)]
    if train.empty or validation.empty or test.empty:
        return {}

    cols = feats["feature_cols"]
    scaler = StandardScaler().fit(train[cols].fillna(0.0))
    scaled_feats = dict(feats)
    scaled_feats["table"] = feats["table"].copy()
    scaled_feats["table"][cols] = scaled_feats["table"][cols].astype(float)
    scaled_feats["table"].loc[:, cols] = scaler.transform(
        feats["table"][cols].fillna(0.0))
    scaled_tt = tt.copy()
    scaled_tt[cols] = scaled_tt[cols].astype(float)
    scaled_tt.loc[:, cols] = scaler.transform(tt[cols].fillna(0.0))
    classification = task.kind in ("categorical", "binary")
    encoder = LabelEncoder().fit(train[y_col].astype(str)) if classification else None
    class_index = ({label: i for i, label in enumerate(encoder.classes_)}
                   if classification else {})
    verbose = getattr(cfg, "gnn_verbose", True)
    log_every = max(1, getattr(cfg, "gnn_log_every", 5))
    context = f"[GNN][{feats.get('log_name', 'unknown')}][{task.key}]"
    if verbose:
        print(f"    {context} device={device}")

    class GCN(nn.Module):
        def __init__(self, output_dim):
            super().__init__()
            hidden = getattr(cfg, "gnn_hidden_dim", 64)
            self.conv1 = GraphConv(len(cols), hidden, allow_zero_in_degree=True)
            self.conv2 = GraphConv(hidden, hidden, allow_zero_in_degree=True)
            self.head = nn.Sequential(nn.ReLU(), nn.Dropout(0.2),
                                      nn.Linear(hidden, output_dim))

        def forward(self, graph):
            h = torch.relu(self.conv1(graph, graph.ndata["x"]))
            h = torch.relu(self.conv2(graph, h))
            counts = graph.batch_num_nodes().detach().cpu().tolist()
            offsets = np.cumsum([0] + counts[:-1])
            cut_nodes = torch.stack([h[int(o + n - 1)]
                                     for o, n in zip(offsets, counts)])
            return self.head(cut_nodes)

    def prepare(frame, k):
        ids = set(frame["event_id"].astype(str))
        source = scaled_tt[scaled_tt["event_id"].astype(str).isin(ids)]
        samples = extract_k_prefix_graphs(scaled_feats, source, y_col, k)
        pairs = []
        # An unseen validation/test class remains a distinct true label; the
        # train-fitted head cannot predict it, which is the honest error rather
        # than leaking the held-out vocabulary into model construction.
        values = ([class_index.get(str(s.y), len(class_index)) for s in samples]
                  if classification
                  else [float(s.y) for s in samples])
        for sample, value in zip(samples, values):
            src, dst = zip(*sample.edges)
            graph = dgl.graph((src, dst), num_nodes=len(sample.node_ids))
            graph.ndata["x"] = torch.as_tensor(sample.x, dtype=torch.float32)
            dtype = torch.long if classification else torch.float32
            pairs.append((graph, torch.as_tensor(value, dtype=dtype)))
        return pairs, samples

    def collate(batch):
        graphs, labels = zip(*batch)
        return dgl.batch(graphs), torch.stack(labels)

    def train_for(k):
        training, _ = prepare(train, k)
        validating, _ = prepare(validation, k)
        if not training or not validating:
            if verbose:
                print(f"    {context} k={k}: skipped (insufficient subgraphs)")
            return None
        if verbose:
            print(f"    {context} k={k}: train={len(training)}, "
                  f"validation={len(validating)}")
        model = GCN(len(encoder.classes_) if classification else 1).to(device)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=getattr(cfg, "gnn_learning_rate", 0.001))
        loss_fn = nn.CrossEntropyLoss() if classification else nn.MSELoss()
        loader = DataLoader(training, batch_size=getattr(cfg, "gnn_batch_size", 32),
                            shuffle=True, collate_fn=collate)
        epochs = getattr(cfg, "gnn_epochs", 20)
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            batches = 0
            for graph, labels in loader:
                graph = graph.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                output = model(graph)
                loss = (loss_fn(output, labels) if classification else
                        loss_fn(output.squeeze(-1), labels))
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu())
                batches += 1
            if verbose and (epoch == 1 or epoch % log_every == 0 or epoch == epochs):
                print(f"      epoch {epoch}/{epochs} "
                      f"loss={total_loss / max(1, batches):.6f}")
        model.eval()
        with torch.no_grad():
            graph, labels = collate(validating)
            graph = graph.to(device)
            labels = labels.to(device)
            prediction = model(graph)
        actual = labels.detach().cpu().numpy()
        predicted = prediction.detach().cpu().numpy()
        objective = (-f1_score(actual, predicted.argmax(1),
                               average="macro", zero_division=0)
                     if classification else
                     mean_absolute_error(actual, predicted.squeeze(-1)))
        if verbose:
            metric_name = "f1_macro" if classification else "mae"
            shown = -objective if classification else objective
            print(f"    {context} k={k}: validation_{metric_name}={shown:.6f}")
        state = {name: value.detach().cpu().clone()
                 for name, value in model.state_dict().items()}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return objective, state

    candidates = []
    for k in getattr(cfg, "gnn_k_values", tuple(range(2, 9))):
        result = train_for(k)
        if result is not None:
            candidates.append((result[0], k, result[1]))
    if not candidates:
        return {}
    _, best_k, best_state = min(candidates, key=lambda candidate: candidate[0])
    model = GCN(len(encoder.classes_) if classification else 1).to(device)
    model.load_state_dict(best_state)
    if verbose:
        print(f"    {context} selected_k={best_k}")
    testing, samples = prepare(test, best_k)
    if not testing:
        return {}
    model.eval()
    with torch.no_grad():
        graph, labels = collate(testing)
        graph = graph.to(device)
        labels = labels.to(device)
        prediction = model(graph)
    if classification:
        actual = labels.detach().cpu().numpy()
        predicted = prediction.argmax(1).detach().cpu().numpy()
        majority = encoder.transform([train[y_col].astype(str).mode().iloc[0]])[0]
        metric = f1_score(actual, predicted, average="macro", zero_division=0)
        baseline = f1_score(actual, np.full(len(actual), majority),
                            average="macro", zero_division=0)
    else:
        actual = labels.detach().cpu().numpy()
        predicted = prediction.squeeze(-1).detach().cpu().numpy()
        median = float(train[y_col].astype(float).median())
        metric = mean_absolute_error(actual, predicted)
        baseline = mean_absolute_error(actual, np.full(len(actual), median))
    if verbose:
        metric_name = "f1_macro" if classification else "mae"
        print(f"    {context} test_{metric_name}={metric:.6f} "
              f"baseline={baseline:.6f} n={len(samples)}")
    return {"metric": float(metric), "baseline": float(baseline),
            "n_test": len(samples), "best_k": int(best_k)}
