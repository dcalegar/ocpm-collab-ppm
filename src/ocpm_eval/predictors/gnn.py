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
    target_index: int
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
    """Extract variable-size induced graphs from structural predecessors.

    ``k`` includes the cut event, as in prefix-based predictive monitoring.
    The other nodes are its closest ancestors in the OCPA event graph. Initial
    prefixes shorter than k are retained at their natural size.
    """
    if k < 1:
        raise ValueError("k must be positive")
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
        chosen.sort(key=order.get)
        local = {event_id: i for i, event_id in enumerate(chosen)}
        edges = tuple((local[s], local[t]) for s, t in raw_edges
                      if s in local and t in local)
        edges += tuple((i, i) for i in range(len(chosen)))
        samples.append(EventGraphSample(
            target_id, tuple(chosen), edges, local[target_id],
            full.loc[chosen, cols].to_numpy(dtype=np.float32),
            labelled.loc[target_id, y_col]))
    return samples


def fit_and_score_fold(feats, tt, y_col, task, train_mask, test_mask, cfg):
    """Select the configured k on validation executions, then evaluate test once."""
    import dgl
    import torch
    import torch.nn as nn
    from dgl.nn import GlobalAttentionPooling, GraphConv
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
    # Include validation-only classes for the final development refit, without
    # using any label information from the test fold.
    encoder = (LabelEncoder().fit(development[y_col].astype(str))
               if classification else None)
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
            self.pool = GlobalAttentionPooling(nn.Linear(hidden, 1))
            self.head = nn.Sequential(
                nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(hidden, output_dim))

        def forward(self, graph, target_indices):
            h = torch.relu(self.conv1(graph, graph.ndata["x"]))
            h = torch.relu(self.conv2(graph, h))
            counts = graph.batch_num_nodes().detach().cpu().tolist()
            offsets = np.cumsum([0] + counts[:-1])
            cut_nodes = torch.stack([
                h[int(offset + target)]
                for offset, target in zip(offsets, target_indices.tolist())
            ])
            graph_context = self.pool(graph, h)
            return self.head(torch.cat([cut_nodes, graph_context], dim=1))

    def prepare(frame, k):
        ids = set(frame["event_id"].astype(str))
        source = scaled_tt[scaled_tt["event_id"].astype(str).isin(ids)]
        samples = extract_k_prefix_graphs(scaled_feats, source, y_col, k)
        pairs = []
        # Only test labels can be outside the development-fitted vocabulary.
        # Keep them as an extra true-label index so they count as errors without
        # leaking test classes into the model output space.
        values = ([class_index.get(str(s.y), len(class_index)) for s in samples]
                  if classification
                  else [float(s.y) for s in samples])
        for sample, value in zip(samples, values):
            src, dst = zip(*sample.edges)
            graph = dgl.graph((src, dst), num_nodes=len(sample.node_ids))
            graph.ndata["x"] = torch.as_tensor(sample.x, dtype=torch.float32)
            dtype = torch.long if classification else torch.float32
            pairs.append((graph, torch.as_tensor(value, dtype=dtype),
                          sample.target_index))
        return pairs, samples

    def collate(batch):
        graphs, labels, target_indices = zip(*batch)
        return (dgl.batch(graphs), torch.stack(labels),
                torch.as_tensor(target_indices, dtype=torch.long))

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
        if classification:
            training_labels = np.asarray([
                int(label.item()) for _, label, _ in training])
            counts = np.bincount(training_labels,
                                 minlength=len(encoder.classes_))
            weights = np.zeros(len(counts), dtype=np.float32)
            present = counts > 0
            weights[present] = (len(training_labels) /
                                (present.sum() * counts[present]))
            loss_fn = nn.CrossEntropyLoss(
                weight=torch.as_tensor(weights, device=device))
        else:
            loss_fn = nn.HuberLoss(
                delta=getattr(cfg, "gnn_huber_delta", 1.0))
        loader = DataLoader(training, batch_size=getattr(cfg, "gnn_batch_size", 32),
                            shuffle=True, collate_fn=collate)
        epochs = getattr(cfg, "gnn_epochs", 100)
        patience = max(0, getattr(cfg, "gnn_early_stopping_patience", 10))
        min_delta = max(0.0, getattr(cfg, "gnn_early_stopping_min_delta", 0.0001))
        validation_graph, validation_labels, validation_targets = collate(validating)
        validation_graph = validation_graph.to(device)
        validation_labels = validation_labels.to(device)
        validation_targets = validation_targets.to(device)
        best_objective = float("inf")
        best_state = None
        best_epoch = 0
        epochs_without_improvement = 0
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            batches = 0
            for graph, labels, targets in loader:
                graph = graph.to(device)
                labels = labels.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                output = model(graph, targets)
                loss = (loss_fn(output, labels) if classification else
                        loss_fn(output.squeeze(-1), labels))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                total_loss += float(loss.detach().cpu())
                batches += 1
            model.eval()
            with torch.no_grad():
                validation_prediction = model(validation_graph, validation_targets)
            validation_actual = validation_labels.detach().cpu().numpy()
            validation_predicted = validation_prediction.detach().cpu().numpy()
            if classification:
                validation_metric = f1_score(
                    validation_actual, validation_predicted.argmax(1),
                    average="macro", zero_division=0)
                objective = -validation_metric
                metric_name = "f1_macro"
            else:
                validation_metric = mean_absolute_error(
                    validation_actual, validation_predicted.squeeze(-1))
                objective = validation_metric
                metric_name = "mae"

            if objective < best_objective - min_delta:
                best_objective = objective
                best_epoch = epoch
                epochs_without_improvement = 0
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            else:
                epochs_without_improvement += 1
            if verbose and (epoch == 1 or epoch % log_every == 0 or epoch == epochs):
                print(f"      epoch {epoch}/{epochs} "
                      f"loss={total_loss / max(1, batches):.6f} "
                      f"validation_{metric_name}={validation_metric:.6f}")
            if patience and epochs_without_improvement >= patience:
                if verbose:
                    print(f"      early stopping at epoch {epoch}; "
                          f"best_epoch={best_epoch}")
                break

        if best_state is None:
            return None
        if verbose:
            shown = -best_objective if classification else best_objective
            print(f"    {context} k={k}: best_epoch={best_epoch} "
                  f"validation_{metric_name}={shown:.6f}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return best_objective, best_epoch

    candidates = []
    for k in getattr(cfg, "gnn_k_values", tuple(range(2, 9))):
        result = train_for(k)
        if result is not None:
            candidates.append((result[0], k, result[1]))
    if not candidates:
        return {}
    _, best_k, best_epoch = min(candidates, key=lambda candidate: candidate[0])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = GCN(len(encoder.classes_) if classification else 1).to(device)
    development_data, _ = prepare(development, best_k)
    if not development_data:
        return {}
    if classification:
        development_labels = np.asarray([
            int(label.item()) for _, label, _ in development_data])
        counts = np.bincount(development_labels,
                             minlength=len(encoder.classes_))
        weights = np.zeros(len(counts), dtype=np.float32)
        present = counts > 0
        weights[present] = (len(development_labels) /
                            (present.sum() * counts[present]))
        development_loss = nn.CrossEntropyLoss(
            weight=torch.as_tensor(weights, device=device))
    else:
        development_loss = nn.HuberLoss(
            delta=getattr(cfg, "gnn_huber_delta", 1.0))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=getattr(cfg, "gnn_learning_rate", 0.001))
    loader = DataLoader(
        development_data, batch_size=getattr(cfg, "gnn_batch_size", 32),
        shuffle=True, collate_fn=collate)
    for _ in range(best_epoch):
        model.train()
        for graph, labels, targets in loader:
            graph = graph.to(device)
            labels = labels.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            output = model(graph, targets)
            loss = (development_loss(output, labels) if classification else
                    development_loss(output.squeeze(-1), labels))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
    if verbose:
        print(f"    {context} selected_k={best_k} refit_epochs={best_epoch}")
    testing, samples = prepare(test, best_k)
    if not testing:
        return {}
    model.eval()
    with torch.no_grad():
        graph, labels, targets = collate(testing)
        graph = graph.to(device)
        labels = labels.to(device)
        targets = targets.to(device)
        prediction = model(graph, targets)
    if classification:
        actual = labels.detach().cpu().numpy()
        predicted = prediction.argmax(1).detach().cpu().numpy()
        majority = encoder.transform([
            development[y_col].astype(str).mode().iloc[0]])[0]
        metric = f1_score(actual, predicted, average="macro", zero_division=0)
        baseline = f1_score(actual, np.full(len(actual), majority),
                            average="macro", zero_division=0)
    else:
        actual = labels.detach().cpu().numpy()
        predicted = prediction.squeeze(-1).detach().cpu().numpy()
        median = float(development[y_col].astype(float).median())
        metric = mean_absolute_error(actual, predicted)
        baseline = mean_absolute_error(actual, np.full(len(actual), median))
    if verbose:
        metric_name = "f1_macro" if classification else "mae"
        print(f"    {context} test_{metric_name}={metric:.6f} "
              f"baseline={baseline:.6f} n={len(samples)}")
    return {"metric": float(metric), "baseline": float(baseline),
            "n_test": len(samples), "best_k": int(best_k)}
