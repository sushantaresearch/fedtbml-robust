"""
gnn_ref.py — GNN reference system for the FedTBML benchmark.

A 2-layer Graph Convolutional Network (message passing over the entity-linkage
graph) as the benchmark's *reference system*, with logistic regression as the
floor. No PyTorch-Geometric needed — message passing is the normalised dense
adjacency product Â X W (fine at benchmark scale; flower_gnn.py is the PyG +
Flower production version).

The headline experiment is the ABLATION over relational_strength:
  - When laundering is individually anomalous (rs=0), LR and GNN both do well.
  - When laundering is purely relational — mild per-declaration anomaly hidden in
    honest noise, but concentrated in dense entity rings (rs=1) — LR collapses to
    near the prevalence floor while the GNN still detects it.
That gap is the evidence that the graph reference system is necessary, not
decorative — and it justifies the entity-graph framing of the paper.

A second demo federates the GNN (FedAvg over per-client GNN weights, with an
optional poisoned client) so the reference system plugs into the robustness story.

    pip install torch scikit-learn        (CPU torch is enough)
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
import synth_trade as st


def normalized_adj(edge_index: np.ndarray, n: int) -> torch.Tensor:
    """Symmetric-normalised adjacency with self-loops: D^-1/2 (A+I) D^-1/2."""
    A = torch.zeros((n, n), dtype=torch.float32)
    if edge_index.shape[1] > 0:
        A[torch.tensor(edge_index[0]), torch.tensor(edge_index[1])] = 1.0
    A.fill_diagonal_(1.0)
    deg = A.sum(1)
    dinv = deg.pow(-0.5)
    dinv[torch.isinf(dinv)] = 0.0
    return dinv.unsqueeze(1) * A * dinv.unsqueeze(0)


class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden)
        self.l2 = nn.Linear(hidden, 2)

    def forward(self, x, Ahat):
        h = F.relu(Ahat @ self.l1(x))
        h = F.dropout(h, p=0.3, training=self.training)
        return Ahat @ self.l2(h)


def _split_masks(y: np.ndarray, seed: int = 0, test_frac: float = 0.3):
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    cut = int(n * (1 - test_frac))
    tr = np.zeros(n, bool); te = np.zeros(n, bool)
    tr[idx[:cut]] = True; te[idx[cut:]] = True
    return tr, te


def train_gcn(X, edge_index, y, train_mask, test_mask,
              epochs: int = 200, class_weight: float = 10.0, seed: int = 0) -> float:
    torch.manual_seed(seed)
    Xt = torch.tensor(X)
    yt = torch.tensor(y)
    Ahat = normalized_adj(edge_index, X.shape[0])
    model = GCN(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    w = torch.tensor([1.0, class_weight])
    trm = torch.tensor(train_mask)
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        out = model(Xt, Ahat)
        F.cross_entropy(out[trm], yt[trm], weight=w).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        prob = F.softmax(model(Xt, Ahat), dim=1)[:, 1].numpy()
    return average_precision_score(y[test_mask], prob[test_mask])


def lr_baseline(X, y, train_mask, test_mask) -> float:
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X[train_mask], y[train_mask])
    p = clf.predict_proba(X[test_mask])[:, 1]
    return average_precision_score(y[test_mask], p)


def ablation(seed: int = 0, entity_overlap: float = 0.5):
    """LR floor vs GNN as laundering moves from individual to relational, at a
    realistic level of ring camouflage (entity_overlap)."""
    print(f"Reference-system ablation at entity_overlap={entity_overlap} "
          f"(realistic camouflage)\n")
    print(f"  {'relational_strength':>20} | {'LR PR-AUC':>10} | {'GNN PR-AUC':>10} | {'GNN lift':>9}")
    print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*10}-+-{'-'*9}")
    for rs in (0.0, 0.5, 1.0):
        df = st.generate_relational(relational_strength=rs, entity_overlap=entity_overlap, seed=seed)
        X, ei, y = st.build_graph(df)
        tr, te = _split_masks(y, seed=seed)
        lr = lr_baseline(X, y, tr, te)
        gnn = train_gcn(X, ei, y, tr, te, seed=seed)
        print(f"  {rs:>20.1f} | {lr:>10.3f} | {gnn:>10.3f} | {gnn - lr:>+9.3f}")


def overlap_sweep(seed: int = 0):
    """Realism cost: how ring camouflage (shared legitimate entities) erodes
    detection when laundering is purely relational (relational_strength=1.0)."""
    print("\nRealism cost: detection vs ring camouflage (relational_strength=1.0)\n")
    print(f"  {'entity_overlap':>14} | {'prevalence':>10} | {'LR PR-AUC':>10} | {'GNN PR-AUC':>10} | {'GNN lift':>9}")
    print(f"  {'-'*14}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*9}")
    for ov in (0.0, 0.5, 1.0):
        df = st.generate_relational(relational_strength=1.0, entity_overlap=ov, seed=seed)
        X, ei, y = st.build_graph(df)
        tr, te = _split_masks(y, seed=seed)
        lr = lr_baseline(X, y, tr, te)
        gnn = train_gcn(X, ei, y, tr, te, seed=seed)
        print(f"  {ov:>14.1f} | {y.mean():>10.4f} | {lr:>10.3f} | {gnn:>10.3f} | {gnn - lr:>+9.3f}")
    print("\n  Reading: as legitimate entities are shared into rings (overlap -> 1),")
    print("  the ring stops being a separable island and GNN detection falls toward")
    print("  a realistic ceiling — while still beating the tabular floor.")


# --------------------------------------------------------------------------- #
# Federated GNN (FedAvg over GNN weights) with an optional poisoned client      #
# --------------------------------------------------------------------------- #
def _get_flat(model):
    return torch.cat([p.data.flatten() for p in model.parameters()])


def _set_flat(model, vec):
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(vec[i:i + n].view_as(p)); i += n


def federated_gnn(relational_strength=1.0, rounds=10, local_epochs=3,
                  agg="fedavg", poison_client=None, boost=8.0, seed=0):
    """FedAvg over per-client GNN weights; each client trains on its own subgraph.
    poison_client: client id running a model-replacement attack (or None)."""
    torch.manual_seed(seed)
    df = st.generate_relational(relational_strength=relational_strength, seed=seed)
    clients = sorted(df["client_id"].unique())
    graphs, masks = {}, {}
    for c in clients:
        sub = df[df["client_id"] == c]
        X, ei, y = st.build_graph(sub)
        tr, te = _split_masks(y, seed=seed)
        graphs[c] = (torch.tensor(X), normalized_adj(ei, X.shape[0]), torch.tensor(y), X, y)
        masks[c] = (tr, te)

    in_dim = next(iter(graphs.values()))[0].shape[1]
    global_model = GCN(in_dim)
    w = torch.tensor([1.0, 10.0])

    for r in range(1, rounds + 1):
        updates = []
        for c in clients:
            Xt, Ahat, yt, _, _ = graphs[c]
            trm = torch.tensor(masks[c][0])
            m = GCN(in_dim); _set_flat(m, _get_flat(global_model))
            opt = torch.optim.Adam(m.parameters(), lr=1e-2, weight_decay=5e-4)
            for _ in range(local_epochs):
                m.train(); opt.zero_grad()
                F.cross_entropy(m(Xt, Ahat)[trm], yt[trm], weight=w).backward()
                opt.step()
            upd = _get_flat(m)
            if c == poison_client:
                g = _get_flat(global_model)
                upd = g - boost * len(clients) * (upd - g)   # model replacement
            updates.append(upd)
        U = torch.stack(updates)
        if agg == "trimmed":
            k = max(1, int(0.2 * U.shape[0]))
            U_sorted, _ = torch.sort(U, dim=0)
            new = U_sorted[k:U.shape[0] - k].mean(0) if 2 * k < U.shape[0] else U_sorted.mean(0)
        else:
            new = U.mean(0)
        _set_flat(global_model, new)

    # pooled test PR-AUC
    probs, ys = [], []
    global_model.eval()
    with torch.no_grad():
        for c in clients:
            Xt, Ahat, _, _, y = graphs[c]
            p = F.softmax(global_model(Xt, Ahat), dim=1)[:, 1].numpy()
            te = masks[c][1]
            probs.append(p[te]); ys.append(y[te])
    return average_precision_score(np.concatenate(ys), np.concatenate(probs))


def federated_demo(seed=0):
    print("\nFederated GNN reference system (relational laundering, 6 clients)\n")
    clean = federated_gnn(agg="fedavg", poison_client=None, seed=seed)
    pois = federated_gnn(agg="fedavg", poison_client=0, seed=seed)
    rob = federated_gnn(agg="trimmed", poison_client=0, seed=seed)
    print(f"  clean FedAvg            : {clean:.3f} PR-AUC")
    print(f"  poisoned FedAvg         : {pois:.3f}  (degradation {clean - pois:+.3f})")
    print(f"  poisoned + trimmed-mean : {rob:.3f}  (recovered {rob - pois:+.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GNN reference system for FedTBML")
    ap.add_argument("--federated", action="store_true",
                    help="run the federated-GNN-under-poisoning demo too")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ablation(seed=a.seed)
    overlap_sweep(seed=a.seed)
    if a.federated:
        federated_demo(seed=a.seed)
