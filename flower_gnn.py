"""
flower_gnn.py — PRODUCTION SCAFFOLD (not executed by the demo).

This is the stack you graduate to once the logic in fed_core.py is validated:
  - Flower (flwr) for the real federated orchestration / simulation,
  - PyTorch-Geometric for the GNN reference system (an entity-linkage graph over
    declarations, consistent with your macro-meso-micro entity-graph line),
  - a CUSTOM robust Strategy (coordinate-wise trimmed mean) so the Byzantine /
    strategic-poisoning experiments from fed_core carry over unchanged.

It mirrors fed_core.py 1:1:
    local_train(...)            -> GNNClient.fit(...)
    agg_trimmed_mean(...)       -> TrimmedMeanStrategy.aggregate_fit(...)
    poison_update(boost/sign)   -> MaliciousClient.fit(...)

Tested against flwr ~1.x and torch-geometric ~2.x. Install:
    pip install "flwr[simulation]" torch torch-geometric

NOTE: graph construction (how tabular declarations become a graph) is the one
real design choice. The default below links declarations that share an importer,
exporter, or HS6 — an entity-linkage graph — then does node classification.
Swap in your production schema where marked TODO.
"""
from __future__ import annotations
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

import synth_trade as st


# --------------------------------------------------------------------------- #
# 1. Graph construction (entity-linkage graph over declarations)              #
# --------------------------------------------------------------------------- #
def build_entity_graph(df, feature_cols=None) -> Data:
    """Build a PyG node-classification graph for one client's declarations,
    reusing the validated entity-linkage builder in synth_trade (edges connect
    declarations sharing an importer or exporter).

    TODO(prod): replace synth_trade.generate_relational with your real declarations
    and beneficial-owner / consignor / port edges.
    """
    X, edge_index, y = st.build_graph(df)
    return Data(
        x=torch.tensor(X, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.tensor(y, dtype=torch.long),
    )


# --------------------------------------------------------------------------- #
# 2. GNN reference model                                                       #
# --------------------------------------------------------------------------- #
class GraphSAGEDetector(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.head = torch.nn.Linear(hidden, 2)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.3, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        return self.head(h)


def get_weights(model) -> list[np.ndarray]:
    return [v.cpu().numpy() for v in model.state_dict().values()]


def set_weights(model, weights: list[np.ndarray]) -> None:
    sd = OrderedDict(
        (k, torch.tensor(v)) for k, v in zip(model.state_dict().keys(), weights)
    )
    model.load_state_dict(sd, strict=True)


# --------------------------------------------------------------------------- #
# 3. Flower clients (honest + malicious)                                       #
# --------------------------------------------------------------------------- #
class GNNClient(fl.client.NumPyClient):
    def __init__(self, data: Data, class_weight: float = 10.0, epochs: int = 5):
        self.data = data
        self.model = GraphSAGEDetector(data.x.shape[1])
        self.epochs = epochs
        # up-weight the rare laundering class to fight extreme imbalance
        self.loss_w = torch.tensor([1.0, class_weight], dtype=torch.float)

    def get_parameters(self, config):
        return get_weights(self.model)

    def fit(self, parameters, config):
        set_weights(self.model, parameters)
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-2, weight_decay=1e-4)
        self.model.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            out = self.model(self.data.x, self.data.edge_index)
            loss = F.cross_entropy(out, self.data.y, weight=self.loss_w)
            loss.backward()
            opt.step()
        return get_weights(self.model), int(self.data.num_nodes), {}

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
            loss = F.cross_entropy(out, self.data.y, weight=self.loss_w)
        return float(loss), int(self.data.num_nodes), {}


class MaliciousClient(GNNClient):
    """Launderer-controlled node: boosting / model-replacement attack."""
    def __init__(self, *args, boost: float = 5.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.boost = boost

    def fit(self, parameters, config):
        honest, n, _ = super().fit(parameters, config)
        # push the update in the opposite direction, scaled (mirrors fed_core 'boost')
        poisoned = [g - self.boost * (h - g) for g, h in zip(parameters, honest)]
        return poisoned, n, {}


# --------------------------------------------------------------------------- #
# 4. Robust aggregation strategy (coordinate-wise trimmed mean)               #
# --------------------------------------------------------------------------- #
class TrimmedMeanStrategy(FedAvg):
    def __init__(self, beta: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        layers = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        m = len(layers)
        k = int(np.floor(self.beta * m))
        agg = []
        for li in range(len(layers[0])):
            stacked = np.stack([layers[c][li] for c in range(m)])  # (m, ...)
            s = np.sort(stacked, axis=0)
            if 2 * k >= m:
                k_eff = max(0, (m - 1) // 2)
            else:
                k_eff = k
            trimmed = s[k_eff : m - k_eff] if k_eff > 0 else s
            agg.append(trimmed.mean(axis=0))
        return ndarrays_to_parameters(agg), {}


# --------------------------------------------------------------------------- #
# 5. Simulation entry point                                                    #
# --------------------------------------------------------------------------- #
def make_client_fn(n_clients=6, per_client=600, poison_clients=(0,), seed=7):
    df = st.generate_relational(n_clients=n_clients, n_per_client=per_client,
                                relational_strength=1.0, seed=seed)
    graphs = {c: build_entity_graph(df[df["client_id"] == c]) for c in range(n_clients)}

    def client_fn(cid: str):
        c = int(cid)
        if c in poison_clients:
            return MaliciousClient(graphs[c]).to_client()
        return GNNClient(graphs[c]).to_client()

    return client_fn


def main(robust: bool = True, rounds: int = 10, n_clients: int = 6):
    strategy = (TrimmedMeanStrategy(beta=0.2) if robust
                else FedAvg())
    fl.simulation.start_simulation(
        client_fn=make_client_fn(n_clients=n_clients),
        num_clients=n_clients,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    # Requires: pip install "flwr[simulation]" torch torch-geometric
    main(robust=True)
