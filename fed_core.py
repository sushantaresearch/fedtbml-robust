"""
fed_core.py — Runnable federated-learning harness for FedTBML (Paper B core).

What it demonstrates end-to-end, with no heavy dependencies (numpy + sklearn metrics):
  1. NON-IID partition of customs declarations across jurisdiction-clients.
  2. Federated training of a logistic-regression detector (manual FedAvg loop;
     maps 1:1 onto flwr later — see flower_gnn.py).
  3. A POISONED client (a launderer-controlled customs node) under three attacks:
       - label : flip labels locally to suppress its own corridor
       - signflip : send the negated local update
       - boost : model-replacement / boosting attack (push opposite the honest mean)
  4. ROBUST AGGREGATION (coordinate-wise trimmed mean, multi-Krum) vs vanilla FedAvg.
  5. Optional DP-SGD (per-example gradient clipping + Gaussian noise); pair with
     dp_accountant.py to report the (epsilon, delta) budget.
  6. PR-AUC (average precision) as the headline metric — appropriate under the
     extreme class imbalance of TBML detection.

The Paper-B claim this lets you pre-test: poisoning visibly degrades FedAvg, and
trimmed-mean / Krum recover most of the lost detection — on trade data, under a
strategically-motivated adversary that the prior art (TaxFL, SafeLogFL, Suzumura,
the FedGraphNN/SplitNN line) does not model.
"""
from __future__ import annotations
import argparse
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import synth_trade as st


# --------------------------------------------------------------------------- #
# Logistic-regression primitives (numpy, so we fully control FedAvg / DP-SGD)  #
# --------------------------------------------------------------------------- #
def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def predict_proba(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])  # bias column
    return _sigmoid(Xb @ w)


def local_train(
    w0: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 3,
    lr: float = 0.2,
    batch_size: int = 256,
    l2: float = 1e-4,
    dp: dict | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """One client's local optimisation from the broadcast global weights w0.

    If dp is given ({'clip': C, 'noise_mult': sigma}), runs DP-SGD: per-example
    gradients are L2-clipped to C, summed, perturbed with N(0, (sigma*C)^2 I),
    then averaged — the standard Abadi et al. (2016) mechanism.
    """
    rng = rng or np.random.default_rng()
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    w = w0.copy()
    n = Xb.shape[0]
    for _ in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, batch_size):
            b = idx[s : s + batch_size]
            Xb_b, y_b = Xb[b], y[b]
            preds = _sigmoid(Xb_b @ w)
            err = (preds - y_b)[:, None]              # (B,1)
            per_ex = err * Xb_b                       # (B,d) per-example gradients
            if dp is not None:
                C = dp["clip"]
                norms = np.linalg.norm(per_ex, axis=1, keepdims=True) + 1e-12
                per_ex = per_ex * np.minimum(1.0, C / norms)         # clip
                g = per_ex.sum(axis=0)
                g = g + rng.normal(0.0, dp["noise_mult"] * C, size=g.shape)  # noise
                g = g / len(b)
            else:
                g = per_ex.mean(axis=0)
            g = g + l2 * w
            w = w - lr * g
    return w


# --------------------------------------------------------------------------- #
# Aggregators                                                                  #
# --------------------------------------------------------------------------- #
def agg_fedavg(updates: list[np.ndarray], sizes: list[int]) -> np.ndarray:
    W = np.stack(updates)
    weights = np.asarray(sizes, dtype=float)
    weights /= weights.sum()
    return (W * weights[:, None]).sum(axis=0)


def agg_trimmed_mean(updates: list[np.ndarray], beta: float = 0.2) -> np.ndarray:
    """Coordinate-wise trimmed mean: drop the top & bottom beta fraction per dim."""
    W = np.stack(updates)
    m = W.shape[0]
    k = int(np.floor(beta * m))
    Ws = np.sort(W, axis=0)
    if 2 * k >= m:
        k = max(0, (m - 1) // 2)
    trimmed = Ws[k : m - k] if k > 0 else Ws
    return trimmed.mean(axis=0)


def agg_krum(updates: list[np.ndarray], n_byzantine: int = 1, multi: int = 1) -> np.ndarray:
    """(Multi-)Krum: pick the update(s) closest to their m-f-2 nearest neighbours."""
    W = np.stack(updates)
    m = W.shape[0]
    d2 = np.sum((W[:, None, :] - W[None, :, :]) ** 2, axis=2)
    closest = max(1, m - n_byzantine - 2)
    scores = np.array([np.sort(d2[i])[1 : 1 + closest].sum() for i in range(m)])
    chosen = np.argsort(scores)[:multi]
    return W[chosen].mean(axis=0)


# --------------------------------------------------------------------------- #
# Federation                                                                   #
# --------------------------------------------------------------------------- #
def _split_client(Xc, yc, rng, test_frac=0.3):
    n = len(yc)
    idx = rng.permutation(n)
    cut = int(n * (1 - test_frac))
    tr, te = idx[:cut], idx[cut:]
    return Xc[tr], yc[tr], Xc[te], yc[te]


def run_federation(
    df,
    rounds: int = 10,
    agg: str = "fedavg",
    poison_clients: tuple[int, ...] = (),
    attack: str = "boost",
    boost_factor: float = 2.0,
    dp: dict | None = None,
    beta: float = 0.2,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    rng = np.random.default_rng(seed)
    feats = st.FEATURES
    client_ids = sorted(df["client_id"].unique())
    d = len(feats) + 1  # + bias

    # Per-client train/test split; pool test sets for a global evaluation set.
    train, Xtest, ytest = {}, [], []
    for c in client_ids:
        sub = df[df["client_id"] == c]
        Xc = sub[feats].to_numpy(); yc = sub["label"].to_numpy().astype(float)
        Xtr, ytr, Xte, yte = _split_client(Xc, yc, rng)
        train[c] = (Xtr, ytr)
        Xtest.append(Xte); ytest.append(yte)
    Xtest = np.vstack(Xtest); ytest = np.concatenate(ytest)

    w_global = np.zeros(d)
    history = []
    for r in range(1, rounds + 1):
        updates, sizes = [], []
        for c in client_ids:
            Xc, yc = train[c]
            yc_use = yc.copy()
            if c in poison_clients and attack == "label":
                yc_use = 1.0 - yc_use                       # local label flip
            w_c = local_train(w_global, Xc, yc_use, dp=dp,
                              rng=np.random.default_rng(seed + r * 100 + c))
            if c in poison_clients and attack == "signflip":
                w_c = w_global - (w_c - w_global)           # negate the update (subtle)
            elif c in poison_clients and attack == "boost":
                # model-replacement: scale by #clients so it dominates the average
                w_c = w_global - boost_factor * len(client_ids) * (w_c - w_global)
            updates.append(w_c); sizes.append(len(yc))

        if agg == "fedavg":
            w_global = agg_fedavg(updates, sizes)
        elif agg == "trimmed":
            w_global = agg_trimmed_mean(updates, beta=beta)
        elif agg == "krum":
            w_global = agg_krum(updates, n_byzantine=max(1, len(poison_clients)), multi=2)
        else:
            raise ValueError(f"unknown aggregator: {agg}")

        p = predict_proba(w_global, Xtest)
        ap = average_precision_score(ytest, p)
        roc = roc_auc_score(ytest, p)
        history.append((r, ap, roc))
        if verbose:
            print(f"  round {r:2d}  PR-AUC={ap:.4f}  ROC-AUC={roc:.4f}")

    return {"history": history, "final_ap": history[-1][1], "final_roc": history[-1][2],
            "w": w_global}


def _demo():
    """Reproduce the headline Paper-B comparison on synthetic trade data."""
    df = st.generate(n_clients=8, n_per_client=4000, seed=42)
    prev = df["label"].mean()
    print(f"dataset: {len(df)} declarations, {df['client_id'].nunique()} jurisdictions, "
          f"prevalence={prev:.4f} (PR-AUC random baseline = {prev:.4f})\n")

    print("[1] Clean federation, FedAvg")
    clean = run_federation(df, rounds=10, agg="fedavg", poison_clients=(), seed=1)

    print("\n[2] One launderer-controlled node, FedAvg (boosting attack)")
    pois = run_federation(df, rounds=10, agg="fedavg", poison_clients=(3,),
                          attack="boost", seed=1)

    print("\n[3] Same attack, trimmed-mean aggregation")
    rob = run_federation(df, rounds=10, agg="trimmed", poison_clients=(3,),
                         attack="boost", beta=0.2, seed=1)

    print("\n================  SUMMARY (final PR-AUC)  ================")
    print(f"  clean FedAvg              : {clean['final_ap']:.4f}")
    print(f"  poisoned FedAvg          : {pois['final_ap']:.4f}   "
          f"(degradation {clean['final_ap'] - pois['final_ap']:+.4f})")
    print(f"  poisoned + trimmed-mean  : {rob['final_ap']:.4f}   "
          f"(recovered {rob['final_ap'] - pois['final_ap']:+.4f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="FedTBML federated harness (Paper B core)")
    ap.add_argument("--demo", action="store_true", help="run the clean/poisoned/robust comparison")
    ap.add_argument("--clients", type=int, default=8)
    ap.add_argument("--per-client", type=int, default=4000)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--agg", choices=["fedavg", "trimmed", "krum"], default="fedavg")
    ap.add_argument("--poison", type=int, default=0, help="number of malicious clients")
    ap.add_argument("--attack", choices=["label", "signflip", "boost"], default="boost")
    ap.add_argument("--beta", type=float, default=0.2, help="trim fraction for trimmed-mean")
    ap.add_argument("--prevalence-scale", type=float, default=1.0)
    ap.add_argument("--dp", action="store_true", help="train clients with DP-SGD")
    ap.add_argument("--noise", type=float, default=1.0, help="DP noise multiplier (sigma)")
    ap.add_argument("--clip", type=float, default=1.0, help="DP per-example clip norm C")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.demo:
        _demo()
    else:
        df = st.generate(n_clients=a.clients, n_per_client=a.per_client,
                         prevalence_scale=a.prevalence_scale, seed=a.seed)
        dp = {"clip": a.clip, "noise_mult": a.noise} if a.dp else None
        pc = tuple(range(a.poison))
        print(f"clients={a.clients} agg={a.agg} poison={pc} attack={a.attack} dp={bool(dp)}")
        run_federation(df, rounds=a.rounds, agg=a.agg, poison_clients=pc,
                       attack=a.attack, beta=a.beta, dp=dp, seed=a.seed)
