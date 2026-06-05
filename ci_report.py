"""
ci_report.py — multi-seed confidence intervals for the headline FedTBML results.

Turns the single-run numbers into mean ± 95% CI (Student-t) over N seeds — the
step between "validated" and "a reviewer will accept it". Covers the three
headline results:
  1. benchmark retention frontier (aggregator x attack),
  2. the GNN-vs-LR reference-system ablation (across relational_strength),
  3. (optional, --full) the federated GNN under poisoning.

    python ci_report.py --seeds 5          # benchmark + ablation CIs
    python ci_report.py --seeds 10 --full  # + federated GNN CIs (slower)

For the manuscript, use >=10 seeds. CI half-widths shrink ~1/sqrt(N).
"""
from __future__ import annotations
import argparse
import numpy as np
from scipy import stats
import synth_trade as st
from fed_core import run_federation
from gnn_ref import lr_baseline, train_gcn, _split_masks, federated_gnn


def mean_ci(vals, conf: float = 0.95):
    a = np.asarray(vals, float)
    a = a[~np.isnan(a)]
    m = float(a.mean()) if len(a) else float("nan")
    if len(a) < 2:
        return m, 0.0
    sem = a.std(ddof=1) / np.sqrt(len(a))
    h = float(sem * stats.t.ppf(0.5 + conf / 2, len(a) - 1))
    return m, h


def benchmark_ci(seeds, het=1.0, fraction=0.25, attacks=("boost", "alie"),
                 aggs=("fedavg", "trimmed", "krum"), rounds=8,
                 n_clients=8, per_client=3000):
    print(f"[1] Benchmark retention — mean ± 95% CI over {len(seeds)} seeds "
          f"(het={het}, {int(fraction * n_clients)}/{n_clients} malicious)\n")
    pc = tuple(range(int(round(fraction * n_clients))))
    cols = {(agg, atk): [] for agg in aggs for atk in attacks}
    for s in seeds:
        df = st.generate(n_clients=n_clients, n_per_client=per_client,
                         heterogeneity=het, seed=s)
        for agg in aggs:
            clean = run_federation(df, rounds=rounds, agg=agg, poison_clients=(),
                                   seed=s, verbose=False)["final_ap"]
            for atk in attacks:
                a = run_federation(df, rounds=rounds, agg=agg, poison_clients=pc,
                                   attack=atk, seed=s, verbose=False)["final_ap"]
                cols[(agg, atk)].append(a / clean if clean > 1e-9 else np.nan)
    print(f"  {'aggregator':>10} | " + " | ".join(f"{('retention vs ' + atk):>20}" for atk in attacks))
    print(f"  {'-'*10}-+-" + "-+-".join("-" * 20 for _ in attacks))
    for agg in aggs:
        cells = []
        for atk in attacks:
            m, h = mean_ci(cols[(agg, atk)])
            cells.append(f"{m:.3f} ± {h:.3f}")
        print(f"  {agg:>10} | " + " | ".join(f"{c:>20}" for c in cells))


def ablation_ci(seeds, overlap=0.5, rs_values=(0.0, 0.5, 1.0), epochs=150, per_client=700):
    print(f"\n[2] GNN ablation — mean ± 95% CI over {len(seeds)} seeds "
          f"(entity_overlap={overlap}, epochs={epochs})\n")
    print(f"  {'relational_strength':>20} | {'LR PR-AUC':>18} | {'GNN PR-AUC':>18}")
    print(f"  {'-'*20}-+-{'-'*18}-+-{'-'*18}")
    for rs in rs_values:
        lrs, gnns = [], []
        for s in seeds:
            df = st.generate_relational(relational_strength=rs, entity_overlap=overlap,
                                        n_per_client=per_client, seed=s)
            X, ei, y = st.build_graph(df)
            tr, te = _split_masks(y, seed=s)
            lrs.append(lr_baseline(X, y, tr, te))
            gnns.append(train_gcn(X, ei, y, tr, te, epochs=epochs, seed=s))
        lm, lh = mean_ci(lrs)
        gm, gh = mean_ci(gnns)
        print(f"  {rs:>20.1f} | {f'{lm:.3f} ± {lh:.3f}':>18} | {f'{gm:.3f} ± {gh:.3f}':>18}")


def federated_ci(seeds):
    print(f"\n[3] Federated GNN under poisoning — mean ± 95% CI over {len(seeds)} seeds\n")
    conds = {
        "clean FedAvg": dict(agg="fedavg", poison_client=None),
        "poisoned FedAvg": dict(agg="fedavg", poison_client=0),
        "poisoned + trimmed": dict(agg="trimmed", poison_client=0),
    }
    for name, kw in conds.items():
        m, h = mean_ci([federated_gnn(seed=s, **kw) for s in seeds])
        print(f"  {name:>20} : {m:.3f} ± {h:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Multi-seed CIs for FedTBML headline results")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--full", action="store_true", help="also run federated GNN CIs (slower)")
    a = ap.parse_args()
    seeds = list(range(a.seeds))
    benchmark_ci(seeds)
    ablation_ci(seeds)
    if a.full:
        federated_ci(seeds)
