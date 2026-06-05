"""
benchmark.py — the FedTBML-Robust benchmark (the paper's core artifact).

A standardized stress grid for robust federated TBML detection. Each cell fixes:
    heterogeneity  (non-IID severity)  x  attack  x  malicious_fraction
    x  aggregator  x  DP noise (sigma)
and reports detection PR-AUC plus the headline benchmark metric:

    RETENTION = PR-AUC(under attack) / PR-AUC(clean, same aggregator & setting)

Retention isolates *robustness*: 1.0 means the defence fully neutralised the
launderer-controlled node; near 0 means the federation was subverted. The
leaderboard ranks aggregators by WORST-CASE retention across attacks — the
quantity a regulator actually cares about.

This is what converts the method into the benchmark the special issue asks for
("principled benchmarking and evaluation frameworks reflecting heterogeneous,
non-ideal, and failure-prone environments").

Run:
    python benchmark.py --quick      # small grid, ~1 min, for validation
    python benchmark.py              # fuller grid -> results.csv + leaderboard
"""
from __future__ import annotations
import argparse
import csv
import itertools
import numpy as np
import synth_trade as st
from fed_core import run_federation

# Full benchmark axes -------------------------------------------------------- #
HETEROGENEITY = [0.3, 1.0, 1.8]                 # near-IID -> strongly non-IID
ATTACKS = ["label", "signflip", "boost", "alie"]  # incl. adaptive defence-aware attack
FRACTIONS = [0.125, 0.25]                        # share of malicious customs nodes
AGGREGATORS = ["fedavg", "trimmed", "krum"]      # vanilla vs robust
DP_SIGMAS = [0.0, 1.0, 2.0]                      # 0 = no DP


def _poison_clients(n_clients: int, fraction: float) -> tuple[int, ...]:
    return tuple(range(int(round(fraction * n_clients))))


def run_cell(heterogeneity, aggregator, dp_sigma, attack, fraction,
             n_clients=8, per_client=3000, rounds=8, seed=0) -> float:
    """Return final PR-AUC for one configuration (attack with 0 fraction == clean)."""
    df = st.generate(n_clients=n_clients, n_per_client=per_client,
                     heterogeneity=heterogeneity, seed=seed)
    dp = {"clip": 1.0, "noise_mult": dp_sigma} if dp_sigma > 0 else None
    pc = _poison_clients(n_clients, fraction)
    eff_attack = attack if pc else "none"
    res = run_federation(df, rounds=rounds, agg=aggregator, poison_clients=pc,
                         attack=eff_attack, dp=dp, seed=seed, verbose=False)
    return res["final_ap"]


def run_grid(heterogeneity, attacks, fractions, aggregators, dp_sigmas,
             n_clients=8, per_client=3000, rounds=8, seed=0):
    """Run the full cartesian grid; return a list of result dicts."""
    rows = []
    for het, agg, sig in itertools.product(heterogeneity, aggregators, dp_sigmas):
        # clean baseline for this (het, agg, sigma) cell
        clean_ap = run_cell(het, agg, sig, attack="boost", fraction=0.0,
                            n_clients=n_clients, per_client=per_client,
                            rounds=rounds, seed=seed)
        for attack, frac in itertools.product(attacks, fractions):
            ap = run_cell(het, agg, sig, attack=attack, fraction=frac,
                          n_clients=n_clients, per_client=per_client,
                          rounds=rounds, seed=seed)
            retention = ap / clean_ap if clean_ap > 1e-9 else float("nan")
            rows.append(dict(
                heterogeneity=het, aggregator=agg, dp_sigma=sig,
                attack=attack, malicious_fraction=frac,
                clean_prauc=round(clean_ap, 4), attacked_prauc=round(ap, 4),
                retention=round(retention, 4),
            ))
    return rows


def leaderboard(rows) -> None:
    """Rank aggregators by worst-case retention across all attack cells."""
    print("\n================  LEADERBOARD (worst-case retention)  ================")
    print(f"  {'aggregator':>10} | {'min retention':>13} | {'mean retention':>14}")
    print(f"  {'-'*10}-+-{'-'*13}-+-{'-'*14}")
    aggs = sorted({r["aggregator"] for r in rows})
    ranking = []
    for a in aggs:
        rets = [r["retention"] for r in rows if r["aggregator"] == a
                and not np.isnan(r["retention"])]
        ranking.append((a, min(rets), float(np.mean(rets))))
    for a, mn, mean in sorted(ranking, key=lambda t: t[1], reverse=True):
        print(f"  {a:>10} | {mn:>13.3f} | {mean:>14.3f}")
    best = max(ranking, key=lambda t: t[1])
    print(f"\n  -> Most robust under worst-case attack: {best[0]} "
          f"(retains {best[1]:.0%} of clean PR-AUC at its weakest point)")


def main():
    ap = argparse.ArgumentParser(description="FedTBML-Robust benchmark grid")
    ap.add_argument("--quick", action="store_true",
                    help="small grid for validation (~1 min)")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.quick:
        het, atk, frac, sig = [1.0], ["boost", "alie"], [0.25], [0.0]
    else:
        het, atk, frac, sig = HETEROGENEITY, ATTACKS, FRACTIONS, DP_SIGMAS

    n_cells = len(het) * len(AGGREGATORS) * len(sig) * (1 + len(atk) * len(frac))
    print(f"Running FedTBML-Robust benchmark: {n_cells} configurations "
          f"({'quick' if a.quick else 'full'} grid)\n")

    rows = run_grid(het, atk, frac, AGGREGATORS, sig, rounds=a.rounds, seed=a.seed)

    # Print per-cell table
    print(f"  {'het':>4} {'agg':>8} {'sigma':>5} {'attack':>9} {'frac':>5} "
          f"{'clean':>6} {'attacked':>8} {'retention':>9}")
    for r in rows:
        print(f"  {r['heterogeneity']:>4} {r['aggregator']:>8} {r['dp_sigma']:>5} "
              f"{r['attack']:>9} {r['malicious_fraction']:>5} "
              f"{r['clean_prauc']:>6} {r['attacked_prauc']:>8} {r['retention']:>9}")

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  wrote {len(rows)} rows -> {a.out}")

    leaderboard(rows)


if __name__ == "__main__":
    main()
