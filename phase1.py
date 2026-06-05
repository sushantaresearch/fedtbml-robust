#!/usr/bin/env python3
"""
phase1.py — final multi-seed sweeps for FedTBML-Robust.

Each sweep CHECKPOINTS every (config, seed) result to results/<sweep>.csv as it
runs, and skips rows already present — so a timeout never loses work and re-running
resumes. Numpy federated sweeps are fast; the GNN sweep is chunked.

    python phase1.py frontier --seeds 10
    python phase1.py beta     --seeds 10
    python phase1.py fraction --seeds 10
    python phase1.py zsweep   --seeds 10
    python phase1.py dp       --seeds 10
    python phase1.py gnn      --seeds 10            # heavy; resumable
    python phase1.py tables                          # print all mean +/- 95% CI tables
    python phase1.py figures                         # write results/*.pdf
"""
from __future__ import annotations
import os, csv, argparse
import numpy as np
import pandas as pd
from scipy import stats
import synth_trade as st
from fed_core import run_federation
from gnn_ref import lr_baseline, train_gcn, _split_masks
import dp_accountant as dpa

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

HET, PER, ROUNDS, CLIP = 1.0, 3000, 8, 1.0
AGGS = ("fedavg", "trimmed", "krum")
# DP accountant config matching the fed_core runs (per_client=3000 -> ~2100 train)
DP_NLOCAL, DP_BATCH, DP_EPOCHS = 2100, 256, 3


def mean_ci(vals, conf=0.95):
    a = np.asarray(vals, float); a = a[~np.isnan(a)]
    m = float(a.mean()) if len(a) else float("nan")
    if len(a) < 2:
        return m, 0.0
    sem = a.std(ddof=1) / np.sqrt(len(a))
    return m, float(sem * stats.t.ppf(0.5 + conf / 2, len(a) - 1))


def _done(path, keys):
    if not os.path.exists(path):
        return set()
    return {tuple(str(r[k]) for k in keys) for r in csv.DictReader(open(path))}


def _append(path, fields, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


def _ap(df, agg, attack, seed, beta=0.2, z=1.0, frac=2, dp=None):
    pc = tuple(range(frac))
    if attack == "clean":
        return run_federation(df, rounds=ROUNDS, agg=agg, poison_clients=(),
                              beta=beta, dp=dp, seed=seed, verbose=False)["final_ap"]
    return run_federation(df, rounds=ROUNDS, agg=agg, poison_clients=pc, attack=attack,
                          beta=beta, alie_z=z, dp=dp, seed=seed, verbose=False)["final_ap"]


# --------------------------------------------------------------------------- #
# Sweeps (numpy federated)                                                     #
# --------------------------------------------------------------------------- #
def sweep_frontier(seeds):
    path = os.path.join(RES, "frontier.csv"); keys = ["agg", "attack", "seed"]
    fields = keys + ["ap"]; done = _done(path, keys)
    for s in seeds:
        df = None
        for agg in AGGS:
            for attack in ("clean", "boost", "alie"):
                if (agg, attack, str(s)) in done:
                    continue
                if df is None:
                    df = st.generate(n_clients=8, n_per_client=PER, heterogeneity=HET, seed=s)
                ap = _ap(df, agg, attack, s)
                _append(path, fields, {"agg": agg, "attack": attack, "seed": s, "ap": ap})
                print("frontier", agg, attack, s, round(ap, 3), flush=True)


def sweep_beta(seeds):
    path = os.path.join(RES, "beta.csv"); keys = ["beta", "attack", "seed"]
    fields = keys + ["ap"]; done = _done(path, keys)
    for s in seeds:
        df = None
        for beta in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4):
            for attack in ("clean", "boost", "alie"):
                if (str(beta), attack, str(s)) in done:
                    continue
                if df is None:
                    df = st.generate(n_clients=8, n_per_client=PER, heterogeneity=HET, seed=s)
                ap = _ap(df, "trimmed", attack, s, beta=beta)
                _append(path, fields, {"beta": beta, "attack": attack, "seed": s, "ap": ap})
                print("beta", beta, attack, s, round(ap, 3), flush=True)


def sweep_fraction(seeds):
    path = os.path.join(RES, "fraction.csv"); keys = ["agg", "frac", "attack", "seed"]
    fields = keys + ["ap"]; done = _done(path, keys)
    for s in seeds:
        df = None
        for agg in AGGS:
            jobs = [(0, "clean")] + [(f, a) for f in (1, 2, 3) for a in ("boost", "alie")]
            for frac, attack in jobs:
                if (agg, str(frac), attack, str(s)) in done:
                    continue
                if df is None:
                    df = st.generate(n_clients=8, n_per_client=PER, heterogeneity=HET, seed=s)
                ap = _ap(df, agg, attack, s, frac=frac)
                _append(path, fields, {"agg": agg, "frac": frac, "attack": attack, "seed": s, "ap": ap})
                print("fraction", agg, frac, attack, s, round(ap, 3), flush=True)


def sweep_zsweep(seeds):
    path = os.path.join(RES, "zsweep.csv"); keys = ["agg", "z", "seed"]
    fields = keys + ["ap"]; done = _done(path, keys)
    for s in seeds:
        df = None
        for agg in AGGS:
            for z in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
                if (agg, str(z), str(s)) in done:
                    continue
                if df is None:
                    df = st.generate(n_clients=8, n_per_client=PER, heterogeneity=HET, seed=s)
                ap = _ap(df, agg, "alie", s, z=z)
                _append(path, fields, {"agg": agg, "z": z, "seed": s, "ap": ap})
                print("zsweep", agg, z, s, round(ap, 3), flush=True)


def sweep_dp(seeds, sigmas=(0.0, 0.5, 0.8, 1.0, 1.5, 2.0, 4.0)):
    path = os.path.join(RES, "dp.csv"); keys = ["sigma", "seed"]
    fields = keys + ["ap"]; done = _done(path, keys)
    for s in seeds:
        df = None
        for sig in sigmas:
            if (str(sig), str(s)) in done:
                continue
            if df is None:
                df = st.generate(n_clients=8, n_per_client=PER, heterogeneity=HET, seed=s)
            dp = None if sig == 0.0 else {"clip": CLIP, "noise_mult": sig}
            ap = _ap(df, "fedavg", "clean", s, dp=dp)
            _append(path, fields, {"sigma": sig, "seed": s, "ap": ap})
            print("dp", sig, s, round(ap, 3), flush=True)


# --------------------------------------------------------------------------- #
# Sweep (GNN; heavy, chunked)                                                  #
# --------------------------------------------------------------------------- #
GNN_CELLS = [(0.0, 0.5), (0.5, 0.5), (1.0, 0.5), (1.0, 0.0), (1.0, 1.0)]

def sweep_gnn(seeds, cells=GNN_CELLS, epochs=150):
    path = os.path.join(RES, "gnn.csv"); keys = ["rs", "overlap", "seed"]
    fields = keys + ["lr_ap", "gnn_ap"]; done = _done(path, keys)
    for s in seeds:
        for rs, ov in cells:
            if (str(rs), str(ov), str(s)) in done:
                continue
            df = st.generate_relational(relational_strength=rs, entity_overlap=ov, seed=s)
            X, ei, y = st.build_graph(df)
            tr, te = _split_masks(y, seed=s)
            lr = lr_baseline(X, y, tr, te)
            g = train_gcn(X, ei, y, tr, te, epochs=epochs, seed=s)
            _append(path, fields, {"rs": rs, "overlap": ov, "seed": s, "lr_ap": lr, "gnn_ap": g})
            print("gnn", rs, ov, s, round(lr, 3), round(g, 3), flush=True)


# --------------------------------------------------------------------------- #
# Tables                                                                       #
# --------------------------------------------------------------------------- #
def _ret_table(df, group_cols):
    """Compute retention = ap/clean per seed within each group, return mean,ci."""
    out = {}
    for key, g in df.groupby(group_cols):
        m, h = mean_ci(g["retention"].values)
        out[key] = (m, h, len(g))
    return out


def tables():
    n_all = []
    # ---- frontier ----
    p = os.path.join(RES, "frontier.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        clean = d[d.attack == "clean"][["agg", "seed", "ap"]].rename(columns={"ap": "clean"})
        atk = d[d.attack != "clean"].merge(clean, on=["agg", "seed"])
        atk["retention"] = atk["ap"] / atk["clean"]
        nseed = atk["seed"].nunique(); n_all.append(nseed)
        print(f"\n[1] Robustness frontier — retention, mean +/- 95% CI ({nseed} seeds, 2/8 malicious, het={HET})\n")
        print(f"  {'aggregator':>10} | {'vs boost (naive)':>20} | {'vs ALIE (adaptive)':>20}")
        print(f"  {'-'*10}-+-{'-'*20}-+-{'-'*20}")
        for agg in AGGS:
            cells = []
            for attack in ("boost", "alie"):
                g = atk[(atk["agg"] == agg) & (atk.attack == attack)]
                m, h = mean_ci(g["retention"].values)
                cells.append(f"{m:.3f} +/- {h:.3f}")
            print(f"  {agg:>10} | {cells[0]:>20} | {cells[1]:>20}")
    # ---- beta ----
    p = os.path.join(RES, "beta.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        clean = d[d.attack == "clean"][["beta", "seed", "ap"]].rename(columns={"ap": "clean"})
        atk = d[d.attack != "clean"].merge(clean, on=["beta", "seed"])
        atk["retention"] = atk["ap"] / atk["clean"]
        print(f"\n[2] Trimmed-mean: retention vs trim fraction beta (2/8 = 0.25 malicious)\n")
        print(f"  {'beta':>6} | {'vs boost':>18} | {'vs ALIE':>18}")
        print(f"  {'-'*6}-+-{'-'*18}-+-{'-'*18}")
        for beta in sorted(atk.beta.unique()):
            cells = []
            for attack in ("boost", "alie"):
                g = atk[(atk.beta == beta) & (atk.attack == attack)]
                m, h = mean_ci(g["retention"].values)
                cells.append(f"{m:.3f} +/- {h:.3f}")
            flag = "  <- beta >= frac" if beta >= 0.25 else ""
            print(f"  {beta:>6} | {cells[0]:>18} | {cells[1]:>18}{flag}")
    # ---- fraction ----
    p = os.path.join(RES, "fraction.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        clean = d[d.attack == "clean"][["agg", "seed", "ap"]].rename(columns={"ap": "clean"})
        atk = d[d.attack != "clean"].merge(clean, on=["agg", "seed"])
        atk["retention"] = atk["ap"] / atk["clean"]
        print(f"\n[3] Retention vs Byzantine fraction (out of 8 nodes)\n")
        for attack in ("boost", "alie"):
            print(f"  attack = {attack}")
            print(f"    {'aggregator':>10} | " + " | ".join(f"{str(f)+'/8':>14}" for f in (1, 2, 3)))
            print(f"    {'-'*10}-+-" + "-+-".join("-"*14 for _ in (1, 2, 3)))
            for agg in AGGS:
                cells = []
                for f in (1, 2, 3):
                    g = atk[(atk["agg"] == agg) & (atk.frac == f) & (atk.attack == attack)]
                    m, h = mean_ci(g["retention"].values)
                    cells.append(f"{m:.2f}+/-{h:.2f}")
                print(f"    {agg:>10} | " + " | ".join(f"{c:>14}" for c in cells))
            print()
    # ---- zsweep ----
    p = os.path.join(RES, "zsweep.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        fp = os.path.join(RES, "frontier.csv")
        clean = pd.read_csv(fp)
        clean = clean[clean.attack == "clean"][["agg", "seed", "ap"]].rename(columns={"ap": "clean"})
        atk = d.merge(clean, on=["agg", "seed"])
        atk["retention"] = atk["ap"] / atk["clean"]
        print(f"\n[4] ALIE: retention vs attack strength z (2/8 malicious)\n")
        zs = sorted(atk.z.unique())
        print(f"  {'aggregator':>10} | " + " | ".join(f"z={z:>4}" for z in zs))
        print(f"  {'-'*10}-+-" + "-+-".join("-"*6 for _ in zs))
        for agg in AGGS:
            cells = []
            for z in zs:
                g = atk[(atk["agg"] == agg) & (atk.z == z)]
                m, _ = mean_ci(g["retention"].values)
                cells.append(f"{m:>5.2f}")
            print(f"  {agg:>10} | " + " | ".join(f"{c:>6}" for c in cells))
    # ---- dp ----
    p = os.path.join(RES, "dp.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        steps, q = dpa.steps_from_config(DP_NLOCAL, DP_BATCH, DP_EPOCHS, ROUNDS)
        print(f"\n[5] DP-SGD privacy-utility (clean FedAvg; q={q:.4f}, steps={steps}, delta=1e-5)\n")
        print(f"  {'sigma':>6} | {'epsilon':>10} | {'PR-AUC':>16}")
        print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*16}")
        for sig in sorted(d.sigma.unique()):
            g = d[d.sigma == sig]
            m, h = mean_ci(g["ap"].values)
            eps = "inf (no DP)" if sig == 0 else f"{dpa.compute_epsilon(sig, q, steps):.2f}"
            print(f"  {sig:>6} | {eps:>10} | {f'{m:.3f} +/- {h:.3f}':>16}")
    # ---- gnn ----
    p = os.path.join(RES, "gnn.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        print(f"\n[6] GNN reference-system ablation — mean +/- 95% CI\n")
        print(f"  {'relational_strength':>20} | {'entity_overlap':>14} | {'LR PR-AUC':>16} | {'GNN PR-AUC':>16}")
        print(f"  {'-'*20}-+-{'-'*14}-+-{'-'*16}-+-{'-'*16}")
        for (rs, ov), g in d.groupby(["rs", "overlap"]):
            lm, lh = mean_ci(g["lr_ap"].values); gm, gh = mean_ci(g["gnn_ap"].values)
            print(f"  {rs:>20} | {ov:>14} | {f'{lm:.3f} +/- {lh:.3f}':>16} | {f'{gm:.3f} +/- {gh:.3f}':>16}  (n={len(g)})")


# --------------------------------------------------------------------------- #
# Figures                                                                      #
# --------------------------------------------------------------------------- #
def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "figure.dpi": 150,
                         "axes.spines.top": False, "axes.spines.right": False})
    C = {"fedavg": "#444444", "trimmed": "#1f77b4", "krum": "#d62728"}

    # Fig: fraction sweep (boost + alie)
    p = os.path.join(RES, "fraction.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        clean = d[d.attack == "clean"][["agg", "seed", "ap"]].rename(columns={"ap": "clean"})
        atk = d[d.attack != "clean"].merge(clean, on=["agg", "seed"])
        atk["retention"] = atk["ap"] / atk["clean"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, attack in zip(axes, ("boost", "alie")):
            for agg in AGGS:
                xs, ys, es = [], [], []
                for f in (1, 2, 3):
                    g = atk[(atk["agg"] == agg) & (atk.frac == f) & (atk.attack == attack)]
                    m, h = mean_ci(g["retention"].values)
                    xs.append(f / 8); ys.append(m); es.append(h)
                ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, color=C[agg], label=agg)
            ax.axhline(1.0, ls=":", c="#999")
            ax.set_title(f"{'naive (boost)' if attack=='boost' else 'adaptive (ALIE)'}")
            ax.set_xlabel("Byzantine fraction")
        axes[0].set_ylabel("retention (PR-AUC attacked / clean)")
        axes[0].legend(frameon=False)
        fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_fraction.pdf")); plt.close(fig)
        print("wrote fig_fraction.pdf")

    # Fig: ALIE z sweep (the inversion)
    p = os.path.join(RES, "zsweep.csv")
    fp = os.path.join(RES, "frontier.csv")
    if os.path.exists(p) and os.path.exists(fp):
        d = pd.read_csv(p)
        clean = pd.read_csv(fp); clean = clean[clean.attack == "clean"][["agg", "seed", "ap"]].rename(columns={"ap": "clean"})
        atk = d.merge(clean, on=["agg", "seed"]); atk["retention"] = atk["ap"] / atk["clean"]
        fig, ax = plt.subplots(figsize=(6, 4))
        for agg in AGGS:
            xs, ys, es = [], [], []
            for z in sorted(atk.z.unique()):
                g = atk[(atk["agg"] == agg) & (atk.z == z)]
                m, h = mean_ci(g["retention"].values)
                xs.append(z); ys.append(m); es.append(h)
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, color=C[agg], label=agg)
        ax.set_xlabel("ALIE attack strength z"); ax.set_ylabel("retention (PR-AUC attacked / clean)")
        ax.set_title("Adaptive attack: defence ranking inverts"); ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_zsweep.pdf")); plt.close(fig)
        print("wrote fig_zsweep.pdf")

    # Fig: DP privacy-utility
    p = os.path.join(RES, "dp.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        steps, q = dpa.steps_from_config(DP_NLOCAL, DP_BATCH, DP_EPOCHS, ROUNDS)
        rows = []
        for sig in sorted(d.sigma.unique()):
            if sig == 0:
                continue
            g = d[d.sigma == sig]; m, h = mean_ci(g["ap"].values)
            rows.append((dpa.compute_epsilon(sig, q, steps), m, h, sig))
        rows.sort()
        eps = [r[0] for r in rows]; ys = [r[1] for r in rows]; es = [r[2] for r in rows]
        nodp = d[d.sigma == 0]; m0, _ = mean_ci(nodp["ap"].values)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(eps, ys, yerr=es, marker="o", capsize=3, color="#1f77b4")
        if not np.isnan(m0):
            ax.axhline(m0, ls=":", c="#999", label="no DP")
        ax.set_xscale("log"); ax.set_xlabel("privacy budget epsilon (log scale)")
        ax.set_ylabel("PR-AUC"); ax.set_title("Privacy-utility trade-off (clean FedAvg)")
        ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_dp.pdf")); plt.close(fig)
        print("wrote fig_dp.pdf")

    # Fig: GNN ablation + overlap
    p = os.path.join(RES, "gnn.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        # left: ablation across rs at overlap 0.5
        ab = d[d.overlap == 0.5].sort_values("rs")
        rss = sorted(ab.rs.unique()); x = np.arange(len(rss)); w = 0.35
        lrm = [mean_ci(ab[ab.rs == r]["lr_ap"].values) for r in rss]
        gnm = [mean_ci(ab[ab.rs == r]["gnn_ap"].values) for r in rss]
        axes[0].bar(x - w/2, [a[0] for a in lrm], w, yerr=[a[1] for a in lrm], capsize=3, label="LR", color="#999")
        axes[0].bar(x + w/2, [a[0] for a in gnm], w, yerr=[a[1] for a in gnm], capsize=3, label="GCN", color="#1f77b4")
        axes[0].set_xticks(x); axes[0].set_xticklabels([f"{r}" for r in rss])
        axes[0].set_xlabel("relational_strength (overlap=0.5)"); axes[0].set_ylabel("PR-AUC")
        axes[0].set_title("Right model for the regime"); axes[0].legend(frameon=False)
        # right: overlap sweep at rs=1.0
        ov = d[d.rs == 1.0].sort_values("overlap"); ovs = sorted(ov.overlap.unique())
        gno = [mean_ci(ov[ov.overlap == o]["gnn_ap"].values) for o in ovs]
        lro = [mean_ci(ov[ov.overlap == o]["lr_ap"].values) for o in ovs]
        axes[1].errorbar(ovs, [a[0] for a in gno], yerr=[a[1] for a in gno], marker="o", capsize=3, label="GCN", color="#1f77b4")
        axes[1].errorbar(ovs, [a[0] for a in lro], yerr=[a[1] for a in lro], marker="s", capsize=3, label="LR", color="#999")
        axes[1].set_xlabel("entity_overlap (camouflage, rs=1.0)"); axes[1].set_ylabel("PR-AUC")
        axes[1].set_title("Realism cost of ring camouflage"); axes[1].legend(frameon=False)
        fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_gnn.pdf")); plt.close(fig)
        print("wrote fig_gnn.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["frontier", "beta", "fraction", "zsweep", "dp", "gnn", "tables", "figures"])
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    seeds = list(range(a.seeds))
    if a.cmd == "tables":
        tables()
    elif a.cmd == "figures":
        figures()
    else:
        {"frontier": sweep_frontier, "beta": sweep_beta, "fraction": sweep_fraction,
         "zsweep": sweep_zsweep, "dp": sweep_dp, "gnn": sweep_gnn}[a.cmd](seeds)
