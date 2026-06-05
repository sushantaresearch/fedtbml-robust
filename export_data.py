#!/usr/bin/env python3
"""
export_data.py — materialize the canonical FedTBML-Robust synthetic datasets.

Why this exists: the generator seeds numpy.random.default_rng (PCG64), which numpy
does NOT guarantee to be bit-stable across versions (only the legacy Mersenne Twister
is). A benchmark must be comparable regardless of a user's numpy version, so we ship
the exact instances behind the reported results as a frozen snapshot under data/,
with SHA256 sums in data/manifest.json. The generator remains the source of truth for
the full parametric grid; re-running this script under the pinned requirements
reproduces these files, and the checksums let anyone verify their regeneration matches.

    python export_data.py            # write data/*.csv.gz, data/*.npz, data/manifest.json
    python export_data.py --verify   # regenerate in-memory and check against the manifest
"""
from __future__ import annotations
import os, json, hashlib, argparse
import numpy as np
import pandas as pd
import synth_trade as st

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SEEDS = list(range(10))
TAB = dict(n_clients=8, n_per_client=3000, heterogeneity=1.0)
CELLS = [(0.0, 0.5), (0.5, 0.5), (1.0, 0.0), (1.0, 0.5), (1.0, 1.0)]   # (relational_strength, entity_overlap)
CANON_CELL = (1.0, 0.5)  # canonical relational cell whose graph we materialize


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


def _build_tabular():
    frames = []
    for s in SEEDS:
        d = st.generate(seed=s, **TAB)
        d.insert(0, "seed", s)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def _build_relational():
    frames = []
    for rs, ov in CELLS:
        for s in SEEDS:
            d = st.generate_relational(relational_strength=rs, entity_overlap=ov, seed=s)
            d.insert(0, "seed", s)
            d.insert(0, "entity_overlap", ov)
            d.insert(0, "relational_strength", rs)
            frames.append(d)
    return pd.concat(frames, ignore_index=True)


def main():
    os.makedirs(DATA, exist_ok=True)
    manifest = {
        "description": "Frozen canonical synthetic datasets for FedTBML-Robust.",
        "generator": "synth_trade.py (numpy.random.default_rng / PCG64)",
        "note": "default_rng is not guaranteed bit-stable across numpy versions; "
                "these frozen files define the benchmark. Reproduce with the pinned "
                "requirements and verify against the sha256 sums below.",
        "seeds": SEEDS,
        "files": {},
    }

    # 1) Tabular frontier dataset (all seeds)
    tab = _build_tabular()
    p = os.path.join(DATA, "tabular_frontier.csv.gz")
    tab.to_csv(p, index=False, compression="gzip")
    manifest["files"]["tabular_frontier.csv.gz"] = {
        "generator": "generate", "params": TAB, "seeds": SEEDS,
        "rows": int(len(tab)), "columns": list(tab.columns),
        "label_column": "label", "positive_rate": round(float(tab["label"].mean()), 6),
        "sha256": _sha256(p),
    }

    # 2) Relational dataset (5 cells x all seeds)
    rel = _build_relational()
    p = os.path.join(DATA, "relational_cells.csv.gz")
    rel.to_csv(p, index=False, compression="gzip")
    per_cell = {f"rs={rs},ov={ov}": round(float(
        rel[(rel.relational_strength == rs) & (rel.entity_overlap == ov)]["label"].mean()), 6)
        for rs, ov in CELLS}
    manifest["files"]["relational_cells.csv.gz"] = {
        "generator": "generate_relational",
        "params": {"n_clients": 6, "n_per_client": 700, "n_rings": 12, "ring_size": 5,
                   "laundering_rate": 0.015, "cells_(relational_strength,entity_overlap)": CELLS},
        "seeds": SEEDS, "rows": int(len(rel)), "columns": list(rel.columns),
        "label_column": "label", "positive_rate_overall": round(float(rel["label"].mean()), 6),
        "positive_rate_per_cell": per_cell, "sha256": _sha256(p),
    }

    # 3) Canonical relational graph (rs=1.0, ov=0.5, seed=0): X, edge_index, y
    rs, ov = CANON_CELL
    dfg = st.generate_relational(relational_strength=rs, entity_overlap=ov, seed=0)
    X, edge_index, y = st.build_graph(dfg)
    p = os.path.join(DATA, "relational_graph_rs1_ov0p5_seed0.npz")
    np.savez_compressed(p, X=X, edge_index=edge_index, y=y, features=np.array(st.FEATURES))
    manifest["files"]["relational_graph_rs1_ov0p5_seed0.npz"] = {
        "generator": "generate_relational + build_graph",
        "params": {"relational_strength": rs, "entity_overlap": ov, "seed": 0, "max_degree_per_entity": 6},
        "arrays": {"X": list(X.shape), "edge_index": list(edge_index.shape), "y": list(y.shape)},
        "n_features": len(st.FEATURES), "features": list(st.FEATURES),
        "positive_rate": round(float(y.mean()), 6), "sha256": _sha256(p),
    }

    with open(os.path.join(DATA, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("Wrote frozen snapshot to data/:")
    for name, meta in manifest["files"].items():
        size = os.path.getsize(os.path.join(DATA, name)) / 1e6
        print(f"  {name:42}  {size:6.2f} MB  sha256={meta['sha256'][:12]}...")


def verify():
    mpath = os.path.join(DATA, "manifest.json")
    if not os.path.exists(mpath):
        print("No manifest; run without --verify first."); return
    man = json.load(open(mpath))
    ok = True
    for name, meta in man["files"].items():
        p = os.path.join(DATA, name)
        cur = _sha256(p) if os.path.exists(p) else "MISSING"
        match = cur == meta["sha256"]
        ok &= match
        print(f"  {name:42}  {'OK' if match else 'MISMATCH/'+cur[:12]}")
    print("All files match manifest." if ok else "Some files differ from manifest.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    verify() if a.verify else main()
