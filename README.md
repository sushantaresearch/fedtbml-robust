# FedTBML-Robust — a benchmark for Byzantine-robust federated TBML detection under strategic poisoning

Artifact for the paper **"A Benchmark for Byzantine-Robust Federated Trade-Based Money-Laundering Detection under Strategic Poisoning."**

`benchmark.py` is the core deliverable — a standardized stress grid
(heterogeneity × attack × malicious fraction × aggregator × DP budget) with a fixed
protocol and a retention-based leaderboard. `fed_core.py` is the reference system
underneath it. Everything runs on synthetic/simulated data, which is the legitimate
substrate for this domain (AMLworld, SynthAML, and related public efforts do the
same, because confidential customs declarations cannot be shared); a real UN Comtrade
mirror-statistics signal plugs in as an optional realism case study, not the critical
path.

The benchmark targets a threat model absent from prior federated AML/fraud work: a
**strategic poisoning adversary** — a Byzantine client with an incentive to suppress
detection of laundering-linked, minority-class transactions — and measures how robust
aggregation does or does not defend against it, with a GNN reference system for the
relational regime.

---

## Threat model and contribution

A corpus and database scan turned up dense prior art for federated AML/fraud,
cross-jurisdiction federated tax fraud (TaxFL), cross-border customs/logistics risk,
federated graph learning for financial crime, and FedGraphNN/SplitNN fraud pipelines.
None of them model an adversary who controls a participating node to suppress
detection of laundering-linked minority-class transactions. That strategic-poisoning
threat model on trade data is the open contribution this benchmark targets, and it
sits squarely in the special issue's adversarial and system-level robustness theme.

---

## Validated results (10 seeds, Student-t 95% CI; checkpointed to `results/`)

Reproduce with `python phase1.py {frontier,beta,fraction,zsweep,dp,gnn,fedgnn}` then
`tables` / `figures`.

**1. Robustness frontier** (heterogeneity 1.0, 2/8 malicious):

| Aggregator | retention vs naive (boost) | retention vs adaptive (ALIE) |
|---|---|---|
| FedAvg | 0.056 ± 0.005 | 0.786 ± 0.090 |
| trimmed-mean (β=0.2) | 0.058 ± 0.007 | 0.749 ± 0.091 |
| multi-Krum | 0.916 ± 0.115 | 0.715 ± 0.074 |

The ranking is **attack-dependent — no single aggregator dominates**. Multi-Krum is
the only aggregator robust to the naive magnitude attack, but its advantage disappears
under the adaptive ALIE attack, where the three aggregators have overlapping 95%
confidence intervals.

**2. Trimmed-mean and the trim fraction** (2/8 = 0.25 malicious), retention vs boost:

| β | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.40 |
|---|---|---|---|---|---|---|
| retention | 0.056 | 0.058 | 0.058 | 0.643 | 0.643 | 0.661 |

In this implementation, trimmed mean retains detection only when the trim fraction is
at least the realized Byzantine fraction: it is overrun until β reaches the adversary
fraction, then recovers.

**3. Retention vs Byzantine fraction** (boost / ALIE), abbreviated:

| Aggregator | boost 1/8 | boost 2/8 | boost 3/8 | ALIE 1/8 | ALIE 2/8 | ALIE 3/8 |
|---|---|---|---|---|---|---|
| FedAvg | 0.06 | 0.06 | 0.06 | 0.91 | 0.79 | 0.72 |
| trimmed (β=0.2) | 0.88 | 0.06 | 0.06 | 0.89 | 0.75 | 0.67 |
| multi-Krum | 1.04 | 0.92 | 0.93 | 0.85 | 0.71 | 0.65 |

**4. ALIE strength sweep** — the relative ranking is attack-dependent, and multi-Krum's
response is non-monotonic in the attack strength z:

| Aggregator | z=0 | z=0.5 | z=1 | z=1.5 | z=2 | z=3 |
|---|---|---|---|---|---|---|
| FedAvg | 0.91 | 0.85 | 0.79 | 0.73 | 0.68 | 0.60 |
| trimmed | 0.92 | 0.83 | 0.75 | 0.69 | 0.66 | 0.61 |
| multi-Krum | 1.03 | 0.84 | 0.71 | 0.70 | 0.79 | 0.92 |

Multi-Krum has its lowest retention at moderate z (the attacker calibrates updates to
sit just inside the honest cluster) and recovers at large z (the malicious updates
become detectable outliers again) — a pattern only a strength sweep reveals.

**5. DP-SGD privacy–utility** (clean FedAvg; Poisson-subsampled Gaussian, q ≈ 0.122,
216 accountant steps, δ=1e-5):

| σ | 0 (no DP) | 0.5 | 1.0 | 2.0 | 4.0 |
|---|---|---|---|---|---|
| ε | ∞ | 81.0 | 14.3 | 4.78 | 2.01 |
| PR-AUC | 0.615 | 0.064 | 0.065 | 0.066 | 0.069 |

A stark, honest negative: at this benchmark's severe class imbalance, even a weak
example-level DP-SGD setting sharply degrades minority-class PR-AUC — the
privacy–utility tension is severe and worth stating plainly.

**6. GNN reference-system ablation:**

| relational_strength | entity_overlap | LR PR-AUC | GNN PR-AUC |
|---|---|---|---|
| 0.0 (individual) | 0.5 | 1.000 ± 0.000 | 0.895 ± 0.030 |
| 0.5 | 0.5 | 0.971 ± 0.017 | 0.877 ± 0.047 |
| 1.0 (relational) | 0.0 (isolated rings) | 0.046 ± 0.011 | 0.844 ± 0.081 |
| 1.0 (relational) | 0.5 (camouflaged) | 0.049 ± 0.015 | 0.453 ± 0.115 |
| 1.0 (relational) | 1.0 (fully camouflaged) | 0.039 ± 0.009 | 0.090 ± 0.034 |

Use the right model for the regime: when laundering is individually anomalous the
tabular model wins and the GNN's smoothing slightly hurts; the GNN beats the collapsed
LR floor only when laundering is relational, and degrades toward the floor as ring
camouflage increases. The perfect LR score in the individual regime reflects the
deliberately injected per-declaration signal, not label leakage. Intervals do not
overlap where these claims rest.

**7. Federated GCN under model-replacement poisoning** (relational laundering, 6
clients):

| condition | PR-AUC |
|---|---|
| clean (FedAvg) | 0.376 ± 0.152 |
| poisoned (1 malicious client) | 0.012 ± 0.004 |
| trimmed-mean (defended) | 0.329 ± 0.158 |

Federating the graph reference system both destabilises the relational signal (clean
federated 0.376 ± 0.152 vs centralised 0.453 ± 0.115 — cross-jurisdiction rings
fragment across clients) and leaves it exposed: a single model-replacement client
collapses it to the floor, and trimmed-mean recovers it only where a learnable signal
survived federation.

Figures for sweeps 3–7 are in `results/fig_*.pdf`.

> The misinvoicing signal is intentionally non-linear (a large deviation from the HS6
> median in either direction is suspicious), so the linear baseline only learns once
> the magnitude feature `abs_dev_from_hs6_median` is included. Signal design and
> relational structure are informed by AMLworld/SynthAML; the elevated benchmark
> prevalence is a deliberate stress setting for stable ten-seed comparisons, not an
> operational prevalence estimate.

---

## Install & run

```bash
pip install -r requirements.txt          # core only; production deps are commented
python synth_trade.py                     # inspect the synthetic data + per-client prevalence
python fed_core.py --demo                 # quick single-run clean vs poisoned vs robust comparison
python dp_accountant.py                   # DP-SGD privacy budget table
python benchmark.py --quick               # the benchmark stress grid (small, ~1 min)
python benchmark.py                       # fuller grid -> results.csv + leaderboard
python gnn_ref.py                         # GNN-vs-LR ablation (needs torch)
python gnn_ref.py --federated             # + federated GNN under poisoning
python ci_report.py --seeds 5             # quick mean ± 95% CI (benchmark + ablation)
python phase1.py frontier --seeds 10      # final sweeps (also: beta fraction zsweep dp gnn fedgnn)
python phase1.py tables                   # print all 10-seed CI tables
python phase1.py figures                  # write results/fig_*.pdf
```

Sweep the knobs:

```bash
python fed_core.py --clients 12 --poison 2 --attack boost --agg trimmed --rounds 15
python fed_core.py --poison 1 --attack signflip --agg krum
python fed_core.py --dp --noise 1.5 --clip 1.0          # train clients under DP-SGD
```

Runs on Google Colab or a Windows prompt unchanged (core deps are pure-Python).

---

## File map

| File | Role |
|---|---|
| `synth_trade.py` | Synthetic customs declarations + injected over/under-invoicing labels; non-IID by jurisdiction; `generate_relational()` + `build_graph()` for ring-structured laundering; optional `load_comtrade()` real pull |
| `fed_core.py` | Federated loop (manual FedAvg), trimmed-mean + multi-Krum, poisoned client (label / signflip / boost / adaptive ALIE), DP-SGD, PR-AUC reporting, CLI |
| `gnn_ref.py` | GNN reference system (2-layer GCN, no PyG) + LR-vs-GNN ablation + federated-GNN-under-poisoning demo |
| `ci_report.py` | Multi-seed mean ± 95% CI (Student-t) for the retention frontier, the GNN ablation, and (--full) the federated GNN |
| `phase1.py` | Final 10-seed sweeps (frontier, β-threshold, Byzantine-fraction, ALIE-z, DP, GNN) with checkpointing + tables + `results/fig_*.pdf` |
| `benchmark.py` | The benchmark artifact: stress grid (heterogeneity × attack × malicious fraction × aggregator × DP) → retention metric, `results.csv`, leaderboard |
| `dp_accountant.py` | (ε, δ) accounting for the Poisson-subsampled Gaussian via `dp-accounting` |
| `flower_gnn.py` | Production scaffold: Flower `NumPyClient` + custom trimmed-mean `Strategy` + PyG GraphSAGE over an entity-linkage graph; honest + malicious clients |
| `requirements.txt` | Core vs production dependencies |

---

## Data: frozen reference snapshot

The generator seeds `numpy.random.default_rng` (PCG64), which numpy does **not**
guarantee to be bit-identical across versions. So the benchmark stays comparable
regardless of a user's numpy build, `export_data.py` materializes the exact instances
behind the reported results under `data/`, with SHA256 sums in `data/manifest.json`:

- `tabular_frontier.csv.gz` — the 8-client tabular benchmark, seeds 0–9 (≈240k rows, realized positive rate ≈ 3.4%).
- `relational_cells.csv.gz` — the 5 relational cells × seeds 0–9 (≈210k rows, realized positive rate ≈ 1.4%).
- `relational_graph_rs1_ov0p5_seed0.npz` — the canonical entity-linkage graph (X, edge_index, y) for relational_strength=1.0, entity_overlap=0.5, seed 0.

Both rates are elevated benchmark stress settings chosen for stable ten-seed
comparisons, not operational prevalence estimates.

```bash
python export_data.py            # regenerate the frozen snapshot
python export_data.py --verify   # check files against data/manifest.json
```

The snapshot is ~42 MB and is distributed via the Zenodo archive (below); the git repo
carries `export_data.py` and `data/manifest.json` (the checksums) so the data can be
regenerated-and-verified or downloaded from Zenodo. The generator remains the source of
truth for the full parametric grid.

---

## Extending the benchmark

- Add a UN Comtrade realism slice: fill `synth_trade.load_comtrade()`, build the
  CIF/FOB mirror-discrepancy signal for specific corridors, and check that the findings
  hold on a real-data slice.
- Push `laundering_rate` lower with proportionally larger n and the sparse/PyG GCN path
  in `flower_gnn.py` for production-scale runs.

---

## License & citation

Released under the MIT License (see `LICENSE`). Dependencies are pinned in
`requirements.txt` (Python 3.12).

Data and code archive: Zenodo, DOI `10.5281/zenodo.20555008` (all versions).

If you use FedTBML-Robust, please cite the paper (see `CITATION.cff`):

> Sushanta Paul. *A Benchmark for Byzantine-Robust Federated Trade-Based
> Money-Laundering Detection under Strategic Poisoning.* Manuscript under review, 2026.

The paper's Reproducibility appendix lists every hyperparameter, seed, and command;
`python phase1.py {frontier,beta,fraction,zsweep,dp,gnn,fedgnn}` then `tables` /
`figures` regenerates all reported numbers and plots.
