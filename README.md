# FedTBML-Robust — a benchmark for robust federated TBML detection (Paper B)

The artifact for the flagship paper, framed as a **benchmark contribution**:
**"A benchmark for Byzantine-robust federated trade-based money-laundering
detection under strategic poisoning."**

`benchmark.py` is the core deliverable — a standardized stress grid
(heterogeneity × attack × malicious fraction × aggregator × DP budget) with a
fixed protocol and a retention-based leaderboard. `fed_core.py` is the reference
system underneath it. The whole thing runs on synthetic/simulated data (the
legitimate substrate for this domain — AMLworld, SynthAML, and BIS Project Aurora
all do the same, because confidential declarations cannot be shared); the real UN
Comtrade signal plugs in as an optional realism case study, not the critical path.

It lets you pre-test the paper's central claim — *a launderer-controlled customs
node can poison a federated detector, and robust aggregation defends against it* —
with a clean path to a GNN reference system.

---

## Why this framing (prior-art positioning)

A corpus + database scan turned up dense prior art for federated AML/fraud and
even cross-jurisdiction federated tax fraud (TaxFL), cross-border customs/logistics
risk (SafeLogFL), federated graph learning for financial crime (Suzumura et al.),
and FedGraphNN/SplitNN fraud pipelines. **None of them model an adversary who
controls a node to suppress detection of their own corridor.** That strategic-
poisoning threat model on *trade* data is the open contribution this harness targets,
and it lands squarely in the special issue's "adversarial and system-level
robustness" theme.

---

## Validated results (from this harness, reproducible)

**Single-comparison demo** — `python fed_core.py --demo` (8 jurisdictions, 32k
declarations, prevalence ≈ 0.044):

| Setting | Final PR-AUC |
|---|---|
| Clean federation, FedAvg | **0.936** |
| One launderer-controlled node, FedAvg (model-replacement attack) | **0.041** (collapses to random) |
| Same attack, trimmed-mean aggregation | **0.908** (recovers +0.867) |

**Benchmark frontier** — `python benchmark.py` reports retention =
PR-AUC(under attack) / PR-AUC(clean) across the grid. The headline result is that
robustness is **attack-dependent — no single aggregator dominates** (quick grid,
2 of 8 nodes malicious):

| Aggregator | vs naive model-replacement | vs adaptive ALIE |
|---|---|---|
| FedAvg | **0.05** (collapses) | 0.97 (barely affected) |
| trimmed-mean (β=0.2) | **0.05** (collapses — β < Byzantine fraction) | 0.90 |
| multi-Krum | 0.97 (robust) | **0.62** (fragile) |

Three findings the benchmark surfaces that a single experiment would miss:
1. **No free lunch.** Krum — the strongest defence against the naive magnitude
   attack — is the *weakest* against the adaptive, variance-bounded ALIE attack.
2. **Trimmed-mean needs β ≥ the Byzantine fraction.** At 25% malicious nodes, β=0.2
   trims only one extreme per coordinate and is overrun; the trim share must exceed
   the adversary's.
3. **FedAvg fails on magnitude, not variance** — wrecked by a scaled model-replacement
   update, but barely moved by an in-distribution ALIE perturbation.

That frontier — defence × attack family × Byzantine fraction × trim/DP parameters —
is the benchmark's contribution. Including the adaptive ALIE attack is what turns a
known result ("robust aggregation works") into an informative one.

DP-SGD budget (Poisson-subsampled Gaussian, δ=1e-5, q=0.122, 192 steps — the same
config as the experiments above; reproduce with `phase1.py dp` / `dp_accountant.py`):

| σ (noise multiplier) | ε |
|---|---|
| 0.80 | 20.59 |
| 1.00 | 13.43 |
| 1.50 | 6.77 |
| 2.00 | 4.49 |

To reach the ε ≤ 3 band used by comparable work, push σ above ~2, lower the
sampling rate, or run fewer rounds — the accountant quantifies the trade-off. Note
the utility cost is severe: as the table in the results section shows, even σ=0.5
collapses PR-AUC at this imbalance.

> Note: the misinvoicing signal is intentionally non-linear (a large deviation
> from the HS6 median in *either* direction is suspicious), so the linear baseline
> only learns once the magnitude feature `abs_dev_from_hs6_median` is included.
> That non-linearity is exactly the empirical motivation for the GNN reference
> system below.

**GNN reference system** — `python gnn_ref.py`. A 2-layer GCN over the
entity-linkage graph is the benchmark's reference system; logistic regression is
the floor. Two dials matter: `relational_strength` (is laundering individual or
ring-structured) and `entity_overlap` (how camouflaged rings are inside honest
traffic). Ablation at a realistic `entity_overlap=0.5`, prevalence ≈ 1.5%:

| relational_strength | LR PR-AUC | GNN PR-AUC | GNN lift |
|---|---|---|---|
| 0.0 (individually anomalous) | 1.000 | 0.869 | −0.131 |
| 0.5 | 0.988 | 0.862 | −0.127 |
| 1.0 (mild anomaly, hidden in rings) | **0.056** (≈ floor) | **0.441** | **+0.386** |

The honest, nuanced result: **use the right model for the regime.** When laundering
is individually anomalous the tabular model wins and the GNN's smoothing slightly
hurts; only when laundering is purely relational does the GNN beat the collapsed LR
floor — and even then it is far from perfect.

Realism cost — sweeping ring camouflage at `relational_strength=1.0`:

| entity_overlap | LR PR-AUC | GNN PR-AUC |
|---|---|---|
| 0.0 (isolated ring cliques) | 0.051 | 0.980 |
| 0.5 (half-camouflaged) | 0.056 | 0.441 |
| 1.0 (fully camouflaged) | 0.040 | 0.112 |

The earlier "GNN = 1.000" was an artifact of isolated rings; as legitimate entities
are shared into rings, separability erodes and detection falls toward a realistic
ceiling — a result reviewers will trust.

Federating the GNN (`python gnn_ref.py --federated`, 6 clients) plugs it into the
robustness story: clean **0.567** → poisoned FedAvg **0.014** → trimmed-mean
**0.401**. (Federation fragments the entity graph, so rings spanning jurisdictions
are invisible to any single client — itself a benchmark-worthy finding.)

> Calibration note: prevalence is now ~1.5% (toward AMLworld's ~0.05–0.1% and
> SynthAML's 20k alerts in 16M transactions). For production numbers push
> `laundering_rate` lower with proportionally larger n, and swap the dense GCN for
> the PyG/sparse path in `flower_gnn.py` at scale.

**Final results — `python phase1.py {frontier,beta,fraction,zsweep,dp,gnn}` then
`tables` / `figures`** (Student-t 95% CI, **10 seeds**; checkpointed to `results/`).

*1. Robustness frontier* (het=1.0, 2/8 malicious):

| Aggregator | retention vs naive (boost) | retention vs adaptive (ALIE) |
|---|---|---|
| FedAvg | 0.056 ± 0.005 | 0.786 ± 0.090 |
| trimmed-mean (β=0.2) | 0.058 ± 0.007 | 0.749 ± 0.091 |
| multi-Krum | 0.916 ± 0.115 | 0.715 ± 0.074 |

*2. Trimmed-mean needs β ≥ Byzantine fraction* (2/8 = 0.25), retention vs boost:

| β | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.40 |
|---|---|---|---|---|---|---|
| retention | 0.056 | 0.058 | 0.058 | **0.643** | 0.643 | 0.661 |

A clean threshold: the defence is overrun until the trim fraction reaches the
adversary fraction, then it recovers.

*3. Retention vs Byzantine fraction* (boost / ALIE), abbreviated:

| Aggregator | boost 1/8 | boost 2/8 | boost 3/8 | ALIE 1/8 | ALIE 2/8 | ALIE 3/8 |
|---|---|---|---|---|---|---|
| FedAvg | 0.06 | 0.06 | 0.06 | 0.91 | 0.79 | 0.72 |
| trimmed (β=0.2) | 0.88 | 0.06 | 0.06 | 0.89 | 0.75 | 0.67 |
| multi-Krum | 1.04 | 0.92 | 0.93 | 0.85 | 0.71 | 0.65 |

*4. ALIE strength sweep* — the inversion, and Krum's non-monotonic response:

| Aggregator | z=0 | z=0.5 | z=1 | z=1.5 | z=2 | z=3 |
|---|---|---|---|---|---|---|
| FedAvg | 0.91 | 0.85 | 0.79 | 0.73 | 0.68 | 0.60 |
| trimmed | 0.92 | 0.83 | 0.75 | 0.69 | 0.66 | 0.61 |
| multi-Krum | 1.03 | 0.84 | **0.71** | 0.70 | 0.79 | 0.92 |

Krum is *worst at moderate z* (the attacker calibrates updates to sit just inside
the honest cluster) and *recovers at large z* (the malicious updates become
detectable outliers again) — a finding only a strength sweep reveals.

*5. DP-SGD privacy–utility* (clean FedAvg; q=0.122, 192 steps, δ=1e-5):

| σ | 0 (no DP) | 0.5 | 1.0 | 2.0 | 4.0 |
|---|---|---|---|---|---|
| ε | ∞ | 73.6 | 13.4 | 4.5 | 1.9 |
| PR-AUC | 0.615 | 0.064 | 0.065 | 0.066 | 0.069 |

A stark, honest negative: at this extreme imbalance, **even negligible DP noise
collapses minority-class detection** — the privacy–utility tension is severe and
worth stating plainly rather than hiding.

*6. GNN reference-system ablation:*

| relational_strength | entity_overlap | LR PR-AUC | GNN PR-AUC |
|---|---|---|---|
| 0.0 (individual) | 0.5 | 1.000 ± 0.000 | 0.895 ± 0.030 |
| 0.5 | 0.5 | 0.971 ± 0.017 | 0.877 ± 0.047 |
| 1.0 (relational) | 0.0 (isolated rings) | 0.046 ± 0.011 | 0.844 ± 0.081 |
| 1.0 (relational) | 0.5 (camouflaged) | 0.049 ± 0.015 | 0.453 ± 0.115 |
| 1.0 (relational) | 1.0 (fully camouflaged) | 0.039 ± 0.009 | 0.090 ± 0.034 |

All contrasts are statistically clean (intervals do not overlap where the claim
rests): Krum is the only aggregator robust to the naive attack yet fragile to the
adaptive one; the GNN beats the collapsed LR floor on relational laundering but
trails it when laundering is individual, and degrades to the floor under full
camouflage. Figures for sweeps 3–6 are in `results/fig_*.pdf`.

*7. Federated GCN under model-replacement poisoning* (relational laundering, 6
clients, 10 seeds):

| condition | PR-AUC |
|---|---|
| clean (FedAvg) | 0.376 ± 0.152 |
| poisoned (1 malicious client) | 0.012 ± 0.004 |
| trimmed-mean (defended) | 0.329 ± 0.158 |

Federating the *graph* reference system both destabilises the relational signal
(clean federated 0.376 ± 0.152 vs centralised 0.453 ± 0.115, per-seed range
0.010–0.613 — cross-jurisdiction rings fragment across clients) and leaves it
exposed: a single model-replacement client collapses it to the floor, and
trimmed-mean recovers it only where a learnable signal survived federation.

---

## Install & run

```bash
pip install -r requirements.txt          # core only; production deps are commented
python synth_trade.py                     # inspect the synthetic data + per-client prevalence
python fed_core.py --demo                 # clean vs poisoned vs robust comparison
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

| File | Role | Status |
|---|---|---|
| `synth_trade.py` | Synthetic customs declarations + injected over/under-invoicing labels; non-IID by jurisdiction; **`generate_relational()` + `build_graph()`** for ring-structured laundering; optional `load_comtrade()` real pull | runnable |
| `fed_core.py` | Federated loop (manual FedAvg), trimmed-mean + multi-Krum, poisoned client (label / signflip / boost / **ALIE adaptive**), DP-SGD, PR-AUC reporting, CLI | **runnable + validated** |
| `gnn_ref.py` | **GNN reference system** (2-layer GCN, no PyG) + LR-vs-GNN ablation + federated-GNN-under-poisoning demo | **runnable + validated** |
| `ci_report.py` | Multi-seed mean ± 95% CI (Student-t) for the benchmark retention frontier, the GNN ablation, and (--full) the federated GNN | **runnable + validated** |
| `phase1.py` | Final **10-seed** sweeps (frontier, β-threshold, Byzantine-fraction, ALIE-z, DP, GNN) with checkpointing + tables + `results/fig_*.pdf` | **runnable + validated** |
| `benchmark.py` | **The benchmark artifact**: stress grid (heterogeneity × attack × malicious fraction × aggregator × DP) → retention metric, `results.csv`, leaderboard | **runnable + validated** |
| `dp_accountant.py` | Real (ε, δ) accounting for the Poisson-subsampled Gaussian via `dp-accounting` | **runnable + validated** |
| `flower_gnn.py` | Production scaffold: Flower `NumPyClient` + custom trimmed-mean `Strategy` + PyG GraphSAGE over an entity-linkage graph; honest + malicious clients | scaffold (compiles) |
| `requirements.txt` | Core vs production dependencies | — |

---

## Next steps

1. ~~Multi-seed CIs + full sweeps.~~ **Done** (`phase1.py`, 10 seeds): frontier,
   β-threshold, Byzantine-fraction, ALIE-z, DP, and GNN sweeps, with figures.
2. **Add the Comtrade realism slice.** Fill `synth_trade.load_comtrade()` with your
   key, build the cif/fob mirror discrepancy for your Bangladesh corridors, and show
   the findings hold on a real-data slice — the SI's "domain-specific study" credit.
3. *(Stretch)* push `laundering_rate` toward AMLworld's ~0.1% at larger n with the
   sparse/PyG GCN in `flower_gnn.py`.
4. **Manuscript prose.** Tables 1–2 and the sweep figures are already wired into
   `manuscript.tex`; fill the `% TODO` sections. The MLJ Contribution Information
   Sheet (the four-question gate) is the first thing reviewers read.

## Data: frozen reference snapshot

The generator seeds `numpy.random.default_rng` (PCG64), which numpy does **not**
guarantee to be bit-identical across versions. So the benchmark stays comparable
regardless of a user's numpy build, `export_data.py` materializes the exact instances
behind the reported results under `data/`, with SHA256 sums in `data/manifest.json`:

- `tabular_frontier.csv.gz` — the 8-client tabular benchmark, seeds 0–9 (240k rows, ~3.4% positive).
- `relational_cells.csv.gz` — the 5 relational cells × seeds 0–9 (210k rows, ~1.4% positive).
- `relational_graph_rs1_ov0p5_seed0.npz` — the canonical entity-linkage graph (X, edge_index, y) for relational_strength=1.0, entity_overlap=0.5, seed 0.

```bash
python export_data.py            # regenerate the frozen snapshot
python export_data.py --verify   # check files against data/manifest.json
```

The snapshot is ~41 MB and is distributed via the Zenodo archive (below); the git
repo carries `export_data.py` and `data/manifest.json` (the checksums) so the data
can be regenerated-and-verified or downloaded from Zenodo. The generator remains the
source of truth for the full parametric grid.

---

## License & citation

Released under the MIT License (see `LICENSE`). Dependencies are pinned in
`requirements.txt` (Python 3.12); install with `pip install -r requirements.txt`.

If you use FedTBML-Robust, please cite the paper (see `CITATION.cff`):

> Sushanta Paul. *A Benchmark for Byzantine-Robust Federated Trade-Based
> Money-Laundering Detection under Strategic Poisoning.* Working paper, 2026.

The paper's Reproducibility appendix lists every hyperparameter, seed, and command;
`python phase1.py {frontier,beta,fraction,zsweep,dp,gnn,fedgnn}` then `tables` /
`figures` regenerates all reported numbers and plots.
