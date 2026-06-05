"""
synth_trade.py — Synthetic customs trade-declaration generator for FedTBML.

Produces import/export declarations with injected trade-based money-laundering
(TBML) typologies as ground-truth labels:
    - over-invoicing  (declared value >> fair value: moves capital OUT of a country)
    - under-invoicing (declared value << fair value: moves capital IN / evades duty)

The micro-level signal mirrors what mirror-statistics (cif/fob discrepancy)
captures at the macro level (Gara, Giammatteo & Tosti, 2019, The World Economy):
an item priced far from its HS6 peer median is suspicious.

Data are intentionally NON-IID across jurisdictions (each client = a destination
country): every jurisdiction gets its own HS mix, structural price level,
laundering rate and dominant typology, so the federation faces realistic client
heterogeneity. The per-(client, HS6) median deviation is computed *within* each
client, respecting data locality (a jurisdiction can only see its own declarations).

This generator exists so the harness runs with zero external dependencies. The
real cross-jurisdiction signal is pulled via load_comtrade() (needs your UN
Comtrade key); the two layers are designed to be concatenated.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# Canonical feature columns consumed by the model.
FEATURES = [
    "log_declared_unit_value",
    "dev_from_hs6_median",       # signed local mirror signal (over- vs under-invoicing)
    "abs_dev_from_hs6_median",   # <- dominant signal: magnitude of misinvoicing (non-linear)
    "log_quantity",
    "log_weight",
    "value_per_kg_z",
    "value_per_unit_z",
]


def _jurisdiction_profiles(n_clients: int, rng: np.random.Generator,
                           heterogeneity: float = 1.0) -> list[dict]:
    """Distinct HS mix / price level / laundering rate / typology bias per client.
    This is what drives non-IID-ness across the federation.

    `heterogeneity` is the controlled non-IID dial (a benchmark axis):
      ~0.3 -> clients look alike (near-IID); ~1.8 -> strongly heterogeneous.
    """
    h = float(heterogeneity)
    profiles = []
    for k in range(n_clients):
        profiles.append(
            dict(
                client_id=k,
                n_hs=int(rng.integers(8, 20)),                                  # HS6 lines traded
                price_mult=float(np.clip(1.0 + h * rng.uniform(-0.4, 0.8), 0.3, 2.5)),
                laundering_rate=float(np.clip(0.035 + h * rng.uniform(-0.025, 0.025), 0.005, 0.08)),
                over_share=float(np.clip(0.5 + h * rng.uniform(-0.3, 0.3), 0.05, 0.95)),
            )
        )
    return profiles


def generate(
    n_clients: int = 8,
    n_per_client: int = 4000,
    prevalence_scale: float = 1.0,
    heterogeneity: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a DataFrame with columns FEATURES + ['label', 'client_id', ...].

    prevalence_scale < 1.0 pushes the laundering base rate down toward the
    extreme-imbalance regime seen in real customs data (where PR-AUC ~ 0.008).
    heterogeneity controls non-IID severity across jurisdictions (benchmark axis).
    """
    rng = np.random.default_rng(seed)
    profiles = _jurisdiction_profiles(n_clients, rng, heterogeneity=heterogeneity)

    # Global HS6 fair-value table (shared "true" prices before per-country multiplier).
    hs_universe = rng.integers(100000, 999999, size=60)
    hs_fair = {int(h): float(np.exp(rng.normal(3.0, 1.0))) for h in hs_universe}

    rows = []
    for p in profiles:
        hs_lines = rng.choice(list(hs_fair.keys()), size=p["n_hs"], replace=False)
        rate = p["laundering_rate"] * prevalence_scale
        for _ in range(n_per_client):
            hs = int(rng.choice(hs_lines))
            fair = hs_fair[hs] * p["price_mult"]
            qty = float(np.exp(rng.normal(4.0, 1.0)))
            unit_weight = float(np.exp(rng.normal(0.0, 0.5)))
            weight = qty * unit_weight
            laundering = rng.random() < rate
            if laundering:
                if rng.random() < p["over_share"]:
                    factor = float(rng.uniform(2.0, 6.0))     # over-invoice
                else:
                    factor = float(rng.uniform(0.1, 0.45))    # under-invoice
            else:
                factor = float(np.exp(rng.normal(0.0, 0.18))) # honest pricing noise
            declared = fair * factor
            rows.append((p["client_id"], hs, fair, declared, qty, weight, int(laundering)))

    df = pd.DataFrame(
        rows,
        columns=["client_id", "hs6", "fair_value", "declared_unit_value",
                 "quantity", "weight", "label"],
    )

    # --- Feature engineering -------------------------------------------------
    # Median deviation computed WITHIN each (client, hs6): the local mirror signal.
    df["hs6_median"] = df.groupby(["client_id", "hs6"])["declared_unit_value"].transform("median")
    df["dev_from_hs6_median"] = np.log(df["declared_unit_value"] / df["hs6_median"])
    df["abs_dev_from_hs6_median"] = df["dev_from_hs6_median"].abs()
    df["log_declared_unit_value"] = np.log(df["declared_unit_value"])
    df["log_quantity"] = np.log(df["quantity"])
    df["log_weight"] = np.log(df["weight"].clip(lower=1e-3))

    vpk = np.log((df["declared_unit_value"] / df["weight"].clip(lower=1e-3)))
    vpu = np.log((df["declared_unit_value"] / df["quantity"].clip(lower=1e-3)))
    df["value_per_kg_z"] = (vpk - vpk.mean()) / (vpk.std() + 1e-9)
    df["value_per_unit_z"] = (vpu - vpu.mean()) / (vpu.std() + 1e-9)
    return df


def load_comtrade(
    reporters: list[str],
    partners: list[str],
    period: str,
    hs_level: int = 6,
    subscription_key: str | None = None,
) -> "pd.DataFrame":
    """OPTIONAL real-data path: pull bilateral merchandise trade and build the
    macro mirror-discrepancy signal (each reporter = a federated client).

    Requires `pip install comtradeapicall` and a (free-tier OK) UN Comtrade key.
    Mirror discrepancy per (HS6, partner): log(import_value_reported_by_A /
    export_value_reported_by_B) — the Gara et al. (2019) signal, federated by
    keeping each reporter's rows on that reporter's client.

    This is a scaffold: wire your key, choose reporters/partners/period, then
    concatenate the resulting per-reporter frames as additional clients.
    """
    try:
        import comtradeapicall  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Install with `pip install comtradeapicall` and pass your "
            "subscription_key. See https://pypi.org/project/comtradeapicall/"
        ) from e
    raise NotImplementedError(
        "Fill in comtradeapicall.getFinalData(...) for your reporters/partners/"
        "period, compute the cif/fob mirror discrepancy per HS6, and return a "
        "frame with one row per declaration-proxy and a 'client_id' = reporter."
    )


if __name__ == "__main__":
    d = generate(n_clients=8, n_per_client=2000, seed=0)
    print(d[["client_id", "label"]].groupby("client_id").mean().round(4).T)
    print("rows:", len(d), "| overall prevalence:", round(d["label"].mean(), 4))
    print("features:", FEATURES)
