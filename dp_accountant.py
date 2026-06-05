"""
dp_accountant.py — (epsilon, delta) accounting for the DP-SGD used in fed_core.py.

Uses Google's `dp-accounting` (the engine behind TF-Privacy) to compute the
privacy budget of the Poisson-subsampled Gaussian mechanism — i.e. real DP-SGD,
INCLUDING privacy amplification by subsampling, which the closed-form
Dwork-Roth bound (epsilon <= sqrt(2 ln(1.25/delta))/sigma) ignores.

Maps onto fed_core's local_train(dp={'clip': C, 'noise_mult': sigma}):
  - noise_multiplier = sigma          (same sigma passed to the trainer)
  - sample_rate      = batch_size / n_local      (Poisson sampling rate q)
  - steps            = local_epochs * (n_local / batch_size) * federation_rounds

Note: per-client DP composes over that client's own steps; with secure
aggregation the server never sees individual updates, so the client-level
guarantee is what each jurisdiction can certify to its own regulator.

    pip install dp-accounting
"""
from __future__ import annotations
import argparse
import logging
logging.getLogger("absl").setLevel(logging.ERROR)  # hide low-order RDP convergence notices
from dp_accounting import dp_event, rdp


def compute_epsilon(noise_multiplier: float, sample_rate: float,
                    steps: int, delta: float = 1e-5) -> float:
    """Return epsilon for `steps` rounds of the Poisson-subsampled Gaussian
    mechanism at the given noise multiplier and sampling rate."""
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    accountant = rdp.RdpAccountant(orders)
    event = dp_event.PoissonSampledDpEvent(
        sample_rate, dp_event.GaussianDpEvent(noise_multiplier)
    )
    accountant.compose(event, steps)
    return accountant.get_epsilon(delta)


def steps_from_config(n_local: int, batch_size: int, local_epochs: int,
                      rounds: int) -> tuple[int, float]:
    """Translate a fed_core run into (total_steps, sample_rate)."""
    steps_per_epoch = max(1, n_local // batch_size)
    total_steps = steps_per_epoch * local_epochs * rounds
    sample_rate = batch_size / n_local
    return total_steps, sample_rate


def budget_table(sample_rate: float, steps: int, delta: float = 1e-5,
                 sigmas=(0.5, 0.8, 1.0, 1.5, 2.0)) -> None:
    print(f"  q (sample rate) = {sample_rate:.4f}   steps = {steps}   delta = {delta}")
    print(f"  {'sigma':>6} | {'epsilon':>10}")
    print(f"  {'-'*6}-+-{'-'*10}")
    for s in sigmas:
        eps = compute_epsilon(s, sample_rate, steps, delta)
        print(f"  {s:>6.2f} | {eps:>10.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="DP-SGD privacy accountant for FedTBML")
    ap.add_argument("--n-local", type=int, default=2800,
                    help="local TRAIN examples per client (after 70/30 split)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--local-epochs", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--delta", type=float, default=1e-5)
    a = ap.parse_args()

    steps, q = steps_from_config(a.n_local, a.batch_size, a.local_epochs, a.rounds)
    print("DP-SGD budget for the fed_core default configuration\n")
    budget_table(q, steps, a.delta)
    print("\n(For comparison, the non-subsampled Dwork-Roth bound at sigma=1, "
          "delta=1e-5 is epsilon ~ 4.84 per single query — subsampling + tight "
          "accounting is dramatically cheaper.)")
