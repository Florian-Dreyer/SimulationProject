"""
Draw M parameter samples from the prior, simulate R replicates each, and save
all raw results to disk for offline summary-statistic experiments.

Output: data/prior_simulations.npz
  thetas              float64  (M, 3)        — [beta, gamma, rho] per draw
  infected_fractions  float64  (M, R, T+1)   — infected fraction time series
  rewire_counts       int64    (M, R, T+1)   — rewiring events per step
  degree_histograms   int64    (M, R, 31)    — final degree histogram

Usage
-----
    python scripts/simulate_prior.py                  # defaults: M=10000, R=40
    python scripts/simulate_prior.py --M 50000 --seed 123
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Allow running from repo root or scripts/
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from simulator import simulate_batch  # noqa: E402

# Prior bounds: beta, gamma, rho
PRIOR_LOW = np.array([0.05, 0.02, 0.0])
PRIOR_HIGH = np.array([0.50, 0.20, 0.8])


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample prior and run simulations.")
    parser.add_argument("--M", type=int, default=10_000, help="Number of prior draws")
    parser.add_argument("--R", type=int, default=40, help="Replicates per draw")
    parser.add_argument("--seed", type=int, default=0, help="Master RNG seed")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "prior_simulations.npz",
        help="Output .npz path",
    )
    args = parser.parse_args()

    M, R, seed = args.M, args.R, args.seed
    out_path: Path = args.out

    rng = np.random.default_rng(seed)

    # ── Sample parameters from the prior ─────────────────────────────────────
    thetas = rng.uniform(PRIOR_LOW, PRIOR_HIGH, size=(M, 3))  # (M, 3)

    print(f"Running {M:,} × {R} = {M * R:,} simulations (seed={seed}) …")
    t0 = time.perf_counter()

    results = simulate_batch(thetas, n_replicates=R, rng=rng)

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s  ({elapsed / (M * R) * 1000:.3f} ms/sim)")

    # ── Pack list[list[SimResult]] into dense arrays ──────────────────────────
    # SimResult = (infected_fraction (T+1,), rewire_counts (T+1,), degree_histogram (31,))
    T_plus_1 = results[0][0][0].shape[0]  # T+1 = 201

    infected_fractions = np.empty((M, R, T_plus_1), dtype=np.float64)
    rewire_counts = np.empty((M, R, T_plus_1), dtype=np.int64)
    degree_histograms = np.empty((M, R, 31), dtype=np.int64)

    for i in range(M):
        for r in range(R):
            inf_frac, rew, deg = results[i][r]
            infected_fractions[i, r] = inf_frac
            rewire_counts[i, r] = rew
            degree_histograms[i, r] = deg

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        thetas=thetas,
        infected_fractions=infected_fractions,
        rewire_counts=rewire_counts,
        degree_histograms=degree_histograms,
        seed=np.array(seed),
        M=np.array(M),
        R=np.array(R),
    )

    size_mb = out_path.stat().st_size / 1024**2
    print(f"Saved → {out_path}  ({size_mb:.1f} MB)")
    print(
        f"Arrays: thetas {thetas.shape}, "
        f"infected_fractions {infected_fractions.shape}, "
        f"rewire_counts {rewire_counts.shape}, "
        f"degree_histograms {degree_histograms.shape}"
    )


if __name__ == "__main__":
    main()
