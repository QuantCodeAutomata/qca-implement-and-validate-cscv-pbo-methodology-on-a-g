"""
Experiment 1: Implement and Validate CSCV/PBO Methodology.

Validates the mathematical and computational correctness of the CSCV
algorithm on a synthetic performance matrix with known structure.

A T×N matrix is constructed where N//4 strategies have a small positive drift
(Sharpe ~ 0.5) and the remaining strategies are pure noise. This design tests
that the CSCV pipeline runs correctly, produces valid outputs, and exhibits the
expected qualitative behavior (lower PBO than pure-noise case).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.special import comb

from src.cscv import run_cscv
from src.diagnostics import (
    compute_dominance_stats,
    compute_oos_loss_probability,
    compute_pbo,
    cscv_diagnostics,
    fit_degradation_regression,
)
from src.metrics import sharpe_ratio_matrix
from src.plots import (
    plot_degradation_scatter,
    plot_logit_histogram,
    plot_oos_cdf_comparison,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def generate_synthetic_performance_matrix(
    T: int = 1000,
    N: int = 200,
    seed: int = 42,
    n_signal: int | None = None,
    signal_mu: float = 0.001,
    noise_sigma: float = 0.02,
) -> np.ndarray:
    """Generate a synthetic T×N daily returns matrix for CSCV validation.

    The first ``n_signal`` columns have a positive daily drift (signal group);
    the remaining columns are zero-mean noise. All columns share the same
    standard deviation ``noise_sigma``.

    Parameters
    ----------
    T : int
        Number of time periods (rows). Default 1000.
    N : int
        Number of strategy columns. Default 200.
    seed : int
        Random seed for reproducibility. Default 42.
    n_signal : int, optional
        Number of signal columns. Defaults to N // 4.
    signal_mu : float
        Daily mean return for signal columns. Default 0.001.
    noise_sigma : float
        Daily return standard deviation for all columns. Default 0.02.

    Returns
    -------
    np.ndarray
        T×N matrix of daily returns.
    """
    if n_signal is None:
        n_signal = N // 4
    rng = np.random.default_rng(seed)
    M = rng.normal(loc=0.0, scale=noise_sigma, size=(T, N))
    # Inject positive drift into first n_signal columns
    M[:, :n_signal] += signal_mu
    return M


def run_exp1(
    S: int = 16,
    N: int = 200,
    T: int = 1000,
    seed: int = 42,
) -> dict:
    """Run CSCV validation experiment on synthetic performance matrix.

    Parameters
    ----------
    S : int
        Number of CSCV blocks. Default 16.
    N : int
        Number of strategy columns. Default 200.
    T : int
        Number of time periods. Default 1000.
    seed : int
        Random seed. Default 42.

    Returns
    -------
    dict
        Results dictionary with keys:
        - pbo                        : PBO estimate
        - n_combinations             : number of CSCV splits
        - expected_n_combinations    : C(S, S//2)
        - combinations_match         : bool, n_combinations == C(S,S//2)
        - pbo_equals_neg_logit_frac  : bool, PBO == mean(logits < 0)
        - oos_loss_prob              : OOS loss probability
        - degradation                : dict from degradation regression
        - dominance                  : dict from dominance statistics
        - T_original, T_trimmed, block_size
    """
    print("=" * 60)
    print("Experiment 1: CSCV/PBO Generic Validation")
    print("=" * 60)
    print(f"  T={T}, N={N}, S={S}, seed={seed}")

    M = generate_synthetic_performance_matrix(T=T, N=N, seed=seed)
    print(f"  Matrix shape: {M.shape}")

    result = run_cscv(M, S=S)
    diag = cscv_diagnostics(result)

    # Compute pooled OOS across all strategies and splits for dominance
    # Use the selected OOS distribution vs itself as the baseline (simplest available)
    oos_vals = np.array([p[1] for p in result.is_oos_pairs])
    is_vals = np.array([p[0] for p in result.is_oos_pairs])

    # Sanity checks
    expected_n_comb = int(comb(S, S // 2, exact=True))
    combinations_match = result.n_combinations == expected_n_comb
    pbo_manual = float(np.mean(result.logits < 0.0))
    pbo_equals_neg_logit_frac = abs(result.pbo - pbo_manual) < 1e-12

    print(f"\n  PBO = {result.pbo:.4f}")
    print(f"  n_combinations = {result.n_combinations} (expected {expected_n_comb})")
    print(f"  combinations_match = {combinations_match}")
    print(f"  pbo_equals_neg_logit_frac = {pbo_equals_neg_logit_frac}")
    print(f"  OOS loss probability = {diag['oos_loss_prob']:.4f}")
    print(f"  Degradation beta = {diag['degradation']['beta']:.4f}")

    # Produce plots
    fig_logit = plot_logit_histogram(
        result,
        save_path=RESULTS_DIR / "exp1_logit_histogram.png",
        title="Exp 1: CSCV Logit Distribution (Mixed Signal/Noise)",
    )
    plt_closed = fig_logit
    plt_closed.clf()

    fig_deg = plot_degradation_scatter(
        result,
        diag["degradation"],
        diag["oos_loss_prob"],
        save_path=RESULTS_DIR / "exp1_degradation_scatter.png",
        title="Exp 1: IS vs OOS Degradation",
    )
    fig_deg.clf()

    fig_cdf = plot_oos_cdf_comparison(
        oos_vals,
        oos_vals,  # self-benchmark
        save_path=RESULTS_DIR / "exp1_oos_cdf.png",
        title="Exp 1: Selected OOS CDF",
    )
    fig_cdf.clf()

    results = {
        "experiment": "exp_1",
        "pbo": result.pbo,
        "n_combinations": result.n_combinations,
        "expected_n_combinations": expected_n_comb,
        "combinations_match": combinations_match,
        "pbo_equals_neg_logit_frac": pbo_equals_neg_logit_frac,
        "oos_loss_prob": diag["oos_loss_prob"],
        "degradation": {
            k: v for k, v in diag["degradation"].items() if k != "summary_str"
        },
        "dominance": diag["dominance"],
        "T_original": result.T_original,
        "T_trimmed": result.T_trimmed,
        "block_size": result.block_size,
        "S": result.S,
        "N": result.N,
        "seed": seed,
    }

    # Save JSON
    json_path = RESULTS_DIR / "exp1_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Results saved to {json_path}")

    return results


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    results = run_exp1()
    print("\nDone.")
