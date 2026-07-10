"""
Experiment 3: Engineered Monthly Seasonal Effect with CSCV.

Reproduces the paper's true-signal demonstration:
- Generate a daily return series with an injected +0.25σ effect on the
  first 5 days of each stylized 22-day month.
- Run the same 8,800-strategy seasonal search as Experiment 2.
- Apply CSCV and show materially lower PBO vs the noise-only case.

Expected result: lower PBO (~13% vs ~55%), fewer negative OOS Sharpes,
and positive IS-to-OOS relationship.

Reference: Bailey et al. (2014), Section 4.2.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.cscv import run_cscv
from src.diagnostics import cscv_diagnostics
from src.metrics import sharpe_ratio_matrix
from src.plots import (
    plot_cumulative_performance,
    plot_degradation_scatter,
    plot_logit_histogram,
    plot_oos_cdf_comparison,
)
# Reuse strategy simulation from exp_2
from exp.exp_2_noise_only import (
    STRATEGY_PARAMS,
    N_STRATEGIES,
    T_TRIM,
    MONTH_SIZE,
    build_strategy_matrix,
    find_best_strategy,
)

# ── Constants ────────────────────────────────────────────────────────────────
SEED: int = 42
T: int = 1_000
SIGMA: float = 0.01
S: int = 16
SEASONAL_EFFECT_MAGNITUDE: float = 0.25  # +0.25 * sigma on seasonal days
SEASONAL_DAYS: int = 5                    # first 5 days of each 22-day month

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Seasonal data generation ─────────────────────────────────────────────────
def generate_seasonal_returns(
    T: int,
    sigma: float,
    seed: int,
    effect_magnitude: float = SEASONAL_EFFECT_MAGNITUDE,
    seasonal_days: int = SEASONAL_DAYS,
    month_size: int = MONTH_SIZE,
) -> np.ndarray:
    """Generate daily returns with an engineered monthly seasonal edge.

    Base returns are i.i.d. N(0, sigma²). For the first ``seasonal_days``
    observations of each stylized ``month_size``-day month, ``effect_magnitude
    * sigma`` is added to the return. This injects a small but genuine edge
    consistent with an early-month long trade.

    Parameters
    ----------
    T : int
        Total number of daily observations.
    sigma : float
        Base return standard deviation.
    seed : int
        Random seed.
    effect_magnitude : float
        Additive effect in units of sigma for seasonal days. Default 0.25.
    seasonal_days : int
        Number of leading days per month that receive the effect. Default 5.
    month_size : int
        Stylized trading days per month. Default 22.

    Returns
    -------
    np.ndarray
        1-D array of shape (T,) with modified returns.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=0.0, scale=sigma, size=T)

    # Identify first seasonal_days of each month
    shift = effect_magnitude * sigma
    for m_start in range(0, T, month_size):
        end_effect = min(m_start + seasonal_days, T)
        returns[m_start:end_effect] += shift

    return returns


# ── Main experiment function ─────────────────────────────────────────────────
def run_exp3(seed: int = SEED) -> dict:
    """Run the engineered-seasonal-effect CSCV experiment.

    Parameters
    ----------
    seed : int
        Random seed. Default 42.

    Returns
    -------
    dict
        Results including PBO, OOS loss probability, best strategy params,
        degradation regression, and dominance statistics.
    """
    print("=" * 60)
    print("Experiment 3: Engineered Monthly Seasonal Effect")
    print("=" * 60)
    print(
        f"  T={T}, N_STRATEGIES={N_STRATEGIES}, S={S}, T_TRIM={T_TRIM}, seed={seed}"
        f"\n  seasonal_effect = +{SEASONAL_EFFECT_MAGNITUDE}σ on first {SEASONAL_DAYS} days/month"
    )

    # Step 1: Generate seasonal returns
    print("\n  [1/5] Generating seasonal returns...")
    returns = generate_seasonal_returns(T, SIGMA, seed)
    sigma_unit = SIGMA  # use the known sigma_unit (same as generation sigma)
    print(f"        sigma_unit = {sigma_unit:.6f}")

    # Verify effect: first-5-day mean vs overall mean
    first5_mask = np.zeros(T, dtype=bool)
    for m_start in range(0, T, MONTH_SIZE):
        first5_mask[m_start: min(m_start + SEASONAL_DAYS, T)] = True
    print(f"        Mean return (seasonal days): {np.mean(returns[first5_mask]):.6f}")
    print(f"        Mean return (other days):    {np.mean(returns[~first5_mask]):.6f}")

    # Step 2: Build strategy matrix
    print("  [2/5] Simulating 8,800 seasonal strategies (may take a few minutes)...")
    M = build_strategy_matrix(returns, sigma_unit=sigma_unit)
    print(f"        Strategy matrix shape: {M.shape}")

    # Step 3: Find full-sample best strategy on trimmed matrix
    print("  [3/5] Finding full-sample best strategy on trimmed matrix...")
    M_trim = M[:T_TRIM, :]
    best = find_best_strategy(M_trim)
    print(
        f"        Best: entry_day={best['entry_day']}, hp={best['holding_period']}, "
        f"stop_k={best['stop_k']}, side={best['side']}, "
        f"ann_sharpe={best['annualized_sharpe']:.4f}"
    )

    # Step 4: Run CSCV
    print("  [4/5] Running CSCV (S=16, 12,780 splits)...")
    result = run_cscv(M_trim, S=S)
    print(f"        PBO = {result.pbo:.4f}")
    print(f"        n_combinations = {result.n_combinations}")

    # Step 5: Compute diagnostics
    print("  [5/5] Computing diagnostics...")
    diag = cscv_diagnostics(result)
    oos_vals = np.array([p[1] for p in result.is_oos_pairs])
    is_vals = np.array([p[0] for p in result.is_oos_pairs])
    print(f"        OOS loss prob = {diag['oos_loss_prob']:.4f}")
    print(f"        Degradation beta = {diag['degradation']['beta']:.4f}")

    # Produce plots
    best_strat_returns = M_trim[:, best["strategy_index"]]

    fig_cum = plot_cumulative_performance(
        best_strat_returns,
        underlying_returns=returns[:T_TRIM],
        save_path=RESULTS_DIR / "exp3_cumulative_performance.png",
        title="Exp 3 (Seasonal): Best Strategy Cumulative Return",
        strategy_label=f"Best (SR={best['annualized_sharpe']:.2f})",
    )
    fig_cum.clf()

    fig_log = plot_logit_histogram(
        result,
        save_path=RESULTS_DIR / "exp3_logit_histogram.png",
        title=f"Exp 3 (Seasonal): CSCV Logits  PBO={result.pbo:.3f}",
    )
    fig_log.clf()

    fig_deg = plot_degradation_scatter(
        result,
        diag["degradation"],
        diag["oos_loss_prob"],
        save_path=RESULTS_DIR / "exp3_degradation_scatter.png",
        title="Exp 3 (Seasonal): IS vs OOS Sharpe Degradation",
    )
    fig_deg.clf()

    fig_cdf = plot_oos_cdf_comparison(
        oos_vals,
        oos_vals,
        save_path=RESULTS_DIR / "exp3_oos_cdf.png",
        title="Exp 3 (Seasonal): Selected OOS Sharpe CDF",
    )
    fig_cdf.clf()

    results = {
        "experiment": "exp_3",
        "seed": seed,
        "sigma": SIGMA,
        "seasonal_effect_magnitude": SEASONAL_EFFECT_MAGNITUDE,
        "seasonal_days": SEASONAL_DAYS,
        "T_original": T,
        "T_trimmed": T_TRIM,
        "N_strategies": N_STRATEGIES,
        "S": S,
        "pbo": result.pbo,
        "n_combinations": result.n_combinations,
        "oos_loss_prob": diag["oos_loss_prob"],
        "best_strategy": best,
        "degradation": {k: v for k, v in diag["degradation"].items() if k != "summary_str"},
        "dominance": diag["dominance"],
    }

    json_path = RESULTS_DIR / "exp3_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Results saved to {json_path}")

    return results


if __name__ == "__main__":
    results = run_exp3()
    print("\nDone.")
