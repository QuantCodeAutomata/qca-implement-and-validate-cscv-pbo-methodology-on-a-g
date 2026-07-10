"""
Experiment 2: Noise-Only Seasonal Strategy Search with CSCV.

Reproduces the paper's synthetic overfitting demonstration:
- Generate a pure noise daily return series (T = 1,000)
- Enumerate 8,800 monthly seasonal strategy configurations
- Build the T×N performance matrix
- Apply CSCV (S=16, trimmed to 992 rows)
- Estimate PBO and ancillary diagnostics

Expected result: high PBO despite potentially attractive full-sample best-strategy Sharpe,
demonstrating that CSCV correctly detects overfitting in a noise-only setting.

Reference: Bailey et al. (2014), Section 4.1.
"""

from __future__ import annotations

import itertools
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

# ── Constants ────────────────────────────────────────────────────────────────
SEED: int = 42
T: int = 1_000
SIGMA: float = 0.01          # daily return std (sigma_unit)
MONTH_SIZE: int = 22         # stylized trading days per month
S: int = 16                  # CSCV partition count
T_TRIM: int = 992            # floor(1000 / 16) * 16 = 992
N_STRATEGIES: int = 8_800    # 22 × 20 × 10 × 2

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Precompute strategy parameter grid ───────────────────────────────────────
def _build_strategy_params() -> np.ndarray:
    """Return an (8800, 4) array of (entry_day, holding_period, stop_k, side)."""
    params = list(
        itertools.product(
            range(1, 23),   # entry_day:       1 .. 22
            range(1, 21),   # holding_period:  1 .. 20
            range(1, 11),   # stop_loss_k:     1 .. 10
            [-1, 1],        # side
        )
    )
    assert len(params) == N_STRATEGIES, f"Expected 8800 strategies, got {len(params)}"
    return np.array(params, dtype=np.int32)


STRATEGY_PARAMS: np.ndarray = _build_strategy_params()


# ── Data generation ──────────────────────────────────────────────────────────
def generate_noise_returns(T: int, sigma: float, seed: int) -> np.ndarray:
    """Generate T i.i.d. daily returns from N(0, sigma²).

    Parameters
    ----------
    T : int
        Number of observations.
    sigma : float
        Standard deviation of returns.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        1-D array of shape (T,) containing i.i.d. Gaussian returns.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=sigma, size=T)


# ── Strategy simulation ──────────────────────────────────────────────────────
def build_strategy_matrix(
    returns: np.ndarray,
    sigma_unit: float | None = None,
    month_size: int = MONTH_SIZE,
) -> np.ndarray:
    """Simulate all 8,800 seasonal strategies and return the performance matrix.

    For each of the 8,800 configurations (entry_day, holding_period, stop_k, side):
      - Partition the return series into stylized 22-day months.
      - For each month with at least ``entry_day`` observations, enter a position
        at observation ``entry_day - 1`` (0-indexed within the month).
      - Maintain the position for ``holding_period`` days unless the cumulative
        signed return (``side * cumsum(returns_in_window)``) falls below
        ``-stop_k * sigma_unit``, in which case close at end of that day.
      - Record daily strategy returns (= side × underlying return on active days,
        0 on inactive days).

    Stop-loss convention: end-of-day triggering based on cumulative signed returns.
    sigma_unit defaults to the unconditional std of the supplied returns series.

    Parameters
    ----------
    returns : np.ndarray
        1-D array of T daily returns.
    sigma_unit : float, optional
        Volatility scale for stop-loss. Defaults to std(returns, ddof=1).
    month_size : int, optional
        Number of trading days per stylized month. Default 22.

    Returns
    -------
    np.ndarray
        T × 8800 matrix of daily strategy returns.
    """
    T_series = len(returns)
    if sigma_unit is None:
        sigma_unit = float(np.std(returns, ddof=1))

    # Precompute month boundaries
    month_starts = list(range(0, T_series, month_size))

    # Output matrix
    strat_rets = np.zeros((T_series, N_STRATEGIES), dtype=np.float64)

    # Process month by month; within each month, vectorize over strategies
    for m_start in month_starts:
        m_end = min(m_start + month_size, T_series)
        month_len = m_end - m_start

        # For each entry_day (1..22), batch process all strategies with that entry_day
        for entry_day in range(1, month_size + 1):
            if entry_day > month_len:
                continue

            # 0-indexed absolute entry position
            entry_pos = m_start + entry_day - 1

            # Maximum window available (clipped to month and series end)
            max_window_len = min(20, m_end - entry_pos)  # max holding_period = 20
            if max_window_len <= 0:
                continue

            # Returns during the maximum window
            window_returns = returns[entry_pos: entry_pos + max_window_len]
            cum_ret = np.cumsum(window_returns)   # shape (max_window_len,)

            # Strategies with this entry_day: mask over STRATEGY_PARAMS
            mask = STRATEGY_PARAMS[:, 0] == entry_day
            strat_indices = np.where(mask)[0]  # indices into 8800 strategies

            for si in strat_indices:
                _, hp, stop_k, side = STRATEGY_PARAMS[si]
                hp = int(hp)
                stop_k = int(stop_k)
                side = int(side)

                # Clip holding period to available window
                actual_hp = min(hp, max_window_len)
                if actual_hp <= 0:
                    continue

                # Signed cumulative return within holding period
                signed_cum = side * cum_ret[:actual_hp]
                stop_threshold = -stop_k * sigma_unit

                # Find first day where stop is triggered
                stop_days = np.where(signed_cum < stop_threshold)[0]

                if len(stop_days) > 0:
                    # Close at end of that day (return on stop day is realized)
                    actual_exit = entry_pos + int(stop_days[0]) + 1
                else:
                    actual_exit = entry_pos + actual_hp

                # Record strategy returns for active days
                n_active = actual_exit - entry_pos
                strat_rets[entry_pos:actual_exit, si] = (
                    side * returns[entry_pos:actual_exit]
                )

    return strat_rets


# ── Full-sample best strategy identification ─────────────────────────────────
def find_best_strategy(
    strat_matrix: np.ndarray,
    params: np.ndarray = STRATEGY_PARAMS,
) -> dict:
    """Identify the strategy with the highest Sharpe on the full matrix.

    Parameters
    ----------
    strat_matrix : np.ndarray
        T × N strategy return matrix.
    params : np.ndarray
        N × 4 parameter array (entry_day, holding_period, stop_k, side).

    Returns
    -------
    dict
        Keys: strategy_index, entry_day, holding_period, stop_k, side, sharpe.
    """
    sharpes = sharpe_ratio_matrix(strat_matrix, ddof=1, annualize=True)
    best_idx = int(np.argmax(sharpes))
    return {
        "strategy_index": best_idx,
        "entry_day": int(params[best_idx, 0]),
        "holding_period": int(params[best_idx, 1]),
        "stop_k": int(params[best_idx, 2]),
        "side": int(params[best_idx, 3]),
        "annualized_sharpe": float(sharpes[best_idx]),
    }


# ── Main experiment function ─────────────────────────────────────────────────
def run_exp2(seed: int = SEED) -> dict:
    """Run the noise-only seasonal strategy CSCV experiment.

    Parameters
    ----------
    seed : int
        Random seed for synthetic data generation. Default 42.

    Returns
    -------
    dict
        Results including PBO, OOS loss probability, best strategy params,
        degradation regression, and dominance statistics.
    """
    print("=" * 60)
    print("Experiment 2: Noise-Only Seasonal Strategy Search")
    print("=" * 60)
    print(f"  T={T}, N_STRATEGIES={N_STRATEGIES}, S={S}, T_TRIM={T_TRIM}, seed={seed}")

    # Step 1: Generate noise returns
    print("\n  [1/5] Generating noise returns...")
    returns = generate_noise_returns(T, SIGMA, seed)
    sigma_unit = float(np.std(returns, ddof=1))
    print(f"        sigma_unit = {sigma_unit:.6f}")

    # Step 2: Build strategy matrix
    print("  [2/5] Simulating 8,800 seasonal strategies (may take a few minutes)...")
    M = build_strategy_matrix(returns, sigma_unit=sigma_unit)
    print(f"        Strategy matrix shape: {M.shape}")

    # Step 3: Find full-sample best strategy
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
        save_path=RESULTS_DIR / "exp2_cumulative_performance.png",
        title="Exp 2 (Noise-Only): Best Strategy Cumulative Return",
        strategy_label=f"Best (SR={best['annualized_sharpe']:.2f})",
    )
    fig_cum.clf()

    fig_log = plot_logit_histogram(
        result,
        save_path=RESULTS_DIR / "exp2_logit_histogram.png",
        title=f"Exp 2 (Noise-Only): CSCV Logits  PBO={result.pbo:.3f}",
    )
    fig_log.clf()

    fig_deg = plot_degradation_scatter(
        result,
        diag["degradation"],
        diag["oos_loss_prob"],
        save_path=RESULTS_DIR / "exp2_degradation_scatter.png",
        title="Exp 2 (Noise-Only): IS vs OOS Sharpe Degradation",
    )
    fig_deg.clf()

    fig_cdf = plot_oos_cdf_comparison(
        oos_vals,
        oos_vals,
        save_path=RESULTS_DIR / "exp2_oos_cdf.png",
        title="Exp 2 (Noise-Only): Selected OOS Sharpe CDF",
    )
    fig_cdf.clf()

    results = {
        "experiment": "exp_2",
        "seed": seed,
        "sigma": SIGMA,
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

    json_path = RESULTS_DIR / "exp2_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Results saved to {json_path}")

    return results


if __name__ == "__main__":
    results = run_exp2()
    print("\nDone.")
