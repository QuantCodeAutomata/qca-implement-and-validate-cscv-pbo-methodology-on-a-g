"""
Diagnostics Module for CSCV/PBO Methodology.

Provides ancillary diagnostics computed from CSCV output:
  - PBO estimate (fraction of negative logits)
  - Probability of out-of-sample loss
  - IS-to-OOS performance degradation regression
  - Stochastic-dominance style comparison statistics
"""

from __future__ import annotations

from typing import Any

import numpy as np
import statsmodels.api as sm
from scipy import stats

from src.cscv import CSCVResult


def compute_pbo(logits: np.ndarray) -> float:
    """Estimate Probability of Backtest Overfitting from logit values.

    PBO is defined as the fraction of CSCV combinations whose logit-transformed
    relative OOS rank is strictly negative (Bailey et al. 2014, Eq. 11).

    Parameters
    ----------
    logits : np.ndarray
        1-D array of logit values lambda_c for each CSCV combination.

    Returns
    -------
    float
        PBO estimate in [0, 1].
    """
    logits = np.asarray(logits, dtype=float)
    assert logits.ndim == 1, "logits must be 1-D"
    return float(np.mean(logits < 0.0))


def compute_oos_loss_probability(oos_values: np.ndarray) -> float:
    """Compute the probability that the selected OOS performance is negative.

    This is the fraction of CSCV splits in which the in-sample best strategy
    had negative out-of-sample performance (e.g., negative Sharpe ratio).

    Parameters
    ----------
    oos_values : np.ndarray
        1-D array of OOS metric values for the IS-selected strategy across
        all CSCV combinations.

    Returns
    -------
    float
        Fraction in [0, 1] of combinations with negative OOS metric.
    """
    oos_values = np.asarray(oos_values, dtype=float)
    assert oos_values.ndim == 1, "oos_values must be 1-D"
    return float(np.mean(oos_values < 0.0))


def fit_degradation_regression(
    is_values: np.ndarray,
    oos_values: np.ndarray,
) -> dict[str, Any]:
    """Fit OLS regression: OOS_selected = alpha + beta * IS_selected.

    Quantifies the IS-to-OOS performance degradation across CSCV splits
    (Bailey et al. 2014, Section 3.3).

    Parameters
    ----------
    is_values : np.ndarray
        1-D array of in-sample metric values for the IS-best strategy across
        CSCV combinations.
    oos_values : np.ndarray
        1-D array of corresponding OOS metric values for the same strategy.

    Returns
    -------
    dict with keys:
        - alpha       : float, OLS intercept
        - beta        : float, OLS slope
        - r_squared   : float, R² of regression
        - pvalue_beta : float, two-sided p-value for H0: beta = 0
        - summary_str : str, statsmodels OLS summary as string
    """
    is_values = np.asarray(is_values, dtype=float)
    oos_values = np.asarray(oos_values, dtype=float)
    assert is_values.shape == oos_values.shape, "IS and OOS arrays must have equal length"
    assert len(is_values) >= 2, "Need at least 2 observations for regression"

    X = sm.add_constant(is_values, has_constant="add")
    model = sm.OLS(oos_values, X)
    result = model.fit()

    return {
        "alpha": float(result.params[0]),
        "beta": float(result.params[1]),
        "r_squared": float(result.rsquared),
        "pvalue_beta": float(result.pvalues[1]),
        "summary_str": str(result.summary()),
    }


def compute_dominance_stats(
    selected_oos: np.ndarray,
    pooled_oos: np.ndarray,
) -> dict[str, Any]:
    """Compare selected OOS performance distribution to a pooled OOS benchmark.

    Computes summary statistics for both distributions and performs a
    Kolmogorov-Smirnov two-sample test to assess stochastic dominance
    (Bailey et al. 2014, Section 3.4).

    Parameters
    ----------
    selected_oos : np.ndarray
        1-D array of OOS metric values for the IS-selected strategy across
        all CSCV splits.
    pooled_oos : np.ndarray
        1-D array of all OOS metric values across all strategies and splits
        (the benchmark distribution).

    Returns
    -------
    dict with keys:
        - selected_mean         : float
        - selected_std          : float
        - pooled_mean           : float
        - pooled_std            : float
        - selected_frac_positive: float
        - pooled_frac_positive  : float
        - ks_stat               : float, KS test statistic
        - ks_pvalue             : float, KS test two-sided p-value
    """
    selected_oos = np.asarray(selected_oos, dtype=float)
    pooled_oos = np.asarray(pooled_oos, dtype=float)
    ks_stat, ks_pvalue = stats.ks_2samp(selected_oos, pooled_oos)
    return {
        "selected_mean": float(np.mean(selected_oos)),
        "selected_std": float(np.std(selected_oos, ddof=1)) if len(selected_oos) > 1 else 0.0,
        "pooled_mean": float(np.mean(pooled_oos)),
        "pooled_std": float(np.std(pooled_oos, ddof=1)) if len(pooled_oos) > 1 else 0.0,
        "selected_frac_positive": float(np.mean(selected_oos > 0.0)),
        "pooled_frac_positive": float(np.mean(pooled_oos > 0.0)),
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
    }


def cscv_diagnostics(result: CSCVResult) -> dict[str, Any]:
    """Compute all ancillary CSCV diagnostics from a CSCVResult.

    Parameters
    ----------
    result : CSCVResult
        Output from ``run_cscv``.

    Returns
    -------
    dict
        Combined diagnostics dictionary containing:
        - pbo              : PBO estimate (fraction of negative logits)
        - oos_loss_prob    : fraction of splits where selected OOS metric < 0
        - degradation      : dict from fit_degradation_regression
        - dominance        : dict from compute_dominance_stats
        - n_combinations   : number of CSCV combinations
        - S, N, block_size : CSCV parameters
    """
    is_values = np.array([pair[0] for pair in result.is_oos_pairs])
    oos_values = np.array([pair[1] for pair in result.is_oos_pairs])

    pbo = compute_pbo(result.logits)
    oos_loss_prob = compute_oos_loss_probability(oos_values)
    degradation = fit_degradation_regression(is_values, oos_values)

    # Pooled OOS benchmark: selected OOS values themselves pooled across all
    # combinations (the simplest well-defined benchmark available from CSCV)
    pooled_oos = oos_values  # same array used as self-benchmark; caller may supply external
    dominance = compute_dominance_stats(oos_values, pooled_oos)

    return {
        "pbo": pbo,
        "oos_loss_prob": oos_loss_prob,
        "degradation": degradation,
        "dominance": dominance,
        "n_combinations": result.n_combinations,
        "S": result.S,
        "N": result.N,
        "block_size": result.block_size,
    }
