"""
Performance Metrics Module for CSCV/PBO Methodology.

Provides columnwise Sharpe ratio computation and rank transformation
following the ranking convention of Bailey et al. (2014): the best
(highest-metric) strategy receives rank N, the worst receives rank 1.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import rankdata


def sharpe_ratio(
    returns: np.ndarray,
    ddof: int = 1,
    annualize: bool = False,
    periods_per_year: int = 252,
) -> float:
    """Compute the Sharpe ratio of a return series.

    Returns mean(returns) / std(returns, ddof=ddof). Returns 0.0 for series
    with fewer than two observations or zero standard deviation. Optionally
    annualizes by multiplying by sqrt(periods_per_year).

    Parameters
    ----------
    returns : np.ndarray
        1-D array of period returns.
    ddof : int, optional
        Delta degrees of freedom for standard deviation. Default is 1.
    annualize : bool, optional
        If True, multiply by sqrt(periods_per_year). Default is False.
    periods_per_year : int, optional
        Trading periods per year used for annualization. Default is 252.

    Returns
    -------
    float
        Sharpe ratio of the return series.
    """
    returns = np.asarray(returns, dtype=float).ravel()
    if len(returns) < 2:
        return 0.0
    # Use peak-to-peak check: numerically robust zero-variance detection.
    # np.std can return a tiny non-zero value (~1e-18) for identical inputs
    # due to floating-point rounding in the two-pass variance algorithm.
    if float(np.max(returns)) == float(np.min(returns)):
        return 0.0
    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=ddof))
    if not math.isfinite(sigma) or sigma == 0.0:
        return 0.0
    sr = mu / sigma
    if annualize:
        sr *= math.sqrt(periods_per_year)
    return float(sr)


def sharpe_ratio_matrix(
    M: np.ndarray,
    ddof: int = 1,
    annualize: bool = False,
    periods_per_year: int = 252,
) -> np.ndarray:
    """Compute Sharpe ratios for every column of a performance matrix.

    Parameters
    ----------
    M : np.ndarray
        T×N matrix of strategy return series (each column is one strategy).
    ddof : int, optional
        Delta degrees of freedom for standard deviation. Default is 1.
    annualize : bool, optional
        If True, annualize Sharpe by sqrt(periods_per_year). Default is False.
    periods_per_year : int, optional
        Trading periods per year. Default is 252.

    Returns
    -------
    np.ndarray
        Array of shape (N,) with Sharpe ratios for each column.
    """
    M = np.asarray(M, dtype=float)
    assert M.ndim == 2, "M must be 2-D"
    T, N = M.shape
    sharpes = np.zeros(N, dtype=float)
    if T < 2:
        return sharpes
    mu = np.mean(M, axis=0)
    sigma = np.std(M, axis=0, ddof=ddof)
    # Use peak-to-peak for zero-variance detection (robust to FP rounding)
    valid = (np.max(M, axis=0) != np.min(M, axis=0)) & (sigma > 0.0)
    sharpes[valid] = mu[valid] / sigma[valid]
    if annualize:
        sharpes *= math.sqrt(periods_per_year)
    return sharpes


def rank_performance(perf: np.ndarray) -> np.ndarray:
    """Rank a performance vector following Bailey et al. (2014) convention.

    The BEST (highest) value receives rank N and the worst receives rank 1.
    Ties are broken by the average rank method for deterministic and symmetric
    treatment.

    Parameters
    ----------
    perf : np.ndarray
        1-D array of N performance values.

    Returns
    -------
    np.ndarray
        Integer-valued rank array of shape (N,) where max rank equals N.

    Examples
    --------
    >>> rank_performance(np.array([0.5, 1.1, 0.7]))
    array([1, 3, 2])
    """
    perf = np.asarray(perf, dtype=float)
    assert perf.ndim == 1, "perf must be 1-D"
    # scipy rankdata assigns rank 1 to smallest; this gives ranks 1..N with
    # best = N, which matches the paper's convention directly.
    ranks = rankdata(perf, method="average").astype(float)
    return ranks
