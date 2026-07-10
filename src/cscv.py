"""
Combinatorially Symmetric Cross-Validation (CSCV) Module.

Implements the CSCV algorithm for estimating the Probability of Backtest
Overfitting (PBO) as described in:

    Bailey, D. H., Borwein, J., Lopez de Prado, M., & Zhu, Q. J. (2014).
    The Probability of Backtest Overfitting.
    Journal of Computational Finance, 20(2), 39–70.

# Custom — Context7 found no library equivalent (paper Eq. from Bailey et al. 2014)
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.special import comb

from src.data_validation import (
    check_complementarity,
    partition_matrix_into_blocks,
    validate_performance_matrix,
)
from src.metrics import rank_performance, sharpe_ratio_matrix


@dataclass
class CSCVResult:
    """Container for all outputs of a CSCV run.

    Attributes
    ----------
    logits : np.ndarray
        Array of shape (n_combinations,) with logit-transformed relative
        out-of-sample ranks lambda_c = logit(omega_bar_c).
    pbo : float
        Probability of Backtest Overfitting: fraction of combinations with
        lambda_c < 0 (i.e., selected strategy ranked below median OOS).
    is_oos_pairs : list of tuple
        List of (IS_metric, OOS_metric) for the in-sample best strategy in
        each CSCV split.
    omega_bars : np.ndarray
        Array of shape (n_combinations,) with relative OOS ranks omega_bar_c
        in (0, 1).
    n_combinations : int
        Total number of CSCV combinations used, equal to C(S, S//2).
    S : int
        Number of equal-sized blocks used in the partition.
    N : int
        Number of strategy columns.
    block_size : int
        Number of rows in each block (T_trimmed // S).
    T_trimmed : int
        Number of rows after trimming (= S * block_size).
    T_original : int
        Original row count before trimming.
    """

    logits: np.ndarray
    pbo: float
    is_oos_pairs: list[tuple[float, float]]
    omega_bars: np.ndarray
    n_combinations: int
    S: int
    N: int
    block_size: int
    T_trimmed: int
    T_original: int


def run_cscv(
    M: np.ndarray,
    S: int = 16,
    metric_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    annualize: bool = False,
    periods_per_year: int = 252,
) -> CSCVResult:
    """Run the full CSCV algorithm on a performance matrix.

    Implements the CSCV procedure from Bailey et al. (2014):
      1. Trim M to T_trim rows (divisible by S).
      2. Partition into S equal contiguous blocks.
      3. Enumerate all C(S, S//2) half-sample combinations.
      4. For each combination, form IS and OOS matrices (complementary blocks).
      5. Compute the performance metric for all strategies on IS and OOS.
      6. Identify the IS-best strategy and record its OOS rank.
      7. Convert to relative rank omega_bar and logit lambda.
      8. Estimate PBO as the fraction of negative logits.

    Parameters
    ----------
    M : np.ndarray
        T×N numeric matrix (rows = time periods, columns = strategies).
    S : int, optional
        Even number of blocks for partitioning. Default is 16. The paper
        recommends S = 16 as the baseline.
    metric_fn : callable, optional
        Function mapping a (T_sub, N) array to an (N,) metric vector.
        Default is ``sharpe_ratio_matrix`` (non-annualized).
    annualize : bool, optional
        If True and using the default metric, annualize Sharpe ratios.
        Annualization does not affect ranks when applied identically to IS
        and OOS; it is False by default for rank-invariance transparency.
    periods_per_year : int, optional
        Periods per year for annualization. Default is 252.

    Returns
    -------
    CSCVResult
        Dataclass containing logits, PBO, IS/OOS pairs, and metadata.

    Raises
    ------
    AssertionError
        If input dimensions are incompatible or internal consistency checks
        fail.
    """
    # --- default metric ---
    if metric_fn is None:
        def metric_fn(m: np.ndarray) -> np.ndarray:
            return sharpe_ratio_matrix(m, ddof=1, annualize=annualize,
                                       periods_per_year=periods_per_year)

    # --- 1. Validate and trim ---
    M_trimmed, meta = validate_performance_matrix(M, S)
    T_trimmed = meta["T_trimmed"]
    T_original = meta["T_original"]
    N = meta["N"]
    block_size = meta["block_size"]

    # --- 2. Partition into S blocks ---
    blocks = partition_matrix_into_blocks(M_trimmed, S)

    # --- 3. Enumerate all C(S, S//2) half-sample combinations ---
    half = S // 2
    expected_n_comb = int(comb(S, half, exact=True))
    all_combinations = list(itertools.combinations(range(S), half))
    assert len(all_combinations) == expected_n_comb, (
        f"Expected {expected_n_comb} combinations, generated {len(all_combinations)}"
    )

    logits: list[float] = []
    omega_bars: list[float] = []
    is_oos_pairs: list[tuple[float, float]] = []

    # --- 4-7. Process each combination ---
    for is_indices in all_combinations:
        is_set = set(is_indices)
        oos_indices = sorted(i for i in range(S) if i not in is_set)

        # Verify complementarity
        assert check_complementarity(list(is_indices), oos_indices, S), (
            f"IS/OOS sets are not complementary for combination {is_indices}"
        )

        # Concatenate IS blocks in chronological order
        is_matrix = np.concatenate([blocks[i] for i in sorted(is_indices)], axis=0)
        # Concatenate OOS blocks in chronological order
        oos_matrix = np.concatenate([blocks[i] for i in oos_indices], axis=0)

        # Verify row counts
        assert is_matrix.shape == (half * block_size, N), (
            f"IS matrix shape {is_matrix.shape} != ({half * block_size}, {N})"
        )
        assert oos_matrix.shape == (half * block_size, N), (
            f"OOS matrix shape {oos_matrix.shape} != ({half * block_size}, {N})"
        )

        # Compute performance metrics
        is_metric: np.ndarray = metric_fn(is_matrix)   # shape (N,)
        oos_metric: np.ndarray = metric_fn(oos_matrix)  # shape (N,)

        # Rank OOS (higher = better, best = rank N)
        oos_rank = rank_performance(oos_metric)  # shape (N,)

        # Identify IS-best strategy by raw metric (not by rank)
        n_star = int(np.argmax(is_metric))

        # Relative OOS rank: omega_bar in (0, 1)
        omega_bar = float(oos_rank[n_star]) / (N + 1)
        assert 0.0 < omega_bar < 1.0, (
            f"omega_bar={omega_bar} must be strictly in (0, 1)"
        )

        # Logit transform
        lambda_c = math.log(omega_bar / (1.0 - omega_bar))

        logits.append(lambda_c)
        omega_bars.append(omega_bar)
        is_oos_pairs.append((float(is_metric[n_star]), float(oos_metric[n_star])))

    logits_arr = np.array(logits, dtype=float)
    omega_bars_arr = np.array(omega_bars, dtype=float)

    # --- 8. PBO = fraction of negative logits ---
    pbo = float(np.mean(logits_arr < 0.0))

    # Final count assertion
    assert len(logits_arr) == expected_n_comb, (
        f"Final logit count {len(logits_arr)} != expected {expected_n_comb}"
    )

    return CSCVResult(
        logits=logits_arr,
        pbo=pbo,
        is_oos_pairs=is_oos_pairs,
        omega_bars=omega_bars_arr,
        n_combinations=expected_n_comb,
        S=S,
        N=N,
        block_size=block_size,
        T_trimmed=T_trimmed,
        T_original=T_original,
    )
