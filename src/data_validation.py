"""
Data Validation Module for CSCV/PBO Methodology.

Provides validation and preprocessing utilities for the performance matrix M
used in Combinatorially Symmetric Cross-Validation (CSCV).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def validate_performance_matrix(
    M: np.ndarray,
    S: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate and trim a performance matrix for CSCV.

    Checks that M is a 2-D numeric T×N matrix compatible with CSCV using S
    partitions. Trims rows to T_trim = floor(T / S) * S, preserving
    chronological order. Documents removed rows in metadata.

    Parameters
    ----------
    M : np.ndarray
        T×N matrix of strategy performance series (columns = strategies).
    S : int
        Even number of partitions (blocks). Must satisfy S >= 2, S even, S <= T.

    Returns
    -------
    M_trimmed : np.ndarray
        Row-trimmed matrix of shape (T_trim, N).
    metadata : dict
        Dictionary with keys:
        - T_original  : original row count
        - T_trimmed   : trimmed row count
        - N           : column count
        - S           : partition count
        - block_size  : rows per block (T_trimmed // S)
        - n_removed   : number of rows trimmed

    Raises
    ------
    AssertionError
        If any validation check fails.
    """
    assert isinstance(M, np.ndarray), "M must be a numpy ndarray"
    assert M.ndim == 2, f"M must be 2-D, got shape {M.shape}"
    assert np.issubdtype(M.dtype, np.floating) or np.issubdtype(
        M.dtype, np.integer
    ), "M must contain numeric data"
    assert M.shape[0] > 0, "M must have at least one row"
    assert M.shape[1] > 1, f"M must have at least 2 columns (strategies), got {M.shape[1]}"
    assert isinstance(S, int) and S >= 2, f"S must be an integer >= 2, got {S}"
    assert S % 2 == 0, f"S must be even, got {S}"
    assert S <= M.shape[0], f"S={S} exceeds number of rows T={M.shape[0]}"

    T_original = M.shape[0]
    N = M.shape[1]
    T_trimmed = (T_original // S) * S
    n_removed = T_original - T_trimmed

    M_trimmed = M[:T_trimmed, :]

    metadata: dict[str, Any] = {
        "T_original": T_original,
        "T_trimmed": T_trimmed,
        "N": N,
        "S": S,
        "block_size": T_trimmed // S,
        "n_removed": n_removed,
    }
    return M_trimmed, metadata


def validate_block_structure(
    blocks: list[np.ndarray],
    S: int,
    N: int,
    block_size: int,
) -> None:
    """Validate that S contiguous blocks have the expected dimensions.

    Parameters
    ----------
    blocks : list of np.ndarray
        List of S block matrices.
    S : int
        Expected number of blocks.
    N : int
        Expected number of columns in each block.
    block_size : int
        Expected number of rows in each block.

    Raises
    ------
    AssertionError
        If any block has incorrect shape or the list length differs from S.
    """
    assert len(blocks) == S, f"Expected {S} blocks, got {len(blocks)}"
    for i, block in enumerate(blocks):
        assert block.shape == (block_size, N), (
            f"Block {i} has shape {block.shape}, expected ({block_size}, {N})"
        )


def check_complementarity(
    is_indices: list[int],
    oos_indices: list[int],
    S: int,
) -> bool:
    """Check that IS and OOS block index sets partition {0, ..., S-1}.

    Parameters
    ----------
    is_indices : list of int
        Block indices assigned to the in-sample (IS) set.
    oos_indices : list of int
        Block indices assigned to the out-of-sample (OOS) set.
    S : int
        Total number of blocks.

    Returns
    -------
    bool
        True if is_indices and oos_indices are disjoint and together cover
        exactly {0, 1, ..., S-1}.
    """
    is_set = set(is_indices)
    oos_set = set(oos_indices)
    full_set = set(range(S))
    return (
        is_set.isdisjoint(oos_set)
        and is_set | oos_set == full_set
        and len(is_indices) == S // 2
        and len(oos_indices) == S // 2
    )


def partition_matrix_into_blocks(
    M: np.ndarray,
    S: int,
) -> list[np.ndarray]:
    """Partition a trimmed matrix into S equal contiguous row blocks.

    Parameters
    ----------
    M : np.ndarray
        T_trim×N matrix where T_trim is divisible by S.
    S : int
        Number of blocks.

    Returns
    -------
    list of np.ndarray
        List of S arrays each of shape (T_trim // S, N).

    Raises
    ------
    AssertionError
        If T_trim is not divisible by S.
    """
    T, N = M.shape
    assert T % S == 0, f"T={T} must be divisible by S={S}"
    block_size = T // S
    return [M[i * block_size : (i + 1) * block_size, :] for i in range(S)]
