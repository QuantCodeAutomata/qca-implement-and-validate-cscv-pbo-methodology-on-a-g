"""
Unit tests for src/data_validation.py.

Validates:
  - Matrix validation and trimming
  - Block partitioning
  - Complementarity checks
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data_validation import (
    check_complementarity,
    partition_matrix_into_blocks,
    validate_block_structure,
    validate_performance_matrix,
)


# ── validate_performance_matrix ───────────────────────────────────────────────

def test_validate_no_trim_needed():
    """When T is divisible by S, no rows are removed."""
    M = np.random.default_rng(0).normal(size=(160, 10))
    M_trimmed, meta = validate_performance_matrix(M, S=16)
    assert M_trimmed.shape == (160, 10)
    assert meta["n_removed"] == 0
    assert meta["T_trimmed"] == 160


def test_validate_trim_required():
    """When T is not divisible by S, trailing rows are trimmed."""
    M = np.random.default_rng(1).normal(size=(100, 5))
    M_trimmed, meta = validate_performance_matrix(M, S=16)
    # floor(100 / 16) * 16 = 96
    assert meta["T_trimmed"] == 96
    assert meta["n_removed"] == 4
    assert M_trimmed.shape == (96, 5)
    # Verify chronological order preserved
    np.testing.assert_array_equal(M_trimmed, M[:96, :])


def test_validate_metadata_fields():
    """Metadata dictionary contains all expected keys with correct types."""
    M = np.ones((64, 8))
    _, meta = validate_performance_matrix(M, S=8)
    for key in ["T_original", "T_trimmed", "N", "S", "block_size", "n_removed"]:
        assert key in meta
    assert meta["block_size"] == 8
    assert meta["S"] == 8
    assert meta["N"] == 8


def test_validate_odd_S_raises():
    """Odd S raises AssertionError."""
    M = np.ones((100, 5))
    with pytest.raises(AssertionError):
        validate_performance_matrix(M, S=7)


def test_validate_S_too_large_raises():
    """S larger than T raises AssertionError."""
    M = np.ones((10, 5))
    with pytest.raises(AssertionError):
        validate_performance_matrix(M, S=12)


def test_validate_single_column_raises():
    """Single-column matrix raises AssertionError (need at least 2 strategies)."""
    M = np.ones((32, 1))
    with pytest.raises(AssertionError):
        validate_performance_matrix(M, S=4)


def test_validate_float_dtype():
    """Float64 input accepted without error."""
    M = np.random.default_rng(2).random((32, 4)).astype(np.float64)
    M_trim, _ = validate_performance_matrix(M, S=4)
    assert M_trim.shape == (32, 4)


def test_validate_integer_dtype():
    """Integer input accepted (int32)."""
    M = np.ones((32, 4), dtype=np.int32)
    M_trim, _ = validate_performance_matrix(M, S=4)
    assert M_trim.shape == (32, 4)


# ── partition_matrix_into_blocks ──────────────────────────────────────────────

def test_partition_equal_sizes():
    """All blocks must have equal row count."""
    M = np.arange(80 * 5).reshape(80, 5).astype(float)
    blocks = partition_matrix_into_blocks(M, S=8)
    assert len(blocks) == 8
    for b in blocks:
        assert b.shape == (10, 5)


def test_partition_chronological_order():
    """Blocks concatenated in order reconstruct the original matrix."""
    M = np.random.default_rng(3).normal(size=(64, 6))
    blocks = partition_matrix_into_blocks(M, S=8)
    reconstructed = np.concatenate(blocks, axis=0)
    np.testing.assert_array_equal(reconstructed, M)


def test_partition_non_divisible_raises():
    """Non-divisible T raises AssertionError."""
    M = np.ones((65, 5))
    with pytest.raises(AssertionError):
        partition_matrix_into_blocks(M, S=8)


# ── check_complementarity ─────────────────────────────────────────────────────

def test_complementarity_valid():
    """Correct IS/OOS partition returns True."""
    assert check_complementarity([0, 2, 4, 6], [1, 3, 5, 7], S=8) is True


def test_complementarity_invalid_overlap():
    """Overlapping IS and OOS returns False."""
    assert check_complementarity([0, 1, 2, 3], [2, 3, 4, 5], S=8) is False


def test_complementarity_incomplete():
    """IS + OOS not covering all blocks returns False."""
    assert check_complementarity([0, 1], [2, 3], S=8) is False


def test_complementarity_S2():
    """S=2 minimal case."""
    assert check_complementarity([0], [1], S=2) is True
    assert check_complementarity([1], [0], S=2) is True


# ── validate_block_structure ──────────────────────────────────────────────────

def test_validate_block_structure_ok():
    """Valid blocks pass without raising."""
    M = np.ones((48, 4))
    blocks = partition_matrix_into_blocks(M, S=6)
    validate_block_structure(blocks, S=6, N=4, block_size=8)  # no exception


def test_validate_block_structure_wrong_count():
    """Wrong block count raises AssertionError."""
    blocks = [np.ones((8, 4))] * 5
    with pytest.raises(AssertionError):
        validate_block_structure(blocks, S=6, N=4, block_size=8)


def test_validate_block_structure_wrong_shape():
    """Block with wrong shape raises AssertionError."""
    blocks = [np.ones((8, 4))] * 5 + [np.ones((9, 4))]
    with pytest.raises(AssertionError):
        validate_block_structure(blocks, S=6, N=4, block_size=8)
