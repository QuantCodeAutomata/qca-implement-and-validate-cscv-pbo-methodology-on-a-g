"""
Unit tests for src/metrics.py.

Validates:
  - Sharpe ratio computation (single series and matrix)
  - Rank orientation: best strategy gets rank N
  - Edge cases: empty, single observation, zero variance
"""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics import rank_performance, sharpe_ratio, sharpe_ratio_matrix


# ── sharpe_ratio ─────────────────────────────────────────────────────────────

def test_sharpe_ratio_positive():
    """Positive mean / positive std → positive Sharpe."""
    r = np.array([0.01, 0.02, 0.015, 0.018, 0.012])
    sr = sharpe_ratio(r)
    assert sr > 0.0


def test_sharpe_ratio_zero_variance():
    """Constant series → zero Sharpe (no ZeroDivision)."""
    r = np.full(10, 0.01)
    assert sharpe_ratio(r) == 0.0


def test_sharpe_ratio_single_observation():
    """Single observation → zero Sharpe (insufficient data)."""
    assert sharpe_ratio(np.array([0.05])) == 0.0


def test_sharpe_ratio_empty():
    """Empty array → zero Sharpe."""
    assert sharpe_ratio(np.array([])) == 0.0


def test_sharpe_ratio_known_value():
    """Known series: mean=1, std=1 (ddof=1), Sharpe=1."""
    r = np.array([0.0, 1.0, 2.0])  # mean=1, var=1, std=1
    sr = sharpe_ratio(r, ddof=1, annualize=False)
    assert abs(sr - 1.0) < 1e-10


def test_sharpe_ratio_annualized():
    """Annualization multiplies by sqrt(periods_per_year)."""
    r = np.array([0.0, 1.0, 2.0])
    sr_raw = sharpe_ratio(r, annualize=False)
    sr_ann = sharpe_ratio(r, annualize=True, periods_per_year=252)
    assert abs(sr_ann - sr_raw * np.sqrt(252)) < 1e-10


def test_sharpe_ratio_negative_mean():
    """Negative mean → negative Sharpe."""
    r = np.array([-0.01, -0.02, -0.015])
    assert sharpe_ratio(r) < 0.0


# ── sharpe_ratio_matrix ───────────────────────────────────────────────────────

def test_sharpe_ratio_matrix_shape():
    """Output shape matches number of columns."""
    M = np.random.default_rng(0).normal(size=(100, 10))
    out = sharpe_ratio_matrix(M)
    assert out.shape == (10,)


def test_sharpe_ratio_matrix_values():
    """Column-wise Sharpe matches individually computed values."""
    rng = np.random.default_rng(42)
    M = rng.normal(size=(50, 5))
    batch = sharpe_ratio_matrix(M, ddof=1)
    for j in range(5):
        expected = sharpe_ratio(M[:, j], ddof=1)
        assert abs(batch[j] - expected) < 1e-12


def test_sharpe_ratio_matrix_zero_variance_column():
    """Zero-variance column → Sharpe = 0, no exception."""
    M = np.ones((20, 3))
    M[:, 1] = np.random.default_rng(0).normal(size=20)
    out = sharpe_ratio_matrix(M)
    assert out[0] == 0.0
    assert out[2] == 0.0
    assert out[1] != 0.0


def test_sharpe_ratio_matrix_too_few_rows():
    """Only 1 row → all zeros (insufficient data)."""
    M = np.array([[1.0, 2.0, 3.0]])
    out = sharpe_ratio_matrix(M)
    np.testing.assert_array_equal(out, np.zeros(3))


# ── rank_performance ─────────────────────────────────────────────────────────

def test_rank_performance_basic():
    """Example from methodology: (0.5, 1.1, 0.7) → (1, 3, 2)."""
    perf = np.array([0.5, 1.1, 0.7])
    ranks = rank_performance(perf)
    np.testing.assert_array_equal(ranks, [1.0, 3.0, 2.0])


def test_rank_performance_best_is_N():
    """Best strategy always receives rank N."""
    rng = np.random.default_rng(7)
    perf = rng.normal(size=50)
    ranks = rank_performance(perf)
    assert int(np.max(ranks)) == 50
    assert ranks[int(np.argmax(perf))] == 50.0


def test_rank_performance_worst_is_1():
    """Worst strategy always receives rank 1."""
    rng = np.random.default_rng(8)
    perf = rng.normal(size=30)
    ranks = rank_performance(perf)
    assert ranks[int(np.argmin(perf))] == 1.0


def test_rank_performance_ties_average():
    """Tied values get average rank (deterministic)."""
    perf = np.array([1.0, 1.0, 2.0])  # two-way tie for positions 1 and 2
    ranks = rank_performance(perf)
    # The two tied entries should share average rank 1.5
    assert ranks[0] == 1.5
    assert ranks[1] == 1.5
    assert ranks[2] == 3.0


def test_rank_performance_2d_raises():
    """2-D input raises AssertionError."""
    with pytest.raises(AssertionError):
        rank_performance(np.ones((3, 3)))


def test_rank_performance_single():
    """Single element gets rank 1."""
    assert rank_performance(np.array([42.0]))[0] == 1.0
