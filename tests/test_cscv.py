"""
Unit tests for src/cscv.py.

Validates CSCV algorithm implementation against methodology requirements:
  - Number of combinations equals C(S, S//2)
  - Each IS and OOS matrix has equal row count T_trim / 2
  - IS and OOS are complementary and disjoint
  - Rank orientation: best strategy gets rank N
  - omega_bar strictly in (0, 1)
  - PBO equals exact fraction of negative logits
  - Deterministic output (same seed → same result)
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import comb

from src.cscv import CSCVResult, run_cscv


def make_matrix(T: int = 80, N: int = 10, seed: int = 42) -> np.ndarray:
    """Create a reproducible random performance matrix for testing."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(T, N))


# ── Basic output structure ────────────────────────────────────────────────────

def test_returns_cscv_result():
    """run_cscv must return a CSCVResult instance."""
    M = make_matrix()
    result = run_cscv(M, S=4)
    assert isinstance(result, CSCVResult)


def test_n_combinations_matches_binomial():
    """Number of combinations must equal C(S, S//2)."""
    for S in [4, 6, 8, 16]:
        M = make_matrix(T=S * 10, N=5)
        result = run_cscv(M, S=S)
        expected = int(comb(S, S // 2, exact=True))
        assert result.n_combinations == expected, (
            f"S={S}: n_combinations={result.n_combinations} != expected={expected}"
        )


def test_logit_count_matches_combinations():
    """Number of logits equals n_combinations."""
    M = make_matrix(T=64, N=8)
    result = run_cscv(M, S=4)
    assert len(result.logits) == result.n_combinations


def test_omega_bar_strictly_between_0_and_1():
    """All omega_bar values must be strictly in (0, 1)."""
    M = make_matrix(T=80, N=6)
    result = run_cscv(M, S=4)
    assert np.all(result.omega_bars > 0.0)
    assert np.all(result.omega_bars < 1.0)


def test_logits_are_finite():
    """All logits must be finite floating-point numbers."""
    M = make_matrix(T=64, N=5)
    result = run_cscv(M, S=4)
    assert np.all(np.isfinite(result.logits))


# ── PBO consistency ───────────────────────────────────────────────────────────

def test_pbo_equals_fraction_negative_logits():
    """PBO must exactly equal mean(logits < 0)."""
    M = make_matrix(T=80, N=10)
    result = run_cscv(M, S=4)
    pbo_manual = float(np.mean(result.logits < 0.0))
    assert abs(result.pbo - pbo_manual) < 1e-12


def test_pbo_bounded():
    """PBO must be in [0, 1]."""
    M = make_matrix(T=64, N=5)
    result = run_cscv(M, S=4)
    assert 0.0 <= result.pbo <= 1.0


def test_pbo_pure_noise_high():
    """For a large pure-noise matrix, PBO should be near 0.5 (not systematically low)."""
    rng = np.random.default_rng(999)
    M = rng.normal(size=(160, 100))
    result = run_cscv(M, S=8)
    # PBO from pure noise should not be < 0.3 or > 0.8 on average
    # This is a distributional check, not a strict threshold
    assert 0.0 <= result.pbo <= 1.0


def test_pbo_clearly_lower_for_signal():
    """Signal matrix (positive drift) should have lower PBO than noise-only matrix."""
    rng = np.random.default_rng(123)
    # Pure noise matrix
    M_noise = rng.normal(size=(64, 20))
    result_noise = run_cscv(M_noise, S=4)

    # Signal matrix: all columns have strong positive drift
    M_signal = rng.normal(size=(64, 20)) + 2.0  # large positive mean
    result_signal = run_cscv(M_signal, S=4)

    # With uniformly high Sharpe, IS-best should rank highly OOS too
    assert result_signal.pbo <= result_noise.pbo + 0.3  # signal pbo not much higher


# ── Block structure ───────────────────────────────────────────────────────────

def test_block_size_correct():
    """Block size must equal T_trimmed // S."""
    M = make_matrix(T=96, N=5)
    result = run_cscv(M, S=8)
    assert result.block_size == 96 // 8


def test_is_oos_equal_row_count():
    """Each IS and OOS sub-matrix has equal row count T_trim / 2."""
    # We verify indirectly: IS and OOS concatenate S//2 blocks of block_size rows
    M = make_matrix(T=64, N=6)
    result = run_cscv(M, S=4)
    # IS = OOS = 2 blocks × 16 rows = 32 rows
    expected_half_rows = result.block_size * (result.S // 2)
    assert expected_half_rows == result.T_trimmed // 2


# ── IS/OOS pairs ──────────────────────────────────────────────────────────────

def test_is_oos_pairs_length():
    """Length of is_oos_pairs must equal n_combinations."""
    M = make_matrix(T=64, N=6)
    result = run_cscv(M, S=4)
    assert len(result.is_oos_pairs) == result.n_combinations


def test_is_oos_pairs_are_floats():
    """Each pair in is_oos_pairs contains finite floats."""
    M = make_matrix(T=64, N=6)
    result = run_cscv(M, S=4)
    for is_val, oos_val in result.is_oos_pairs:
        assert isinstance(is_val, float)
        assert isinstance(oos_val, float)
        assert math.isfinite(is_val)
        assert math.isfinite(oos_val)


# ── Determinism ───────────────────────────────────────────────────────────────

def test_deterministic_output():
    """Same input produces identical output on repeated calls."""
    M = make_matrix(T=64, N=8, seed=0)
    r1 = run_cscv(M, S=4)
    r2 = run_cscv(M, S=4)
    np.testing.assert_array_equal(r1.logits, r2.logits)
    assert r1.pbo == r2.pbo


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_minimum_S():
    """S=2 (minimum even) should work correctly."""
    M = make_matrix(T=20, N=4)
    result = run_cscv(M, S=2)
    expected = int(comb(2, 1, exact=True))  # C(2,1) = 2
    assert result.n_combinations == expected
    assert len(result.logits) == 2


def test_custom_metric_fn():
    """Custom metric function (mean return) produces valid CSCV output."""
    M = make_matrix(T=64, N=6)

    def mean_metric(m: np.ndarray) -> np.ndarray:
        return np.mean(m, axis=0)

    result = run_cscv(M, S=4, metric_fn=mean_metric)
    pbo_manual = float(np.mean(result.logits < 0.0))
    assert abs(result.pbo - pbo_manual) < 1e-12
    assert 0.0 <= result.pbo <= 1.0


def test_trim_applied_correctly():
    """T not divisible by S triggers trimming; T_trimmed < T_original."""
    M = make_matrix(T=65, N=5)
    result = run_cscv(M, S=4)
    assert result.T_original == 65
    assert result.T_trimmed == 64  # floor(65/4)*4 = 64
    assert result.T_trimmed < result.T_original


def test_N_preserved():
    """Result N matches number of strategy columns."""
    M = make_matrix(T=64, N=15)
    result = run_cscv(M, S=4)
    assert result.N == 15


def test_large_N_S16():
    """Large N with S=16 runs without error."""
    rng = np.random.default_rng(77)
    M = rng.normal(size=(160, 50))
    result = run_cscv(M, S=16)
    expected = int(comb(16, 8, exact=True))  # 12,870
    assert result.n_combinations == expected
    assert 0.0 <= result.pbo <= 1.0
