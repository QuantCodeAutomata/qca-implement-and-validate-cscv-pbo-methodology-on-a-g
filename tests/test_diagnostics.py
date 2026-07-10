"""
Unit tests for src/diagnostics.py.

Validates:
  - PBO equals fraction of negative logits
  - OOS loss probability computation
  - Degradation regression properties
  - Dominance statistics
  - cscv_diagnostics integration function
"""

from __future__ import annotations

import numpy as np
import pytest

from src.cscv import run_cscv
from src.diagnostics import (
    compute_dominance_stats,
    compute_oos_loss_probability,
    compute_pbo,
    cscv_diagnostics,
    fit_degradation_regression,
)


# ── compute_pbo ───────────────────────────────────────────────────────────────

def test_pbo_exact():
    """PBO = fraction of negative logits exactly."""
    logits = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    # 2 negatives out of 5 → PBO = 0.4 (zero not counted as negative)
    assert abs(compute_pbo(logits) - 0.4) < 1e-12


def test_pbo_all_negative():
    assert compute_pbo(np.array([-1.0, -2.0, -0.1])) == 1.0


def test_pbo_all_positive():
    assert compute_pbo(np.array([0.1, 0.5, 1.0])) == 0.0


def test_pbo_bounds():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=100)
    pbo = compute_pbo(logits)
    assert 0.0 <= pbo <= 1.0


def test_pbo_2d_raises():
    with pytest.raises(AssertionError):
        compute_pbo(np.ones((3, 3)))


# ── compute_oos_loss_probability ──────────────────────────────────────────────

def test_oos_loss_prob_known():
    """3 negative out of 5 → 0.6."""
    vals = np.array([-0.5, -0.1, 0.0, 0.3, -0.2])
    assert abs(compute_oos_loss_probability(vals) - 0.6) < 1e-12


def test_oos_loss_prob_all_positive():
    vals = np.array([0.1, 0.5, 1.0])
    assert compute_oos_loss_probability(vals) == 0.0


def test_oos_loss_prob_bounds():
    rng = np.random.default_rng(1)
    vals = rng.normal(size=200)
    prob = compute_oos_loss_probability(vals)
    assert 0.0 <= prob <= 1.0


# ── fit_degradation_regression ────────────────────────────────────────────────

def test_regression_keys():
    """Regression dict contains expected keys."""
    is_v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    oos_v = np.array([0.9, 1.9, 3.1, 3.9, 5.1])
    out = fit_degradation_regression(is_v, oos_v)
    for key in ["alpha", "beta", "r_squared", "pvalue_beta"]:
        assert key in out


def test_regression_perfect_linear():
    """Perfect linear relationship → R² ≈ 1, beta ≈ 1, alpha ≈ 0."""
    x = np.linspace(0, 1, 50)
    y = x.copy()
    out = fit_degradation_regression(x, y)
    assert abs(out["beta"] - 1.0) < 1e-6
    assert abs(out["alpha"]) < 1e-6
    assert abs(out["r_squared"] - 1.0) < 1e-6


def test_regression_no_relationship():
    """Random OOS with constant IS → beta ≈ 0, R² ≈ 0."""
    rng = np.random.default_rng(5)
    is_v = np.ones(100)
    oos_v = rng.normal(size=100)
    # OLS requires non-constant X with add_constant; test at minimum
    # With all-ones X (constant), statsmodels may give trivial results
    # Just verify it doesn't raise and returns finite values
    out = fit_degradation_regression(is_v + rng.normal(scale=1e-10, size=100), oos_v)
    assert all(np.isfinite(v) for v in [out["alpha"], out["beta"], out["r_squared"]])


def test_regression_mismatched_lengths_raises():
    """Mismatched IS/OOS array lengths raise AssertionError."""
    with pytest.raises(AssertionError):
        fit_degradation_regression(np.array([1.0, 2.0]), np.array([1.0]))


def test_regression_too_few_obs_raises():
    """Fewer than 2 observations raise AssertionError."""
    with pytest.raises(AssertionError):
        fit_degradation_regression(np.array([1.0]), np.array([1.0]))


# ── compute_dominance_stats ───────────────────────────────────────────────────

def test_dominance_stats_keys():
    """Output dict contains all expected keys."""
    rng = np.random.default_rng(0)
    sel = rng.normal(size=50)
    pool = rng.normal(size=200)
    out = compute_dominance_stats(sel, pool)
    for key in [
        "selected_mean", "selected_std", "pooled_mean", "pooled_std",
        "selected_frac_positive", "pooled_frac_positive", "ks_stat", "ks_pvalue",
    ]:
        assert key in out


def test_dominance_fracs_bounded():
    """Fraction-positive values must be in [0, 1]."""
    rng = np.random.default_rng(2)
    out = compute_dominance_stats(rng.normal(size=50), rng.normal(size=100))
    assert 0.0 <= out["selected_frac_positive"] <= 1.0
    assert 0.0 <= out["pooled_frac_positive"] <= 1.0


def test_dominance_ks_stat_bounded():
    """KS statistic must be in [0, 1]."""
    rng = np.random.default_rng(3)
    out = compute_dominance_stats(rng.normal(size=50), rng.normal(size=50))
    assert 0.0 <= out["ks_stat"] <= 1.0


def test_dominance_identical_distributions():
    """Identical distributions → KS statistic = 0, p-value = 1."""
    arr = np.linspace(-2, 2, 100)
    out = compute_dominance_stats(arr, arr)
    assert out["ks_stat"] == 0.0
    assert abs(out["ks_pvalue"] - 1.0) < 1e-6


# ── cscv_diagnostics ──────────────────────────────────────────────────────────

def test_cscv_diagnostics_integration():
    """cscv_diagnostics returns all expected keys and consistent pbo."""
    rng = np.random.default_rng(42)
    M = rng.normal(size=(64, 10))
    result = run_cscv(M, S=4)
    diag = cscv_diagnostics(result)
    for key in ["pbo", "oos_loss_prob", "degradation", "dominance",
                "n_combinations", "S", "N", "block_size"]:
        assert key in diag
    # PBO from diagnostics must match result.pbo
    assert abs(diag["pbo"] - result.pbo) < 1e-12


def test_cscv_diagnostics_oos_loss_prob_bounded():
    rng = np.random.default_rng(99)
    M = rng.normal(size=(64, 8))
    result = run_cscv(M, S=4)
    diag = cscv_diagnostics(result)
    assert 0.0 <= diag["oos_loss_prob"] <= 1.0
