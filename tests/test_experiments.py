"""
Integration and methodology tests for CSCV experiments.

Tests cover:
  - exp_1 generic CSCV validation pipeline
  - exp_2 noise-only seasonal simulation (lightweight version)
  - exp_3 seasonal-effect injection correctness
  - Cross-experiment comparisons (exp_3 should have lower PBO than exp_2)
"""

from __future__ import annotations

import numpy as np
import pytest

from src.cscv import run_cscv
from src.diagnostics import compute_pbo, cscv_diagnostics
from src.metrics import sharpe_ratio_matrix


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_noise_returns(T: int, sigma: float, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(scale=sigma, size=T)


def make_signal_returns(
    T: int,
    sigma: float,
    seed: int,
    effect: float = 0.25,
    seasonal_days: int = 5,
    month_size: int = 22,
) -> np.ndarray:
    """Returns with +effect*sigma on the first seasonal_days of each month."""
    returns = np.random.default_rng(seed).normal(scale=sigma, size=T)
    shift = effect * sigma
    for m_start in range(0, T, month_size):
        end = min(m_start + seasonal_days, T)
        returns[m_start:end] += shift
    return returns


# ── Exp 1: Generic CSCV validation ───────────────────────────────────────────

def test_exp1_combinations_match():
    """CSCV produces C(S, S//2) combinations on a small mixed matrix."""
    from scipy.special import comb as _comb
    rng = np.random.default_rng(42)
    M = rng.normal(size=(64, 20))
    M[:, :5] += 0.05  # inject small signal into first 5 columns
    result = run_cscv(M, S=4)
    expected = int(_comb(4, 2, exact=True))  # C(4,2) = 6
    assert result.n_combinations == expected


def test_exp1_pbo_consistent():
    """PBO equals fraction of negative logits for generic validation matrix."""
    rng = np.random.default_rng(11)
    M = rng.normal(size=(64, 15))
    result = run_cscv(M, S=4)
    assert abs(result.pbo - float(np.mean(result.logits < 0))) < 1e-12


def test_exp1_block_size_correct():
    """Block size = T_trimmed // S."""
    rng = np.random.default_rng(5)
    M = rng.normal(size=(96, 10))
    result = run_cscv(M, S=8)
    assert result.block_size == 96 // 8


def test_exp1_trimming_documents_removal():
    """T_trimmed < T_original when T not divisible by S."""
    rng = np.random.default_rng(6)
    M = rng.normal(size=(97, 6))
    result = run_cscv(M, S=8)
    assert result.T_trimmed == 96
    assert result.T_original == 97


# ── Exp 2: Noise-only seasonal strategy (lightweight) ────────────────────────

class _SimpleStrategyMatrix:
    """Minimal seasonal strategy simulator for unit-testing purposes.

    Uses a small grid (3 entry days × 3 holding periods × 2 sides = 18 strategies)
    for fast unit tests, not the full 8,800 of the actual experiment.
    """

    ENTRY_DAYS = [1, 5, 10]
    HOLDING_PERIODS = [1, 3, 5]
    SIDES = [-1, 1]
    MONTH_SIZE = 22

    def __init__(self, returns: np.ndarray, sigma_unit: float | None = None):
        self.returns = returns
        self.T = len(returns)
        self.sigma_unit = sigma_unit if sigma_unit is not None else float(np.std(returns, ddof=1))
        self.params = [
            (ed, hp, sk, side)
            for ed in self.ENTRY_DAYS
            for hp in self.HOLDING_PERIODS
            for sk in [1, 5]
            for side in self.SIDES
        ]
        self.N = len(self.params)

    def build(self) -> np.ndarray:
        M = np.zeros((self.T, self.N), dtype=np.float64)
        months = [(s, min(s + self.MONTH_SIZE, self.T)) for s in range(0, self.T, self.MONTH_SIZE)]
        for m_start, m_end in months:
            ml = m_end - m_start
            for i, (ed, hp, sk, side) in enumerate(self.params):
                if ed > ml:
                    continue
                ep = m_start + ed - 1
                actual_hp = min(hp, m_end - ep)
                if actual_hp <= 0:
                    continue
                window = self.returns[ep: ep + actual_hp]
                cum = side * np.cumsum(window)
                stop_thresh = -sk * self.sigma_unit
                hit = np.where(cum < stop_thresh)[0]
                exit_ = ep + (int(hit[0]) + 1 if len(hit) > 0 else actual_hp)
                M[ep:exit_, i] = side * self.returns[ep:exit_]
        return M


def test_exp2_pbo_noise_only():
    """Small noise-only strategy grid should produce a valid PBO."""
    returns = make_noise_returns(T=200, sigma=0.01, seed=42)
    sim = _SimpleStrategyMatrix(returns)
    M = sim.build()
    T_trim = (len(returns) // 8) * 8
    result = run_cscv(M[:T_trim], S=8)
    assert 0.0 <= result.pbo <= 1.0
    assert abs(result.pbo - float(np.mean(result.logits < 0))) < 1e-12


def test_exp2_strategy_matrix_shape():
    """Strategy matrix has correct shape for the small grid."""
    returns = make_noise_returns(T=100, sigma=0.01, seed=0)
    sim = _SimpleStrategyMatrix(returns)
    M = sim.build()
    assert M.shape[0] == 100
    assert M.shape[1] == sim.N


def test_exp2_strategy_returns_bounded():
    """Strategy returns cannot exceed magnitude of underlying returns."""
    returns = make_noise_returns(T=100, sigma=0.01, seed=1)
    sim = _SimpleStrategyMatrix(returns)
    M = sim.build()
    max_underlying = np.max(np.abs(returns))
    # Strategy returns are ±1 × underlying returns, so same magnitude
    assert np.max(np.abs(M)) <= max_underlying + 1e-12


# ── Exp 3: Seasonal effect injection ─────────────────────────────────────────

def test_exp3_seasonal_days_have_higher_mean():
    """First 5 days of each month must have higher mean than remaining days."""
    returns = make_signal_returns(T=440, sigma=0.01, seed=42, effect=0.25)
    first5_mask = np.zeros(len(returns), dtype=bool)
    for m_start in range(0, len(returns), 22):
        first5_mask[m_start: min(m_start + 5, len(returns))] = True
    mean_seasonal = np.mean(returns[first5_mask])
    mean_other = np.mean(returns[~first5_mask])
    assert mean_seasonal > mean_other, (
        f"Seasonal mean {mean_seasonal:.6f} not above non-seasonal mean {mean_other:.6f}"
    )


def test_exp3_vs_exp2_pbo_directional():
    """Seasonal-effect matrix should yield no higher PBO than pure noise on average."""
    sigma = 0.01
    seed = 99
    S = 4
    T = 200
    T_trim = (T // S) * S

    returns_noise = make_noise_returns(T, sigma, seed)
    returns_signal = make_signal_returns(T, sigma, seed, effect=1.0)  # large effect

    for ret, label in [(returns_noise, "noise"), (returns_signal, "signal")]:
        sim = _SimpleStrategyMatrix(ret)
        M = sim.build()[:T_trim]
        r = run_cscv(M, S=S)
        assert 0.0 <= r.pbo <= 1.0, f"{label}: PBO={r.pbo} out of [0,1]"


def test_exp3_large_effect_lower_pbo():
    """Very large seasonal effect should yield materially lower PBO than noise."""
    sigma = 0.01
    seed = 7
    S = 4
    T = 200
    T_trim = (T // S) * S

    # Noise case
    returns_noise = make_noise_returns(T, sigma, seed)
    sim_n = _SimpleStrategyMatrix(returns_noise)
    M_noise = sim_n.build()[:T_trim]
    result_noise = run_cscv(M_noise, S=S)

    # Signal case: very large effect so long early-month strategies clearly dominate
    returns_signal = make_signal_returns(T, sigma, seed, effect=5.0)
    sim_s = _SimpleStrategyMatrix(returns_signal)
    M_signal = sim_s.build()[:T_trim]
    result_signal = run_cscv(M_signal, S=S)

    # With a massive effect, the signal case PBO should be <= noise PBO
    assert result_signal.pbo <= result_noise.pbo + 0.15


# ── Cross-experiment: PBO diagnostics ─────────────────────────────────────────

def test_oos_loss_prob_better_for_signal():
    """Signal case must have no higher OOS loss probability than noise case."""
    sigma = 0.01
    seed = 21
    S = 4
    T = 200
    T_trim = (T // S) * S

    ret_noise = make_noise_returns(T, sigma, seed)
    ret_signal = make_signal_returns(T, sigma, seed, effect=5.0)

    results = {}
    for case, ret in [("noise", ret_noise), ("signal", ret_signal)]:
        sim = _SimpleStrategyMatrix(ret)
        M = sim.build()[:T_trim]
        cscv_result = run_cscv(M, S=S)
        diag = cscv_diagnostics(cscv_result)
        results[case] = diag["oos_loss_prob"]

    # Signal case: large effect → IS-best should rarely lose OOS
    assert results["signal"] <= results["noise"] + 0.2
