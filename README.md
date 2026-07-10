# CSCV / PBO — Probability of Backtest Overfitting

A complete, self-contained Python implementation of **Combinatorially Symmetric
Cross-Validation (CSCV)** to estimate the **Probability of Backtest Overfitting
(PBO)** as described in:

> Bailey, D. H., Borwein, J., Lopez de Prado, M., & Zhu, Q. J. (2014).
> *The Probability of Backtest Overfitting.*
> Journal of Computational Finance, 20(2), 39–70.

---

## Experiments

| ID | Title | Goal |
|----|-------|------|
| exp_1 | Generic CSCV Validation | Verify algorithm correctness on synthetic matrix |
| exp_2 | Noise-Only Seasonal Search | Demonstrate overfitting on pure-noise process |
| exp_3 | Engineered Seasonal Effect | Show CSCV distinguishes genuine edge from noise |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all experiments (≈ 5–15 min depending on hardware)
python exp/run_all.py

# Run tests only
pytest tests/ -v

# Run a single experiment
python exp/exp_1_cscv_validation.py
python exp/exp_2_noise_only.py
python exp/exp_3_seasonal_effect.py
```

Results (JSON metrics, plots, RESULTS.md) are saved in `results/`.

---

## Repository Structure

```
repo/
├── data/                  # Data utilities
├── src/                   # Core CSCV implementation
│   ├── data_validation.py # Matrix validation and trimming
│   ├── metrics.py         # Sharpe ratio and rank utilities
│   ├── cscv.py            # CSCV algorithm (run_cscv)
│   ├── diagnostics.py     # PBO, degradation, dominance diagnostics
│   └── plots.py           # Visualization functions
├── exp/                   # Experiment scripts
│   ├── exp_1_cscv_validation.py
│   ├── exp_2_noise_only.py
│   ├── exp_3_seasonal_effect.py
│   └── run_all.py         # Master runner
├── tests/                 # pytest unit and integration tests
├── results/               # Output plots, JSON, RESULTS.md
├── README.md
├── requirements.txt
└── .gitignore
```

---

## CSCV Algorithm Summary

1. Accept a numeric matrix **M** of shape T×N (T time periods, N strategies).
2. Choose an even partition count **S** (default 16).
3. Trim M to T_trim = floor(T/S) × S rows.
4. Partition M into S equal contiguous blocks M₁, …, Mₛ.
5. Enumerate all **C(S, S/2)** half-sample combinations.
6. For each combination c:
   - Form IS (in-sample) and OOS (out-of-sample) matrices from complementary block sets.
   - Compute the performance metric (Sharpe ratio) on each column.
   - Select the IS-best strategy n*.
   - Compute its relative OOS rank ω̄ₙ* = rank_oos(n*) / (N+1).
   - Compute the logit λ_c = ln(ω̄ / (1 - ω̄)).
7. **PBO** = fraction of combinations with λ_c < 0.

---

## Implementation Notes

- **Custom CSCV**: Context7 confirmed no library equivalent; implemented from scratch
  following Bailey et al. (2014).
- **Rank convention**: Best strategy → rank N (highest). Implemented via `scipy.stats.rankdata`.
- **Performance metric**: Sharpe = mean / std(ddof=1); optional annualization by √252.
- **Stop-loss**: End-of-day triggering on cumulative signed return < −k × σ_unit.
- **Calendar**: Stylized 22-trading-day months.
- **Dependencies**: `numpy`, `scipy`, `statsmodels`, `matplotlib`, `seaborn`, `pytest`.
