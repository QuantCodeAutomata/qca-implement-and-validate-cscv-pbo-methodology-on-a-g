"""
Master experiment runner: executes all three CSCV/PBO experiments in sequence
and writes a consolidated RESULTS.md summary.

Usage:
    python exp/run_all.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_all() -> None:
    from exp.exp_1_cscv_validation import run_exp1
    from exp.exp_2_noise_only import run_exp2
    from exp.exp_3_seasonal_effect import run_exp3

    all_results: dict[str, dict] = {}

    print("\n" + "=" * 70)
    print("Running Experiment 1: Generic CSCV Validation")
    print("=" * 70)
    t0 = time.time()
    all_results["exp_1"] = run_exp1()
    print(f"  Elapsed: {time.time() - t0:.1f}s")

    print("\n" + "=" * 70)
    print("Running Experiment 2: Noise-Only Seasonal Strategy Search")
    print("=" * 70)
    t0 = time.time()
    all_results["exp_2"] = run_exp2()
    print(f"  Elapsed: {time.time() - t0:.1f}s")

    print("\n" + "=" * 70)
    print("Running Experiment 3: Engineered Seasonal Effect")
    print("=" * 70)
    t0 = time.time()
    all_results["exp_3"] = run_exp3()
    print(f"  Elapsed: {time.time() - t0:.1f}s")

    # Save combined JSON
    combined_path = RESULTS_DIR / "all_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\n  All results saved to {combined_path}")

    # Write RESULTS.md
    write_results_md(all_results)
    print(f"  Summary saved to {RESULTS_DIR / 'RESULTS.md'}")


def write_results_md(results: dict[str, dict]) -> None:
    """Write a Markdown summary of all experiment results."""
    e1 = results.get("exp_1", {})
    e2 = results.get("exp_2", {})
    e3 = results.get("exp_3", {})

    def _fmt(v, fmt=".4f"):
        if v is None:
            return "N/A"
        try:
            return format(float(v), fmt)
        except (TypeError, ValueError):
            return str(v)

    lines = [
        "# RESULTS — CSCV/PBO Methodology Implementation",
        "",
        "## Overview",
        "",
        "This repository implements and validates the **Combinatorially Symmetric",
        "Cross-Validation (CSCV)** algorithm for estimating the **Probability of",
        "Backtest Overfitting (PBO)** as described in:",
        "",
        "> Bailey, D. H., Borwein, J., Lopez de Prado, M., & Zhu, Q. J. (2014).",
        "> *The Probability of Backtest Overfitting.*",
        "> Journal of Computational Finance, 20(2), 39–70.",
        "",
        "Three experiments are run on synthetic data to validate correctness and",
        "reproduce the paper's qualitative demonstrations.",
        "",
        "---",
        "",
        "## Experiment 1 — Generic CSCV Validation",
        "",
        "**Purpose**: Verify the mathematical and computational correctness of the",
        "CSCV pipeline on a synthetic T×N matrix with known structure (N/4 signal",
        "columns + pure noise).",
        "",
        "| Metric | Value |",
        "| ------ | ----- |",
        f"| T (original) | {e1.get('T_original', 'N/A')} |",
        f"| T (trimmed)  | {e1.get('T_trimmed', 'N/A')} |",
        f"| N strategies | {e1.get('N', 'N/A')} |",
        f"| S (blocks)   | {e1.get('S', 'N/A')} |",
        f"| Block size   | {e1.get('block_size', 'N/A')} |",
        f"| n_combinations | {e1.get('n_combinations', 'N/A')} |",
        f"| C(S,S/2) expected | {e1.get('expected_n_combinations', 'N/A')} |",
        f"| Combinations match? | {e1.get('combinations_match', 'N/A')} |",
        f"| PBO = frac(λ<0)? | {e1.get('pbo_equals_neg_logit_frac', 'N/A')} |",
        f"| PBO | {_fmt(e1.get('pbo'))} |",
        f"| OOS loss probability | {_fmt(e1.get('oos_loss_prob'))} |",
        f"| Degradation α | {_fmt(e1.get('degradation', {}).get('alpha'))} |",
        f"| Degradation β | {_fmt(e1.get('degradation', {}).get('beta'))} |",
        f"| Degradation R² | {_fmt(e1.get('degradation', {}).get('r_squared'))} |",
        "",
        "**Internal consistency checks passed**:",
        f"- `combinations_match = {e1.get('combinations_match', '?')}`",
        f"- `pbo_equals_neg_logit_frac = {e1.get('pbo_equals_neg_logit_frac', '?')}`",
        "",
        "**Plots**:",
        "- `results/exp1_logit_histogram.png`",
        "- `results/exp1_degradation_scatter.png`",
        "- `results/exp1_oos_cdf.png`",
        "",
        "---",
        "",
        "## Experiment 2 — Noise-Only Seasonal Strategy Search",
        "",
        "**Purpose**: Reproduce the paper's overfitting demonstration: optimize 8,800",
        "seasonal strategies over a pure-noise random walk and show that CSCV detects",
        "the overfitting despite an attractive full-sample Sharpe ratio.",
        "",
        "**Data**: Synthetic Gaussian i.i.d. returns, T=1,000, σ=" + str(e2.get("sigma", "N/A")) + ".",
        "Trimmed to 992 rows for CSCV with S=16.",
        "",
        "| Metric | Value |",
        "| ------ | ----- |",
        f"| T (original) | {e2.get('T_original', 'N/A')} |",
        f"| T (trimmed)  | {e2.get('T_trimmed', 'N/A')} |",
        f"| N strategies | {e2.get('N_strategies', 'N/A')} |",
        f"| S (blocks)   | {e2.get('S', 'N/A')} |",
        f"| n_combinations | {e2.get('n_combinations', 'N/A')} |",
        f"| PBO | {_fmt(e2.get('pbo'))} |",
        f"| OOS loss probability | {_fmt(e2.get('oos_loss_prob'))} |",
        f"| Best strategy ann. Sharpe | {_fmt(e2.get('best_strategy', {}).get('annualized_sharpe'))} |",
        f"| Best strategy entry_day | {e2.get('best_strategy', {}).get('entry_day', 'N/A')} |",
        f"| Best strategy holding_period | {e2.get('best_strategy', {}).get('holding_period', 'N/A')} |",
        f"| Best strategy stop_k | {e2.get('best_strategy', {}).get('stop_k', 'N/A')} |",
        f"| Best strategy side | {e2.get('best_strategy', {}).get('side', 'N/A')} |",
        f"| Degradation β | {_fmt(e2.get('degradation', {}).get('beta'))} |",
        f"| Degradation R² | {_fmt(e2.get('degradation', {}).get('r_squared'))} |",
        "",
        "**Paper reference**: PBO ≈ 55%, OOS loss prob ≈ 53%, ann. Sharpe ≈ 1.27.",
        "",
        "**Plots**:",
        "- `results/exp2_cumulative_performance.png`",
        "- `results/exp2_logit_histogram.png`",
        "- `results/exp2_degradation_scatter.png`",
        "- `results/exp2_oos_cdf.png`",
        "",
        "---",
        "",
        "## Experiment 3 — Engineered Monthly Seasonal Effect",
        "",
        "**Purpose**: Reproduce the paper's true-signal demonstration by injecting a",
        "+0.25σ daily return boost on the first 5 days of each 22-day stylized month",
        "and showing that CSCV assigns materially lower PBO than in the noise-only case.",
        "",
        "**Data**: Same base distribution as Exp 2 with +0.25σ effect on seasonal days.",
        "",
        "| Metric | Value |",
        "| ------ | ----- |",
        f"| T (original) | {e3.get('T_original', 'N/A')} |",
        f"| T (trimmed)  | {e3.get('T_trimmed', 'N/A')} |",
        f"| N strategies | {e3.get('N_strategies', 'N/A')} |",
        f"| S (blocks)   | {e3.get('S', 'N/A')} |",
        f"| n_combinations | {e3.get('n_combinations', 'N/A')} |",
        f"| Seasonal effect magnitude | {e3.get('seasonal_effect_magnitude', 'N/A')}σ |",
        f"| Seasonal days per month | {e3.get('seasonal_days', 'N/A')} |",
        f"| PBO | {_fmt(e3.get('pbo'))} |",
        f"| OOS loss probability | {_fmt(e3.get('oos_loss_prob'))} |",
        f"| Best strategy ann. Sharpe | {_fmt(e3.get('best_strategy', {}).get('annualized_sharpe'))} |",
        f"| Best strategy entry_day | {e3.get('best_strategy', {}).get('entry_day', 'N/A')} |",
        f"| Best strategy holding_period | {e3.get('best_strategy', {}).get('holding_period', 'N/A')} |",
        f"| Best strategy stop_k | {e3.get('best_strategy', {}).get('stop_k', 'N/A')} |",
        f"| Best strategy side | {e3.get('best_strategy', {}).get('side', 'N/A')} |",
        f"| Degradation β | {_fmt(e3.get('degradation', {}).get('beta'))} |",
        f"| Degradation R² | {_fmt(e3.get('degradation', {}).get('r_squared'))} |",
        "",
        "**Paper reference**: PBO ≈ 13%, OOS loss prob ≈ 13%, ann. Sharpe ≈ 1.54.",
        "",
        "**Plots**:",
        "- `results/exp3_cumulative_performance.png`",
        "- `results/exp3_logit_histogram.png`",
        "- `results/exp3_degradation_scatter.png`",
        "- `results/exp3_oos_cdf.png`",
        "",
        "---",
        "",
        "## Comparison: Noise-Only vs Seasonal-Effect",
        "",
        "| Metric | Exp 2 (Noise-Only) | Exp 3 (Seasonal) | Direction |",
        "| ------ | ------------------ | ----------------- | --------- |",
        f"| PBO | {_fmt(e2.get('pbo'))} | {_fmt(e3.get('pbo'))} |"
        f" {'Exp3 < Exp2 ✓' if e3.get('pbo', 1.0) < e2.get('pbo', 0.0) else 'Exp3 >= Exp2'} |",
        f"| OOS loss prob | {_fmt(e2.get('oos_loss_prob'))} | {_fmt(e3.get('oos_loss_prob'))} |"
        f" {'Exp3 < Exp2 ✓' if e3.get('oos_loss_prob', 1.0) < e2.get('oos_loss_prob', 0.0) else 'Exp3 >= Exp2'} |",
        f"| Best ann. Sharpe | {_fmt(e2.get('best_strategy', {}).get('annualized_sharpe'))} |"
        f" {_fmt(e3.get('best_strategy', {}).get('annualized_sharpe'))} | — |",
        "",
        "---",
        "",
        "## Implementation Notes",
        "",
        "- **CSCV algorithm**: Custom implementation (Context7 confirmed no library equivalent)",
        "  following Bailey et al. (2014) exactly. Uses `itertools.combinations(range(S), S//2)`",
        "  for exhaustive enumeration of half-sample splits.",
        "- **Performance metric**: Sharpe ratio = mean(r) / std(r, ddof=1), optionally",
        "  annualized by √252. Annualization does not affect CSCV ranks when applied",
        "  identically to IS and OOS.",
        "- **Rank convention**: Best strategy receives rank N (highest), worst receives",
        "  rank 1 (lowest), using scipy `rankdata` with average tie-breaking.",
        "- **Stop-loss convention**: End-of-day triggering based on cumulative signed",
        "  returns. Stop triggered when `side × cumsum(returns) < -k × σ_unit`.",
        "- **Calendar**: Stylized 22-business-day months; no exchange calendar required.",
        "- **Trimming rule**: `T_trim = floor(T / S) × S`, preserving chronological order.",
        "- **Seed**: 42 for all synthetic data generation.",
        "",
        "---",
        "",
        "## Repository Structure",
        "",
        "```",
        "repo/",
        "├── data/             # Data utilities (synthetic generation)",
        "├── src/              # Core CSCV implementation",
        "│   ├── data_validation.py",
        "│   ├── metrics.py",
        "│   ├── cscv.py",
        "│   ├── diagnostics.py",
        "│   └── plots.py",
        "├── exp/              # Experiment scripts",
        "│   ├── exp_1_cscv_validation.py",
        "│   ├── exp_2_noise_only.py",
        "│   ├── exp_3_seasonal_effect.py",
        "│   └── run_all.py",
        "├── tests/            # pytest unit and integration tests",
        "│   ├── test_metrics.py",
        "│   ├── test_data_validation.py",
        "│   ├── test_cscv.py",
        "│   ├── test_diagnostics.py",
        "│   └── test_experiments.py",
        "├── results/          # Output plots, JSON metrics, RESULTS.md",
        "├── README.md",
        "├── requirements.txt",
        "└── .gitignore",
        "```",
    ]

    md_path = RESULTS_DIR / "RESULTS.md"
    md_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    run_all()
