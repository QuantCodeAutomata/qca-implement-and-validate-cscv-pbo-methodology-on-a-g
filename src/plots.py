"""
Plotting Module for CSCV/PBO Methodology.

Produces publication-quality diagnostic figures from CSCV results:
  - Logit histogram with PBO annotation
  - IS vs OOS degradation scatter with regression line
  - CDF comparison of selected OOS vs pooled OOS
  - Cumulative strategy performance chart
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for reproducible saves
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.cscv import CSCVResult

sns.set_theme(style="whitegrid", palette="muted")


def plot_logit_histogram(
    result: CSCVResult,
    save_path: str | Path | None = None,
    title: str = "CSCV Logit Distribution",
) -> plt.Figure:
    """Plot histogram of logit-transformed relative OOS ranks with PBO annotation.

    Parameters
    ----------
    result : CSCVResult
        Output from ``run_cscv``.
    save_path : str or Path, optional
        If provided, save the figure to this path.
    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(result.logits, bins=40, edgecolor="white", alpha=0.85, color="steelblue",
            label="Logit distribution")
    ax.axvline(x=0.0, color="crimson", linestyle="--", linewidth=2, label="λ = 0")
    ax.set_xlabel("Logit of relative OOS rank (λ)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"{title}\nPBO = {result.pbo:.3f}", fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_degradation_scatter(
    result: CSCVResult,
    degradation: dict[str, Any],
    oos_loss_prob: float,
    save_path: str | Path | None = None,
    title: str = "IS vs OOS Performance Degradation",
    metric_label: str = "Sharpe Ratio",
) -> plt.Figure:
    """Plot IS vs OOS Sharpe scatter with OLS regression line.

    Parameters
    ----------
    result : CSCVResult
        Output from ``run_cscv``.
    degradation : dict
        Dictionary from ``fit_degradation_regression``.
    oos_loss_prob : float
        Fraction of splits with negative OOS performance.
    save_path : str or Path, optional
        If provided, save the figure to this path.
    title : str, optional
        Figure title.
    metric_label : str, optional
        Label for axes (e.g., 'Sharpe Ratio').

    Returns
    -------
    matplotlib.figure.Figure
    """
    is_vals = np.array([p[0] for p in result.is_oos_pairs])
    oos_vals = np.array([p[1] for p in result.is_oos_pairs])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(is_vals, oos_vals, alpha=0.3, s=8, color="steelblue", label="CSCV splits")

    # Regression line
    x_range = np.linspace(is_vals.min(), is_vals.max(), 200)
    y_fit = degradation["alpha"] + degradation["beta"] * x_range
    ax.plot(x_range, y_fit, color="crimson", linewidth=2,
            label=f"OLS: α={degradation['alpha']:.3f}, β={degradation['beta']:.3f}, "
                  f"R²={degradation['r_squared']:.3f}")

    ax.axhline(y=0.0, color="gray", linestyle=":", linewidth=1.5)
    ax.set_xlabel(f"IS {metric_label}", fontsize=12)
    ax.set_ylabel(f"OOS {metric_label}", fontsize=12)
    ax.set_title(
        f"{title}\nOOS Loss Prob = {oos_loss_prob:.3f}", fontsize=13
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_oos_cdf_comparison(
    selected_oos: np.ndarray,
    pooled_oos: np.ndarray,
    save_path: str | Path | None = None,
    title: str = "OOS Performance: Selected vs Pooled CDF",
    metric_label: str = "Sharpe Ratio",
) -> plt.Figure:
    """Plot empirical CDF of selected OOS vs pooled OOS performance.

    Parameters
    ----------
    selected_oos : np.ndarray
        OOS metric values for the IS-selected strategy across CSCV splits.
    pooled_oos : np.ndarray
        All OOS metric values (benchmark distribution).
    save_path : str or Path, optional
        If provided, save the figure to this path.
    title : str, optional
        Figure title.
    metric_label : str, optional
        Label for the x-axis.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for arr, label, color in [
        (selected_oos, "Selected OOS", "steelblue"),
        (pooled_oos, "Pooled OOS", "darkorange"),
    ]:
        sorted_arr = np.sort(arr)
        cdf = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr)
        ax.plot(sorted_arr, cdf, label=label, color=color, linewidth=2)

    ax.axvline(x=0.0, color="crimson", linestyle="--", linewidth=1.5, label="Zero threshold")
    ax.set_xlabel(f"OOS {metric_label}", fontsize=12)
    ax.set_ylabel("Empirical CDF", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_cumulative_performance(
    strategy_returns: np.ndarray,
    underlying_returns: np.ndarray | None = None,
    save_path: str | Path | None = None,
    title: str = "Cumulative Strategy Performance",
    strategy_label: str = "Best Strategy",
) -> plt.Figure:
    """Plot cumulative return of the best strategy and the underlying price series.

    Parameters
    ----------
    strategy_returns : np.ndarray
        Daily return series of the selected strategy.
    underlying_returns : np.ndarray, optional
        Daily return series of the underlying asset/price process.
    save_path : str or Path, optional
        If provided, save the figure to this path.
    title : str, optional
        Figure title.
    strategy_label : str, optional
        Label for the strategy line.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    cum_strategy = np.cumsum(strategy_returns)
    ax.plot(cum_strategy, label=strategy_label, color="steelblue", linewidth=2)

    if underlying_returns is not None:
        cum_underlying = np.cumsum(underlying_returns)
        ax.plot(cum_underlying, label="Underlying", color="darkorange",
                linewidth=1.5, alpha=0.75)

    ax.axhline(y=0.0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Trading Day", fontsize=12)
    ax.set_ylabel("Cumulative Return", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig
