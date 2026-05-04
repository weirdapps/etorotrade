"""
Statistical Utilities for Backtesting

Provides:
- Bootstrap confidence intervals for hit rates and returns
- Walk-forward train/test splitting
- Benjamini-Hochberg FDR correction for multiple testing
"""

from collections.abc import Callable
from typing import Any

import numpy as np


def bootstrap_ci(
    data: np.ndarray,
    stat_fn: Callable = np.mean,
    n_boot: int = 2000,
    ci: float = 0.90,
    seed: int | None = None,
) -> tuple[float, float]:
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        data: 1D array of observations.
        stat_fn: Statistic function (e.g., np.mean, np.median).
        n_boot: Number of bootstrap resamples.
        ci: Confidence level (default 0.90 = 90% CI).
        seed: Random seed for reproducibility.

    Returns:
        (lower_bound, upper_bound) of the CI.
    """
    if len(data) == 0:
        return (np.nan, np.nan)

    rng = np.random.default_rng(seed)
    boot_stats = np.array(
        [stat_fn(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    )

    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_stats, alpha * 100))
    hi = float(np.percentile(boot_stats, (1 - alpha) * 100))
    return (lo, hi)


def hit_rate_ci(
    hits: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.90,
) -> tuple[float, float, float]:
    """
    Compute hit rate with bootstrap confidence interval.

    Args:
        hits: 1D array of 0s and 1s (1 = hit, 0 = miss).
        n_boot: Number of bootstrap resamples.
        ci: Confidence level.

    Returns:
        (hit_rate_pct, ci_lower_pct, ci_upper_pct)
    """
    if len(hits) == 0:
        return (np.nan, np.nan, np.nan)

    rate = float(np.mean(hits)) * 100
    lo, hi = bootstrap_ci(hits, stat_fn=lambda x: np.mean(x) * 100, n_boot=n_boot, ci=ci)
    return (rate, lo, hi)


def walk_forward_split(
    entries: list[dict[str, Any]],
    train_ratio: float = 0.7,
    date_key: str = "date",
    min_test: int = 2,
) -> tuple[list[dict], list[dict]]:
    """
    Split time-ordered entries into train and test sets.

    Ensures temporal ordering: all train entries come before test entries.

    Args:
        entries: List of dicts with a date key.
        train_ratio: Fraction of data for training (default 0.7).
        date_key: Key name for date field.
        min_test: Minimum entries in test set.

    Returns:
        (train_entries, test_entries)
    """
    if len(entries) <= 1:
        return (list(entries), [])

    sorted_entries = sorted(entries, key=lambda e: e.get(date_key, ""))
    split_idx = max(
        1,
        min(
            int(len(sorted_entries) * train_ratio),
            len(sorted_entries) - min_test,
        ),
    )

    return (sorted_entries[:split_idx], sorted_entries[split_idx:])


def fdr_correction(
    p_values: dict[str, float],
    alpha: float = 0.05,
) -> list[str]:
    """
    Benjamini-Hochberg FDR correction for multiple testing.

    Args:
        p_values: Dict mapping test name to p-value.
        alpha: Family-wise error rate.

    Returns:
        List of test names that remain significant after correction.
    """
    if not p_values:
        return []

    sorted_tests = sorted(p_values.items(), key=lambda x: x[1])
    m = len(sorted_tests)
    significant = []

    for rank, (name, p) in enumerate(sorted_tests, start=1):
        threshold = alpha * rank / m
        if p <= threshold:
            significant.append(name)
        else:
            break  # All subsequent are rejected too

    return significant
