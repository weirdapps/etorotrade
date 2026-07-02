"""
Statistical Utilities for Backtesting

Provides:
- Bootstrap confidence intervals for hit rates and returns
- Walk-forward train/test splitting (simple and rolling expanding-window)
- Benjamini-Hochberg FDR correction for multiple testing
"""

import datetime
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


def block_bootstrap_ci(
    data: np.ndarray,
    block_size: int = 30,
    stat_fn: Callable = np.mean,
    n_boot: int = 2000,
    ci: float = 0.90,
    seed: int | None = None,
) -> tuple[float, float]:
    """
    Non-overlapping block bootstrap for autocorrelated data.

    Resamples contiguous blocks of size `block_size` rather than
    individual observations.  Produces wider (more honest) CIs when
    observations are not independent (e.g. overlapping T+30 windows).

    Falls back to i.i.d. bootstrap when data length < block_size.

    Args:
        data: 1D array of observations.
        block_size: Length of each contiguous block.
        stat_fn: Statistic function (e.g., np.mean).
        n_boot: Number of bootstrap resamples.
        ci: Confidence level (default 0.90 = 90% CI).
        seed: Random seed for reproducibility.

    Returns:
        (lower_bound, upper_bound) of the CI.
    """
    arr = np.asarray(data, dtype=float)
    n = len(arr)

    if n == 0:
        return (np.nan, np.nan)

    if n < block_size:
        return bootstrap_ci(arr, stat_fn=stat_fn, n_boot=n_boot, ci=ci, seed=seed)

    rng = np.random.default_rng(seed)
    n_blocks = max(1, n // block_size)
    block_starts = np.arange(0, n - block_size + 1)

    stats = np.empty(n_boot)
    for i in range(n_boot):
        chosen = rng.choice(block_starts, size=n_blocks, replace=True)
        sample = np.concatenate([arr[s : s + block_size] for s in chosen])
        stats[i] = stat_fn(sample)

    alpha = (1 - ci) / 2
    lo = float(np.percentile(stats, alpha * 100))
    hi = float(np.percentile(stats, (1 - alpha) * 100))
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


def rolling_walk_forward(
    items: list[dict[str, Any]],
    n_folds: int = 5,
    embargo_days: int = 30,
    date_key: str = "signal_date",
) -> list[tuple[list, list]]:
    """
    Expanding-window walk-forward split with purge/embargo buffer.

    Sorts items by date, computes n_folds+1 cut points across unique dates,
    and produces up to n_folds (train, test) pairs where:
      - train = all items with date < cut_k
      - test  = items with date in [cut_k + embargo_days, cut_{k+1})

    Folds whose test set is empty are skipped.

    Args:
        items:        List of dicts each containing a date field (ISO string or date).
        n_folds:      Number of folds to attempt.
        embargo_days: Days to exclude after the train cutoff before test starts.
        date_key:     Key name for the date field.

    Returns:
        List of (train_items, test_items) tuples; folds with empty test omitted.
    """
    if not items:
        return []

    def _to_date(v: Any) -> datetime.date:
        if isinstance(v, datetime.date):
            return v
        return datetime.date.fromisoformat(str(v)[:10])

    # Sort by date
    sorted_items = sorted(items, key=lambda x: _to_date(x[date_key]))
    unique_dates = sorted({_to_date(x[date_key]) for x in sorted_items})
    n_dates = len(unique_dates)

    if n_dates < 2:
        return []

    # Compute n_folds+1 cut points across unique dates (indices, evenly spaced)
    # cuts[0] is index 0, cuts[n_folds] is the last index
    n_cuts = n_folds + 1
    # Spread n_cuts+1 boundary indices across [0, n_dates-1]
    step = n_dates / n_cuts
    cut_indices = [int(round(i * step)) for i in range(n_cuts + 1)]
    # Clamp to valid range
    cut_indices = [min(max(idx, 0), n_dates - 1) for idx in cut_indices]
    cut_dates = [unique_dates[i] for i in cut_indices]

    folds = []
    for k in range(n_folds):
        train_cutoff = cut_dates[k + 1]  # train: date < train_cutoff
        test_start = train_cutoff + datetime.timedelta(days=embargo_days)
        test_end = cut_dates[k + 2]  # test: date < test_end

        train = [x for x in sorted_items if _to_date(x[date_key]) < train_cutoff]
        test = [x for x in sorted_items if test_start <= _to_date(x[date_key]) < test_end]

        if not test:
            continue
        folds.append((train, test))

    return folds
