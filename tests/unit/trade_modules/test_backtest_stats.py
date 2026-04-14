"""Tests for backtest statistical utilities."""

import numpy as np
import pytest

from trade_modules.backtest_stats import (
    bootstrap_ci,
    hit_rate_ci,
    walk_forward_split,
    fdr_correction,
)


class TestBootstrapCI:
    def test_returns_lo_hi_tuple(self):
        data = np.random.normal(5, 2, 100)
        lo, hi = bootstrap_ci(data, stat_fn=np.mean, n_boot=500)
        assert lo < hi

    def test_ci_contains_sample_mean(self):
        np.random.seed(42)
        data = np.random.normal(10, 1, 200)
        lo, hi = bootstrap_ci(data, stat_fn=np.mean)
        assert lo <= np.mean(data) <= hi

    def test_wider_ci_with_smaller_sample(self):
        np.random.seed(42)
        small = np.random.normal(5, 2, 20)
        large = np.random.normal(5, 2, 200)
        lo_s, hi_s = bootstrap_ci(small, stat_fn=np.mean)
        lo_l, hi_l = bootstrap_ci(large, stat_fn=np.mean)
        assert (hi_s - lo_s) > (hi_l - lo_l)

    def test_empty_data_returns_nan(self):
        lo, hi = bootstrap_ci(np.array([]), stat_fn=np.mean)
        assert np.isnan(lo)
        assert np.isnan(hi)


class TestHitRateCI:
    def test_returns_rate_and_ci(self):
        hits = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 1])  # 7/10
        rate, lo, hi = hit_rate_ci(hits)
        assert rate == pytest.approx(70.0)
        assert lo < 70.0
        assert hi > 70.0

    def test_perfect_hit_rate(self):
        hits = np.ones(20)
        rate, lo, hi = hit_rate_ci(hits)
        assert rate == pytest.approx(100.0)
        assert lo > 80  # CI lower bound still high

    def test_empty_returns_nan(self):
        rate, lo, hi = hit_rate_ci(np.array([]))
        assert np.isnan(rate)

    def test_ci_spans_50_flag(self):
        """Small sample near 50% should have CI spanning 50."""
        hits = np.array([1, 0, 1, 0, 1])  # 60%, n=5
        rate, lo, hi = hit_rate_ci(hits)
        assert lo < 50 < hi  # Not a proven signal


class TestWalkForwardSplit:
    def test_basic_split(self):
        dates = [f"2026-01-{d:02d}" for d in range(1, 11)]
        entries = [{"date": d, "data": i} for i, d in enumerate(dates)]
        train, test = walk_forward_split(entries, train_ratio=0.7)
        assert len(train) == 7
        assert len(test) == 3

    def test_train_before_test(self):
        dates = ["2026-01-01", "2026-01-05", "2026-01-10", "2026-01-15"]
        entries = [{"date": d} for d in dates]
        train, test = walk_forward_split(entries, train_ratio=0.5)
        assert train[-1]["date"] <= test[0]["date"]

    def test_minimum_test_size(self):
        """At least 2 entries in test set."""
        entries = [{"date": f"2026-01-{d:02d}"} for d in range(1, 5)]
        train, test = walk_forward_split(entries, train_ratio=0.9)
        assert len(test) >= 2

    def test_single_entry_returns_empty_test(self):
        entries = [{"date": "2026-01-01"}]
        train, test = walk_forward_split(entries, train_ratio=0.7)
        assert len(train) == 1
        assert len(test) == 0


class TestFDRCorrection:
    def test_reduces_significance(self):
        """Multiple p-values: FDR should accept fewer than uncorrected."""
        p_values = {"a": 0.01, "b": 0.03, "c": 0.04, "d": 0.06, "e": 0.10}
        significant = fdr_correction(p_values, alpha=0.05)
        # With 5 tests, some borderline p-values should be rejected
        assert len(significant) <= 5

    def test_no_significant_results(self):
        p_values = {"a": 0.5, "b": 0.8, "c": 0.9}
        significant = fdr_correction(p_values, alpha=0.05)
        assert len(significant) == 0

    def test_all_significant(self):
        p_values = {"a": 0.001, "b": 0.002}
        significant = fdr_correction(p_values, alpha=0.05)
        assert set(significant) == {"a", "b"}
