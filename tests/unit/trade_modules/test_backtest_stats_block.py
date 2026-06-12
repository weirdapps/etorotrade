"""Tests for block bootstrap CI in backtest_stats."""

import numpy as np

from trade_modules.backtest_stats import block_bootstrap_ci, bootstrap_ci


class TestBlockBootstrapCI:
    def test_returns_tuple_of_floats(self):
        data = np.random.default_rng(42).standard_normal(100)
        lo, hi = block_bootstrap_ci(data, block_size=30, n_boot=500)
        assert isinstance(lo, float)
        assert isinstance(hi, float)
        assert lo < hi

    def test_wider_than_iid_for_autocorrelated(self):
        """Block bootstrap CI should be >= i.i.d. CI for AR(1) data."""
        rng = np.random.default_rng(42)
        n = 300
        ar = np.empty(n)
        ar[0] = 0.0
        for i in range(1, n):
            ar[i] = 0.9 * ar[i - 1] + rng.standard_normal()

        iid_lo, iid_hi = bootstrap_ci(ar, seed=42)
        blk_lo, blk_hi = block_bootstrap_ci(ar, block_size=30, seed=42)
        iid_width = iid_hi - iid_lo
        blk_width = blk_hi - blk_lo
        # Block should generally be wider for autocorrelated data.
        # Use a generous tolerance since bootstrap is stochastic.
        assert blk_width >= iid_width * 0.8

    def test_falls_back_for_small_data(self):
        """When data < block_size, should fall back to i.i.d. bootstrap."""
        data = np.array([1.0, 2.0, 3.0])
        lo, hi = block_bootstrap_ci(data, block_size=30)
        # Should work without error (falls back to bootstrap_ci)
        assert lo < hi

    def test_empty_data_returns_nan(self):
        lo, hi = block_bootstrap_ci(np.array([]), block_size=30)
        assert np.isnan(lo)
        assert np.isnan(hi)

    def test_deterministic_with_seed(self):
        data = np.random.default_rng(99).standard_normal(200)
        lo1, hi1 = block_bootstrap_ci(data, block_size=20, seed=123)
        lo2, hi2 = block_bootstrap_ci(data, block_size=20, seed=123)
        assert lo1 == lo2
        assert hi1 == hi2

    def test_different_seeds_differ(self):
        data = np.random.default_rng(99).standard_normal(200)
        lo1, hi1 = block_bootstrap_ci(data, block_size=20, seed=1)
        lo2, hi2 = block_bootstrap_ci(data, block_size=20, seed=2)
        assert (lo1 != lo2) or (hi1 != hi2)

    def test_accepts_list_input(self):
        """Should accept plain Python lists, not only ndarrays."""
        data = list(np.random.default_rng(42).standard_normal(100))
        lo, hi = block_bootstrap_ci(data, block_size=20, n_boot=500)
        assert lo < hi

    def test_ci_contains_sample_mean(self):
        rng = np.random.default_rng(42)
        data = rng.normal(10, 1, 200)
        lo, hi = block_bootstrap_ci(data, block_size=20)
        assert lo <= np.mean(data) <= hi

    def test_custom_stat_fn(self):
        data = np.random.default_rng(42).standard_normal(120)
        lo, hi = block_bootstrap_ci(data, block_size=20, stat_fn=np.median)
        assert lo < hi

    def test_block_size_equals_data_length(self):
        """Edge case: block_size == len(data) gives exactly 1 block per resample."""
        data = np.random.default_rng(42).standard_normal(30)
        lo, hi = block_bootstrap_ci(data, block_size=30, n_boot=500)
        # With 1 block, every resample is the same slice → degenerate CI
        assert lo == hi  # Only one possible block to choose
