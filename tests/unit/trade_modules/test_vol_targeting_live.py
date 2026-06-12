"""Tests for estimate_portfolio_vol_from_history and compute_vol_scale."""

import numpy as np

from trade_modules.vol_targeting import (
    compute_vol_scale,
    estimate_portfolio_vol_from_history,
)


class TestEstimatePortfolioVolFromHistory:
    def test_estimate_returns_float(self):
        np.random.seed(42)
        weights = {"A": 0.4, "B": 0.3, "C": 0.3}
        returns = {t: np.random.randn(120) * 0.02 for t in weights}
        vol = estimate_portfolio_vol_from_history(weights, returns)
        assert vol is not None
        assert 0.05 < vol < 0.80

    def test_estimate_none_on_insufficient_data(self):
        weights = {"A": 0.5, "B": 0.5}
        returns = {"A": np.array([0.01, 0.02]), "B": np.array([0.01])}
        vol = estimate_portfolio_vol_from_history(weights, returns)
        assert vol is None

    def test_estimate_none_on_single_ticker(self):
        weights = {"A": 1.0}
        returns = {"A": np.random.randn(100) * 0.02}
        vol = estimate_portfolio_vol_from_history(weights, returns)
        assert vol is None

    def test_estimate_none_when_no_returns(self):
        weights = {"A": 0.5, "B": 0.5}
        vol = estimate_portfolio_vol_from_history(weights, None)
        assert vol is None

    def test_estimate_none_when_empty_weights(self):
        returns = {"A": np.random.randn(100) * 0.02}
        vol = estimate_portfolio_vol_from_history({}, returns)
        assert vol is None


class TestComputeVolScale:
    def test_none_returns_1(self):
        assert compute_vol_scale(None) == 1.0

    def test_at_target(self):
        assert abs(compute_vol_scale(0.12) - 1.0) < 0.01

    def test_high_vol_scales_down(self):
        scale = compute_vol_scale(0.24, target_vol=0.12)
        assert scale == 0.5  # 0.12/0.24 = 0.5

    def test_low_vol_scales_up(self):
        scale = compute_vol_scale(0.08, target_vol=0.12)
        assert scale == 1.5  # 0.12/0.08 = 1.5, capped at 1.5

    def test_clamped_floor(self):
        assert compute_vol_scale(1.0, target_vol=0.12) == 0.5  # floor

    def test_clamped_cap(self):
        assert compute_vol_scale(0.01, target_vol=0.12) == 1.5  # cap

    def test_zero_vol_returns_1(self):
        assert compute_vol_scale(0.0) == 1.0

    def test_negative_vol_returns_1(self):
        assert compute_vol_scale(-0.05) == 1.0
