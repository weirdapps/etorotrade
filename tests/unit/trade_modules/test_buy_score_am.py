"""Tests for analyst momentum (AM) integration in BUY conviction scoring."""

import numpy as np

from trade_modules.analysis.signals import calculate_buy_score

DEFAULT_CONFIG = {
    "weight_upside": 0.22,
    "weight_consensus": 0.13,
    "weight_momentum": 0.20,
    "weight_valuation": 0.18,
    "weight_fundamental": 0.17,
    "weight_analyst_momentum": 0.10,
}

# Common baseline inputs for isolation tests
BASELINE = {
    "upside": 20,
    "buy_pct": 80,
    "pct_52w": 85,
    "price_200dma_pct": 110,
    "pef": 18,
    "pet": 20,
    "roe": 20,
    "de": 40,
    "fcf_yield": 5,
    "buy_scoring_config": DEFAULT_CONFIG,
}


class TestBuyScoreAMComponent:
    """Test analyst momentum scoring in buy conviction score."""

    def test_rising_am_increases_buy_score(self):
        score_rising = calculate_buy_score(**BASELINE, analyst_momentum=10)
        score_flat = calculate_buy_score(**BASELINE, analyst_momentum=0)
        assert score_rising > score_flat

    def test_declining_am_decreases_buy_score(self):
        score_declining = calculate_buy_score(**BASELINE, analyst_momentum=-10)
        score_flat = calculate_buy_score(**BASELINE, analyst_momentum=0)
        assert score_declining < score_flat

    def test_nan_am_gives_neutral_score(self):
        score_nan = calculate_buy_score(**BASELINE, analyst_momentum=np.nan)
        score_neutral = calculate_buy_score(**BASELINE, analyst_momentum=0)
        # NaN defaults to 50 (neutral), same as AM=0
        assert abs(score_nan - score_neutral) < 0.01

    def test_score_stays_in_range(self):
        for am in [-20, -10, -5, 0, 5, 10, 20]:
            score = calculate_buy_score(**BASELINE, analyst_momentum=am)
            assert 0 <= score <= 100, f"Score {score} out of range for AM={am}"

    def test_am_score_tiers_monotonic(self):
        scores = []
        for am in [-15, -10, -5, 0, 5, 10, 15]:
            score = calculate_buy_score(**BASELINE, analyst_momentum=am)
            scores.append(score)
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1], f"AM buy scoring not monotonic at index {i}"

    def test_backward_compatible_without_am(self):
        # Calling without analyst_momentum should work (defaults to NaN)
        score = calculate_buy_score(**BASELINE)
        assert 0 <= score <= 100

    def test_weights_sum_to_one(self):
        total = (
            DEFAULT_CONFIG["weight_upside"]
            + DEFAULT_CONFIG["weight_consensus"]
            + DEFAULT_CONFIG["weight_momentum"]
            + DEFAULT_CONFIG["weight_valuation"]
            + DEFAULT_CONFIG["weight_fundamental"]
            + DEFAULT_CONFIG["weight_analyst_momentum"]
        )
        assert abs(total - 1.0) < 0.001

    def test_strong_am_improves_marginal_buy(self):
        # A stock with modest fundamentals but strong rising AM
        modest = {
            "upside": 10,
            "buy_pct": 70,
            "pct_52w": 75,
            "price_200dma_pct": 105,
            "pef": 25,
            "pet": 22,
            "roe": 12,
            "de": 80,
            "fcf_yield": 2,
            "buy_scoring_config": DEFAULT_CONFIG,
        }
        score_rising = calculate_buy_score(**modest, analyst_momentum=15)
        score_declining = calculate_buy_score(**modest, analyst_momentum=-15)
        # The 10% AM weight should create meaningful differentiation
        assert score_rising - score_declining >= 5
