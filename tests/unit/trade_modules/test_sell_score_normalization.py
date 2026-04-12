"""Tests for SELL score continuous scoring (np.interp) and component bounds.

Verifies that each scoring component in calculate_sell_score stays within 0-100,
scores are monotonic, and np.interp produces no cliff artifacts.
"""

import numpy as np

from trade_modules.analysis.signals import calculate_sell_score

DEFAULT_CONFIG = {
    'weight_analyst': 0.20,
    'weight_momentum': 0.25,
    'weight_valuation': 0.20,
    'weight_fundamental': 0.25,
    'weight_analyst_momentum': 0.10,
}

class TestSellScoreComponentBounds:
    """Verify each component normalizes to 0-100 range."""

    def test_all_bullish_gives_low_score(self):
        score, factors = calculate_sell_score(
            upside=50, buy_pct=90,
            pct_52w=95, pef=15, pet=20,
            roe=25, de=30, sell_scoring_config=DEFAULT_CONFIG,
        )
        assert 0 <= score <= 100
        assert score < 10  # Very bullish inputs should give low sell score

    def test_all_bearish_gives_high_score(self):
        score, factors = calculate_sell_score(
            upside=-15, buy_pct=20,
            pct_52w=25, pef=100, pet=90,
            roe=-10, de=400, sell_scoring_config=DEFAULT_CONFIG,
        )
        assert 0 <= score <= 100
        assert score > 70  # Very bearish inputs should give high sell score

    def test_score_never_exceeds_100(self):
        score, _ = calculate_sell_score(
            upside=-20, buy_pct=10,
            pct_52w=10, pef=-5, pet=-10,
            roe=-50, de=500, sell_scoring_config=DEFAULT_CONFIG,
            analyst_momentum=-20,
        )
        assert score <= 100

    def test_score_never_below_zero(self):
        score, _ = calculate_sell_score(
            upside=100, buy_pct=99,
            pct_52w=99, pef=15, pet=20,
            roe=50, de=10, sell_scoring_config=DEFAULT_CONFIG,
            analyst_momentum=20,
        )
        assert score >= 0

    def test_nan_inputs_handled(self):
        score, factors = calculate_sell_score(
            upside=5, buy_pct=50,
            pct_52w=np.nan, pef=np.nan, pet=np.nan,
            roe=np.nan, de=np.nan, sell_scoring_config=DEFAULT_CONFIG,
        )
        assert 0 <= score <= 100

class TestContinuousScoring:
    """Verify np.interp produces continuous, monotonic scores with no cliffs."""

    def test_analyst_monotonic(self):
        """Worse analyst sentiment → higher sell score (monotonic)."""
        scores = []
        for upside in [20, 10, 5, 0, -5, -10]:
            score, _ = calculate_sell_score(
                upside=upside, buy_pct=60,
                pct_52w=np.nan, pef=np.nan, pet=np.nan,
                roe=np.nan, de=np.nan, sell_scoring_config=DEFAULT_CONFIG,
            )
            scores.append(score)
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1], f"Analyst not monotonic: {scores}"

    def test_no_large_cliffs(self):
        """Adjacent upside values should not produce >10pt score jumps."""
        prev_score = None
        for upside_10x in range(-100, 200, 5):  # -10 to 20 in 0.5 steps
            upside = upside_10x / 10.0
            score, _ = calculate_sell_score(
                upside=upside, buy_pct=60,
                pct_52w=70, pef=20, pet=25,
                roe=15, de=50, sell_scoring_config=DEFAULT_CONFIG,
            )
            if prev_score is not None:
                assert abs(score - prev_score) <= 10, \
                    f"Cliff detected at upside={upside}: {prev_score:.1f} → {score:.1f}"
            prev_score = score

    def test_fundamental_isolation(self):
        """Bad fundamentals produce higher sell score than good."""
        score_bad, _ = calculate_sell_score(
            upside=15, buy_pct=70,
            pct_52w=np.nan, pef=np.nan, pet=np.nan,
            roe=-10, de=400, sell_scoring_config=DEFAULT_CONFIG,
        )
        score_good, _ = calculate_sell_score(
            upside=15, buy_pct=70,
            pct_52w=np.nan, pef=np.nan, pet=np.nan,
            roe=25, de=30, sell_scoring_config=DEFAULT_CONFIG,
        )
        assert score_bad > score_good

class TestSellScoreAMComponent:
    """Test analyst momentum scoring in sell score."""

    def test_declining_am_increases_sell_score(self):
        score_declining, _ = calculate_sell_score(
            upside=5, buy_pct=50,
            pct_52w=70, pef=20, pet=18,
            roe=15, de=50, sell_scoring_config=DEFAULT_CONFIG,
            analyst_momentum=-10,
        )
        score_flat, _ = calculate_sell_score(
            upside=5, buy_pct=50,
            pct_52w=70, pef=20, pet=18,
            roe=15, de=50, sell_scoring_config=DEFAULT_CONFIG,
            analyst_momentum=0,
        )
        assert score_declining > score_flat

    def test_rising_am_reduces_sell_score(self):
        score_rising, _ = calculate_sell_score(
            upside=5, buy_pct=50,
            pct_52w=70, pef=20, pet=18,
            roe=15, de=50, sell_scoring_config=DEFAULT_CONFIG,
            analyst_momentum=10,
        )
        score_declining, _ = calculate_sell_score(
            upside=5, buy_pct=50,
            pct_52w=70, pef=20, pet=18,
            roe=15, de=50, sell_scoring_config=DEFAULT_CONFIG,
            analyst_momentum=-10,
        )
        assert score_rising < score_declining

    def test_am_decline_factor_reported(self):
        _, factors = calculate_sell_score(
            upside=5, buy_pct=50,
            pct_52w=70, pef=20, pet=18,
            roe=15, de=50, sell_scoring_config=DEFAULT_CONFIG,
            analyst_momentum=-10,
        )
        assert any("am_decline" in f for f in factors)

    def test_am_score_monotonic(self):
        """More declining AM = higher sell score (monotonic)."""
        scores = []
        for am in [5, 0, -3, -7, -12, -20]:
            score, _ = calculate_sell_score(
                upside=10, buy_pct=60,
                pct_52w=80, pef=20, pet=18,
                roe=15, de=50, sell_scoring_config=DEFAULT_CONFIG,
                analyst_momentum=am,
            )
            scores.append(score)
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1], f"AM scoring not monotonic at index {i}"
