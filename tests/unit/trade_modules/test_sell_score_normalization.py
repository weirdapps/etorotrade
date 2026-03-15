"""Tests for SELL score normalization constants and component scoring bounds.

Verifies that each scoring component in calculate_sell_score stays within 0-100
after normalization, and that the named constants match the actual max raw scores.
"""

import numpy as np
import pytest

from trade_modules.analysis.signals import (
    calculate_sell_score,
    SELL_ANALYST_MAX_RAW,
    SELL_MOMENTUM_MAX_RAW,
    SELL_VALUATION_MAX_RAW,
    SELL_FUNDAMENTAL_MAX_RAW,
)


DEFAULT_CONFIG = {
    'weight_analyst': 0.315,
    'weight_momentum': 0.225,
    'weight_valuation': 0.18,
    'weight_fundamental': 0.18,
    'weight_analyst_momentum': 0.10,
}


class TestNormalizationConstants:
    """Verify the named constants match documented derivations."""

    def test_analyst_max_raw(self):
        # upside penalty max 40 + buy% penalty max 40 + EXRET penalty max 20
        assert SELL_ANALYST_MAX_RAW == 100

    def test_momentum_max_raw(self):
        # 52-week decline max 50 + PEF/PET deterioration max 35
        assert SELL_MOMENTUM_MAX_RAW == 85

    def test_valuation_max_raw(self):
        # high PE penalty max 40 + negative PE penalty max 30
        assert SELL_VALUATION_MAX_RAW == 70

    def test_fundamental_max_raw(self):
        # low ROE penalty max 40 + high D/E penalty max 40
        assert SELL_FUNDAMENTAL_MAX_RAW == 80


class TestSellScoreComponentBounds:
    """Verify each component normalizes to 0-100 range."""

    def test_all_zeros_gives_zero(self):
        score, factors = calculate_sell_score(
            upside=50, buy_pct=90, exret=20,
            pct_52w=95, pef=15, pet=20,
            roe=25, de=30, sell_scoring_config=DEFAULT_CONFIG,
        )
        assert 0 <= score <= 100
        assert score < 10  # Very bullish inputs should give low sell score

    def test_all_max_gives_high_score(self):
        score, factors = calculate_sell_score(
            upside=-15, buy_pct=20, exret=-5,
            pct_52w=25, pef=100, pet=15,
            roe=-10, de=400, sell_scoring_config=DEFAULT_CONFIG,
        )
        assert 0 <= score <= 100
        assert score > 80  # Very bearish inputs should give high sell score

    def test_score_never_exceeds_100(self):
        # Worst-case scenario for every factor
        score, _ = calculate_sell_score(
            upside=-20, buy_pct=10, exret=-10,
            pct_52w=10, pef=-5, pet=15,
            roe=-50, de=500, sell_scoring_config=DEFAULT_CONFIG,
            analyst_momentum=-20,
        )
        assert score <= 100

    def test_score_never_below_zero(self):
        score, _ = calculate_sell_score(
            upside=100, buy_pct=99, exret=50,
            pct_52w=99, pef=15, pet=20,
            roe=50, de=10, sell_scoring_config=DEFAULT_CONFIG,
            analyst_momentum=20,
        )
        assert score >= 0

    def test_nan_inputs_handled(self):
        score, factors = calculate_sell_score(
            upside=5, buy_pct=50, exret=3,
            pct_52w=np.nan, pef=np.nan, pet=np.nan,
            roe=np.nan, de=np.nan, sell_scoring_config=DEFAULT_CONFIG,
        )
        assert 0 <= score <= 100

    def test_analyst_component_isolation(self):
        # Only analyst factors should contribute; rest are neutral
        score_bad_analyst, _ = calculate_sell_score(
            upside=-15, buy_pct=20, exret=-5,
            pct_52w=np.nan, pef=np.nan, pet=np.nan,
            roe=np.nan, de=np.nan, sell_scoring_config=DEFAULT_CONFIG,
        )
        score_good_analyst, _ = calculate_sell_score(
            upside=30, buy_pct=90, exret=20,
            pct_52w=np.nan, pef=np.nan, pet=np.nan,
            roe=np.nan, de=np.nan, sell_scoring_config=DEFAULT_CONFIG,
        )
        assert score_bad_analyst > score_good_analyst

    def test_momentum_component_isolation(self):
        # Only momentum factors should differ
        score_bad_momentum, _ = calculate_sell_score(
            upside=15, buy_pct=70, exret=10,
            pct_52w=25, pef=50, pet=20,
            roe=np.nan, de=np.nan, sell_scoring_config=DEFAULT_CONFIG,
        )
        score_good_momentum, _ = calculate_sell_score(
            upside=15, buy_pct=70, exret=10,
            pct_52w=95, pef=18, pet=20,
            roe=np.nan, de=np.nan, sell_scoring_config=DEFAULT_CONFIG,
        )
        assert score_bad_momentum > score_good_momentum

    def test_fundamental_component_isolation(self):
        score_bad_fund, _ = calculate_sell_score(
            upside=15, buy_pct=70, exret=10,
            pct_52w=np.nan, pef=np.nan, pet=np.nan,
            roe=-10, de=400, sell_scoring_config=DEFAULT_CONFIG,
        )
        score_good_fund, _ = calculate_sell_score(
            upside=15, buy_pct=70, exret=10,
            pct_52w=np.nan, pef=np.nan, pet=np.nan,
            roe=25, de=30, sell_scoring_config=DEFAULT_CONFIG,
        )
        assert score_bad_fund > score_good_fund


class TestSellScoreAMComponent:
    """Test analyst momentum scoring in sell score."""

    def test_declining_am_increases_sell_score(self):
        score_declining, factors = calculate_sell_score(
            upside=5, buy_pct=50, exret=3,
            pct_52w=70, pef=20, pet=18,
            roe=15, de=50, sell_scoring_config=DEFAULT_CONFIG,
            analyst_momentum=-10,
        )
        score_flat, _ = calculate_sell_score(
            upside=5, buy_pct=50, exret=3,
            pct_52w=70, pef=20, pet=18,
            roe=15, de=50, sell_scoring_config=DEFAULT_CONFIG,
            analyst_momentum=0,
        )
        assert score_declining > score_flat

    def test_rising_am_reduces_sell_score(self):
        score_rising, _ = calculate_sell_score(
            upside=5, buy_pct=50, exret=3,
            pct_52w=70, pef=20, pet=18,
            roe=15, de=50, sell_scoring_config=DEFAULT_CONFIG,
            analyst_momentum=10,
        )
        score_declining, _ = calculate_sell_score(
            upside=5, buy_pct=50, exret=3,
            pct_52w=70, pef=20, pet=18,
            roe=15, de=50, sell_scoring_config=DEFAULT_CONFIG,
            analyst_momentum=-10,
        )
        assert score_rising < score_declining

    def test_nan_am_treated_as_neutral(self):
        score_nan, _ = calculate_sell_score(
            upside=5, buy_pct=50, exret=3,
            pct_52w=70, pef=20, pet=18,
            roe=15, de=50, sell_scoring_config=DEFAULT_CONFIG,
            analyst_momentum=np.nan,
        )
        # NaN AM gives 0 score for the AM component,
        # should be lower than declining AM
        score_declining, _ = calculate_sell_score(
            upside=5, buy_pct=50, exret=3,
            pct_52w=70, pef=20, pet=18,
            roe=15, de=50, sell_scoring_config=DEFAULT_CONFIG,
            analyst_momentum=-15,
        )
        assert score_nan < score_declining

    def test_severe_am_decline_factor_reported(self):
        _, factors = calculate_sell_score(
            upside=5, buy_pct=50, exret=3,
            pct_52w=70, pef=20, pet=18,
            roe=15, de=50, sell_scoring_config=DEFAULT_CONFIG,
            analyst_momentum=-15,
        )
        assert any("severe_am_decline" in f for f in factors)

    def test_am_score_tiers(self):
        # Verify scoring tiers are monotonically increasing for declining AM
        scores = []
        for am in [5, 0, -3, -7, -12, -20]:
            score, _ = calculate_sell_score(
                upside=10, buy_pct=60, exret=6,
                pct_52w=80, pef=20, pet=18,
                roe=15, de=50, sell_scoring_config=DEFAULT_CONFIG,
                analyst_momentum=am,
            )
            scores.append(score)
        # Each successive value should be >= the previous (more declining AM = higher sell score)
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1], f"AM scoring not monotonic at index {i}"
