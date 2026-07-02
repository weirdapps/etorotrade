"""Tests for trade_modules/regime_detector.py.

Focused coverage for the fail-safe behavior of the bear-market dampener:
when `spy_2yr_return` is missing (SPY data outage), `bear_market_active`
must be True so the momentum-BUY track is disabled while the system is blind.
"""

from trade_modules.regime_detector import RegimeDetector


class TestBearMarketActiveFailSafe:
    """`bear_market_active` is FAIL-SAFE: missing data = bear."""

    def _features(self):
        # Neutral feature set — keeps the classifier on a deterministic path.
        return {
            "vix_score": 50.0,
            "term_structure": 50.0,
            "spy_momentum": 50.0,
            "spy_drawdown": 50.0,
            "vix_trend": 50.0,
        }

    def test_missing_spy_2yr_return_treated_as_bear(self):
        detector = RegimeDetector()
        result = detector.classify(features=self._features(), data={})
        assert result["spy_2yr_return"] is None
        assert result["bear_market_active"] is True

    def test_none_spy_2yr_return_treated_as_bear(self):
        detector = RegimeDetector()
        result = detector.classify(features=self._features(), data={"spy_2yr_return": None})
        assert result["bear_market_active"] is True

    def test_negative_spy_2yr_return_is_bear(self):
        detector = RegimeDetector()
        result = detector.classify(features=self._features(), data={"spy_2yr_return": -0.05})
        assert result["bear_market_active"] is True

    def test_positive_spy_2yr_return_is_not_bear(self):
        detector = RegimeDetector()
        result = detector.classify(features=self._features(), data={"spy_2yr_return": 0.15})
        assert result["bear_market_active"] is False

    def test_zero_spy_2yr_return_is_not_bear(self):
        # Boundary: exactly 0% is not negative, so not bear (behavior preserved).
        detector = RegimeDetector()
        result = detector.classify(features=self._features(), data={"spy_2yr_return": 0.0})
        assert result["bear_market_active"] is False
