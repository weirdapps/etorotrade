from trade_modules.trailing_stop import (
    TrailingStopConfig,
    _threshold_for_tier,
    check_portfolio_stops,
    check_trailing_stop,
)


class TestThresholdForTier:
    def test_mega_wider_band(self):
        config = TrailingStopConfig()
        assert _threshold_for_tier("MEGA", False, config) == 25.0

    def test_mid_default(self):
        config = TrailingStopConfig()
        assert _threshold_for_tier("MID", False, config) == 20.0

    def test_crypto_widest(self):
        config = TrailingStopConfig()
        assert _threshold_for_tier("MID", True, config) == 35.0

    def test_micro_tighter(self):
        config = TrailingStopConfig()
        assert _threshold_for_tier("MICRO", False, config) == 15.0

    def test_unknown_tier_uses_default(self):
        config = TrailingStopConfig()
        assert _threshold_for_tier("UNKNOWN", False, config) == 20.0


class TestCheckTrailingStop:
    def test_no_trigger_within_band(self):
        result = check_trailing_stop("AAPL", 180.0, 200.0, tier="MEGA")
        assert result.triggered is False
        assert result.drawdown_from_high_pct == 10.0

    def test_trigger_beyond_band(self):
        result = check_trailing_stop("AAPL", 155.0, 200.0, tier="MID")
        assert result.triggered is True
        assert result.drawdown_from_high_pct == 22.5

    def test_mega_cap_wider_band(self):
        # 22.5% drawdown — within MEGA's 25% band
        result = check_trailing_stop("AAPL", 155.0, 200.0, tier="MEGA")
        assert result.triggered is False

    def test_crypto_detected_by_suffix(self):
        result = check_trailing_stop("BTC-USD", 40000, 60000)
        # 33.3% drawdown, crypto threshold 35% — not triggered
        assert result.triggered is False

    def test_crypto_triggered(self):
        result = check_trailing_stop("ETH-USD", 1800, 3000)
        # 40% drawdown > 35% crypto threshold
        assert result.triggered is True

    def test_disabled_config(self):
        config = TrailingStopConfig(enabled=False)
        result = check_trailing_stop("AAPL", 100, 200, config=config)
        assert result.triggered is False

    def test_zero_high(self):
        result = check_trailing_stop("AAPL", 100, 0)
        assert result.triggered is False

    def test_exact_threshold(self):
        # Exactly at 20% drawdown — should trigger (>=)
        result = check_trailing_stop("TEST", 80.0, 100.0, tier="MID")
        assert result.triggered is True
        assert result.drawdown_from_high_pct == 20.0

    def test_result_carries_ticker(self):
        result = check_trailing_stop("NVDA", 100, 120, tier="MEGA")
        assert result.ticker == "NVDA"


class TestCheckPortfolioStops:
    def test_portfolio_batch(self):
        positions = [
            {"ticker": "AAPL", "current_price": 180, "high_since_entry": 200, "tier": "MEGA"},
            {"ticker": "SMALL1", "current_price": 5, "high_since_entry": 10, "tier": "SMALL"},
        ]
        results = check_portfolio_stops(positions)
        assert len(results) == 2
        assert results[0].triggered is False  # AAPL: 10% < 25% MEGA
        assert results[1].triggered is True  # SMALL1: 50% > 20%

    def test_empty_portfolio(self):
        results = check_portfolio_stops([])
        assert results == []
