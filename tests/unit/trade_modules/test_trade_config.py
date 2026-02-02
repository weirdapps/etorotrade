#!/usr/bin/env python3
"""
ITERATION 5: TradeConfig and Configuration System Tests
Target: Test tier classification, region detection, threshold retrieval
"""

import pytest
from trade_modules.trade_config import TradeConfig


class TestTierClassification:
    """Test market cap tier classification logic."""

    @pytest.fixture
    def config(self):
        """Create TradeConfig instance."""
        return TradeConfig()

    def test_mega_tier_classification(self, config):
        """Classify ≥$500B as MEGA tier."""
        assert config.get_tier_from_market_cap(500_000_000_000) == "mega"
        assert config.get_tier_from_market_cap(3_000_000_000_000) == "mega"

    def test_large_tier_classification(self, config):
        """Classify $100B-$500B as LARGE tier."""
        assert config.get_tier_from_market_cap(100_000_000_000) == "large"
        assert config.get_tier_from_market_cap(250_000_000_000) == "large"
        assert config.get_tier_from_market_cap(499_999_999_999) == "large"

    def test_mid_tier_classification(self, config):
        """Classify $10B-$100B as MID tier."""
        assert config.get_tier_from_market_cap(10_000_000_000) == "mid"
        assert config.get_tier_from_market_cap(50_000_000_000) == "mid"
        assert config.get_tier_from_market_cap(99_999_999_999) == "mid"

    def test_small_tier_classification(self, config):
        """Classify $2B-$10B as SMALL tier."""
        assert config.get_tier_from_market_cap(2_000_000_000) == "small"
        assert config.get_tier_from_market_cap(5_000_000_000) == "small"
        assert config.get_tier_from_market_cap(9_999_999_999) == "small"

    def test_micro_tier_classification(self, config):
        """Classify <$2B as MICRO tier."""
        assert config.get_tier_from_market_cap(1_999_999_999) == "micro"
        assert config.get_tier_from_market_cap(500_000_000) == "micro"
        assert config.get_tier_from_market_cap(100_000_000) == "micro"

    def test_tier_boundary_precision(self, config):
        """Test exact boundary values."""
        # Exact boundaries
        assert config.get_tier_from_market_cap(500_000_000_000) == "mega"  # ≥500B
        assert config.get_tier_from_market_cap(499_999_999_999) == "large"  # <500B
        assert config.get_tier_from_market_cap(100_000_000_000) == "large"  # ≥100B
        assert config.get_tier_from_market_cap(99_999_999_999) == "mid"  # <100B
        assert config.get_tier_from_market_cap(10_000_000_000) == "mid"  # ≥10B
        assert config.get_tier_from_market_cap(9_999_999_999) == "small"  # <10B
        assert config.get_tier_from_market_cap(2_000_000_000) == "small"  # ≥2B
        assert config.get_tier_from_market_cap(1_999_999_999) == "micro"  # <2B


class TestRegionDetection:
    """Test region detection from ticker suffixes."""

    @pytest.fixture
    def config(self):
        """Create TradeConfig instance."""
        return TradeConfig()

    def test_us_region_no_suffix(self, config):
        """Detect US region for tickers without suffix."""
        assert config.get_region_from_ticker("AAPL") == "us"
        assert config.get_region_from_ticker("MSFT") == "us"
        assert config.get_region_from_ticker("GOOGL") == "us"

    def test_eu_region_de_suffix(self, config):
        """Detect EU region for .DE suffix."""
        assert config.get_region_from_ticker("SAP.DE") == "eu"
        assert config.get_region_from_ticker("BMW.DE") == "eu"

    def test_eu_region_pa_suffix(self, config):
        """Detect EU region for .PA (Paris) suffix."""
        assert config.get_region_from_ticker("MC.PA") == "eu"
        assert config.get_region_from_ticker("OR.PA") == "eu"

    def test_hk_region_hk_suffix(self, config):
        """Detect HK region for .HK suffix."""
        assert config.get_region_from_ticker("BABA.HK") == "hk"
        assert config.get_region_from_ticker("0700.HK") == "hk"

    def test_case_insensitive_suffix(self, config):
        """Region detection should be case-insensitive."""
        assert config.get_region_from_ticker("SAP.de") == "eu"
        assert config.get_region_from_ticker("BABA.hk") == "hk"


class TestThresholdRetrieval:
    """Test threshold retrieval for different tier×region combinations."""

    @pytest.fixture
    def config(self):
        """Create TradeConfig instance."""
        return TradeConfig()

    def test_get_tier_thresholds_mega_us_buy(self, config):
        """Retrieve MEGA-US BUY thresholds."""
        thresholds = config.get_tier_thresholds("mega", "buy")

        assert "min_upside" in thresholds
        assert "min_buy_percentage" in thresholds
        assert "min_exret" in thresholds
        assert isinstance(thresholds["min_upside"], (int, float))

    def test_get_tier_thresholds_mega_us_sell(self, config):
        """Retrieve MEGA-US SELL thresholds."""
        thresholds = config.get_tier_thresholds("mega", "sell")

        assert "max_upside" in thresholds
        assert isinstance(thresholds["max_upside"], (int, float))

    def test_universal_thresholds(self, config):
        """Check universal thresholds exist."""
        assert hasattr(config, "UNIVERSAL_THRESHOLDS")
        assert "min_analyst_count" in config.UNIVERSAL_THRESHOLDS
        assert "min_price_targets" in config.UNIVERSAL_THRESHOLDS
        # Standard analyst count for $5B+ stocks
        assert config.UNIVERSAL_THRESHOLDS["min_analyst_count"] == 4
        assert config.UNIVERSAL_THRESHOLDS["min_price_targets"] == 4
        # $2B hard floor for market cap
        assert config.UNIVERSAL_THRESHOLDS["min_market_cap"] == 2_000_000_000
        # Tiered analyst requirements
        assert config.UNIVERSAL_THRESHOLDS["small_cap_threshold"] == 5_000_000_000
        assert config.UNIVERSAL_THRESHOLDS["small_cap_min_analysts"] == 6


class TestTierThresholdConsistency:
    """Ensure threshold consistency across tiers."""

    @pytest.fixture
    def config(self):
        """Create TradeConfig instance."""
        return TradeConfig()

    def test_buy_thresholds_increase_with_risk(self, config):
        """Higher-risk tiers should have stricter (higher) BUY thresholds.

        Risk order: MEGA < LARGE < MID < SMALL < MICRO
        So: SMALL.min_upside > MID.min_upside > LARGE.min_upside > MEGA.min_upside
        """
        mega_buy = config.get_tier_thresholds("mega", "buy")
        large_buy = config.get_tier_thresholds("large", "buy")
        mid_buy = config.get_tier_thresholds("mid", "buy")
        small_buy = config.get_tier_thresholds("small", "buy")

        # Upside requirements should increase with risk
        assert small_buy.get("min_upside", 0) >= mid_buy.get("min_upside", 0)
        assert mid_buy.get("min_upside", 0) >= large_buy.get("min_upside", 0)
        assert large_buy.get("min_upside", 0) >= mega_buy.get("min_upside", 0)

    def test_sell_thresholds_present(self, config):
        """All tiers should have SELL thresholds."""
        for tier in ["mega", "large", "mid", "small"]:
            sell_thresholds = config.get_tier_thresholds(tier, "sell")
            assert "max_upside" in sell_thresholds, f"{tier} SELL should have max_upside"
