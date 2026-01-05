#!/usr/bin/env python3
"""
ITERATION 8: Config Interfaces Tests
Target: Test configuration interfaces and dependency injection context
"""

import pytest
from trade_modules.config_interfaces import (
    TradeAction,
    IConfigProvider,
    ITradingCriteriaProvider,
    IDisplayProvider,
    IRateLimitProvider,
    IPathProvider,
    ConfigurationContext,
    global_config_context,
    get_config_context,
)


class TestTradeAction:
    """Test TradeAction enum."""

    def test_trade_action_values(self):
        """Verify trade action enum values."""
        assert TradeAction.BUY.value == "B"
        assert TradeAction.SELL.value == "S"
        assert TradeAction.HOLD.value == "H"
        assert TradeAction.INCONCLUSIVE.value == "I"

    def test_trade_action_members(self):
        """Verify all trade action members exist."""
        actions = [action.name for action in TradeAction]
        assert "BUY" in actions
        assert "SELL" in actions
        assert "HOLD" in actions
        assert "INCONCLUSIVE" in actions


class MockConfigProvider(IConfigProvider):
    """Mock configuration provider for testing."""

    def get_setting(self, key: str, default=None):
        """Get a configuration setting."""
        return "mock_value"

    def get_file_path(self, file_key: str) -> str:
        """Get a file path from configuration."""
        return f"/mock/path/{file_key}"


class MockTradingCriteriaProvider(ITradingCriteriaProvider):
    """Mock trading criteria provider for testing."""

    def get_thresholds(self, option: str, action: str, tier: str = None):
        """Get trading thresholds."""
        return {"min_upside": 10.0}

    def get_universal_thresholds(self):
        """Get universal thresholds."""
        return {"min_analysts": 4}

    def get_tier_thresholds(self, tier: str, action: str):
        """Get tier-specific thresholds."""
        return {"tier": tier, "action": action}


class MockDisplayProvider(IDisplayProvider):
    """Mock display provider for testing."""

    def get_display_columns(self, option: str, sub_option: str = None,
                           output_type: str = "console"):
        """Get display columns."""
        return ["ticker", "price", "upside"]

    def get_sort_config(self, option: str, sub_option: str = None):
        """Get sorting configuration."""
        return {"column": "upside", "reverse": True}

    def get_format_rule(self, column: str):
        """Get formatting rule for a column."""
        return {"precision": 2}


class MockRateLimitProvider(IRateLimitProvider):
    """Mock rate limit provider for testing."""

    def get_rate_limit_config(self):
        """Get rate limiting configuration."""
        return {"requests_per_second": 10}

    def get_concurrent_limits(self):
        """Get concurrent request limits."""
        return {"max_concurrent": 25}


class MockPathProvider(IPathProvider):
    """Mock path provider for testing."""

    def get_input_dir(self) -> str:
        """Get input directory path."""
        return "/mock/input"

    def get_output_dir(self) -> str:
        """Get output directory path."""
        return "/mock/output"

    def get_log_dir(self) -> str:
        """Get log directory path."""
        return "/mock/logs"


class TestConfigurationContext:
    """Test ConfigurationContext dependency injection."""

    @pytest.fixture
    def context(self):
        """Create fresh ConfigurationContext."""
        return ConfigurationContext()

    def test_initialize_context(self, context):
        """Context initializes with None providers."""
        assert context.config is None
        assert context.trading_criteria is None
        assert context.display is None
        assert context.rate_limit is None
        assert context.paths is None

    def test_set_config_provider(self, context):
        """Set and retrieve config provider."""
        provider = MockConfigProvider()
        context.set_config_provider(provider)
        assert context.config == provider

    def test_set_trading_criteria_provider(self, context):
        """Set and retrieve trading criteria provider."""
        provider = MockTradingCriteriaProvider()
        context.set_trading_criteria_provider(provider)
        assert context.trading_criteria == provider

    def test_set_display_provider(self, context):
        """Set and retrieve display provider."""
        provider = MockDisplayProvider()
        context.set_display_provider(provider)
        assert context.display == provider

    def test_set_rate_limit_provider(self, context):
        """Set and retrieve rate limit provider."""
        provider = MockRateLimitProvider()
        context.set_rate_limit_provider(provider)
        assert context.rate_limit == provider

    def test_set_path_provider(self, context):
        """Set and retrieve path provider."""
        provider = MockPathProvider()
        context.set_path_provider(provider)
        assert context.paths == provider

    def test_all_providers_can_be_set(self, context):
        """Set all providers at once."""
        config = MockConfigProvider()
        criteria = MockTradingCriteriaProvider()
        display = MockDisplayProvider()
        rate_limit = MockRateLimitProvider()
        paths = MockPathProvider()

        context.set_config_provider(config)
        context.set_trading_criteria_provider(criteria)
        context.set_display_provider(display)
        context.set_rate_limit_provider(rate_limit)
        context.set_path_provider(paths)

        assert context.config == config
        assert context.trading_criteria == criteria
        assert context.display == display
        assert context.rate_limit == rate_limit
        assert context.paths == paths


class TestGlobalConfigContext:
    """Test global configuration context."""

    def test_global_context_exists(self):
        """Global context is accessible."""
        assert global_config_context is not None
        assert isinstance(global_config_context, ConfigurationContext)

    def test_get_config_context(self):
        """get_config_context returns global context."""
        context = get_config_context()
        assert context is global_config_context


class TestMockProviders:
    """Test mock providers implement interfaces correctly."""

    def test_mock_config_provider(self):
        """Mock config provider works."""
        provider = MockConfigProvider()
        assert provider.get_setting("key") == "mock_value"
        assert provider.get_file_path("test") == "/mock/path/test"

    def test_mock_trading_criteria_provider(self):
        """Mock trading criteria provider works."""
        provider = MockTradingCriteriaProvider()
        assert provider.get_thresholds("market", "BUY")["min_upside"] == 10.0
        assert provider.get_universal_thresholds()["min_analysts"] == 4
        assert provider.get_tier_thresholds("MEGA", "BUY")["tier"] == "MEGA"

    def test_mock_display_provider(self):
        """Mock display provider works."""
        provider = MockDisplayProvider()
        assert "ticker" in provider.get_display_columns("market")
        assert provider.get_sort_config("market")["column"] == "upside"
        assert provider.get_format_rule("price")["precision"] == 2

    def test_mock_rate_limit_provider(self):
        """Mock rate limit provider works."""
        provider = MockRateLimitProvider()
        assert provider.get_rate_limit_config()["requests_per_second"] == 10
        assert provider.get_concurrent_limits()["max_concurrent"] == 25

    def test_mock_path_provider(self):
        """Mock path provider works."""
        provider = MockPathProvider()
        assert provider.get_input_dir() == "/mock/input"
        assert provider.get_output_dir() == "/mock/output"
        assert provider.get_log_dir() == "/mock/logs"
